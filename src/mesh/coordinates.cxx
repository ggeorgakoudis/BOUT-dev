/**************************************************************************
 * Differential geometry
 * Calculates the covariant metric tensor, and christoffel symbol terms
 * given the contravariant metric tensor terms
 **************************************************************************/

#include <bout/assert.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/msg_stack.hxx>
#include <bout/sys/timer.hxx>
#include <bout/utils.hxx>

#include <bout/derivs.hxx>
#include <bout/fft.hxx>
#include <bout/interpolation.hxx>

#include <bout/globals.hxx>

#include "parallel/fci.hxx"
#include "parallel/shiftedmetricinterp.hxx"
#include "bout/ContravariantMetricTensor.hxx"

// use anonymous namespace so this utility function is not available outside this file
namespace {
template <typename T, typename... Ts>
// Use sendY()/sendX() and wait() instead of Mesh::communicate() to ensure we
// don't try to calculate parallel slices as Coordinates are not constructed yet
void communicate(T& t, Ts... ts) {
  FieldGroup g(t, ts...);
  auto h = t.getMesh()->sendY(g);
  t.getMesh()->wait(h);
  h = t.getMesh()->sendX(g);
  t.getMesh()->wait(h);
}

/// Interpolate a Field2D to a new CELL_LOC with interp_to.
/// Communicates to set internal guard cells.
/// Boundary guard cells are set by extrapolating from the grid, like
/// 'free_o3' boundary conditions
/// Corner guard cells are set to BoutNaN
const Field2D interpolateAndExtrapolate(const Field2D& f, CELL_LOC location,
                                        bool extrapolate_x, bool extrapolate_y,
                                        bool no_extra_interpolate,
                                        ParallelTransform* UNUSED(pt) = nullptr) {

  Mesh* localmesh = f.getMesh();
  Field2D result = interp_to(f, location, "RGN_NOBNDRY");
  // Ensure result's data is unique. Otherwise result might be a duplicate of
  // f (if no interpolation is needed, e.g. if interpolation is in the
  // z-direction); then f would be communicated. Since this function is used
  // on geometrical quantities that might not be periodic in y even on closed
  // field lines (due to dependence on integrated shear), we don't want to
  // communicate f. We will sort out result's boundary guard cells below, but
  // not f's so we don't want to change f.
  result.allocate();
  communicate(result);

  // Extrapolate into boundaries (if requested) so that differential geometry
  // terms can be interpolated if necessary
  // Note: cannot use applyBoundary("free_o3") here because applyBoundary()
  // would try to create a new Coordinates object since we have not finished
  // initializing yet, leading to an infinite recursion.
  // Also, here we interpolate for the boundary points at xstart/ystart and
  // (xend+1)/(yend+1) instead of extrapolating.
  for (auto& bndry : localmesh->getBoundaries()) {
    if ((extrapolate_x and bndry->bx != 0) or (extrapolate_y and bndry->by != 0)) {
      int extrap_start = 0;
      if (not no_extra_interpolate) {
        // Can use no_extra_interpolate argument to skip the extra interpolation when we
        // want to extrapolate the Christoffel symbol terms which come from derivatives so
        // don't have the extra point set already
        if ((location == CELL_XLOW) && (bndry->bx > 0)) {
          extrap_start = 1;
        } else if ((location == CELL_YLOW) && (bndry->by > 0)) {
          extrap_start = 1;
        }
      }
      for (bndry->first(); !bndry->isDone(); bndry->next1d()) {
        // interpolate extra boundary point that is missed by interp_to, if
        // necessary.
        // Only interpolate this point if we are actually changing location. E.g.
        // when we use this function to extrapolate J and Bxy on staggered grids,
        // this point should already be set correctly because the metric
        // components have been interpolated to here.
        if (extrap_start > 0 and f.getLocation() != location) {
          ASSERT1(bndry->bx == 0 or localmesh->xstart > 1);
          ASSERT1(bndry->by == 0 or localmesh->ystart > 1);
          // note that either bx or by is >0 here
          result(bndry->x, bndry->y) =
              (9.
                   * (f(bndry->x - bndry->bx, bndry->y - bndry->by)
                      + f(bndry->x, bndry->y))
               - f(bndry->x - 2 * bndry->bx, bndry->y - 2 * bndry->by)
               - f(bndry->x + bndry->bx, bndry->y + bndry->by))
              / 16.;
        }

        // set boundary guard cells
        if ((bndry->bx != 0 && localmesh->GlobalNx - 2 * bndry->width >= 3)
            || (bndry->by != 0
                && localmesh->GlobalNy - localmesh->numberOfYBoundaries() * bndry->width
                       >= 3)) {
          if (bndry->bx != 0 && localmesh->LocalNx == 1 && bndry->width == 1) {
            throw BoutException(
                "Not enough points in the x-direction on this "
                "processor for extrapolation needed to use staggered grids. "
                "Increase number of x-guard cells MXG or decrease number of "
                "processors in the x-direction NXPE.");
          }
          if (bndry->by != 0 && localmesh->LocalNy == 1 && bndry->width == 1) {
            throw BoutException(
                "Not enough points in the y-direction on this "
                "processor for extrapolation needed to use staggered grids. "
                "Increase number of y-guard cells MYG or decrease number of "
                "processors in the y-direction NYPE.");
          }
          // extrapolate into boundary guard cells if there are enough grid points
          for (int i = extrap_start; i < bndry->width; i++) {
            int xi = bndry->x + i * bndry->bx;
            int yi = bndry->y + i * bndry->by;
            result(xi, yi) = 3.0 * result(xi - bndry->bx, yi - bndry->by)
                             - 3.0 * result(xi - 2 * bndry->bx, yi - 2 * bndry->by)
                             + result(xi - 3 * bndry->bx, yi - 3 * bndry->by);
          }
        } else {
          // not enough grid points to extrapolate, set equal to last grid point
          for (int i = extrap_start; i < bndry->width; i++) {
            result(bndry->x + i * bndry->bx, bndry->y + i * bndry->by) =
                result(bndry->x - bndry->bx, bndry->y - bndry->by);
          }
        }
      }
    }
  }
#if CHECK > 0
  if (not(
          // if include_corner_cells=true, then we extrapolate valid data into the
          // corner cells if they are not already filled
          localmesh->include_corner_cells

          // if we are not extrapolating at all, the corner cells should contain valid
          // data
          or (not extrapolate_x and not extrapolate_y))) {
    // Invalidate corner guard cells
    for (int i = 0; i < localmesh->xstart; i++) {
      for (int j = 0; j < localmesh->ystart; j++) {
        result(i, j) = BoutNaN;
        result(i, localmesh->LocalNy - 1 - j) = BoutNaN;
        result(localmesh->LocalNx - 1 - i, j) = BoutNaN;
        result(localmesh->LocalNx - 1 - i, localmesh->LocalNy - 1 - j) = BoutNaN;
      }
    }
  }
#endif

  return result;
}

#if BOUT_USE_METRIC_3D
Field3D interpolateAndExtrapolate(const Field3D& f_, CELL_LOC location,
                                  bool extrapolate_x, bool extrapolate_y,
                                  bool no_extra_interpolate, ParallelTransform* pt_) {

  Mesh* localmesh = f_.getMesh();
  Field3D result;
  Field3D f = f_;
  ParallelTransform* pt_f;
  if (f.getCoordinates() == nullptr) {
    // if input f is member of the Coordinates we are currently constructing, it will not
    // have Coordinates and needs to use the passed-in ParallelTransform
    pt_f = pt_;
  } else {
    // if input f is from Coordinates at a different location, it will have its own
    // Coordinates, and we should use its ParallelTransform
    pt_f = &f.getCoordinates()->getParallelTransform();
  }
  if (f.getDirectionY() != YDirectionType::Standard) {
    if (pt_f->canToFromFieldAligned()) {
      f = pt_f->fromFieldAligned(f);
    } else {
      f.setDirectionY(YDirectionType::Standard);
    }
  }
  if (location == CELL_YLOW and f.getLocation() != CELL_YLOW) {
    auto f_aligned = pt_f->toFieldAligned(f, "RGN_NOX");
    result = interp_to(f_aligned, location, "RGN_NOBNDRY");
    ParallelTransform* pt_result;
    if (result.getCoordinates() == nullptr) {
      pt_result = pt_;
    } else {
      pt_result = &result.getCoordinates()->getParallelTransform();
    }
    result = pt_result->fromFieldAligned(result, "RGN_NOBNDRY");
  } else {
    result = interp_to(f, location, "RGN_NOBNDRY");
  }
  // Ensure result's data is unique. Otherwise result might be a duplicate of
  // f (if no interpolation is needed, e.g. if interpolation is in the
  // z-direction); then f would be communicated. Since this function is used
  // on geometrical quantities that might not be periodic in y even on closed
  // field lines (due to dependence on integrated shear), we don't want to
  // communicate f. We will sort out result's boundary guard cells below, but
  // not f's so we don't want to change f.
  result.allocate();
  communicate(result);

  // Extrapolate into boundaries (if requested) so that differential geometry
  // terms can be interpolated if necessary
  // Note: cannot use applyBoundary("free_o3") here because applyBoundary()
  // would try to create a new Coordinates object since we have not finished
  // initializing yet, leading to an infinite recursion.
  // Also, here we interpolate for the boundary points at xstart/ystart and
  // (xend+1)/(yend+1) instead of extrapolating.
  for (auto& bndry : localmesh->getBoundaries()) {
    if ((extrapolate_x and bndry->bx != 0) or (extrapolate_y and bndry->by != 0)) {
      int extrap_start = 0;
      if (not no_extra_interpolate) {
        // Can use no_extra_interpolate argument to skip the extra interpolation when we
        // want to extrapolate the Christoffel symbol terms which come from derivatives so
        // don't have the extra point set already
        if ((location == CELL_XLOW) && (bndry->bx > 0)) {
          extrap_start = 1;
        } else if ((location == CELL_YLOW) && (bndry->by > 0)) {
          extrap_start = 1;
        }
      }
      for (bndry->first(); !bndry->isDone(); bndry->next1d()) {
        // interpolate extra boundary point that is missed by interp_to, if
        // necessary.
        // Only interpolate this point if we are actually changing location. E.g.
        // when we use this function to extrapolate J and Bxy on staggered grids,
        // this point should already be set correctly because the metric
        // components have been interpolated to here.
        if (extrap_start > 0 and f.getLocation() != location) {
          ASSERT1(bndry->bx == 0 or localmesh->xstart > 1);
          ASSERT1(bndry->by == 0 or localmesh->ystart > 1);
          // note that either bx or by is >0 here
          for (int zi = 0; zi < localmesh->LocalNz; ++zi) {
            result(bndry->x, bndry->y, zi) =
                (9.
                     * (f(bndry->x - bndry->bx, bndry->y - bndry->by, zi)
                        + f(bndry->x, bndry->y, zi))
                 - f(bndry->x - 2 * bndry->bx, bndry->y - 2 * bndry->by, zi)
                 - f(bndry->x + bndry->bx, bndry->y + bndry->by, zi))
                / 16.;
          }
        }
        // set boundary guard cells
        if ((bndry->bx != 0 && localmesh->GlobalNx - 2 * bndry->width >= 3)
            || (bndry->by != 0 && localmesh->GlobalNy - 2 * bndry->width >= 3)) {
          if (bndry->bx != 0 && localmesh->LocalNx == 1 && bndry->width == 1) {
            throw BoutException(
                "Not enough points in the x-direction on this "
                "processor for extrapolation needed to use staggered grids. "
                "Increase number of x-guard cells MXG or decrease number of "
                "processors in the x-direction NXPE.");
          }
          if (bndry->by != 0 && localmesh->LocalNy == 1 && bndry->width == 1) {
            throw BoutException(
                "Not enough points in the y-direction on this "
                "processor for extrapolation needed to use staggered grids. "
                "Increase number of y-guard cells MYG or decrease number of "
                "processors in the y-direction NYPE.");
          }
          // extrapolate into boundary guard cells if there are enough grid points
          for (int i = extrap_start; i < bndry->width; i++) {
            int xi = bndry->x + i * bndry->bx;
            int yi = bndry->y + i * bndry->by;
            for (int zi = 0; zi < localmesh->LocalNz; ++zi) {
              result(xi, yi, zi) =
                  3.0 * result(xi - bndry->bx, yi - bndry->by, zi)
                  - 3.0 * result(xi - 2 * bndry->bx, yi - 2 * bndry->by, zi)
                  + result(xi - 3 * bndry->bx, yi - 3 * bndry->by, zi);
            }
          }
        } else {
          // not enough grid points to extrapolate, set equal to last grid point
          for (int i = extrap_start; i < bndry->width; i++) {
            for (int zi = 0; zi < localmesh->LocalNz; ++zi) {
              result(bndry->x + i * bndry->bx, bndry->y + i * bndry->by, zi) =
                  result(bndry->x - bndry->bx, bndry->y - bndry->by, zi);
            }
          }
        }
      }
    }
  }
#if CHECK > 0
  if (not(
          // if include_corner_cells=true, then we extrapolate valid data into the
          // corner cells if they are not already filled
          localmesh->include_corner_cells

          // if we are not extrapolating at all, the corner cells should contain valid
          // data
          or (not extrapolate_x and not extrapolate_y))) {
    // Invalidate corner guard cells
    for (int i = 0; i < localmesh->xstart; i++) {
      for (int j = 0; j < localmesh->ystart; j++) {
        for (int k = 0; k < localmesh->LocalNz; ++k) {
          result(i, j, k) = BoutNaN;
          result(i, localmesh->LocalNy - 1 - j, k) = BoutNaN;
          result(localmesh->LocalNx - 1 - i, j, k) = BoutNaN;
          result(localmesh->LocalNx - 1 - i, localmesh->LocalNy - 1 - j, k) = BoutNaN;
        }
      }
    }
  }
#endif // CHECK > 0

  return result;
}
#endif // BOUT_USE_METRIC_3D

// If the CELL_CENTRE variable was read, the staggered version is required to
// also exist for consistency
void checkStaggeredGet(Mesh* mesh, const std::string& name, const std::string& suffix) {
  if (mesh->sourceHasVar(name) != mesh->sourceHasVar(name + suffix)) {
    throw BoutException("Attempting to read staggered fields from grid, but " + name
                        + " is not present in both CELL_CENTRE and staggered versions.");
  }
}

// convenience function for repeated code
int getAtLoc(Mesh* mesh, Coordinates::FieldMetric& var, const std::string& name,
             const std::string& suffix, CELL_LOC location, BoutReal default_value = 0.) {

  checkStaggeredGet(mesh, name, suffix);
  int result = mesh->get(var, name + suffix, default_value, false, location);

  return result;
}

int getAtLocAndFillGuards(Mesh* mesh, Coordinates::FieldMetric& var,
                          const std::string& name, const std::string& suffix,
                          CELL_LOC location, BoutReal default_value, bool extrapolate_x,
                          bool extrapolate_y, bool no_extra_interpolate,
                          ParallelTransform* pt) {
  auto ret = getAtLoc(mesh, var, name, suffix, location, default_value);
  var = interpolateAndExtrapolate(var, location, extrapolate_x, extrapolate_y,
                                  no_extra_interpolate, pt);
  return ret;
}

std::string getLocationSuffix(CELL_LOC location) {
  switch (location) {
  case CELL_CENTRE: {
    return "";
  }
  case CELL_XLOW: {
    return "_xlow";
  }
  case CELL_YLOW: {
    return "_ylow";
  }
  case CELL_ZLOW: {
    // in 2D metric, same as CELL_CENTRE
    return bout::build::use_metric_3d ? "_zlow" : "";
  }
  default: {
    throw BoutException(
        "Incorrect location passed to "
        "Coordinates(Mesh*,const CELL_LOC,const Coordinates*) constructor.");
  }
  }
}

} // anonymous namespace

Coordinates::Coordinates(Mesh* mesh, FieldMetric dx, FieldMetric dy, FieldMetric dz,
                         FieldMetric J, FieldMetric Bxy, FieldMetric g11, FieldMetric g22,
                         FieldMetric g33, FieldMetric g12, FieldMetric g13,
                         FieldMetric g23, FieldMetric g_11, FieldMetric g_22,
                         FieldMetric g_33, FieldMetric g_12, FieldMetric g_13,
                         FieldMetric g_23, FieldMetric ShiftTorsion,
                         FieldMetric IntShiftTorsion)
    : dx(std::move(dx)), dy(std::move(dy)), dz(dz), J(std::move(J)), Bxy(std::move(Bxy)),
      contravariantMetricTensor(g11, g22, g33, g12, g13, g23),
      covariantMetricTensor(g_11, g_22, g_33, g_12, g_13, g_23),
      ShiftTorsion(std::move(ShiftTorsion)), IntShiftTorsion(std::move(IntShiftTorsion)),
      nz(mesh->LocalNz), localmesh(mesh), location(CELL_CENTRE) {}

Coordinates::Coordinates(Mesh* mesh, Options* options)
    : dx(1., mesh), dy(1., mesh), dz(1., mesh), d1_dx(mesh), d1_dy(mesh), d1_dz(mesh),
      J(1., mesh), Bxy(1., mesh), contravariantMetricTensor(1., 1., 1., 0, 0, 0, mesh),
      covariantMetricTensor(1., 1., 1., 0, 0, 0, mesh), G1_11(mesh), G1_22(mesh),
      G1_33(mesh), G1_12(mesh), G1_13(mesh), G1_23(mesh), G2_11(mesh), G2_22(mesh),
      G2_33(mesh), G2_12(mesh), G2_13(mesh), G2_23(mesh), G3_11(mesh), G3_22(mesh),
      G3_33(mesh), G3_12(mesh), G3_13(mesh), G3_23(mesh), G1(mesh), G2(mesh), G3(mesh),
      ShiftTorsion(mesh), IntShiftTorsion(mesh),
      // Identity metric tensor
      localmesh(mesh), location(CELL_CENTRE) {

  if (options == nullptr) {
    options = Options::getRoot()->getSection("mesh");
  }

  // Note: If boundary cells were not loaded from the grid file, use
  // 'interpolateAndExtrapolate' to set them. Ensures that derivatives are
  // smooth at all the boundaries.

  const bool extrapolate_x =
      (*options)["extrapolate_x"].withDefault(not mesh->sourceHasXBoundaryGuards());
  const bool extrapolate_y =
      (*options)["extrapolate_y"].withDefault(not mesh->sourceHasYBoundaryGuards());

  if (extrapolate_x) {
    output_warn.write(_("WARNING: extrapolating input mesh quantities into x-boundary "
                        "cells. Set option extrapolate_x=false to disable this.\n"));
  }

  if (extrapolate_y) {
    output_warn.write(_("WARNING: extrapolating input mesh quantities into y-boundary "
                        "cells. Set option extrapolate_y=false to disable this.\n"));
  }

  mesh->get(dx, "dx", 1.0, false);
  mesh->get(dy, "dy", 1.0, false);

  nz = mesh->LocalNz;

  {
    auto& options = Options::root();
    const bool has_zperiod = options.isSet("zperiod");
    const auto zmin = has_zperiod ? 0.0 : options["ZMIN"].withDefault(0.0);
    const auto zmax = has_zperiod ? 1.0 / options["zperiod"].withDefault(1.0)
                                  : options["ZMAX"].withDefault(1.0);

    const auto default_dz = (zmax - zmin) * TWOPI / nz;

    mesh->get(dz, "dz", default_dz, false);
  }

  // required early for differentiation.
  setParallelTransform(options);

  dz = interpolateAndExtrapolate(dz, location, extrapolate_x, extrapolate_y, false,
                                 transform.get());
  dx = interpolateAndExtrapolate(dx, location, extrapolate_x, extrapolate_y, false,
                                 transform.get());

  if (mesh->periodicX) {
    communicate(dx);
  }

  dy = interpolateAndExtrapolate(dy, location, extrapolate_x, extrapolate_y, false,
                                 transform.get());

  auto getUnaligned = [this](const std::string& name, BoutReal default_value) {
    auto field = localmesh->get(name, default_value, false);
    if (field.getDirectionY() == YDirectionType::Aligned
        and transform->canToFromFieldAligned()) {
      return transform->fromFieldAligned(field);
    } else {
      field.setDirectionY(YDirectionType::Standard);
      return field;
    }
  };

  auto getUnalignedAtLocation = [this, extrapolate_x, extrapolate_y, getUnaligned](
                                    const std::string& name, BoutReal default_value) {
    auto field = getUnaligned(name, default_value);
    return interpolateAndExtrapolate(field, location, extrapolate_x, extrapolate_y, false,
                                     transform.get());
  };

  FieldMetric g11, g22, g33, g12, g13, g23;

  // Diagonal components of metric tensor g^{ij} (default to 1)
  g11 = getUnalignedAtLocation("g11", 1.0);
  g22 = getUnalignedAtLocation("g22", 1.0);
  g33 = getUnalignedAtLocation("g33", 1.0);

  // Off-diagonal elements. Default to 0
  g12 = getUnalignedAtLocation("g12", 0.0);
  g13 = getUnalignedAtLocation("g13", 0.0);
  g23 = getUnalignedAtLocation("g23", 0.0);

  setContravariantMetricTensor(ContravariantMetricTensor(g11, g22, g33, g12, g13, g23));

  // Check input metrics
  checkContravariant();

  /// Find covariant metric components
  auto covariant_component_names = {"g_11", "g_22", "g_33", "g_12", "g_13", "g_23"};
  auto source_has_component = [&mesh](const std::string& name) {
    return mesh->sourceHasVar(name);
  };
  // Check if any of the components are present
  if (std::any_of(begin(covariant_component_names), end(covariant_component_names),
                  source_has_component)) {
    // Check that all components are present
    if (std::all_of(begin(covariant_component_names), end(covariant_component_names),
                    source_has_component)) {

      FieldMetric old_g_11 = covariantMetricTensor.Getg_11(); // non-const
      FieldMetric old_g_22 = covariantMetricTensor.Getg_22(); // non-const
      FieldMetric old_g_33 = covariantMetricTensor.Getg_33(); // non-const
      FieldMetric old_g_12 = covariantMetricTensor.Getg_12(); // non-const
      FieldMetric old_g_13 = covariantMetricTensor.Getg_13(); // non-const
      FieldMetric old_g_23 = covariantMetricTensor.Getg_23(); // non-const

      FieldMetric g_11, g_22, g_33, g_12, g_13, g_23;
      //      g_11 = getUnaligned(old_g_11, "g_11", 1.0);
      //      g_22 = getUnaligned(old_g_22, "g_22", 1.0);
      //      g_33 = getUnaligned(old_g_33, "g_33", 1.0);
      //      g_12 = getUnaligned(old_g_12, "g_12", 0.0);
      //      g_13 = getUnaligned(old_g_13, "g_13", 0.0);
      //      g_23 = getUnaligned(old_g_23, "g_23", 0.0);

      covariantMetricTensor.setCovariantMetricTensor(
          CovariantMetricTensor(g_11, g_22, g_33, g_12, g_13, g_23));

      output_warn.write("\tWARNING! Covariant components of metric tensor set manually. "
                        "Contravariant components NOT recalculated\n");

    } else {
      output_warn.write("Not all covariant components of metric tensor found. "
                        "Calculating all from the contravariant tensor\n");
      /// Calculate contravariant metric components if not found
      try {
        calcCovariant("RGN_NOCORNERS");
      } catch (BoutException) {
        throw BoutException("Error in calcCovariant call");
      }
    }
  } else {
    /// Calculate contravariant metric components if not found
    try {
      calcCovariant("RGN_NOCORNERS");
    } catch (BoutException) {
      throw BoutException("Error in calcCovariant call");
    }
  }

  FieldMetric g_11, g_22, g_33, g_12, g_13, g_23;

  // More robust to extrapolate derived quantities directly, rather than
  // deriving from extrapolated covariant metric components
  g_11 = interpolateAndExtrapolate(covariantMetricTensor.Getg_11(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());
  g_22 = interpolateAndExtrapolate(covariantMetricTensor.Getg_22(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());
  g_33 = interpolateAndExtrapolate(covariantMetricTensor.Getg_33(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());
  g_12 = interpolateAndExtrapolate(covariantMetricTensor.Getg_12(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());
  g_13 = interpolateAndExtrapolate(covariantMetricTensor.Getg_13(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());
  g_23 = interpolateAndExtrapolate(covariantMetricTensor.Getg_23(), location,
                                   extrapolate_x, extrapolate_y, false, transform.get());

  covariantMetricTensor.setCovariantMetricTensor(
      CovariantMetricTensor(g_11, g_22, g_33, g_12, g_13, g_23));

  // Check covariant metrics
  checkCovariant();

  /// Calculate Jacobian and Bxy
  if (jacobian()) {
    throw BoutException("Error in jacobian call");
  }

  // Attempt to read J from the grid file
  auto Jcalc = J;
  if (mesh->get(J, "J", 0.0, false)) {
    output_warn.write(
        "\tWARNING: Jacobian 'J' not found. Calculating from metric tensor\n");
    J = Jcalc;
  } else {
    J = interpolateAndExtrapolate(J, location, extrapolate_x, extrapolate_y, false,
                                  transform.get());

    // Compare calculated and loaded values
    output_warn.write("\tMaximum difference in J is {:e}\n", max(abs(J - Jcalc)));

    communicate(J);

    // Re-evaluate Bxy using new J
    Bxy = sqrt(g_22) / J;
  }

  // Check jacobian
  bout::checkFinite(J, "J", "RGN_NOCORNERS");
  bout::checkPositive(J, "J", "RGN_NOCORNERS");
  if (min(abs(J)) < 1.0e-10) {
    throw BoutException("\tERROR: Jacobian becomes very small\n");
  }

  // Attempt to read Bxy from the grid file
  auto Bcalc = Bxy;
  if (mesh->get(Bxy, "Bxy", 0.0, false)) {
    output_warn.write("\tWARNING: Magnitude of B field 'Bxy' not found. Calculating from "
                      "metric tensor\n");
    Bxy = Bcalc;
  } else {

    Bxy = interpolateAndExtrapolate(Bxy, location, extrapolate_x, extrapolate_y, false,
                                    transform.get());
    output_warn.write("\tMaximum difference in Bxy is {:e}\n", max(abs(Bxy - Bcalc)));
  }

  // Check Bxy
  bout::checkFinite(Bxy, "Bxy", "RGN_NOCORNERS");
  bout::checkPositive(Bxy, "Bxy", "RGN_NOCORNERS");

  if (mesh->get(ShiftTorsion, "ShiftTorsion", 0.0, false)) {
    output_warn.write(
        "\tWARNING: No Torsion specified for zShift. Derivatives may not be correct\n");
    ShiftTorsion = 0.0;
  }
  ShiftTorsion = interpolateAndExtrapolate(ShiftTorsion, location, extrapolate_x,
                                           extrapolate_y, false, transform.get());

  //////////////////////////////////////////////////////

  if (mesh->IncIntShear) {
    if (mesh->get(IntShiftTorsion, "IntShiftTorsion", 0.0, false)) {
      output_warn.write("\tWARNING: No Integrated torsion specified\n");
    }
    IntShiftTorsion = interpolateAndExtrapolate(IntShiftTorsion, location, extrapolate_x,
                                                extrapolate_y, false, transform.get());
  } else {
    // IntShiftTorsion will not be used, but set to zero to avoid uninitialized field
    IntShiftTorsion = 0.;
  }
}

Coordinates::Coordinates(Mesh* mesh, Options* options, const CELL_LOC loc,
                         const Coordinates* coords_in, bool force_interpolate_from_centre)
    : dx(1., mesh), dy(1., mesh), dz(1., mesh), d1_dx(mesh), d1_dy(mesh), d1_dz(mesh),
      J(1., mesh), Bxy(1., mesh), contravariantMetricTensor(1., 1., 1., 0, 0, 0, mesh),
      covariantMetricTensor(1., 1., 1., 0, 0, 0, mesh), G1_11(mesh), G1_22(mesh),
      G1_33(mesh), G1_12(mesh), G1_13(mesh), G1_23(mesh), G2_11(mesh), G2_22(mesh),
      G2_33(mesh), G2_12(mesh), G2_13(mesh), G2_23(mesh), G3_11(mesh), G3_22(mesh),
      G3_33(mesh), G3_12(mesh), G3_13(mesh), G3_23(mesh), G1(mesh), G2(mesh), G3(mesh),
      ShiftTorsion(mesh), IntShiftTorsion(mesh),
      // Identity metric tensor
      localmesh(mesh), location(loc) {

  std::string suffix = getLocationSuffix(location);

  nz = mesh->LocalNz;

  // Default to true in case staggered quantities are not read from file
  bool extrapolate_x = true;
  bool extrapolate_y = true;

  if (!force_interpolate_from_centre && mesh->sourceHasVar("dx" + suffix)) {

    extrapolate_x = not mesh->sourceHasXBoundaryGuards();
    extrapolate_y = not mesh->sourceHasYBoundaryGuards();

    if (extrapolate_x) {
      output_warn.write(_("WARNING: extrapolating input mesh quantities into x-boundary "
                          "cells\n"));
    }

    if (extrapolate_y) {
      output_warn.write(_("WARNING: extrapolating input mesh quantities into y-boundary "
                          "cells\n"));
    }

    {
      auto& options = Options::root();
      const bool has_zperiod = options.isSet("zperiod");
      const auto zmin = has_zperiod ? 0.0 : options["ZMIN"].withDefault(0.0);
      const auto zmax = has_zperiod ? 1.0 / options["zperiod"].withDefault(1.0)
                                    : options["ZMAX"].withDefault(1.0);

      const auto default_dz = (zmax - zmin) * TWOPI / nz;
      getAtLoc(mesh, dz, "dz", suffix, location, default_dz);
    }
    setParallelTransform(options);

    dz = interpolateAndExtrapolate(dz, location, extrapolate_x, extrapolate_y, false,
                                   transform.get());

    getAtLocAndFillGuards(mesh, dx, "dx", suffix, location, 1.0, extrapolate_x,
                          extrapolate_y, false, transform.get());

    if (mesh->periodicX) {
      communicate(dx);
    }

    getAtLocAndFillGuards(mesh, dy, "dy", suffix, location, 1.0, extrapolate_x,
                          extrapolate_y, false, transform.get());

    auto contravariant_components =
        contravariantMetricTensor.getContravariantMetricTensor();

    // grid data source has staggered fields, so read instead of interpolating
    // Diagonal components of metric tensor g^{ij} (default to 1)
    //    TODO: Method `getAtLocAndFillGuards` violates command–query separation principle?
    getAtLocAndFillGuards(mesh, contravariant_components.g11, "g11", suffix, location,
                          1.0, extrapolate_x, extrapolate_y, false, transform.get());
    getAtLocAndFillGuards(mesh, contravariant_components.g22, "g22", suffix, location,
                          1.0, extrapolate_x, extrapolate_y, false, transform.get());
    getAtLocAndFillGuards(mesh, contravariant_components.g33, "g33", suffix, location,
                          1.0, extrapolate_x, extrapolate_y, false, transform.get());
    getAtLocAndFillGuards(mesh, contravariant_components.g12, "g12", suffix, location,
                          0.0, extrapolate_x, extrapolate_y, false, transform.get());
    getAtLocAndFillGuards(mesh, contravariant_components.g13, "g13", suffix, location,
                          0.0, extrapolate_x, extrapolate_y, false, transform.get());
    getAtLocAndFillGuards(mesh, contravariant_components.g23, "g23", suffix, location,
                          0.0, extrapolate_x, extrapolate_y, false, transform.get());

    setContravariantMetricTensor(ContravariantMetricTensor(
        contravariant_components.g11, contravariant_components.g22,
        contravariant_components.g33, contravariant_components.g12,
        contravariant_components.g13, contravariant_components.g23));

    // Check input metrics
    checkContravariant();

    /// Find covariant metric components
    auto covariant_component_names = {"g_11", "g_22", "g_33", "g_12", "g_13", "g_23"};
    auto source_has_component = [&suffix, &mesh](const std::string& name) {
      return mesh->sourceHasVar(name + suffix);
    };

    // Check if any of the components are present
    if (std::any_of(begin(covariant_component_names), end(covariant_component_names),
                    source_has_component)) {
      // Check that all components are present
      if (std::all_of(begin(covariant_component_names), end(covariant_component_names),
                      source_has_component)) {

        FieldMetric old_g_11 = covariantMetricTensor.Getg_11(); // non-const
        FieldMetric old_g_22 = covariantMetricTensor.Getg_22(); // non-const
        FieldMetric old_g_33 = covariantMetricTensor.Getg_33(); // non-const
        FieldMetric old_g_12 = covariantMetricTensor.Getg_12(); // non-const
        FieldMetric old_g_13 = covariantMetricTensor.Getg_13(); // non-const
        FieldMetric old_g_23 = covariantMetricTensor.Getg_23(); // non-const

        getAtLoc(mesh, old_g_11, "g_11", suffix, location);
        getAtLoc(mesh, old_g_22, "g_22", suffix, location);
        getAtLoc(mesh, old_g_33, "g_33", suffix, location);
        getAtLoc(mesh, old_g_12, "g_12", suffix, location);
        getAtLoc(mesh, old_g_13, "g_13", suffix, location);
        getAtLoc(mesh, old_g_23, "g_23", suffix, location);

        output_warn.write(
            "\tWARNING! Staggered covariant components of metric tensor set manually. "
            "Contravariant components NOT recalculated\n");

      } else {
        output_warn.write(
            "Not all staggered covariant components of metric tensor found. "
            "Calculating all from the contravariant tensor\n");
        /// Calculate contravariant metric components if not found
        try {
          calcCovariant("RGN_NOCORNERS");
        } catch (BoutException& err) {
          std::cout << err.what();
          throw BoutException("Error in staggered calcCovariant call/n"
                              + std::string(err.what()));
        }
      }
    } else {
      /// Calculate contravariant metric components if not found
      try {
        calcCovariant("RGN_NOCORNERS");
      } catch (BoutException& err) {
        throw BoutException("Error in staggered calcCovariant call/n"
                            + std::string(err.what()));
      }
    }

    FieldMetric g_11, g_22, g_33, g_12, g_13, g_23;

    // More robust to extrapolate derived quantities directly, rather than
    // deriving from extrapolated covariant metric components
    g_11 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_11(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());
    g_22 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_22(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());
    g_33 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_33(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());
    g_12 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_12(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());
    g_13 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_13(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());
    g_23 =
        interpolateAndExtrapolate(covariantMetricTensor.Getg_23(), location,
                                  extrapolate_x, extrapolate_y, false, transform.get());

    // Check covariant metrics
    checkCovariant();

    /// Calculate Jacobian and Bxy
    if (jacobian()) {
      throw BoutException("Error in jacobian call while constructing staggered "
                          "Coordinates");
    }

    // Attempt to read J from the grid file
    auto Jcalc = J;
    if (getAtLoc(mesh, J, "J", suffix, location)) {
      output_warn.write(
          "\tWARNING: Jacobian 'J_{:s}' not found. Calculating from metric tensor\n",
          suffix);
      J = Jcalc;
    } else {
      J = interpolateAndExtrapolate(J, location, extrapolate_x, extrapolate_y, false,
                                    transform.get());

      // Compare calculated and loaded values
      output_warn.write("\tMaximum difference in J is %e\n", max(abs(J - Jcalc)));

      // Re-evaluate Bxy using new J
      Bxy = sqrt(g_22) / J;
    }

    // Check jacobian
    bout::checkFinite(J, "J" + suffix, "RGN_NOCORNERS");
    bout::checkPositive(J, "J" + suffix, "RGN_NOCORNERS");
    if (min(abs(J)) < 1.0e-10) {
      throw BoutException("\tERROR: Jacobian{:s} becomes very small\n", suffix);
    }

    // Attempt to read Bxy from the grid file
    auto Bcalc = Bxy;
    if (getAtLoc(mesh, Bxy, "Bxy", suffix, location)) {
      output_warn.write(
          "\tWARNING: Magnitude of B field 'Bxy_{:s}' not found. Calculating "
          " from metric tensor\n",
          suffix);
      Bxy = Bcalc;
    } else {
      Bxy = interpolateAndExtrapolate(Bxy, location, extrapolate_x, extrapolate_y, false,
                                      transform.get());

      output_warn.write("\tMaximum difference in Bxy is %e\n", max(abs(Bxy - Bcalc)));
    }

    // Check Bxy
    bout::checkFinite(Bxy, "Bxy" + suffix, "RGN_NOCORNERS");
    bout::checkPositive(Bxy, "Bxy" + suffix, "RGN_NOCORNERS");

    checkStaggeredGet(mesh, "ShiftTorsion", suffix);
    if (mesh->get(ShiftTorsion, "ShiftTorsion" + suffix, 0.0, false)) {
      output_warn.write(
          "\tWARNING: No Torsion specified for zShift. Derivatives may not be correct\n");
      ShiftTorsion = 0.0;
    }
    ShiftTorsion.setLocation(location);
    ShiftTorsion = interpolateAndExtrapolate(ShiftTorsion, location, extrapolate_x,
                                             extrapolate_y, false, transform.get());

    //////////////////////////////////////////////////////

    if (mesh->IncIntShear) {
      checkStaggeredGet(mesh, "IntShiftTorsion", suffix);
      if (mesh->get(IntShiftTorsion, "IntShiftTorsion" + suffix, 0.0, false)) {
        output_warn.write("\tWARNING: No Integrated torsion specified\n");
        IntShiftTorsion = 0.0;
      }
      IntShiftTorsion.setLocation(location);
      IntShiftTorsion =
          interpolateAndExtrapolate(IntShiftTorsion, location, extrapolate_x,
                                    extrapolate_y, false, transform.get());
    } else {
      // IntShiftTorsion will not be used, but set to zero to avoid uninitialized field
      IntShiftTorsion = 0.;
    }
  } else {
    // Interpolate fields from coords_in

    if (isUniform(coords_in->dz)) {
      dz = coords_in->dz;
      dz.setLocation(location);
    } else {
      throw BoutException(
          "We are asked to transform dz to get dz before we have a transform, which "
          "might require dz!\nPlease provide a dz for the staggered quantity!");
    }
    setParallelTransform(options);
    dx = interpolateAndExtrapolate(coords_in->dx, location, true, true, false,
                                   transform.get());
    dy = interpolateAndExtrapolate(coords_in->dy, location, true, true, false,
                                   transform.get());
    // not really needed - we have used dz already ...
    dz = interpolateAndExtrapolate(coords_in->dz, location, true, true, false,
                                   transform.get());

    ContravariantMetricTensor::ContravariantComponents metric_tensor =
        coords_in->getContravariantMetricTensor();
    FieldMetric g11, g22, g33, g12, g13, g23;

    // Diagonal components of metric tensor g^{ij}
    g11 = interpolateAndExtrapolate(metric_tensor.g11, location, true, true, false,
                                    transform.get());
    g22 = interpolateAndExtrapolate(metric_tensor.g22, location, true, true, false,
                                    transform.get());
    g33 = interpolateAndExtrapolate(metric_tensor.g33, location, true, true, false,
                                    transform.get());

    // Off-diagonal elements.
    g12 = interpolateAndExtrapolate(metric_tensor.g12, location, true, true, false,
                                    transform.get());
    g13 = interpolateAndExtrapolate(metric_tensor.g13, location, true, true, false,
                                    transform.get());
    g23 = interpolateAndExtrapolate(metric_tensor.g23, location, true, true, false,
                                    transform.get());

    setContravariantMetricTensor(ContravariantMetricTensor(g11, g22, g33, g12, g13, g23),
                                 "RGN_NOBNDRY");

    FieldMetric g_11, g_22, g_33, g_12, g_13, g_23;

    // 3x3 matrix inversion can exaggerate small interpolation errors, so it is
    // more robust to interpolate and extrapolate derived quantities directly,
    // rather than deriving from interpolated/extrapolated covariant metric
    // components
    g_11 = interpolateAndExtrapolate(covariantMetricTensor.Getg_11(), location, true,
                                     true, false, transform.get());
    g_22 = interpolateAndExtrapolate(covariantMetricTensor.Getg_22(), location, true,
                                     true, false, transform.get());
    g_33 = interpolateAndExtrapolate(covariantMetricTensor.Getg_33(), location, true,
                                     true, false, transform.get());
    g_12 = interpolateAndExtrapolate(covariantMetricTensor.Getg_12(), location, true,
                                     true, false, transform.get());
    g_13 = interpolateAndExtrapolate(covariantMetricTensor.Getg_13(), location, true,
                                     true, false, transform.get());
    g_23 = interpolateAndExtrapolate(covariantMetricTensor.Getg_23(), location, true,
                                     true, false, transform.get());

    covariantMetricTensor.setCovariantMetricTensor(
        CovariantMetricTensor(g_11, g_22, g_33, g_12, g_13, g_23));

    // Check input metrics
    checkContravariant();
    checkCovariant();

    J = interpolateAndExtrapolate(coords_in->J, location, true, true, false,
                                  transform.get());
    Bxy = interpolateAndExtrapolate(coords_in->Bxy, location, true, true, false,
                                    transform.get());

    bout::checkFinite(J, "The Jacobian", "RGN_NOCORNERS");
    bout::checkPositive(J, "The Jacobian", "RGN_NOCORNERS");
    bout::checkFinite(Bxy, "Bxy", "RGN_NOCORNERS");
    bout::checkPositive(Bxy, "Bxy", "RGN_NOCORNERS");

    ShiftTorsion = interpolateAndExtrapolate(coords_in->ShiftTorsion, location, true,
                                             true, false, transform.get());

    if (mesh->IncIntShear) {
      IntShiftTorsion = interpolateAndExtrapolate(coords_in->IntShiftTorsion, location,
                                                  true, true, false, transform.get());
    }
  }
}

void Coordinates::outputVars(Options& output_options) {
  Timer time("io");
  const std::string loc_string =
      (location == CELL_CENTRE) ? "" : "_" + toString(location);

  output_options["dx" + loc_string].force(dx, "Coordinates");
  output_options["dy" + loc_string].force(dy, "Coordinates");
  output_options["dz" + loc_string].force(dz, "Coordinates");

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();

  output_options["g11" + loc_string].force(contravariant_components.g11, "Coordinates");
  output_options["g22" + loc_string].force(contravariant_components.g22, "Coordinates");
  output_options["g33" + loc_string].force(contravariant_components.g33, "Coordinates");
  output_options["g12" + loc_string].force(contravariant_components.g12, "Coordinates");
  output_options["g13" + loc_string].force(contravariant_components.g13, "Coordinates");
  output_options["g23" + loc_string].force(contravariant_components.g23, "Coordinates");

  output_options["g_11" + loc_string].force(covariantMetricTensor.Getg_11(),
                                            "Coordinates");
  output_options["g_22" + loc_string].force(covariantMetricTensor.Getg_22(),
                                            "Coordinates");
  output_options["g_33" + loc_string].force(covariantMetricTensor.Getg_33(),
                                            "Coordinates");
  output_options["g_12" + loc_string].force(covariantMetricTensor.Getg_12(),
                                            "Coordinates");
  output_options["g_13" + loc_string].force(covariantMetricTensor.Getg_13(),
                                            "Coordinates");
  output_options["g_23" + loc_string].force(covariantMetricTensor.Getg_23(),
                                            "Coordinates");

  output_options["J" + loc_string].force(J, "Coordinates");
  output_options["Bxy" + loc_string].force(Bxy, "Coordinates");

  output_options["G1" + loc_string].force(G1, "Coordinates");
  output_options["G2" + loc_string].force(G2, "Coordinates");
  output_options["G3" + loc_string].force(G3, "Coordinates");

  getParallelTransform().outputVars(output_options);
}

const Field2D& Coordinates::zlength() const {
  BOUT_OMP(critical)
  if (not zlength_cache) {
    zlength_cache = std::make_unique<Field2D>(0., localmesh);

#if BOUT_USE_METRIC_3D
    BOUT_FOR_SERIAL(i, dz.getRegion("RGN_ALL")) { (*zlength_cache)[i] += dz[i]; }
#else
    (*zlength_cache) = dz * nz;
#endif
  }

  return *zlength_cache;
}

int Coordinates::geometry(bool recalculate_staggered,
                          bool force_interpolate_from_centre) {
  TRACE("Coordinates::geometry");

  auto contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();

  communicate(dx, dy, dz, contravariant_components.g11, contravariant_components.g22,
              contravariant_components.g33, contravariant_components.g12,
              contravariant_components.g13, contravariant_components.g23,
              covariantMetricTensor.Getg_11(), covariantMetricTensor.Getg_22(),
              covariantMetricTensor.Getg_33(), covariantMetricTensor.Getg_12(),
              covariantMetricTensor.Getg_13(), covariantMetricTensor.Getg_23(), J, Bxy);

  output_progress.write("Calculating differential geometry terms\n");

  if (min(abs(dx)) < 1e-8) {
    throw BoutException("dx magnitude less than 1e-8");
  }

  if (min(abs(dy)) < 1e-8) {
    throw BoutException("dy magnitude less than 1e-8");
  }

  if (min(abs(dz)) < 1e-8) {
    throw BoutException("dz magnitude less than 1e-8");
  }

  // Check input metrics
  checkContravariant();
  checkCovariant();
  CalculateChristoffelSymbols();

  auto tmp = J * contravariant_components.g12;
  communicate(tmp);
  G1 = (DDX(J * contravariant_components.g11) + DDY(tmp)
        + DDZ(J * contravariant_components.g13))
       / J;
  tmp = J * contravariant_components.g22;
  communicate(tmp);
  G2 = (DDX(J * contravariant_components.g12) + DDY(tmp)
        + DDZ(J * contravariant_components.g23))
       / J;
  tmp = J * contravariant_components.g23;
  communicate(tmp);
  G3 = (DDX(J * contravariant_components.g13) + DDY(tmp)
        + DDZ(J * contravariant_components.g33))
       / J;

  // Communicate christoffel symbol terms
  output_progress.write("\tCommunicating connection terms\n");

  communicate(G1_11, G1_22, G1_33, G1_12, G1_13, G1_23, G2_11, G2_22, G2_33, G2_12, G2_13,
              G2_23, G3_11, G3_22, G3_33, G3_12, G3_13, G3_23, G1, G2, G3);

  // Set boundary guard cells of Christoffel symbol terms
  // Ideally, when location is staggered, we would set the upper/outer boundary point
  // correctly rather than by extrapolating here: e.g. if location==CELL_YLOW and we are
  // at the upper y-boundary the x- and z-derivatives at yend+1 at the boundary can be
  // calculated because the guard cells are available, while the y-derivative could be
  // calculated from the CELL_CENTRE metric components (which have guard cells available
  // past the boundary location). This would avoid the problem that the y-boundary on the
  // CELL_YLOW grid is at a 'guard cell' location (yend+1).
  // However, the above would require lots of special handling, so just extrapolate for
  // now.
  G1_11 = interpolateAndExtrapolate(G1_11, location, true, true, true, transform.get());
  G1_22 = interpolateAndExtrapolate(G1_22, location, true, true, true, transform.get());
  G1_33 = interpolateAndExtrapolate(G1_33, location, true, true, true, transform.get());
  G1_12 = interpolateAndExtrapolate(G1_12, location, true, true, true, transform.get());
  G1_13 = interpolateAndExtrapolate(G1_13, location, true, true, true, transform.get());
  G1_23 = interpolateAndExtrapolate(G1_23, location, true, true, true, transform.get());

  G2_11 = interpolateAndExtrapolate(G2_11, location, true, true, true, transform.get());
  G2_22 = interpolateAndExtrapolate(G2_22, location, true, true, true, transform.get());
  G2_33 = interpolateAndExtrapolate(G2_33, location, true, true, true, transform.get());
  G2_12 = interpolateAndExtrapolate(G2_12, location, true, true, true, transform.get());
  G2_13 = interpolateAndExtrapolate(G2_13, location, true, true, true, transform.get());
  G2_23 = interpolateAndExtrapolate(G2_23, location, true, true, true, transform.get());

  G3_11 = interpolateAndExtrapolate(G3_11, location, true, true, true, transform.get());
  G3_22 = interpolateAndExtrapolate(G3_22, location, true, true, true, transform.get());
  G3_33 = interpolateAndExtrapolate(G3_33, location, true, true, true, transform.get());
  G3_12 = interpolateAndExtrapolate(G3_12, location, true, true, true, transform.get());
  G3_13 = interpolateAndExtrapolate(G3_13, location, true, true, true, transform.get());
  G3_23 = interpolateAndExtrapolate(G3_23, location, true, true, true, transform.get());

  G1 = interpolateAndExtrapolate(G1, location, true, true, true, transform.get());
  G2 = interpolateAndExtrapolate(G2, location, true, true, true, transform.get());
  G3 = interpolateAndExtrapolate(G3, location, true, true, true, transform.get());

  //////////////////////////////////////////////////////
  /// Non-uniform meshes. Need to use DDX, DDY

  OPTION(Options::getRoot(), non_uniform, true);

  Coordinates::FieldMetric d2x(localmesh), d2y(localmesh),
      d2z(localmesh); // d^2 x / d i^2

  // Read correction for non-uniform meshes
  std::string suffix = getLocationSuffix(location);
  if (location == CELL_CENTRE
      or (!force_interpolate_from_centre and localmesh->sourceHasVar("dx" + suffix))) {
    bool extrapolate_x = not localmesh->sourceHasXBoundaryGuards();
    bool extrapolate_y = not localmesh->sourceHasYBoundaryGuards();

    if (localmesh->get(d2x, "d2x" + suffix, 0.0, false, location)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2x' not found. Calculating from dx\n");
      d1_dx = bout::derivatives::index::DDX(1. / dx); // d/di(1/dx)

      communicate(d1_dx);
      d1_dx =
          interpolateAndExtrapolate(d1_dx, location, true, true, true, transform.get());
    } else {
      d2x.setLocation(location);
      // set boundary cells if necessary
      d2x = interpolateAndExtrapolate(d2x, location, extrapolate_x, extrapolate_y, false,
                                      transform.get());

      d1_dx = -d2x / (dx * dx);
    }

    if (localmesh->get(d2y, "d2y" + suffix, 0.0, false, location)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2y' not found. Calculating from dy\n");
      d1_dy = DDY(1. / dy); // d/di(1/dy)

      communicate(d1_dy);
      d1_dy =
          interpolateAndExtrapolate(d1_dy, location, true, true, true, transform.get());
    } else {
      d2y.setLocation(location);
      // set boundary cells if necessary
      d2y = interpolateAndExtrapolate(d2y, location, extrapolate_x, extrapolate_y, false,
                                      transform.get());

      d1_dy = -d2y / (dy * dy);
    }

#if BOUT_USE_METRIC_3D
    if (localmesh->get(d2z, "d2z" + suffix, 0.0, false)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2z' not found. Calculating from dz\n");
      d1_dz = bout::derivatives::index::DDZ(1. / dz);
      communicate(d1_dz);
      d1_dz =
          interpolateAndExtrapolate(d1_dz, location, true, true, true, transform.get());
    } else {
      d2z.setLocation(location);
      // set boundary cells if necessary
      d2z = interpolateAndExtrapolate(d2z, location, extrapolate_x, extrapolate_y, false,
                                      transform.get());

      d1_dz = -d2z / (dz * dz);
    }
#else
    d1_dz = 0;
#endif
  } else {
    if (localmesh->get(d2x, "d2x", 0.0, false)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2x' not found. Calculating from dx\n");
      d1_dx = bout::derivatives::index::DDX(1. / dx); // d/di(1/dx)

      communicate(d1_dx);
      d1_dx =
          interpolateAndExtrapolate(d1_dx, location, true, true, true, transform.get());
    } else {
      // Shift d2x to our location
      d2x = interpolateAndExtrapolate(d2x, location, true, true, false, transform.get());

      d1_dx = -d2x / (dx * dx);
    }

    if (localmesh->get(d2y, "d2y", 0.0, false)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2y' not found. Calculating from dy\n");
      d1_dy = DDY(1. / dy); // d/di(1/dy)

      communicate(d1_dy);
      d1_dy =
          interpolateAndExtrapolate(d1_dy, location, true, true, true, transform.get());
    } else {
      // Shift d2y to our location
      d2y = interpolateAndExtrapolate(d2y, location, true, true, false, transform.get());

      d1_dy = -d2y / (dy * dy);
    }

#if BOUT_USE_METRIC_3D
    if (localmesh->get(d2z, "d2z", 0.0, false)) {
      output_warn.write(
          "\tWARNING: differencing quantity 'd2z' not found. Calculating from dz\n");
      d1_dz = bout::derivatives::index::DDZ(1. / dz);

      communicate(d1_dz);
      d1_dz =
          interpolateAndExtrapolate(d1_dz, location, true, true, true, transform.get());
    } else {
      // Shift d2z to our location
      d2z = interpolateAndExtrapolate(d2z, location, true, true, false, transform.get());

      d1_dz = -d2z / (dz * dz);
    }
#else
    d1_dz = 0;
#endif
  }
  communicate(d1_dx, d1_dy, d1_dz);

  if (location == CELL_CENTRE && recalculate_staggered) {
    // Re-calculate interpolated Coordinates at staggered locations
    localmesh->recalculateStaggeredCoordinates();
  }

  // Invalidate and recalculate cached variables
  zlength_cache.reset();
  Grad2_par2_DDY_invSgCache.clear();
  invSgCache.reset();

  return 0;
}

void Coordinates::CalculateChristoffelSymbols() {
  // Calculate Christoffel symbol terms (18 independent values)
  // Note: This calculation is completely general: metric
  // tensor can be 2D or 3D. For 2D, all DDZ terms are zero

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();

  G1_11 = 0.5 * contravariant_components.g11 * DDX(covariantMetricTensor.Getg_11())
          + contravariant_components.g12
                * (DDX(covariantMetricTensor.Getg_12())
                   - 0.5 * DDY(covariantMetricTensor.Getg_11()))
          + contravariant_components.g13
                * (DDX(covariantMetricTensor.Getg_13())
                   - 0.5 * DDZ(covariantMetricTensor.Getg_11()));
  G1_22 = contravariant_components.g11
              * (DDY(covariantMetricTensor.Getg_12())
                 - 0.5 * DDX(covariantMetricTensor.Getg_22()))
          + 0.5 * contravariant_components.g12 * DDY(covariantMetricTensor.Getg_22())
          + contravariant_components.g13
                * (DDY(covariantMetricTensor.Getg_23())
                   - 0.5 * DDZ(covariantMetricTensor.Getg_22()));
  G1_33 = contravariant_components.g11
              * (DDZ(covariantMetricTensor.Getg_13())
                 - 0.5 * DDX(covariantMetricTensor.Getg_33()))
          + contravariant_components.g12
                * (DDZ(covariantMetricTensor.Getg_23())
                   - 0.5 * DDY(covariantMetricTensor.Getg_33()))
          + 0.5 * contravariant_components.g13 * DDZ(covariantMetricTensor.Getg_33());
  G1_12 =
      0.5 * contravariant_components.g11 * DDY(covariantMetricTensor.Getg_11())
      + 0.5 * contravariant_components.g12 * DDX(covariantMetricTensor.Getg_22())
      + 0.5 * contravariant_components.g13
            * (DDY(covariantMetricTensor.Getg_13()) + DDX(covariantMetricTensor.Getg_23())
               - DDZ(covariantMetricTensor.Getg_12()));
  G1_13 =
      0.5 * contravariant_components.g11 * DDZ(covariantMetricTensor.Getg_11())
      + 0.5 * contravariant_components.g12
            * (DDZ(covariantMetricTensor.Getg_12()) + DDX(covariantMetricTensor.Getg_23())
               - DDY(covariantMetricTensor.Getg_13()))
      + 0.5 * contravariant_components.g13 * DDX(covariantMetricTensor.Getg_33());
  G1_23 =
      0.5 * contravariant_components.g11
          * (DDZ(covariantMetricTensor.Getg_12()) + DDY(covariantMetricTensor.Getg_13())
             - DDX(covariantMetricTensor.Getg_23()))
      + 0.5 * contravariant_components.g12
            * (DDZ(covariantMetricTensor.Getg_22()) + DDY(covariantMetricTensor.Getg_23())
               - DDY(covariantMetricTensor.Getg_23()))
      // + 0.5 *g13*(DDZ(g_32) + DDY(g_33) - DDZ(g_23));
      // which equals
      + 0.5 * contravariant_components.g13 * DDY(covariantMetricTensor.Getg_33());

  G2_11 = 0.5 * contravariant_components.g12 * DDX(covariantMetricTensor.Getg_11())
          + contravariant_components.g22
                * (DDX(covariantMetricTensor.Getg_12())
                   - 0.5 * DDY(covariantMetricTensor.Getg_11()))
          + contravariant_components.g23
                * (DDX(covariantMetricTensor.Getg_13())
                   - 0.5 * DDZ(covariantMetricTensor.Getg_11()));
  G2_22 = contravariant_components.g12
              * (DDY(covariantMetricTensor.Getg_12())
                 - 0.5 * DDX(covariantMetricTensor.Getg_22()))
          + 0.5 * contravariant_components.g22 * DDY(covariantMetricTensor.Getg_22())
          + contravariant_components.g23
                * (DDY(contravariant_components.g23)
                   - 0.5 * DDZ(covariantMetricTensor.Getg_22()));
  G2_33 = contravariant_components.g12
              * (DDZ(covariantMetricTensor.Getg_13())
                 - 0.5 * DDX(covariantMetricTensor.Getg_33()))
          + contravariant_components.g22
                * (DDZ(covariantMetricTensor.Getg_23())
                   - 0.5 * DDY(covariantMetricTensor.Getg_33()))
          + 0.5 * contravariant_components.g23 * DDZ(covariantMetricTensor.Getg_33());
  G2_12 =
      0.5 * contravariant_components.g12 * DDY(covariantMetricTensor.Getg_11())
      + 0.5 * contravariant_components.g22 * DDX(covariantMetricTensor.Getg_22())
      + 0.5 * contravariant_components.g23
            * (DDY(covariantMetricTensor.Getg_13()) + DDX(covariantMetricTensor.Getg_23())
               - DDZ(covariantMetricTensor.Getg_12()));
  G2_13 =
      // 0.5 *g21*(DDZ(covariantMetricTensor.Getg_11()) + DDX(covariantMetricTensor.Getg_13()) - DDX(covariantMetricTensor.Getg_13()))
      // which equals
      0.5 * contravariant_components.g12
          * (DDZ(covariantMetricTensor.Getg_11()) + DDX(covariantMetricTensor.Getg_13())
             - DDX(covariantMetricTensor.Getg_13()))
      // + 0.5 *g22*(DDZ(covariantMetricTensor.Getg_21()) + DDX(covariantMetricTensor.Getg_23()) - DDY(covariantMetricTensor.Getg_13()))
      // which equals
      + 0.5 * contravariant_components.g22
            * (DDZ(covariantMetricTensor.Getg_12()) + DDX(covariantMetricTensor.Getg_23())
               - DDY(covariantMetricTensor.Getg_13()))
      // + 0.5 *g23*(DDZ(covariantMetricTensor.Getg_31()) + DDX(covariantMetricTensor.Getg_33()) - DDZ(g_13));
      // which equals
      + 0.5 * contravariant_components.g23 * DDX(covariantMetricTensor.Getg_33());
  G2_23 =
      0.5 * contravariant_components.g12
          * (DDZ(covariantMetricTensor.Getg_12()) + DDY(covariantMetricTensor.Getg_13())
             - DDX(covariantMetricTensor.Getg_23()))
      + 0.5 * contravariant_components.g22 * DDZ(covariantMetricTensor.Getg_22())
      + 0.5 * contravariant_components.g23 * DDY(covariantMetricTensor.Getg_33());

  G3_11 = 0.5 * contravariant_components.g13 * DDX(covariantMetricTensor.Getg_11())
          + contravariant_components.g23
                * (DDX(covariantMetricTensor.Getg_12())
                   - 0.5 * DDY(covariantMetricTensor.Getg_11()))
          + contravariant_components.g33
                * (DDX(covariantMetricTensor.Getg_13())
                   - 0.5 * DDZ(covariantMetricTensor.Getg_11()));
  G3_22 = contravariant_components.g13
              * (DDY(covariantMetricTensor.Getg_12())
                 - 0.5 * DDX(covariantMetricTensor.Getg_22()))
          + 0.5 * contravariant_components.g23 * DDY(covariantMetricTensor.Getg_22())
          + contravariant_components.g33
                * (DDY(covariantMetricTensor.Getg_23())
                   - 0.5 * DDZ(covariantMetricTensor.Getg_22()));
  G3_33 = contravariant_components.g13
              * (DDZ(covariantMetricTensor.Getg_13())
                 - 0.5 * DDX(covariantMetricTensor.Getg_33()))
          + contravariant_components.g23
                * (DDZ(covariantMetricTensor.Getg_23())
                   - 0.5 * DDY(covariantMetricTensor.Getg_33()))
          + 0.5 * contravariant_components.g33 * DDZ(covariantMetricTensor.Getg_33());
  G3_12 =
      // 0.5 *g31*(DDY(covariantMetricTensor.Getg_11()) + DDX(covariantMetricTensor.Getg_12()) - DDX(covariantMetricTensor.Getg_12()))
      // which equals to
      0.5 * contravariant_components.g13 * DDY(covariantMetricTensor.Getg_11())
      // + 0.5 *g32*(DDY(covariantMetricTensor.Getg_21()) + DDX(covariantMetricTensor.Getg_22()) - DDY(covariantMetricTensor.Getg_12()))
      // which equals to
      + 0.5 * contravariant_components.g23 * DDX(covariantMetricTensor.Getg_22())
      //+ 0.5 *g33*(DDY(covariantMetricTensor.Getg_31()) + DDX(covariantMetricTensor.Getg_32()) - DDZ(covariantMetricTensor.Getg_12()));
      // which equals to
      + 0.5 * contravariant_components.g33
            * (DDY(covariantMetricTensor.Getg_13()) + DDX(covariantMetricTensor.Getg_23())
               - DDZ(covariantMetricTensor.Getg_12()));
  G3_13 =
      0.5 * contravariant_components.g13 * DDZ(covariantMetricTensor.Getg_11())
      + 0.5 * contravariant_components.g23
            * (DDZ(covariantMetricTensor.Getg_12()) + DDX(covariantMetricTensor.Getg_23())
               - DDY(covariantMetricTensor.Getg_13()))
      + 0.5 * contravariant_components.g33 * DDX(covariantMetricTensor.Getg_33());
  G3_23 =
      0.5 * contravariant_components.g13
          * (DDZ(covariantMetricTensor.Getg_12()) + DDY(covariantMetricTensor.Getg_13())
             - DDX(covariantMetricTensor.Getg_23()))
      + 0.5 * contravariant_components.g23 * DDZ(covariantMetricTensor.Getg_22())
      + 0.5 * contravariant_components.g33 * DDY(covariantMetricTensor.Getg_33());
}

CovariantMetricTensor Coordinates::calcCovariant(const std::string& region) {
  TRACE("Coordinates::calcCovariant");
  covariantMetricTensor.calcCovariant(contravariantMetricTensor, location, region);
  return covariantMetricTensor;
}

ContravariantMetricTensor Coordinates::calcContravariant(const std::string& region) {
  TRACE("Coordinates::calcContravariant");
  contravariantMetricTensor.calcContravariant(covariantMetricTensor, location, region);
  return contravariantMetricTensor;
}

int Coordinates::jacobian() {
  TRACE("Coordinates::jacobian");
  // calculate Jacobian using g^-1 = det[g^ij], J = sqrt(g)

  const bool extrapolate_x = not localmesh->sourceHasXBoundaryGuards();
  const bool extrapolate_y = not localmesh->sourceHasYBoundaryGuards();

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();
  auto g = contravariant_components.g11 * contravariant_components.g22
               * contravariant_components.g33
           + 2.0 * contravariant_components.g12 * contravariant_components.g13
                 * contravariant_components.g23
           - contravariant_components.g11 * contravariant_components.g23
                 * contravariant_components.g23
           - contravariant_components.g22 * contravariant_components.g13
                 * contravariant_components.g13
           - contravariant_components.g33 * contravariant_components.g12
                 * contravariant_components.g12;

  // Check that g is positive
  bout::checkPositive(g, "The determinant of g^ij", "RGN_NOBNDRY");

  J = 1. / sqrt(g);
  // More robust to extrapolate derived quantities directly, rather than
  // deriving from extrapolated covariant metric components
  J = interpolateAndExtrapolate(J, location, extrapolate_x, extrapolate_y, false,
                                transform.get());

  Bxy = sqrt(covariantMetricTensor.Getg_22()) / J;
  Bxy = interpolateAndExtrapolate(Bxy, location, extrapolate_x, extrapolate_y, false,
                                  transform.get());

  return 0;
}

namespace {
// Utility function for fixing up guard cells of zShift
void fixZShiftGuards(Field2D& zShift) {
  auto localmesh = zShift.getMesh();

  // extrapolate into boundary guard cells if necessary
  zShift = interpolateAndExtrapolate(zShift, zShift.getLocation(),
                                     not localmesh->sourceHasXBoundaryGuards(),
                                     not localmesh->sourceHasYBoundaryGuards(), false);

  // make sure zShift has been communicated
  communicate(zShift);

  // Correct guard cells for discontinuity of zShift at poloidal branch cut
  for (int x = 0; x < localmesh->LocalNx; x++) {
    const auto lower = localmesh->hasBranchCutLower(x);
    if (lower.first) {
      for (int y = 0; y < localmesh->ystart; y++) {
        zShift(x, y) -= lower.second;
      }
    }
    const auto upper = localmesh->hasBranchCutUpper(x);
    if (upper.first) {
      for (int y = localmesh->yend + 1; y < localmesh->LocalNy; y++) {
        zShift(x, y) += upper.second;
      }
    }
  }
}
} // namespace

void Coordinates::setParallelTransform(Options* options) {

  auto ptoptions = options->getSection("paralleltransform");

  std::string ptstr;
  ptoptions->get("type", ptstr, "identity");

  // Convert to lower case for comparison
  ptstr = lowercase(ptstr);

  if (ptstr == "identity") {
    // Identity method i.e. no transform needed
    transform =
        bout::utils::make_unique<ParallelTransformIdentity>(*localmesh, ptoptions);

  } else if (ptstr == "shifted" or ptstr == "shiftedinterp") {
    // Shifted metric method

    Field2D zShift{localmesh};

    // Read the zShift angle from the mesh
    std::string suffix = getLocationSuffix(location);
    if (localmesh->sourceHasVar("dx" + suffix)) {
      // Grid file has variables at this location, so should be able to read
      checkStaggeredGet(localmesh, "zShift", suffix);
      if (localmesh->get(zShift, "zShift" + suffix, 0.0, false, location)) {
        // No zShift variable. Try qinty in BOUT grid files
        if (localmesh->get(zShift, "qinty" + suffix, 0.0, false, location)) {
          // Failed to find either variable, cannot use ShiftedMetric
          throw BoutException("Could not read zShift" + suffix + " from grid file");
        }
      }
    } else {
      if (location == CELL_YLOW and bout::build::use_metric_3d) {
        throw BoutException("Cannot interpolate zShift to construct ShiftedMetric when "
                            "using 3d metrics. You must provide zShift_ylow in the grid "
                            "file.");
      }
      Field2D zShift_centre;
      if (localmesh->get(zShift_centre, "zShift", 0.0, false)) {
        // No zShift variable. Try qinty in BOUT grid files
        if (localmesh->get(zShift_centre, "qinty", 0.0, false)) {
          // Failed to find either variable, cannot use ShiftedMetric
          throw BoutException("Could not read zShift from grid file");
        }
      }

      fixZShiftGuards(zShift_centre);

      zShift = interpolateAndExtrapolate(zShift_centre, location, true, true, false,
                                         transform.get());
    }

    fixZShiftGuards(zShift);

    if (ptstr == "shifted") {
      transform = bout::utils::make_unique<ShiftedMetric>(*localmesh, location, zShift,
                                                          getUniform(zlength()));
    } else if (ptstr == "shiftedinterp") {
      transform = bout::utils::make_unique<ShiftedMetricInterp>(
          *localmesh, location, zShift, getUniform(zlength()));
    }

  } else if (ptstr == "fci") {

    if (location != CELL_CENTRE) {
      throw BoutException("FCITransform is not available on staggered grids.");
    }

    // Flux Coordinate Independent method
    const bool fci_zperiodic = (*ptoptions)["z_periodic"].withDefault(true);
    transform =
        bout::utils::make_unique<FCITransform>(*localmesh, dy, fci_zperiodic, ptoptions);

  } else {
    throw BoutException(_("Unrecognised paralleltransform option.\n"
                          "Valid choices are 'identity', 'shifted', 'fci'"));
  }
}

/*******************************************************************************
 * Operators
 *
 *******************************************************************************/

Coordinates::FieldMetric Coordinates::DDX(const Field2D& f, CELL_LOC loc,
                                          const std::string& method,
                                          const std::string& region) {
  ASSERT1(location == loc || loc == CELL_DEFAULT);
  return bout::derivatives::index::DDX(f, loc, method, region) / dx;
}
Field3D Coordinates::DDX(const Field3D& f, CELL_LOC outloc, const std::string& method,
                         const std::string& region) {

  auto result = bout::derivatives::index::DDX(f, outloc, method, region);
  result /= dx;

  if (f.getMesh()->IncIntShear) {
    // Using BOUT-06 style shifting
    result += IntShiftTorsion * DDZ(f, outloc, method, region);
  }

  return result;
};

Coordinates::FieldMetric Coordinates::DDY(const Field2D& f, CELL_LOC loc,
                                          const std::string& method,
                                          const std::string& region) const {
  ASSERT1(location == loc || loc == CELL_DEFAULT);
  return bout::derivatives::index::DDY(f, loc, method, region) / dy;
}

Field3D Coordinates::DDY(const Field3D& f, CELL_LOC outloc, const std::string& method,
                         const std::string& region) const {
#if BOUT_USE_METRIC_3D
  if (!f.hasParallelSlices() and !transform->canToFromFieldAligned()) {
    Field3D f_parallel = f;
    transform->calcParallelSlices(f_parallel);
    f_parallel.applyParallelBoundary("parallel_neumann");
    return bout::derivatives::index::DDY(f_parallel, outloc, method, region);
  }
#endif
  return bout::derivatives::index::DDY(f, outloc, method, region) / dy;
};

Coordinates::FieldMetric Coordinates::DDZ(const Field2D& f, CELL_LOC loc,
                                          const std::string& UNUSED(method),
                                          const std::string& UNUSED(region)) {
  ASSERT1(location == loc || loc == CELL_DEFAULT);
  ASSERT1(f.getMesh() == localmesh);
  if (loc == CELL_DEFAULT) {
    loc = f.getLocation();
  }
  return zeroFrom(f).setLocation(loc);
}
Field3D Coordinates::DDZ(const Field3D& f, CELL_LOC outloc, const std::string& method,
                         const std::string& region) {
  return bout::derivatives::index::DDZ(f, outloc, method, region) / dz;
};

/////////////////////////////////////////////////////////
// Parallel gradient

Coordinates::FieldMetric Coordinates::Grad_par(const Field2D& var,
                                               MAYBE_UNUSED(CELL_LOC outloc),
                                               const std::string& UNUSED(method)) {
  TRACE("Coordinates::Grad_par( Field2D )");
  ASSERT1(location == outloc
          || (outloc == CELL_DEFAULT && location == var.getLocation()));

  return DDY(var) * invSg();
}

Field3D Coordinates::Grad_par(const Field3D& var, CELL_LOC outloc,
                              const std::string& method) {
  TRACE("Coordinates::Grad_par( Field3D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  return ::DDY(var, outloc, method) * invSg();
}

/////////////////////////////////////////////////////////
// Vpar_Grad_par
// vparallel times the parallel derivative along unperturbed B-field

Coordinates::FieldMetric Coordinates::Vpar_Grad_par(const Field2D& v, const Field2D& f,
                                                    MAYBE_UNUSED(CELL_LOC outloc),
                                                    const std::string& UNUSED(method)) {
  ASSERT1(location == outloc || (outloc == CELL_DEFAULT && location == f.getLocation()));

  return VDDY(v, f) * invSg();
}

Field3D Coordinates::Vpar_Grad_par(const Field3D& v, const Field3D& f, CELL_LOC outloc,
                                   const std::string& method) {
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  return VDDY(v, f, outloc, method) * invSg();
}

/////////////////////////////////////////////////////////
// Parallel divergence

Coordinates::FieldMetric Coordinates::Div_par(const Field2D& f, CELL_LOC outloc,
                                              const std::string& method) {
  TRACE("Coordinates::Div_par( Field2D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  // Need Bxy at location of f, which might be different from location of this
  // Coordinates object
  auto Bxy_floc = f.getCoordinates()->Bxy;

  return Bxy * Grad_par(f / Bxy_floc, outloc, method);
}

Field3D Coordinates::Div_par(const Field3D& f, CELL_LOC outloc,
                             const std::string& method) {
  TRACE("Coordinates::Div_par( Field3D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  // Need Bxy at location of f, which might be different from location of this
  // Coordinates object
  auto Bxy_floc = f.getCoordinates()->Bxy;

  if (!f.hasParallelSlices()) {
    // No yup/ydown fields. The Grad_par operator will
    // shift to field aligned coordinates
    return Bxy * Grad_par(f / Bxy_floc, outloc, method);
  }

  // Need to modify yup and ydown fields
  Field3D f_B = f / Bxy_floc;
  f_B.splitParallelSlices();
  for (int i = 0; i < f.getMesh()->ystart; ++i) {
    f_B.yup(i) = f.yup(i) / Bxy_floc.yup(i);
    f_B.ydown(i) = f.ydown(i) / Bxy_floc.ydown(i);
  }
  return Bxy * Grad_par(f_B, outloc, method);
}

/////////////////////////////////////////////////////////
// second parallel derivative (b dot Grad)(b dot Grad)
// Note: For parallel Laplacian use Laplace_par

Coordinates::FieldMetric Coordinates::Grad2_par2(const Field2D& f, CELL_LOC outloc,
                                                 const std::string& method) {
  TRACE("Coordinates::Grad2_par2( Field2D )");
  ASSERT1(location == outloc || (outloc == CELL_DEFAULT && location == f.getLocation()));

  auto result = Grad2_par2_DDY_invSg(outloc, method) * DDY(f, outloc, method)
                + D2DY2(f, outloc, method) / covariantMetricTensor.Getg_22();

  return result;
}

Field3D Coordinates::Grad2_par2(const Field3D& f, CELL_LOC outloc,
                                const std::string& method) {
  TRACE("Coordinates::Grad2_par2( Field3D )");
  if (outloc == CELL_DEFAULT) {
    outloc = f.getLocation();
  }
  ASSERT1(location == outloc);

  Field3D result = ::DDY(f, outloc, method);

  Field3D r2 = D2DY2(f, outloc, method) / covariantMetricTensor.Getg_22();

  result = Grad2_par2_DDY_invSg(outloc, method) * result + r2;

  ASSERT2(result.getLocation() == outloc);

  return result;
}

/////////////////////////////////////////////////////////
// perpendicular Laplacian operator

#include <bout/invert_laplace.hxx> // Delp2 uses same coefficients as inversion code

Coordinates::FieldMetric Coordinates::Delp2(const Field2D& f, CELL_LOC outloc,
                                            bool UNUSED(useFFT)) {
  TRACE("Coordinates::Delp2( Field2D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();
  auto result = G1 * DDX(f, outloc) + contravariant_components.g11 * D2DX2(f, outloc);

  return result;
}

Field3D Coordinates::Delp2(const Field3D& f, CELL_LOC outloc, bool useFFT) {
  TRACE("Coordinates::Delp2( Field3D )");

  if (outloc == CELL_DEFAULT) {
    outloc = f.getLocation();
  }

  ASSERT1(location == outloc);
  ASSERT1(f.getLocation() == outloc);

  if (localmesh->GlobalNx == 1 && localmesh->GlobalNz == 1) {
    // copy mesh, location, etc
    return f * 0;
  }
  ASSERT2(localmesh->xstart > 0); // Need at least one guard cell

  Field3D result{emptyFrom(f).setLocation(outloc)};

  if (useFFT and not bout::build::use_metric_3d) {
    int ncz = localmesh->LocalNz;

    // Allocate memory
    auto ft = Matrix<dcomplex>(localmesh->LocalNx, ncz / 2 + 1);
    auto delft = Matrix<dcomplex>(localmesh->LocalNx, ncz / 2 + 1);

    // Loop over y indices
    // Note: should not include y-guard or y-boundary points here as that would
    // use values from corner cells in dx, which may not be initialised.
    for (int jy = localmesh->ystart; jy <= localmesh->yend; jy++) {

      // Take forward FFT

      for (int jx = 0; jx < localmesh->LocalNx; jx++) {
        rfft(&f(jx, jy, 0), ncz, &ft(jx, 0));
      }

      // Loop over kz
      for (int jz = 0; jz <= ncz / 2; jz++) {

        // No smoothing in the x direction
        for (int jx = localmesh->xstart; jx <= localmesh->xend; jx++) {
          // Perform x derivative

          dcomplex a, b, c;
          laplace_tridag_coefs(jx, jy, jz, a, b, c, nullptr, nullptr, outloc);

          delft(jx, jz) = a * ft(jx - 1, jz) + b * ft(jx, jz) + c * ft(jx + 1, jz);
        }
      }

      // Reverse FFT
      for (int jx = localmesh->xstart; jx <= localmesh->xend; jx++) {

        irfft(&delft(jx, 0), ncz, &result(jx, jy, 0));
      }
    }
  } else {
    auto const contravariant_components =
        contravariantMetricTensor.getContravariantMetricTensor();
    result = G1 * ::DDX(f, outloc) + G3 * ::DDZ(f, outloc)
             + contravariant_components.g11 * ::D2DX2(f, outloc)
             + contravariant_components.g33 * ::D2DZ2(f, outloc)
             + 2 * contravariant_components.g13 * ::D2DXDZ(f, outloc);
  };

  ASSERT2(result.getLocation() == outloc);

  return result;
}

FieldPerp Coordinates::Delp2(const FieldPerp& f, CELL_LOC outloc, bool useFFT) {
  TRACE("Coordinates::Delp2( FieldPerp )");

  if (outloc == CELL_DEFAULT) {
    outloc = f.getLocation();
  }

  ASSERT1(location == outloc);
  ASSERT1(f.getLocation() == outloc);

  if (localmesh->GlobalNx == 1 && localmesh->GlobalNz == 1) {
    // copy mesh, location, etc
    return f * 0;
  }
  ASSERT2(localmesh->xstart > 0); // Need at least one guard cell

  FieldPerp result{emptyFrom(f).setLocation(outloc)};

  int jy = f.getIndex();
  result.setIndex(jy);

  if (useFFT) {
    int ncz = localmesh->LocalNz;

    // Allocate memory
    auto ft = Matrix<dcomplex>(localmesh->LocalNx, ncz / 2 + 1);
    auto delft = Matrix<dcomplex>(localmesh->LocalNx, ncz / 2 + 1);

    // Take forward FFT
    for (int jx = 0; jx < localmesh->LocalNx; jx++) {
      rfft(&f(jx, 0), ncz, &ft(jx, 0));
    }

    // Loop over kz
    for (int jz = 0; jz <= ncz / 2; jz++) {

      // No smoothing in the x direction
      for (int jx = localmesh->xstart; jx <= localmesh->xend; jx++) {
        // Perform x derivative

        dcomplex a, b, c;
        laplace_tridag_coefs(jx, jy, jz, a, b, c);

        delft(jx, jz) = a * ft(jx - 1, jz) + b * ft(jx, jz) + c * ft(jx + 1, jz);
      }
    }

    // Reverse FFT
    for (int jx = localmesh->xstart; jx <= localmesh->xend; jx++) {
      irfft(&delft(jx, 0), ncz, &result(jx, 0));
    }

  } else {
    throw BoutException("Non-fourier Delp2 not currently implented for FieldPerp.");
    // Would be the following but don't have standard derivative operators for FieldPerps
    // yet
    // result = G1 * ::DDX(f, outloc) + G3 * ::DDZ(f, outloc) + g11 * ::D2DX2(f, outloc)
    //          + g33 * ::D2DZ2(f, outloc) + 2 * g13 * ::D2DXDZ(f, outloc);
  };

  return result;
}

Coordinates::FieldMetric Coordinates::Laplace_par(const Field2D& f, CELL_LOC outloc) {
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  return D2DY2(f, outloc) / covariantMetricTensor.Getg_22()
         + DDY(J / covariantMetricTensor.Getg_22(), outloc) * DDY(f, outloc) / J;
}

Field3D Coordinates::Laplace_par(const Field3D& f, CELL_LOC outloc) {
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);
  return D2DY2(f, outloc) / covariantMetricTensor.Getg_22()
         + DDY(J / covariantMetricTensor.Getg_22(), outloc) * ::DDY(f, outloc) / J;
}

// Full Laplacian operator on scalar field

Coordinates::FieldMetric Coordinates::Laplace(const Field2D& f, CELL_LOC outloc,
                                              const std::string& dfdy_boundary_conditions,
                                              const std::string& dfdy_dy_region) {
  TRACE("Coordinates::Laplace( Field2D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();
  auto result = G1 * DDX(f, outloc) + G2 * DDY(f, outloc)
                + contravariant_components.g11 * D2DX2(f, outloc)
                + contravariant_components.g22 * D2DY2(f, outloc)
                + 2.0 * contravariant_components.g12
                      * D2DXDY(f, outloc, "DEFAULT", "RGN_NOBNDRY",
                               dfdy_boundary_conditions, dfdy_dy_region);

  return result;
}

Field3D Coordinates::Laplace(const Field3D& f, CELL_LOC outloc,
                             const std::string& dfdy_boundary_conditions,
                             const std::string& dfdy_dy_region) {
  TRACE("Coordinates::Laplace( Field3D )");
  ASSERT1(location == outloc || outloc == CELL_DEFAULT);

  auto const contravariant_components =
      contravariantMetricTensor.getContravariantMetricTensor();
  Field3D result = G1 * ::DDX(f, outloc) + G2 * ::DDY(f, outloc) + G3 * ::DDZ(f, outloc)
                   + contravariant_components.g11 * D2DX2(f, outloc)
                   + contravariant_components.g22 * D2DY2(f, outloc)
                   + contravariant_components.g33 * D2DZ2(f, outloc)
                   + 2.0
                         * (contravariant_components.g12
                                * D2DXDY(f, outloc, "DEFAULT", "RGN_NOBNDRY",
                                         dfdy_boundary_conditions, dfdy_dy_region)
                            + contravariant_components.g13 * D2DXDZ(f, outloc)
                            + contravariant_components.g23 * D2DYDZ(f, outloc));

  return result;
}

// Full perpendicular Laplacian, in form of inverse of Laplacian operator in LaplaceXY
// solver
Field2D Coordinates::Laplace_perpXY(MAYBE_UNUSED(const Field2D& A),
                                    MAYBE_UNUSED(const Field2D& f)) {
  TRACE("Coordinates::Laplace_perpXY( Field2D )");
#if not(BOUT_USE_METRIC_3D)
  Field2D result;
  result.allocate();
  for (auto i : result.getRegion(RGN_NOBNDRY)) {
    result[i] = 0.;

    // outer x boundary
    const auto outer_x_avg = [&i](const auto& f) { return 0.5 * (f[i] + f[i.xp()]); };
    const BoutReal outer_x_A = outer_x_avg(A);
    const BoutReal outer_x_J = outer_x_avg(J);
    const BoutReal outer_x_g11 =
        outer_x_avg(contravariantMetricTensor.getContravariantMetricTensor().g11);
    const BoutReal outer_x_dx = outer_x_avg(dx);
    const BoutReal outer_x_value =
        outer_x_A * outer_x_J * outer_x_g11 / (J[i] * outer_x_dx * dx[i]);
    result[i] += outer_x_value * (f[i.xp()] - f[i]);

    // inner x boundary
    const auto inner_x_avg = [&i](const auto& f) { return 0.5 * (f[i] + f[i.xm()]); };
    const BoutReal inner_x_A = inner_x_avg(A);
    const BoutReal inner_x_J = inner_x_avg(J);
    const BoutReal inner_x_g11 =
        inner_x_avg(contravariantMetricTensor.getContravariantMetricTensor().g11);
    const BoutReal inner_x_dx = inner_x_avg(dx);
    const BoutReal inner_x_value =
        inner_x_A * inner_x_J * inner_x_g11 / (J[i] * inner_x_dx * dx[i]);
    result[i] += inner_x_value * (f[i.xm()] - f[i]);

    // upper y boundary
    const auto upper_y_avg = [&i](const auto& f) { return 0.5 * (f[i] + f[i.yp()]); };
    auto const contravariant_components =
        contravariantMetricTensor.getContravariantMetricTensor();
    const BoutReal upper_y_A = upper_y_avg(A);
    const BoutReal upper_y_J = upper_y_avg(J);
    const BoutReal upper_y_g_22 = upper_y_avg(covariantMetricTensor.Getg_22());
    const BoutReal upper_y_g23 = upper_y_avg(contravariant_components.g23);
    const BoutReal upper_y_g_23 = upper_y_avg(covariantMetricTensor.Getg_23());
    const BoutReal upper_y_dy = upper_y_avg(dy);
    const BoutReal upper_y_value = -upper_y_A * upper_y_J * upper_y_g23 * upper_y_g_23
                                   / (upper_y_g_22 * J[i] * upper_y_dy * dy[i]);
    result[i] += upper_y_value * (f[i.yp()] - f[i]);

    // lower y boundary
    const auto lower_y_avg = [&i](const auto& f) { return 0.5 * (f[i] + f[i.ym()]); };
    const BoutReal lower_y_A = lower_y_avg(A);
    const BoutReal lower_y_J = lower_y_avg(J);
    const BoutReal lower_y_g_22 = lower_y_avg(covariantMetricTensor.Getg_22());
    const BoutReal lower_y_g23 = lower_y_avg(contravariant_components.g23);
    const BoutReal lower_y_g_23 = lower_y_avg(covariantMetricTensor.Getg_23());
    const BoutReal lower_y_dy = lower_y_avg(dy);
    const BoutReal lower_y_value = -lower_y_A * lower_y_J * lower_y_g23 * lower_y_g_23
                                   / (lower_y_g_22 * J[i] * lower_y_dy * dy[i]);
    result[i] += lower_y_value * (f[i.ym()] - f[i]);
  }

  return result;
#else
  throw BoutException("Coordinates::Laplace_perpXY for 3D metric not implemented");
#endif
}

const Coordinates::FieldMetric& Coordinates::invSg() const {
  if (invSgCache == nullptr) {
    auto ptr = std::make_unique<FieldMetric>();
    (*ptr) = 1.0 / sqrt(covariantMetricTensor.Getg_22());
    invSgCache = std::move(ptr);
  }
  return *invSgCache;
}

const Coordinates::FieldMetric&
Coordinates::Grad2_par2_DDY_invSg(CELL_LOC outloc, const std::string& method) const {
  if (auto search = Grad2_par2_DDY_invSgCache.find(method);
      search != Grad2_par2_DDY_invSgCache.end()) {
    return *search->second;
  }
  invSg();

  // Communicate to get parallel slices
  localmesh->communicate(*invSgCache);
  invSgCache->applyParallelBoundary("parallel_neumann");

  // cache
  auto ptr = std::make_unique<FieldMetric>();
  *ptr = DDY(*invSgCache, outloc, method) * invSg();
  Grad2_par2_DDY_invSgCache[method] = std::move(ptr);
  return *Grad2_par2_DDY_invSgCache[method];
}

void Coordinates::checkCovariant() {
  covariantMetricTensor.checkCovariant(localmesh->ystart);
}

void Coordinates::checkContravariant() {
  contravariantMetricTensor.checkContravariant(localmesh->ystart);
}

void Coordinates::setContravariantMetricTensor(ContravariantMetricTensor metric_tensor,
                                               const std::string& region) {
  contravariantMetricTensor.setContravariantMetricTensor(metric_tensor);
  covariantMetricTensor.calcCovariant(contravariantMetricTensor, location, region);
}

ContravariantMetricTensor::ContravariantComponents
Coordinates::getContravariantMetricTensor() const {
  return contravariantMetricTensor.getContravariantMetricTensor();
}

const CovariantMetricTensor::FieldMetric& Coordinates::g_11() const {
  return covariantMetricTensor.Getg_11();
}
const CovariantMetricTensor::FieldMetric& Coordinates::g_22() const {
  return covariantMetricTensor.Getg_22();
}
const CovariantMetricTensor::FieldMetric& Coordinates::g_33() const {
  return covariantMetricTensor.Getg_33();
}
const CovariantMetricTensor::FieldMetric& Coordinates::g_12() const {
  return covariantMetricTensor.Getg_12();
}
const CovariantMetricTensor::FieldMetric& Coordinates::g_13() const {
  return covariantMetricTensor.Getg_13();
}
const CovariantMetricTensor::FieldMetric& Coordinates::g_23() const {
  return covariantMetricTensor.Getg_23();
}

void Coordinates::setCovariantMetricTensor(CovariantMetricTensor metric_tensor) {
  covariantMetricTensor.setCovariantMetricTensor(metric_tensor);
  contravariantMetricTensor.calcContravariant(covariantMetricTensor, location);
}
