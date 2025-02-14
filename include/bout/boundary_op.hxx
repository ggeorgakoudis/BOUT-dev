
class BoundaryOp;
class BoundaryModifier;

#ifndef __BNDRY_OP__
#define __BNDRY_OP__

#include "bout/boundary_region.hxx"
#include "bout/field2d.hxx"
#include "bout/field3d.hxx"
#include "bout/unused.hxx"
#include "bout/vector2d.hxx"
#include "bout/vector3d.hxx"

#include <cmath>
#include <list>
#include <string>

class BoundaryOpBase {
public:
  BoundaryOpBase() = default;
  virtual ~BoundaryOpBase() = default;

  /// Apply a boundary condition on field f
  virtual void apply(Field2D& f) = 0;
  virtual void apply(Field2D& f, BoutReal UNUSED(t)) { return apply(f); } //JMAD
  virtual void apply(Field3D& f) = 0;
  virtual void apply(Field3D& f, BoutReal UNUSED(t)) { return apply(f); } //JMAD

  virtual void apply(Vector2D& f) {
    apply(f.x);
    apply(f.y);
    apply(f.z);
  }

  virtual void apply(Vector3D& f) {
    apply(f.x);
    apply(f.y);
    apply(f.z);
  }
};

/// An operation on a boundary
class BoundaryOp : public BoundaryOpBase {
public:
  BoundaryOp() {
    bndry = nullptr;
    apply_to_ddt = false;
  }
  BoundaryOp(BoundaryRegion* region) {
    bndry = region;
    apply_to_ddt = false;
  }
  ~BoundaryOp() override = default;

  // Note: All methods must implement clone, except for modifiers (see below)
  virtual BoundaryOp* clone(BoundaryRegion* UNUSED(region),
                            const std::list<std::string>& UNUSED(args)) {
    throw BoutException("BoundaryOp::clone not implemented");
  }

  /// Clone using positional args and keywords
  /// If not implemented, check if keywords are passed, then call two-argument version
  virtual BoundaryOp* clone(BoundaryRegion* region, const std::list<std::string>& args,
                            const std::map<std::string, std::string>& keywords) {
    if (!keywords.empty()) {
      // Given keywords, but not using
      throw BoutException("Keywords ignored in boundary : {:s}", keywords.begin()->first);
    }

    return clone(region, args);
  }

  /// Apply a boundary condition on ddt(f)
  virtual void apply_ddt(Field2D& f) { apply(ddt(f)); }
  virtual void apply_ddt(Field3D& f) { apply(ddt(f)); }
  virtual void apply_ddt(Vector2D& f) { apply(ddt(f)); }
  virtual void apply_ddt(Vector3D& f) { apply(ddt(f)); }

  BoundaryRegion* bndry;
  // True if this boundary condition should be applied on the time derivatives
  // false if it should be applied to the field values
  bool apply_to_ddt;
};

class BoundaryModifier : public BoundaryOp {
public:
  BoundaryModifier() = default;
  BoundaryModifier(BoundaryOp* operation) : BoundaryOp(operation->bndry), op(operation) {}
  virtual BoundaryOp* cloneMod(BoundaryOp* op, const std::list<std::string>& args) = 0;

protected:
  BoundaryOp* op{nullptr};
};

#endif // __BNDRY_OP__
