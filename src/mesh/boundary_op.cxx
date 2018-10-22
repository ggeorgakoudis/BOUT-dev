/***************************************************************************
 * Copyright 2018 B.D. Dudson, J.T. Omotani
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include <boundary_op.hxx>
#include <bout/constants.hxx>
#include <bout/mesh.hxx>

using namespace BoundaryOpUtilities;

BoutReal BoundaryOpUtilities::calcGridSpacing(Mesh* localmesh, CELL_LOC loc, int x, int bx, int y, int by) {
  ASSERT1( (bx != 0 || by != 0) && !(bx !=0 && by != 0) );
  if (localmesh->StaggerGrids) {
    if (loc == CELL_CENTRE || loc == CELL_ZLOW) {
      if (bx > 0) {
        // Outer x-boundary
        // Evaluate dx at the boundary between x and x-1
        return localmesh->getCoordinates(CELL_XLOW)->dx(x, y);
      } else if (bx < 0) {
        // Inner x-boundary
        // Evaluate dx at the boundary between x and x+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinates(CELL_XLOW)->dx(x+1, y);
      } else if (by > 0) {
        // Upper y-boundary
        // Evaluate dy at the boundary between y and y-1
        return localmesh->getCoordinates(CELL_YLOW)->dy(x, y);
      } else /*(by < 0)*/ {
        // Lower y-boundary
        // Evaluate dy at the boundary between y and y+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinates(CELL_YLOW)->dy(x, y+1);
      }
    } else if (loc == CELL_XLOW) {
      if (bx > 0) {
        // Outer x-boundary
        // Evaluate dx at the boundary between x and x-1
        return localmesh->getCoordinates(CELL_XLOW)->dx(x, y);
      } else if (bx < 0) {
        // Inner x-boundary
        // Evaluate dx at the boundary between x and x+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinates(CELL_XLOW)->dx(x+1, y);
      } else if (by > 0) {
        // Upper y-boundary
        // Evaluate dy at the boundary between y and y-1
        return localmesh->getCoordinatesXYCorner()->dy(x, y);
      } else /*(by < 0)*/ {
        // Lower y-boundary
        // Evaluate dy at the boundary between y and y+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinatesXYCorner()->dy(x, y+1);
      }
    } else /*if (loc == CELL_YLOW)*/ {
      if (bx > 0) {
        // Outer x-boundary
        // Evaluate dx at the boundary between x and x-1
        return localmesh->getCoordinatesXYCorner()->dx(x, y);
      } else if (bx < 0) {
        // Inner x-boundary
        // Evaluate dx at the boundary between x and x+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinatesXYCorner()->dx(x+1, y);
      } else if (by > 0) {
        // Upper y-boundary
        // Evaluate dy at the boundary between y and y-1
        return localmesh->getCoordinates(CELL_YLOW)->dy(x, y);
      } else /*(by < 0)*/ {
        // Lower y-boundary
        // Evaluate dy at the boundary between y and y+1
        // Need to multiply by sign of bx so derivative ends up with correct
        // direction/sign
        return -localmesh->getCoordinates(CELL_YLOW)->dy(x, y+1);
      }
    }
  } else {
    // this is not right for non-constant grid spacing, but don't have
    // interpolation code available for unstaggered grids, so leave behaviour
    // unchanged for now
    if (bx != 0) {
      return localmesh->getCoordinates()->dx(x,y);
    } else /*by != 0*/ {
      return localmesh->getCoordinates()->dy(x,y);
    }
  }
}

void BoundaryOp::extrapFurther(Field2D &f, int x, int bx, int y, int by, int z) {
  f(x, y, z) = 2.0*f(x - bx, y - by, z) - f(x - 2*bx, y - 2*by, z);
}
void BoundaryOp::extrapFurther(Field3D &f, int x, int bx, int y, int by, int z) {
  f(x, y, z) = 2.0*f(x - bx, y - by, z) - f(x - 2*bx, y - 2*by, z);
}

template<typename T>
void BoundaryOp::applyTemplate(T &f,BoutReal t) {
  // Set (at 2nd order) the value at the mid-point between the guard cell and the grid cell to be val
  // N.B. Only first guard cells (closest to the grid) should ever be used

  Mesh* localmesh = f.getMesh();

  // Check for staggered grids
  CELL_LOC loc = f.getLocation();

  bndry->first();

  // Decide which generator to use
  std::shared_ptr<FieldGenerator>  fg = gen;
  if(!fg) {
    fg = f.getBndryGenerator(bndry->location);
  }

  BoutReal val = 0.0;

  if (loc == CELL_CENTRE) {
    // no staggering
    for(; !bndry->isDone(); bndry->next1d()) {
      // Get the grid spacing at this boundary
      BoutReal delta = calcGridSpacing(localmesh, CELL_CENTRE, bndry->x, bndry->bx, bndry->y, bndry->by);

      // Calculate the X and Y normalised values half-way between the guard cell and grid cell 
      BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x)  // In the guard cell
          + localmesh->GlobalX(bndry->x - bndry->bx) ); // the grid cell

      BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y)  // In the guard cell
          + localmesh->GlobalY(bndry->y - bndry->by) ); // the grid cell

      for(int z=0; z<f.getNz(); z++) {
        // Calculate the Z normalized value (at either guard cell or grid cell
        BoutReal znorm = localmesh->GlobalZ(z);
        if (fg) {
          val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
        }
        applyAtPoint(f, val, bndry->x, bndry->bx, bndry->y, bndry->by, z, delta);

        // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
        for (int i = 1; i < bndry->width; i++) {
          int x = bndry->x + i*bndry->bx;
          int y = bndry->y + i*bndry->by;
          extrapFurther(f, x, bndry->bx, y, bndry->by, z);
        }
      }
    }
  } else if ( loc == CELL_XLOW ) {
    // field is shifted in X
    if(bndry->bx > 0) {
      // Outer x boundary
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_XLOW, bndry->x, bndry->bx, bndry->y, 0);

        BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x)
            + localmesh->GlobalX(bndry->x - bndry->bx) );
        BoutReal ynorm = localmesh->GlobalY(bndry->y);

        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg){
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }
          applyAtPointStaggered(f, val, bndry->x, bndry->bx, bndry->y, 0, z, delta);

          // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
          for (int i = 1; i < bndry->width; i++) {
            int x = bndry->x + i*bndry->bx;
            extrapFurther(f, x, bndry->bx, bndry->y, 0, z);
          }
        }
      }
    }
    if (bndry->bx < 0){
      // Inner x boundary. Set one point inwards
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_XLOW, bndry->x, bndry->bx, bndry->y, 0);

        BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x)
            + localmesh->GlobalX(bndry->x - bndry->bx) );
        BoutReal ynorm = localmesh->GlobalY(bndry->y);

        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg){
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }
          // Set one point inwards
          applyAtPointStaggered(f, val, bndry->x + 1, bndry->bx, bndry->y, 0, z, delta);

          // Need to set second and third guard cells, as may be used for interpolation or upwinding derivatives
          for (int i = 0; i < bndry->width; i++) {
            int x = bndry->x + i*bndry->bx;
            extrapFurther(f, x, bndry->bx, bndry->y, 0, z);
          }
        }
      }
    }
    if(bndry->by !=0){
      // y boundaries
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_XLOW, bndry->x, 0, bndry->y, bndry->by);

        // x norm is shifted by half a grid point because it is staggered.
        // y norm is located half way between first grid cell and guard cell.
        BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x) + localmesh->GlobalX(bndry->x - 1) );
        BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y) + localmesh->GlobalY(bndry->y - bndry->by) );

        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg){
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }
          applyAtPoint(f, val, bndry->x, 0, bndry->y, bndry->by, z, delta);

          // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
          for(int i=1;i<bndry->width;i++) {
            int y = bndry->y + i*bndry->by;
            extrapFurther(f, bndry->x, 0, y, bndry->by, z);
          }
        }
      }
    }
  } else if( loc == CELL_YLOW ) {
    // Shifted in Y
    if(bndry->by > 0) {
      // Upper y boundary boundary
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_YLOW, bndry->x, 0, bndry->y, bndry->by);

        BoutReal xnorm = localmesh->GlobalX(bndry->x);
        BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y) + localmesh->GlobalY(bndry->y - bndry->by) );
        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg){
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }
          applyAtPointStaggered(f, val, bndry->x, 0, bndry->y, bndry->by, z, delta);

          // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
          for(int i=1;i<bndry->width;i++) {
            int y = bndry->y + i*bndry->by;
            extrapFurther(f, bndry->x, 0, y, bndry->by, z);
          }
        }
      }
    }
    if(bndry->by < 0){
      // Lower y boundary. Set one point inwards
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_YLOW, bndry->x, 0, bndry->y, bndry->by);

        BoutReal xnorm = localmesh->GlobalX(bndry->x);
        BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y) + localmesh->GlobalY(bndry->y - bndry->by) );

        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg){
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }
          applyAtPointStaggered(f, val, bndry->x, 0, bndry->y+1, bndry->by, z, delta);

          // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
          for(int i=0;i<bndry->width;i++) {
            int y = bndry->y + i*bndry->by;
            extrapFurther(f, bndry->x, 0, y, bndry->by, z);
          }
        }
      }
    }
    if(bndry->bx != 0){
      // x boundaries
      for(; !bndry->isDone(); bndry->next1d()) {
        // Get the grid spacing at this boundary
        BoutReal delta = calcGridSpacing(localmesh, CELL_YLOW, bndry->x, bndry->bx, bndry->y, 0);

        // x norm is located half way between first grid cell and guard cell.
        // y norm is shifted by half a grid point because it is staggered.
        BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x) + localmesh->GlobalX(bndry->x - bndry->bx) );
        BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y) + localmesh->GlobalY(bndry->y - 1) );

        for(int z=0; z<f.getNz(); z++) {
          BoutReal znorm = localmesh->GlobalZ(z);
          if(fg) {
            val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
          }

          applyAtPoint(f, val, bndry->x, bndry->bx, bndry->y, 0, z, delta);

          // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
          for(int i=1;i<bndry->width;i++) {
            int x = bndry->x + i*bndry->bx;
            extrapFurther(f, x, bndry->bx, bndry->y, 0, z);
          }
        }
      }
    }
  } else if ( loc == CELL_ZLOW ){
    // Staggered in Z. Note there are no z-boundaries.
    for(; !bndry->isDone(); bndry->next1d()) {
      // Get the grid spacing at this boundary
      BoutReal delta = calcGridSpacing(localmesh, CELL_ZLOW, bndry->x, bndry->bx, bndry->y, bndry->by);

      // Calculate the X and Y normalised values half-way between the guard cell and grid cell 
      BoutReal xnorm = 0.5*(   localmesh->GlobalX(bndry->x)  // In the guard cell
          + localmesh->GlobalX(bndry->x - bndry->bx) ); // the grid cell

      BoutReal ynorm = 0.5*(   localmesh->GlobalY(bndry->y)  // In the guard cell
          + localmesh->GlobalY(bndry->y - bndry->by) ); // the grid cell

      for(int z=0; z<f.getNz(); z++) {
        // It shouldn't matter if znorm<0 because the expression in fg->generate should be periodic in z
        BoutReal znorm = 0.5*( localmesh->GlobalZ(z) + localmesh->GlobalZ(z - 1) ); // znorm is shifted by half a grid point because it is staggered
        if (fg) {
          val = fg->generate(xnorm,TWOPI*ynorm,TWOPI*znorm, t);
        }
        applyAtPoint(f, val, bndry->x, bndry->bx, bndry->y, bndry->by, z, delta);

        // Need to set second guard cell, as may be used for interpolation or upwinding derivatives
        for (int i = 1; i < bndry->width; i++) {
          int x = bndry->x + i*bndry->bx;
          int y = bndry->y + i*bndry->by;
          extrapFurther(f, x, bndry->bx, y, bndry->by, z);
        }
      }
    }
  }
}
//// instantiate template for Field2D and Field3D
//template
//void BoundaryOp::applyTemplate(Field2D &f,BoutReal t);
//template
//void BoundaryOp::applyTemplate(Field3D &f,BoutReal t);

void BoundaryOp::apply_ddt(Field2D &f) {
  Field2D *dt = f.timeDeriv();
  for(bndry->first(); !bndry->isDone(); bndry->next())
    for(int z=0; z<f.getNz(); z++)
      (*dt)(bndry->x, bndry->y, z) = 0.; // Set time derivative to zero
}

void BoundaryOp::apply_ddt(Field3D &f) {
  Field3D *dt = f.timeDeriv();
  for(bndry->first(); !bndry->isDone(); bndry->next())
    for(int z=0; z<f.getNz(); z++)
      (*dt)(bndry->x, bndry->y, z) = 0.; // Set time derivative to zero
}
