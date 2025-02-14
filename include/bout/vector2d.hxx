/*!
 * \file vector2d.hxx
 *
 * \brief Class for 2D vectors. Built on the Field2D class,
 * all operators relating to vectors are here (none in Field classes).
 * As with Field2D, Vector2D are constant in z (toroidal angle)
 * Components are either co- or contra-variant, depending on a flag. By default
 * they are covariant
 *
 * \author B. Dudson, October 2007
 *
 **************************************************************************
 * Copyright 2010 B.D.Dudson, S.Farley, M.V.Umansky, X.Q.Xu
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
 */

class Vector2D;

#pragma once
#ifndef __VECTOR2D_H__
#define __VECTOR2D_H__

class Field2D;
class Field3D;
class Vector3D;

#include <bout/coordinates.hxx>

/*!
 * A vector with three components (x,y,z) which only vary in 2D
 * (x and y). Implemented as a collection of three Field2D objects.
 */
class Vector2D : public FieldData {
public:
  Vector2D(const Vector2D& f);

  /// Many-argument constructor for fully specifying the initialisation of a Vector3D
  Vector2D(Mesh* localmesh = nullptr, bool covariant = true,
           CELL_LOC location = CELL_LOC::centre);

  ~Vector2D() override;

  Coordinates::FieldMetric x, y, z; ///< components

  bool covariant{true}; ///< true if the components are covariant (default)

  /// In-place conversion to covariant form
  void toCovariant();

  /// In-place conversion to contravariant form
  void toContravariant();

  /// Return a pointer to the time-derivative field
  Vector2D* timeDeriv();

  /// Assignment
  Vector2D& operator=(const Vector2D& rhs);

  /*!
   * Assign a BoutReal value. This sets all components
   * to the same value \p val.
   *
   * Vector2D v = 0.0;
   *
   * is equivalent to
   *
   * Vector2D v;
   * v.x = 0.0;
   * v.y = 0.0;
   * v.z = 0.0;
   *
   * The only real use for this is setting vector to zero.
   */
  Vector2D& operator=(BoutReal val);

  // operators

  Vector2D& operator+=(const Vector2D& rhs);

  /// Unary minus, changes sign of all components
  const Vector2D operator-() const;

  /// Subtract another vector
  Vector2D& operator-=(const Vector2D& rhs);

  /// Multiply all components by \p rhs
  Vector2D& operator*=(BoutReal rhs);

  /// Multiply all components by \p rhs
  Vector2D& operator*=(const Field2D& rhs);

  /// Divide all components by \p rhs
  Vector2D& operator/=(BoutReal rhs);

  /// Divide all components by \p rhs
  Vector2D& operator/=(const Field2D& rhs);

  // Binary operators

  const Vector2D operator+(const Vector2D& rhs) const; ///< Addition
  const Vector3D operator+(const Vector3D& rhs) const; ///< Addition

  const Vector2D operator-(const Vector2D& rhs) const; ///< Subtract vector \p rhs
  const Vector3D operator-(const Vector3D& rhs) const; ///< Subtract vector \p rhs

  const Vector2D operator*(BoutReal rhs) const; ///< Multiply all components by \p rhs
  const Vector2D
  operator*(const Field2D& rhs) const; ///< Multiply all components by \p rhs
  const Vector3D
  operator*(const Field3D& rhs) const; ///< Multiply all components by \p rhs

  const Vector2D operator/(BoutReal rhs) const; ///< Divides all components by \p rhs
  const Vector2D
  operator/(const Field2D& rhs) const; ///< Divides all components by \p rhs
  const Vector3D
  operator/(const Field3D& rhs) const; ///< Divides all components by \p rhs

  const Coordinates::FieldMetric operator*(const Vector2D& rhs) const; ///< Dot product
  const Field3D operator*(const Vector3D& rhs) const;                  ///< Dot product

  /// Set component locations consistently
  Vector2D& setLocation(CELL_LOC loc) override;

  /// Get component location
  CELL_LOC getLocation() const override;

  // FieldData virtual functions
  bool is3D() const override { return false; }
  int elementSize() const override { return 3; }

  /// Apply boundary condition to all fields
  void applyBoundary(bool init = false) override;
  void applyBoundary(const std::string& condition) {
    x.applyBoundary(condition);
    y.applyBoundary(condition);
    z.applyBoundary(condition);
  }
  void applyBoundary(const char* condition) { applyBoundary(std::string(condition)); }
  void applyTDerivBoundary() override;

private:
  Vector2D* deriv{nullptr};       ///< Time-derivative, can be NULL
  CELL_LOC location{CELL_CENTRE}; ///< Location of the variable in the cell
};

// Non-member overloaded operators

const Vector2D operator*(BoutReal lhs, const Vector2D& rhs);
const Vector2D operator*(const Field2D& lhs, const Vector2D& rhs);
const Vector3D operator*(const Field3D& lhs, const Vector2D& rhs);

/// Cross product
const Vector2D cross(const Vector2D& lhs, const Vector2D& rhs);
/// Cross product
const Vector3D cross(const Vector2D& lhs, const Vector3D& rhs);

/*!
 * Absolute value (Modulus) of given vector \p v
 *
 * |v| = sqrt( v dot v )
 */
Coordinates::FieldMetric abs(const Vector2D& v, const std::string& region = "RGN_ALL");

/// Transform to and from field-aligned coordinates
inline Vector2D toFieldAligned(Vector2D v,
                               const std::string& UNUSED(region) = "RGN_ALL") {
  // toFieldAligned is a null operation for the Field2D components of v, so return a copy
  // of the argument (hence pass-by-value instead of pass-by-reference)
  return v;
}
inline Vector2D fromFieldAligned(Vector2D v,
                                 const std::string& UNUSED(region) = "RGN_ALL") {
  // fromFieldAligned is a null operation for the Field2D components of v, so return a copy
  // of the argument (hence pass-by-value instead of pass-by-reference)
  return v;
}

/// Create new Vector2D with same attributes as the argument, but uninitialised components
inline Vector2D emptyFrom(const Vector2D& v) {
  auto result = Vector2D(v.getMesh(), v.covariant, v.getLocation());
  result.x = emptyFrom(v.x);
  result.y = emptyFrom(v.y);
  result.z = emptyFrom(v.z);

  return result;
}

/// Create new Vector2D with same attributes as the argument, and zero-initialised components
inline Vector2D zeroFrom(const Vector2D& v) {
  auto result = Vector2D(v.getMesh(), v.covariant, v.getLocation());
  result.x = zeroFrom(v.x);
  result.y = zeroFrom(v.y);
  result.z = zeroFrom(v.z);

  return result;
}

/*!
 * @brief Time derivative of 2D vector field
 */
inline Vector2D& ddt(Vector2D& f) { return *(f.timeDeriv()); }

#endif // __VECTOR2D_H__
