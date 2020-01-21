// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members of the UnplacedEllipticalCone class.
/// @file volumes/EllipticalConeStruct.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_ELLIPTICALCONESTRUCT_H_
#define VECGEOM_VOLUMES_ELLIPTICALCONESTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct with data members of the UnplacedEllipticalCone class
//
template <typename T = double>
struct EllipticalConeStruct {

  // Elliptical Cone parameters: (x/dx)^2 + (y/dy)^2 = (z-dz)^2
  //
  T fDx;   ///< X semi-axis at Z = (height - 1), 1/dx - inclination tangent in X
  T fDy;   ///< Y semi-axis at Z = (height - 1), 1/dy - inclination tangent in Y
  T fDz;   ///< height, Z coordinate of apex
  T fZCut; ///< Z cut

  T fSurfaceArea; ///< area of the surface
  T fCubicVolume; ///< volume

  // Precalculated cached values
  //
  T fRsph;      ///< radius of bounding sphere
  T invDx;      ///< 1/dx
  T invDy;      ///< 1/dy
  T cosAxisMin; ///< min cosine of inclination angle
  T dApex;      ///< apex offset needed for "flying away" check
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
