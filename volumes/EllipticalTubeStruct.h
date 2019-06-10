// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members of the UnplacedEllipticalTube class
/// @file volumes/EllipticalTubeStruct.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_ELLIPTICALTUBESTRUCT_H_
#define VECGEOM_VOLUMES_ELLIPTICALTUBESTRUCT_H_
#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct with data members of the UnplacedEllipticalTube class
template <typename T = double>
struct EllipticalTubeStruct {

  // Elliptical Tube parameters
  //
  T fDx; ///< Semi-axis in X
  T fDy; ///< Semi-axis in Y
  T fDz; ///< Half length in Z

  T fSurfaceArea; ///< Area of the surface
  T fCubicVolume; ///< Volume

  // Precalculated cached values
  //
  T fRsph;    ///< Radius of bounding sphere
  T fDDx;     ///< Dx squared
  T fDDy;     ///< Dy squared
  T fSx;      ///< X scale factor
  T fSy;      ///< Y scale factor
  T fR;       ///< Resulting Radius, after scaling elipse to circle
  T fQ1;      ///< Coefficient in the approximation of dist = Q1*(x^2+y^2) - Q2
  T fQ2;      ///< Coefficient in the approximation of dist = Q1*(x^2+y^2) - Q2
  T fScratch; ///< Half length of scratching segment squared
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
