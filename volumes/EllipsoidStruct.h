// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members of the UnplacedEllipsoid class
/// @file volumes/EllipsoidStruct.h
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_ELLIPSOIDSTRUCT_H_
#define VECGEOM_VOLUMES_ELLIPSOIDSTRUCT_H_
#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct with data members of the UnplacedEllipsoid class
template <typename T = double>
struct EllipsoidStruct {

  // Ellipsoid parameters
  T fDx;         ///< Semi-axis in x
  T fDy;         ///< Semi-axis in y
  T fDz;         ///< Semi-axis in z
  T fZBottomCut; ///< Bottom cut in z
  T fZTopCut;    ///< Top cut in z

  T fSurfaceArea; ///< Area of the surface
  T fCubicVolume; ///< Volume

  // Precalculated cached values
  T fXmax; ///< Extent in x
  T fYmax; ///< Extent in y
  T fRsph; ///< Radius of bounding sphere
  T fR;    ///< Resulting radius, after scaling ellipsoid to sphere
  T fSx;   ///< X scale factor
  T fSy;   ///< Y scale factor
  T fSz;   ///< Z scale factor

  // Scaled cuts
  T fScZBottomCut; ///< Scaled bottom cut in z
  T fScZTopCut;    ///< Scaled top cut in z
  T fScZMidCut;    ///< Scaled middle z position between cuts
  T fScZDimCut;    ///< Scaled half z dimension between cuts

  // Coefficients for approximation of distance near the surface
  T fQ1; ///< 1st coefficient in the approximation of dist = Q1*(x^2+y^2+z^2) - Q2
  T fQ2; ///< 2nd coefficient in the approximation of dist = Q1*(x^2+y^2+z^2) - Q2
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
