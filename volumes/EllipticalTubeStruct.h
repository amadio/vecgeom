/// @file EllipticalTubeStruct.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_ELLIPTICALTUBESTRUCT_H_
#define VECGEOM_VOLUMES_ELLIPTICALTUBESTRUCT_H_
#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct EllipticalTubeStruct {

  // Elliptical Tube parameters
  //
  T fDx; // semi-axis in X
  T fDy; // semi-axis in Y
  T fDz; // half length in Z

  T fSurfaceArea; // area of the surface
  T fCubicVolume; // volume

  // Precalculated cached values
  //
  T fRsph;    // R of bounding sphere
  T fDDx;     // Dx squared
  T fDDy;     // Dy squared
  T fSx;      // X scale factor
  T fSy;      // Y scale factor
  T fR;       // resulting Radius, after scaling elipse to circle
  T fQ1;      // approximation of dist = Q1*(x^2+y^2) - Q2
  T fQ2;      // approximation of dist = Q1*(x^2+y^2) - Q2
  T fScratch; // Half length of scratching segment squared
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
