/// @file EllipticalConeStruct.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_ELLIPTICALCONESTRUCT_H_
#define VECGEOM_VOLUMES_ELLIPTICALCONESTRUCT_H_
#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct EllipticalConeStruct {

  // Elliptical Cone parameters: (x/dx)^2 + (y/dy)^2 = (z-dz)^2
  //
  T fDx;   // semi-axis in X at Z = height-1, 1/dx - inclination tangent in X
  T fDy;   // semi-axis in Y at Z = height-1, 1/dy - inclination tangent in Y
  T fDz;   // height, Z coordinate of apex
  T fZCut; // Z cut

  T fSurfaceArea; // area of the surface
  T fCubicVolume; // volume

  // Precalculated cached values
  //
  T fRsph;      // R of bounding sphere
  T invDx;      // 1/dx
  T invDy;      // 1/dy
  T cosAxisMin; // min cosine of inclination angle
  T dApex;      // apex offset needed for "flying away" check
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
