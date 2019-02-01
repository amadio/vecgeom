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

  // Elliptical Cone parameters
  //


  T fSurfaceArea; // area of the surface
  T fCubicVolume; // volume

  // Precalculated cached values
  //

};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
