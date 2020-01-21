/// @file PlacedSphere.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "VecGeom/volumes/PlacedSphere.h"
#include "VecGeom/volumes/Sphere.h"
#include "VecGeom/volumes/SpecializedSphere.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedSphere::ConvertToUnspecialized() const
{
  return new SimpleSphere(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}
#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedSphere)

#endif // VECCORE_CUDA

} // End global namespace
