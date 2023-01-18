/// @file PlacedCoaxialCones.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "VecGeom/volumes/PlacedCoaxialCones.h"
#include "VecGeom/volumes/SpecializedCoaxialCones.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedCoaxialCones::PrintType() const
{
  printf("PlacedCoaxialCones");
}

void PlacedCoaxialCones::PrintType(std::ostream &s) const
{
  s << "PlacedCoaxialCones";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedCoaxialCones::ConvertToUnspecialized() const
{
  return new SimpleCoaxialCones(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedCoaxialCones::ConvertToRoot() const
{
  return nullptr; // There is no corresponding TGeo shape
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedCoaxialCones::ConvertToGeant4() const
{
  return nullptr; // There is no corresponding G4 shape
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedCoaxialCones)

#endif // VECCORE_CUDA

} // namespace vecgeom
