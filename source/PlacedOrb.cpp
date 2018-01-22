/// @file PlacedOrb.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedOrb.h"
#include "volumes/SpecializedOrb.h"
#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Orb.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedOrb::PrintType() const
{
  printf("PlacedOrb");
}

void PlacedOrb::PrintType(std::ostream &s) const
{
  s << "PlacedOrb";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedOrb::ConvertToUnspecialized() const
{
  return new SimpleOrb(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedOrb::ConvertToRoot() const
{
  return new TGeoSphere(GetLabel().c_str(), 0., GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedOrb::ConvertToGeant4() const
{
  return new G4Orb("", GetRadius());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedOrb)

#endif // VECCORE_CUDA

} // End namespace vecgeom
