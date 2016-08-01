/// @file PlacedOrb.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedOrb.h"
#include "volumes/SpecializedOrb.h"
#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
#include "UOrb.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Orb.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedOrb::PrintType() const
{
  printf("PlacedOrb");
}

void PlacedOrb::PrintType(std::ostream &s) const
{
  s << "PlacedOrb";
}

#ifndef VECGEOM_NVCC

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

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedOrb::ConvertToUSolids() const
{
  return new UOrb(GetLabel(), GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedOrb::ConvertToGeant4() const
{
  return new G4Orb("", GetRadius());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedOrb)

#endif // VECGEOM_NVCC

} // End namespace vecgeom
