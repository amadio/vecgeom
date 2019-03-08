/// @file PlacedGenericPolycone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedGenericPolycone.h"
#include "volumes/SpecializedGenericPolycone.h"
#ifdef VECGEOM_ROOT

#endif
#ifdef VECGEOM_GEANT4
#include "G4GenericPolycone.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedGenericPolycone::PrintType() const
{
  printf("PlacedGenericPolycone");
}

void PlacedGenericPolycone::PrintType(std::ostream &s) const
{
  s << "PlacedGenericPolycone";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedGenericPolycone::ConvertToUnspecialized() const
{
  return new SimpleGenericPolycone(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedGenericPolycone::ConvertToRoot() const
{
  return nullptr; // There is no corresponding TGeo shape
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedGenericPolycone::ConvertToGeant4() const
{
  // Return the Geant4 GenericPolycone object
  return new G4GenericPolycone("", GetSPhi(), GetDPhi(), GetNumRz(), &GetR()[0], &GetZ()[0]);
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedGenericPolycone)

#endif // VECCORE_CUDA

} // namespace vecgeom
