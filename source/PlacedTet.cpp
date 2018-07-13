/// @file PlacedTet.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedTet.h"
#include "volumes/SpecializedTet.h"
#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Tet.hh"
#include "G4ThreeVector.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTet::PrintType() const
{
  printf("PlacedTet");
}

void PlacedTet::PrintType(std::ostream &s) const
{
  s << "PlacedTet";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTet::ConvertToUnspecialized() const
{
  return new SimpleTet(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTet::ConvertToRoot() const
{
  // return new TGeoSphere(GetLabel().c_str(), 0., GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTet::ConvertToGeant4() const
{
  return new G4Tet("", G4ThreeVector(GetAnchor().x(), GetAnchor().y(), GetAnchor().z()),
                   G4ThreeVector(GetP2().x(), GetP2().y(), GetP2().z()),
                   G4ThreeVector(GetP3().x(), GetP3().y(), GetP3().z()),
                   G4ThreeVector(GetP4().x(), GetP4().y(), GetP4().z()));
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTet)

#endif // VECCORE_CUDA

} // End namespace vecgeom
