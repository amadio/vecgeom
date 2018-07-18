/// @file PlacedTet.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/PlacedTet.h"
#include "volumes/SpecializedTet.h"
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
  return nullptr; // There is no suitable TGeo shape 2018.07.18
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTet::ConvertToGeant4() const
{
  Vector3D<Precision> p0, p1, p2, p3;
  GetVertices(p0, p1, p2, p3);
  return new G4Tet("", G4ThreeVector(p0.x(), p0.y(), p0.z()), G4ThreeVector(p1.x(), p1.y(), p1.z()),
                   G4ThreeVector(p2.x(), p2.y(), p2.z()), G4ThreeVector(p3.x(), p3.y(), p3.z()));
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTet)

#endif // VECCORE_CUDA

} // End namespace vecgeom
