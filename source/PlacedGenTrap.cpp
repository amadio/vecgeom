/// \file PlacedGenTrap.cpp
/// \author: swenzel
/// Created on: Aug 3, 2014
///  Modified and completed: mihaela.gheata@cern.ch

#include "VecGeom/volumes/GenTrap.h"

#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4GenericTrap.hh"
#include "G4TwoVector.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void PlacedGenTrap::PrintType() const
{
  printf("PlacedGenTrap");
}

void PlacedGenTrap::PrintType(std::ostream &os) const
{
  os << "PlacedGenTrap";
}

#ifndef VECCORE_CUDA

//______________________________________________________________________________
VPlacedVolume const *PlacedGenTrap::ConvertToUnspecialized() const
{
  return new SimpleGenTrap(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
//______________________________________________________________________________
TGeoShape const *PlacedGenTrap::ConvertToRoot() const
{
  double vertexarray[16];
  for (int i = 0; i < 8; ++i) {
    vertexarray[2 * i]     = GetVertex(i).x();
    vertexarray[2 * i + 1] = GetVertex(i).y();
  }
  return new TGeoArb8(GetLabel().c_str(), GetDZ(), &vertexarray[0]);
}
#endif

#ifdef VECGEOM_GEANT4
//______________________________________________________________________________
G4VSolid const *PlacedGenTrap::ConvertToGeant4() const
{
  std::vector<G4TwoVector> vertices;
  for (int i = 0; i < 8; ++i) {
    vertices.push_back(G4TwoVector(GetVertex(i).x(), GetVertex(i).y()));
  }
  return new G4GenericTrap(GetLabel(), GetDZ(), vertices);
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedGenTrap)

#endif // VECCORE_CUDA

} // End namespace vecgeom
