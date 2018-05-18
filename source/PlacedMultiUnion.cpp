/// @file PlacedMultiUnion.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/MultiUnion.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_GEANT4
#include "G4MultiUnion.hh"
#include "G4ThreeVector.hh"
#include "G4Rotation.hh"
#include "G4Transform3D.hh"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedMultiUnion::ConvertToUnspecialized() const
{
  return new SimpleMultiUnion(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedMultiUnion::ConvertToRoot() const
{
  return nullptr;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedMultiUnion::ConvertToGeant4() const
{
  G4MultiUnion *munion         = new G4MultiUnion("");
  UnplacedMultiUnion *unplaced = GetUnplacedVolume();
  for (int i = 0; i < unplaced->GetNumberOfSolids(); ++i) {
    auto g4solid = unplaced->GetNode(i)->ConvertToGeant4();
    assert(g4solid && "Cannot convert component to Geant4 solid");
    auto trans         = unplaced->GetNode(i)->GetTransformation();
    const double *rotm = trans->Rotation();
    G4Rotation g4rot(G4ThreeVector(rotm[0], rotm[3], rotm[6]), G4ThreeVector(rotm[1], rotm[4], rotm[7]),
                     G4ThreeVector(rotm[2], rotm[5], rotm[8]));
    G4Transform3D *g4trans =
        new G4Transform3D(g4rot, G4ThreeVector(trans->Translation(0), trans->Translation(1), trans->Translation(2)));
    munion->AddNode(g4solid, g4trans);
  }
  munion->Voxelize();
  return munion;
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedMultiUnion)

#endif

} // End global namespace
