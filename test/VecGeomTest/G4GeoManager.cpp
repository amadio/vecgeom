#include "G4GeoManager.h"

#include "G4GDMLParser.hh"
#include "G4GeometryManager.hh"
#include "G4VPhysicalVolume.hh"

#ifdef VECGEOM_ROOT
// from VGM
#include "Geant4GM/volumes/Factory.h"
#include "RootGM/volumes/Factory.h"
// from ROOT
#include "TGeoManager.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

G4GeoManager::G4GeoManager() : fNavigator(nullptr)
{
}

void G4GeoManager::LoadG4Geometry(std::string gdmlfile, bool validate)
{
  G4GDMLParser parser;
  parser.Read(gdmlfile, validate);
  LoadG4Geometry(const_cast<G4VPhysicalVolume *>(parser.GetWorldVolume()));
}

// converts to a G4 geometry from ROOT using VGM
// expects a valid gGeoManager object for the import
// returns world pointer of in-memory representation
G4VPhysicalVolume *G4GeoManager::GetG4GeometryFromROOT()
{
  G4VPhysicalVolume *world(nullptr);
#ifdef VECGEOM_ROOT
  try {
    RootGM::Factory rtFactory;
    // rtFactory.SetDebug(1);
    rtFactory.SetIgnore(true);
    rtFactory.Import(gGeoManager->GetTopNode());

    Geant4GM::Factory g4Factory;

    // enable this to see debug output
    // g4Factory.SetDebug(1);
    // set to avoid crashes due to wrong material, etc. (which should
    // not affect the geometry test
    g4Factory.SetIgnore(true);
    rtFactory.Export(&g4Factory);
    world = g4Factory.World();
  } catch (...) {
    std::cerr << "caught exception in VGM; so return nullptr\n";
    world = nullptr;
  }
#else
// throw or put warning
#endif
  return world;
}

// sets a G4 geometry from existing G4PhysicalVolume
void G4GeoManager::LoadG4Geometry(G4VPhysicalVolume *world)
{
  // if there is an existing geometry
  if (fNavigator != nullptr) delete fNavigator;
  fNavigator = new G4Navigator();
  fNavigator->SetWorldVolume(world);

  // voxelize
  G4GeometryManager::GetInstance()->CloseGeometry(fNavigator->GetWorldVolume());
}
}
} // end namespaces
