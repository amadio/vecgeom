#include "volumes/LogicalVolume.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace vecgeom;

// dummy Region class
class MyRegion {
};

class MyRegion2 {
};

int main()
{
  // testing region concept on ExN03 example geometry
  TGeoManager::SetVerboseLevel(0);
  TGeoManager::Import("ExN03.root");
  RootGeoManager::Instance().LoadRootGeometry();

  GeoManager &geom = GeoManager::Instance();
  // hierarchy of ExN03 geometry is
  // World  - Calorimeter - Layer - [ Lead | liquidArgon ]
  auto leadvolume  = const_cast<LogicalVolume *>(geom.FindLogicalVolume("Lead"));
  auto worldvolume = const_cast<LogicalVolume *>(geom.FindLogicalVolume("World"));
  auto layervolume = const_cast<LogicalVolume *>(geom.FindLogicalVolume("Layer"));
  auto calovolume  = const_cast<LogicalVolume *>(geom.FindLogicalVolume("Calorimeter"));
  auto argonvolume = const_cast<LogicalVolume *>(geom.FindLogicalVolume("liquidArgon"));

  MyRegion *region1 = new MyRegion;
  MyRegion *region2 = new MyRegion;

  leadvolume->SetRegion(region1);
  assert(leadvolume->GetRegion<MyRegion>() == region1);
  assert(worldvolume->GetRegion<MyRegion>() == nullptr);
  assert(argonvolume->GetRegion<MyRegion>() == nullptr);

  // now do something with a pushdown
  calovolume->SetRegion(region2);
  assert(worldvolume->GetRegion<MyRegion>() == nullptr);
  assert(calovolume->GetRegion<MyRegion>() == region2);
  assert(layervolume->GetRegion<MyRegion>() == region2);
  assert(leadvolume->GetRegion<MyRegion>() == region2);
  assert(argonvolume->GetRegion<MyRegion>() == region2);

  // now do something without pushdown
  calovolume->SetRegion(region1, false);
  assert(worldvolume->GetRegion<MyRegion>() == nullptr);
  assert(calovolume->GetRegion<MyRegion>() == region1);
  assert(layervolume->GetRegion<MyRegion>() == region2);
  assert(leadvolume->GetRegion<MyRegion>() == region2);
  assert(argonvolume->GetRegion<MyRegion>() == region2);

  MyRegion const *reg = leadvolume->GetRegion<MyRegion>();
  assert(reg == region2);

  std::cout << "test passed \n";
  return 0;
}
