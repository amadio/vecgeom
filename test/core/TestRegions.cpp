#include "VecGeom/volumes/LogicalVolume.h"

#include "VecGeomTest/RootGeoManager.h"

#include "VecGeom/management/GeoManager.h"
#include "TGeoManager.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace vecgeom {
// for compatibility with CUDA
inline namespace VECGEOM_IMPL_NAMESPACE {
// this is our Region class
class Region {
};
}
}

using namespace vecgeom;

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

  Region *region1 = new Region;
  Region *region2 = new Region;

  leadvolume->SetRegion(region1);
  assert(leadvolume->GetRegion() == region1);
  assert(worldvolume->GetRegion() == nullptr);
  assert(argonvolume->GetRegion() == nullptr);

  // now do something with a pushdown
  calovolume->SetRegion(region2);
  assert(worldvolume->GetRegion() == nullptr);
  assert(calovolume->GetRegion() == region2);
  assert(layervolume->GetRegion() == region2);
  assert(leadvolume->GetRegion() == region2);
  assert(argonvolume->GetRegion() == region2);

  // now do something without pushdown
  calovolume->SetRegion(region1, false);
  assert(worldvolume->GetRegion() == nullptr);
  assert(calovolume->GetRegion() == region1);
  assert(layervolume->GetRegion() == region2);
  assert(leadvolume->GetRegion() == region2);
  assert(argonvolume->GetRegion() == region2);

  Region const *reg = leadvolume->GetRegion();
  assert(reg == region2);

  std::cout << "test passed \n";
  return 0;
}
