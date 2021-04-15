/// \file BVHManager.cpp
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/GeoManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void BVHManager::Init()
{
  auto lvmap = GeoManager::Instance().GetLogicalVolumesMap();
  Instance().fBVHs.resize(lvmap.size());
  for (auto item : lvmap)
    Instance().fBVHs[item.second->id()] = new BVH(*item.second);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
