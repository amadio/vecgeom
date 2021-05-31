/// \file BVHManager.cpp
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/GeoManager.h"

#include <vector>

namespace vecgeom {
#ifdef VECCORE_CUDA
inline
#endif
namespace cuda {

void *AllocateDeviceBVHBuffer(size_t n);
void FreeDeviceBVHBuffer();

VECCORE_ATT_DEVICE
BVH *GetDeviceBVH(int id);

} // namespace cuda

inline namespace VECGEOM_IMPL_NAMESPACE {

static std::vector<BVH*> hBVH;

void BVHManager::Init()
{
  auto lvmap = GeoManager::Instance().GetLogicalVolumesMap();
  hBVH.resize(lvmap.size());
  for (auto item : lvmap)
    hBVH[item.first] = item.second->GetDaughters().size() > 0 ? new BVH(*item.second) : nullptr;
}

#ifdef VECGEOM_CUDA_INTERFACE
void BVHManager::DeviceInit()
{
  int n = hBVH.size();

  BVH *ptr = (BVH*) vecgeom::cuda::AllocateDeviceBVHBuffer(n);

  for (int id = 0; id < n; ++id) {
    if (!hBVH[id])
      continue;

    hBVH[id]->CopyToGpu(&ptr[id]);
  }
}
#endif

BVH const *BVHManager::GetBVH(int id)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  return hBVH[id];
#else
  return GetDeviceBVH(id);
#endif
}

VECCORE_ATT_HOST_DEVICE
BVH const *BVHManager::GetBVH(LogicalVolume const *v)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  return hBVH[v->id()];
#else
  return GetDeviceBVH(v->id());
#endif
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
