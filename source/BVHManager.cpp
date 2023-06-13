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
void BVHManager::Init()
{
  std::vector<LogicalVolume const *> lvols;
  GeoManager::Instance().GetAllLogicalVolumes(lvols);
  // There may be volumes not used in the hierarchy, so the maximum index may be larger
  hBVH.resize(GeoManager::Instance().GetLogicalVolumesMap().size());
  for (auto logical_volume : lvols)
    hBVH[logical_volume->id()] = logical_volume->GetDaughters().size() > 0 ? new BVH(*logical_volume) : nullptr;
}

#ifdef VECGEOM_CUDA_INTERFACE
void BVHManager::DeviceInit()
{
  int n = hBVH.size();

  BVH *ptr = (BVH *)vecgeom::cuda::AllocateDeviceBVHBuffer(n);

  for (int id = 0; id < n; ++id) {
    if (!hBVH[id]) continue;

    hBVH[id]->CopyToGpu(&ptr[id]);
  }
}
#endif

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
