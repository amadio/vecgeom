/// \file UnplacedHype.cpp

#include "volumes/UnplacedHype.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedHype.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

void UnplacedHype::Print() const {
  // NYI
}

void UnplacedHype::Print(std::ostream &os) const {
  // NYI
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedHype::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    return new(placement) SpecializedHype<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
        logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
        logical_volume, transformation);
#endif
  }
  return new SpecializedHype<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedHype::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedHype>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedHype_CopyToGpu(VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedHype::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedHype_CopyToGpu(gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedHype::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedHype>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedHype_ConstructOnGpu(VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedHype();
}

void UnplacedHype_CopyToGpu(VUnplacedVolume *const gpu_ptr) {
  UnplacedHype_ConstructOnGpu<<<1, 1>>>(gpu_ptr);
}

#endif

} // End namespace vecgeom