#include "volumes/UnplacedSExtruVolume.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedSExtru.h"
#include "base/RNG.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void UnplacedSExtruVolume::Print() const
{
  printf("UnplacedSExtruVolume\n");
}

void UnplacedSExtruVolume::Print(std::ostream &os) const
{
  os << "UnplacedSExtruVolume";
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedSExtruVolume::Create(LogicalVolume const *const logical_volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedSExtru<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedSExtru<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedSExtruVolume::SpecializedVolume(LogicalVolume const *const volume,
                                                       Transformation3D const *const transformation,
                                                       const TranslationCode trans_code, const RotationCode rot_code,
                                                       VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedSExtruVolume>(volume, transformation, trans_code, rot_code,
                                                                     placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedSExtruVolume::Create(LogicalVolume const *const logical_volume,
                                            Transformation3D const *const transformation, const int id,
                                            VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedSExtru<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedSExtru<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE VPlacedVolume *UnplacedSExtruVolume::SpecializedVolume(LogicalVolume const *const volume,
                                                                          Transformation3D const *const transformation,
                                                                          const TranslationCode trans_code,
                                                                          const RotationCode rot_code, const int id,
                                                                          VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedSExtruVolume>(volume, transformation, trans_code, rot_code, id,
                                                                     placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedSExtruVolume::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const
{
  auto &vertices         = fPolyShell.fPolygon.GetVertices();
  Precision const *x_cpu = vertices.x();
  Precision const *y_cpu = vertices.y();
  const auto Nvert       = vertices.size();

  // copying the arrays needed for the constructor
  Precision *x_gpu_ptr = AllocateOnGpu<Precision>(Nvert * sizeof(Precision));
  Precision *y_gpu_ptr = AllocateOnGpu<Precision>(Nvert * sizeof(Precision));
  vecgeom::CopyToGpu(x_cpu, x_gpu_ptr, sizeof(Precision) * Nvert);
  vecgeom::CopyToGpu(y_cpu, y_gpu_ptr, sizeof(Precision) * Nvert);

  DevicePtr<cuda::VUnplacedVolume> gpusextru = CopyToGpuImpl<UnplacedSExtruVolume>(
      gpu_ptr, (int)Nvert, x_gpu_ptr, y_gpu_ptr, fPolyShell.fLowerZ, fPolyShell.fUpperZ);

  // remove temporary space from GPU
  FreeFromGpu(x_gpu_ptr);
  FreeFromGpu(y_gpu_ptr);

  return gpusextru;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedSExtruVolume::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedSExtruVolume>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedSExtruVolume>::SizeOf();
template void DevicePtr<cuda::UnplacedSExtruVolume>::Construct(int, double *, double *, Precision, Precision) const;

} // End cxx namespace

#endif

} // End global namespace
