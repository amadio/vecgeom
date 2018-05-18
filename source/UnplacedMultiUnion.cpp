#include "volumes/UnplacedMultiUnion.h"
#include "volumes/SpecializedMultiUnion.h"
#include "base/RNG.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

Precision UnplacedMultiUnion::Capacity() const
{
  if (fMultiUnion.fCapacity >= 0.) {
    return fMultiUnion.fCapacity;
  }
  // Sample in the solid extent and estimate capacity by counting how many points are
  // sampled inside
  const size_t nsamples = 100000;
  Vector3D<double> min, max;
  Extent(min, max);
  size_t ninside = 0;
  for (size_t i = 0; i < nsamples; ++i) {
    Vector3D<double> point(RNG::Instance().uniform(min.x(), max.x()), RNG::Instance().uniform(min.y(), max.y()),
                           RNG::Instance().uniform(min.z(), max.z()));
    if (Contains(point)) ninside++;
  }
  fMultiUnion.fCapacity = (double)ninside / nsamples;
  return fMultiUnion.fCapacity;
}

Precision UnplacedMultiUnion::SurfaceArea() const
{
  if (fMultiUnion.fSurfaceArea >= 0.) {
    return fMultiUnion.fSurfaceArea;
  }
  // Sample points on components, count how many are on the solid surface to correct
  // the surface area of each component
  const size_t nsamples    = 10000;
  fMultiUnion.fSurfaceArea = 0.;
  for (size_t i = 0; i < GetNumberOfSolids(); ++i) {
    size_t nsurf = 0;
    for (size_t ip = 0; ip < nsamples; ++ip) {
      Vector3D<double> point = GetNode(i)->GetTransformation()->InverseTransform(GetNode(i)->SamplePointOnSurface());
      if (Inside(point) == vecgeom::kSurface) nsurf++;
    }
    fMultiUnion.fSurfaceArea += GetNode(i)->SurfaceArea() * nsurf / nsamples;
  }
  return fMultiUnion.fSurfaceArea;
}

Vector3D<Precision> UnplacedMultiUnion::SamplePointOnSurface() const
{
  // Select a random component
  auto counter                = 0;
  VPlacedVolume const *volume = nullptr;
  Vector3D<double> point;
  size_t id = 0;
  do {
    if (counter == 0) {
      id     = (size_t)RNG::Instance().uniform(0., GetNumberOfSolids());
      volume = GetNode(id);
    }
    point   = volume->GetTransformation()->InverseTransform(volume->SamplePointOnSurface());
    counter = (counter + 1) % 1000;
  } while (Inside(point) != vecgeom::kSurface);
  return point;
}

bool UnplacedMultiUnion::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  // Compute normal to solid in a point
  return false;
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedMultiUnion::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedMultiUnion<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedMultiUnion<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedMultiUnion::SpecializedVolume(LogicalVolume const *const volume,
                                                     Transformation3D const *const transformation,
                                                     const TranslationCode trans_code, const RotationCode rot_code,
                                                     VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedMultiUnion>(volume, transformation, trans_code, rot_code,
                                                                   placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedMultiUnion::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation, const int id,
                                          VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedMultiUnion<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedMultiUnion<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedMultiUnion::SpecializedVolume(LogicalVolume const *const volume,
                                                     Transformation3D const *const transformation,
                                                     const TranslationCode trans_code, const RotationCode rot_code,
                                                     const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedMultiUnion>(volume, transformation, trans_code, rot_code, id,
                                                                   placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedMultiUnion::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedMultiUnion>(in_gpu_ptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedMultiUnion::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedMultiUnion>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedMultiUnion>::SizeOf();
template void DevicePtr<cuda::UnplacedMultiUnion>::Construct() const;

} // End cxx namespace

#endif

} // End global namespace
