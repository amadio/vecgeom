/// \file UnplacedScaledShape.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/UnplacedScaledShape.h"

#include "backend/Backend.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedScaledShape.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#endif
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void UnplacedScaledShape::Print() const {
  printf("UnplacedScaledShape: scale:{%g, %g, %g} shape: ",
         fScale.Scale()[0],fScale.Scale()[1], fScale.Scale()[2]);
  UnscaledShape()->Print();	 
}

void UnplacedScaledShape::Print(std::ostream &os) const {
  os << "UnplacedScaledShape: " << fScale.Scale() << *UnscaledShape();
}


#ifndef VECGEOM_NVCC
//______________________________________________________________________________
void UnplacedScaledShape::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  // First get the extent of the unscaled shape
  fPlaced->Extent(aMin, aMax);
  // The center of the extent may not be in the origin
  Vector3D<Precision> pos;
  pos = 0.5 * (aMin+aMax);
  Vector3D<Precision> center, semilengths;
  fScale.InverseTransform(pos, center);
  // The lenghts are also scaled
  pos = 0.5 * (aMax-aMin);
  fScale.InverseTransform(pos, semilengths);
  aMin = center - semilengths;
  aMax = center + semilengths;
}
 
//______________________________________________________________________________
Vector3D<Precision> UnplacedScaledShape::GetPointOnSurface() const
{
  // Sample the scaled shape
  Vector3D<Precision> sampled;
  fScale.InverseTransform(fPlaced->GetPointOnSurface(), sampled);
  return sampled;
}

#endif

#ifndef VECGEOM_NVCC
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedScaledShape::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedScaledShape<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedScaledShape<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedScaledShape::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedScaledShape>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedScaledShape::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedScaledShape<trans_code, rot_code>(logical_volume,
                                                        transformation, id);
    return placement;
  }
  return new SpecializedScaledShape<trans_code, rot_code>(logical_volume,
                                                  transformation, id);
}

__device__
VPlacedVolume* UnplacedScaledShape::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedScaledShape>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedScaledShape::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   Vector3D<Precision> const &scale = fScale.Scale();
   return CopyToGpuImpl<UnplacedScaledShape>(in_gpu_ptr, UnscaledShape(), scale[0], scale[1], scale[2]);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedScaledShape::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedScaledShape>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedScaledShape>::SizeOf();
template void DevicePtr<cuda::UnplacedScaledShape>::Construct(
    VUnplacedVolume const *shape, Precision sx, Precision sy, Precision sz) const;

} // End cxx namespace

#endif

} // End global namespace
