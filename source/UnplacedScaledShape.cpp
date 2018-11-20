/// \file UnplacedScaledShape.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/UnplacedScaledShape.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedScaledShape.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#endif
#include <stdio.h>

#ifdef VECGEOM_CUDA_INTERFACE
#include "management/CudaManager.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void UnplacedScaledShape::Print() const
{
  printf("UnplacedScaledShape: scale:{%g, %g, %g} shape: ", fScaled.fScale.Scale()[0], fScaled.fScale.Scale()[1],
         fScaled.fScale.Scale()[2]);
  //  UnscaledShape()->Print();
}

void UnplacedScaledShape::Print(std::ostream &os) const
{
  os << "UnplacedScaledShape: " << fScaled.fScale.Scale() << *UnscaledShape();
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool UnplacedScaledShape::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  bool valid = false;
  ScaledShapeImplementation::NormalKernel<double>(fScaled, point, normal, valid);
  return valid;
}

//______________________________________________________________________________
void UnplacedScaledShape::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  // First get the extent of the unscaled shape
  fScaled.fPlaced->Extent(aMin, aMax);
  // The center of the extent may not be in the origin
  Vector3D<Precision> pos;
  pos = 0.5 * (aMin + aMax);
  Vector3D<Precision> center, semilengths;
  fScaled.fScale.InverseTransform(pos, center);
  // The lenghts are also scaled
  pos = 0.5 * (aMax - aMin);
  fScaled.fScale.InverseTransform(pos, semilengths);
  aMin = center - semilengths;
  aMax = center + semilengths;
}

//______________________________________________________________________________
Vector3D<Precision> UnplacedScaledShape::SamplePointOnSurface() const
{
  // Sample the scaled shape
  Vector3D<Precision> sampled;
  fScaled.fScale.InverseTransform(fScaled.fPlaced->GetUnplacedVolume()->SamplePointOnSurface(), sampled);
  return sampled;
}

//______________________________________________________________________________
template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedScaledShape::Create(LogicalVolume const *const logical_volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedScaledShape<trans_code, rot_code>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                                 ,
                                                                 id
#endif
    );
    return placement;
  }
  return new SpecializedScaledShape<trans_code, rot_code>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                          ,
                                                          id
#endif
  );
}

//______________________________________________________________________________
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedScaledShape::CreateSpecializedVolume(LogicalVolume const *const volume,
                                                            Transformation3D const *const transformation,
                                                            const TranslationCode trans_code,
                                                            const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                                            const int id,
#endif
                                                            VPlacedVolume *const placement)
{
  return VolumeFactory::CreateByTransformation<UnplacedScaledShape>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                                    id,
#endif
                                                                    placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedScaledShape::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  DevicePtr<cuda::VPlacedVolume> gpuptr = CudaManager::Instance().LookupPlaced(fScaled.fPlaced);
  Vector3D<Precision> const &scl        = fScaled.fScale.Scale();
  return CopyToGpuImpl<UnplacedScaledShape>(in_gpu_ptr, gpuptr, scl[0], scl[1], scl[2], fGlobalConvexity);
}

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedScaledShape::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedScaledShape>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedScaledShape>::SizeOf();
// template void DevicePtr<cuda::UnplacedScaledShape>::Construct(
//    DevicePtr<cuda::VPlacedVolume> gpuptr, Scale3D const scale) const;
template void DevicePtr<cuda::UnplacedScaledShape>::Construct(DevicePtr<cuda::VPlacedVolume> gpuptr, Precision sx,
                                                              Precision sy, Precision sz, bool globalConvexity) const;

} // namespace cxx

#endif

} // namespace vecgeom
