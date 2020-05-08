/// @file UnplacedCoaxialCones.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/UnplacedCoaxialCones.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedCoaxialCones.h"
#include "VecGeom/base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void UnplacedCoaxialCones::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const {}

std::ostream &UnplacedCoaxialCones::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     //  << "     *** Dump for solid - " << GetName() << " ***\n"
     //  << "     ===================================================\n"

     << " Solid type: CoaxialCones\n"
     << " Parameters: \n"

     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

void UnplacedCoaxialCones::Print() const
{
  printf("CoaxialCones");
}

void UnplacedCoaxialCones::Print(std::ostream &os) const
{
  os << "CoaxialCones";
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedCoaxialCones::Create(LogicalVolume const *const logical_volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedCoaxialCones::SpecializedVolume(LogicalVolume const *const volume,
                                                       Transformation3D const *const transformation,
                                                       const TranslationCode trans_code, const RotationCode rot_code,
                                                       VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedCoaxialCones>(volume, transformation, trans_code, rot_code,
                                                                     placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedCoaxialCones::Create(LogicalVolume const *const logical_volume,
                                            Transformation3D const *const transformation, const int id,
                                            const int copy_no, const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement)
        SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedCoaxialCones::SpecializedVolume(LogicalVolume const *const volume,
                                                       Transformation3D const *const transformation,
                                                       const TranslationCode trans_code, const RotationCode rot_code,
                                                       const int id, const int copy_no, const int child_id,
                                                       VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedCoaxialCones>(volume, transformation, trans_code, rot_code, id,
                                                                     copy_no, child_id, placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedCoaxialCones::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  /* Transfer the geometry dimension arrays to the GPU and then use construtor
   * to create the geometry on GPU
   */
  Precision *rmin1_gpu_ptr = AllocateOnGpu<Precision>(fCoaxialCones.fRmin1Vect.size() * sizeof(Precision));
  Precision *rmax1_gpu_ptr = AllocateOnGpu<Precision>(fCoaxialCones.fRmax1Vect.size() * sizeof(Precision));
  Precision *rmin2_gpu_ptr = AllocateOnGpu<Precision>(fCoaxialCones.fRmin2Vect.size() * sizeof(Precision));
  Precision *rmax2_gpu_ptr = AllocateOnGpu<Precision>(fCoaxialCones.fRmax2Vect.size() * sizeof(Precision));

  vecgeom::CopyToGpu(&fCoaxialCones.fRmin1Vect[0], rmin1_gpu_ptr, sizeof(Precision) * fCoaxialCones.fRmin1Vect.size());
  vecgeom::CopyToGpu(&fCoaxialCones.fRmax1Vect[0], rmax1_gpu_ptr, sizeof(Precision) * fCoaxialCones.fRmax1Vect.size());
  vecgeom::CopyToGpu(&fCoaxialCones.fRmin2Vect[0], rmin2_gpu_ptr, sizeof(Precision) * fCoaxialCones.fRmin2Vect.size());
  vecgeom::CopyToGpu(&fCoaxialCones.fRmax2Vect[0], rmax2_gpu_ptr, sizeof(Precision) * fCoaxialCones.fRmax2Vect.size());

  DevicePtr<cuda::VUnplacedVolume> gpuCoaxialCones = CopyToGpuImpl<UnplacedCoaxialCones>(
      in_gpu_ptr, fCoaxialCones.fNumOfCones, rmin1_gpu_ptr, rmax1_gpu_ptr, rmin2_gpu_ptr, rmax2_gpu_ptr,
      fCoaxialCones.fDz, fCoaxialCones.fSPhi, fCoaxialCones.fDPhi);

  FreeFromGpu(rmin1_gpu_ptr);
  FreeFromGpu(rmax1_gpu_ptr);
  FreeFromGpu(rmin2_gpu_ptr);
  FreeFromGpu(rmax2_gpu_ptr);

  return gpuCoaxialCones;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedCoaxialCones::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedCoaxialCones>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedCoaxialCones>::SizeOf();
template void DevicePtr<cuda::UnplacedCoaxialCones>::Construct(
    unsigned int numOfCones, Precision *rmin1, Precision *rmax1, Precision *rmin2, Precision *rmax2, Precision dz,
    Precision sPhi,
    Precision dPhi) const; // const Precision dx, const Precision dy, const Precision dz) const;

} // namespace cxx

#endif
} // namespace vecgeom
