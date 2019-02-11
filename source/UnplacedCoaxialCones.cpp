/// @file UnplacedCoaxialCones.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/EllipticUtilities.h"
#include "volumes/UnplacedCoaxialCones.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedCoaxialCones.h"
#include "base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/* Default constructor kept for reference only,
 * May be removed later
 */
VECCORE_ATT_HOST_DEVICE
UnplacedCoaxialCones::UnplacedCoaxialCones()
{
  // Call the SetParameters function if required
  //SetParameters
  fGlobalConvexity = true;
}

/*
 * All the required Parametric Constructor
 *
 */

VECCORE_ATT_HOST_DEVICE
void UnplacedCoaxialCones::CheckParameters()
{

}

VECCORE_ATT_HOST_DEVICE
void UnplacedCoaxialCones::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
}

Vector3D<Precision> UnplacedCoaxialCones::SamplePointOnSurface() const
{

}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedCoaxialCones::StreamInfo(std::ostream &os) const
// Definition taken from UCoaxialCones
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
	//Provided Elliptical Cone Parameters as done for Tube below
  //printf("CoaxialCones {%.2f, %.2f, %.2f}", fCoaxialCones.fDx, fCoaxialCones.fDy, fCoaxialCones.fDz);
}

void UnplacedCoaxialCones::Print(std::ostream &os) const
{
	//Provided Elliptical Cone Parameters as done for Tube below
  //os << "CoaxialCones {" << fEllipticalTube.fDx << ", " << fEllipticalTube.fDy << ", " << fEllipticalTube.fDz << "}";
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
                                              VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedCoaxialCones<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedCoaxialCones::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         const TranslationCode trans_code, const RotationCode rot_code,
                                                         const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedCoaxialCones>(volume, transformation, trans_code, rot_code, id,
                                                                       placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedCoaxialCones::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedCoaxialCones>(in_gpu_ptr, GetDx(), GetDy(), GetDz());
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
template void DevicePtr<cuda::UnplacedCoaxialCones>::Construct(const Precision dx, const Precision dy,
                                                                 const Precision dz) const;

} // namespace cxx

#endif
} // namespace vecgeom
