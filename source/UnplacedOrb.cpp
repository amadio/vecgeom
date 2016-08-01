/// \file UnplacedOrb.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedOrb.h"
#include "backend/Backend.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedOrb.h"
#include "base/RNG.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
UnplacedOrb::UnplacedOrb() : fCubicVolume(0), fSurfaceArea(0), fEpsilon(2.e-11), fRTolerance(0.)
{
  // default constructor
  fGlobalConvexity = true;
  SetRadialTolerance();
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedOrb::UnplacedOrb(const Precision r) : fOrb(r)
{
  fCubicVolume     = (4 * kPi / 3) * fOrb.fR * fOrb.fR * fOrb.fR;
  fSurfaceArea     = (4 * kPi) * fOrb.fR * fOrb.fR;
  fGlobalConvexity = true;
  SetRadialTolerance();
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedOrb::SetRadius(Precision r)
{
  fOrb.fR      = r;
  fCubicVolume = (4 * kPi / 3) * fOrb.fR * fOrb.fR * fOrb.fR;
  fSurfaceArea = (4 * kPi) * fOrb.fR * fOrb.fR;
  SetRadialTolerance();
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedOrb::SetRadialTolerance()
{
  fRTolerance = Max(kTolerance, fEpsilon * fOrb.fR);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedOrb::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  aMin.Set(-fOrb.fR);
  aMax.Set(fOrb.fR);
}

Vector3D<Precision> UnplacedOrb::GetPointOnSurface() const
{
  //  generate a random number from zero to 2UUtils::kPi...
  Precision phi    = RNG::Instance().uniform(0., 2. * kPi);
  Precision cosphi = std::cos(phi);
  Precision sinphi = std::sin(phi);

  // generate a random point uniform in area
  Precision costheta = RNG::Instance().uniform(-1., 1.);
  Precision sintheta = std::sqrt(1. - (costheta * costheta));

  return Vector3D<Precision>(fOrb.fR * sintheta * cosphi, fOrb.fR * sintheta * sinphi, fOrb.fR * costheta);
}

std::string UnplacedOrb::GetEntityType() const
{
  return "Orb\n";
}

#if defined(VECGEOM_USOLIDS)
VECGEOM_CUDA_HEADER_BOTH
void UnplacedOrb::GetParametersList(int, double *aArray) const
{
  aArray[0] = GetRadius();
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedOrb *UnplacedOrb::Clone() const
{
  return new UnplacedOrb(fOrb.fR);
}

// VECGEOM_CUDA_HEADER_BOTH
std::ostream &UnplacedOrb::StreamInfo(std::ostream &os) const
// Definition taken from UOrb
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     //  << "     *** Dump for solid - " << GetName() << " ***\n"
     //  << "     ===================================================\n"

     << " Solid type: UOrb\n"
     << " Parameters: \n"

     << "       outer radius: " << fOrb.fR << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}
#endif

void UnplacedOrb::Print() const
{
  printf("UnplacedOrb {%.2f}", GetRadius());
}

void UnplacedOrb::Print(std::ostream &os) const
{
  os << "UnplacedOrb {" << GetRadius() << "}";
}

#ifndef VECGEOM_NVCC
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedOrb::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedOrb<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedOrb<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedOrb::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedOrb>(volume, transformation, trans_code, rot_code, placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume *UnplacedOrb::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, const int id,
                                   VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedOrb<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedOrb<trans_code, rot_code>(logical_volume, transformation, id);
}

__device__ VPlacedVolume *UnplacedOrb::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         const TranslationCode trans_code, const RotationCode rot_code,
                                                         const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedOrb>(volume, transformation, trans_code, rot_code, id,
                                                            placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedOrb::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedOrb>(in_gpu_ptr, GetRadius());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedOrb::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedOrb>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedOrb>::SizeOf();
template void DevicePtr<cuda::UnplacedOrb>::Construct(const Precision r) const;

} // End cxx namespace

#endif
} // End global namespace
