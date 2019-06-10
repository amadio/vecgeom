// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedParaboloid.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedParaboloid.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedParaboloid.h"
#include "base/RNG.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedParaboloid::UnplacedParaboloid() : fCubicVolume(0), fSurfaceArea(0)
{
  // default constructor
  fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
UnplacedParaboloid::UnplacedParaboloid(const Precision rlo, const Precision rhi, const Precision dz)
    : fParaboloid(rlo, rhi, dz)
{
  CalcCapacity();
  CalcSurfaceArea();
  fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedParaboloid::CalcCapacity()
{
  fCubicVolume = kPi * fParaboloid.fDz * (fParaboloid.fRlo * fParaboloid.fRlo + fParaboloid.fRhi * fParaboloid.fRhi);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedParaboloid::CalcSurfaceArea()
{

  Precision h1, h2, A1, A2;
  h1 = -fParaboloid.fB + fParaboloid.fDz;
  h2 = -fParaboloid.fB - fParaboloid.fDz;

  // Calculate surface area for the paraboloid full paraboloid
  // cutoff at z = dz (not the cutoff area though).
  A1 = fParaboloid.fRhi2 + 4 * h1 * h1;
  A1 *= (A1 * A1); // Sets A1 = A1^3
  A1 = kPi * fParaboloid.fRhi / 6 / (h1 * h1) * (sqrt(A1) - fParaboloid.fRhi2 * fParaboloid.fRhi);

  // Calculate surface area for the paraboloid full paraboloid
  // cutoff at z = -dz (not the cutoff area though).
  A2 = fParaboloid.fRlo2 + 4 * (h2 * h2);
  A2 *= (A2 * A2); // Sets A2 = A2^3

  if (h2 != 0)
    A2 = kPi * fParaboloid.fRlo / 6 / (h2 * h2) * (Sqrt(A2) - fParaboloid.fRlo2 * fParaboloid.fRlo);
  else
    A2 = 0.;
  fSurfaceArea = (A1 - A2 + (fParaboloid.fRlo2 + fParaboloid.fRhi2) * kPi);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedParaboloid::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  aMin.x() = -fParaboloid.fDx;
  aMax.x() = fParaboloid.fDx;
  aMin.y() = -fParaboloid.fDy;
  aMax.y() = fParaboloid.fDy;
  aMin.z() = -fParaboloid.fDz;
  aMax.z() = fParaboloid.fDz;
}

Vector3D<Precision> UnplacedParaboloid::SamplePointOnSurface() const
{
  // G4 implementation
  Precision A   = SurfaceArea();
  Precision z   = RNG::Instance().uniform(0., 1.);
  Precision phi = RNG::Instance().uniform(0., 2 * kPi);
  if (kPi * (fParaboloid.fRlo2 + fParaboloid.fRhi2) / A >= z) {
    Precision rho;
    // points on the cutting circle surface at -dZ
    if (kPi * fParaboloid.fRlo2 / A > z) {
      rho = fParaboloid.fRlo * Sqrt(RNG::Instance().uniform(0., 1.));
      return Vector3D<Precision>(rho * cos(phi), rho * sin(phi), -fParaboloid.fDz);
    }
    // points on the cutting circle surface at dZ
    else {
      rho = fParaboloid.fRhi * Sqrt(RNG::Instance().uniform(0., 1.));
      return Vector3D<Precision>(rho * cos(phi), rho * sin(phi), fParaboloid.fDz);
    }
  }
  // points on the paraboloid surface
  else {
    z = RNG::Instance().uniform(0., 1.) * 2 * fParaboloid.fDz - fParaboloid.fDz;
    return Vector3D<Precision>(Sqrt(z * fParaboloid.fInvA - fParaboloid.fB * fParaboloid.fInvA) * cos(phi),
                               Sqrt(z * fParaboloid.fInvA - fParaboloid.fB * fParaboloid.fInvA) * sin(phi), z);
  }
}

std::string UnplacedParaboloid::GetEntityType() const
{
  return "Paraboloid\n";
}

/*
VECCORE_ATT_HOST_DEVICE
void UnplacedParaboloid::GetParametersList(int, double *aArray) const
{
  aArray[0] = GetRadius();
}
*/

VECCORE_ATT_HOST_DEVICE
UnplacedParaboloid *UnplacedParaboloid::Clone() const
{
  return new UnplacedParaboloid(fParaboloid.fRlo, fParaboloid.fRhi, fParaboloid.fDz);
}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedParaboloid::StreamInfo(std::ostream &os) const
// Definition taken from UParaboloid
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Paraboloid\n"
     << " Parameters: \n"
     << "     Paraboloid Radii Rlo=" << fParaboloid.fRlo << "mm, Rhi" << fParaboloid.fRhi << "mm \n"
     << "     Half-length Dz = " << fParaboloid.fDz << "mm\n";
  os << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

void UnplacedParaboloid::Print() const
{
  // printf("UnplacedParaboloid {%.2f}", GetRadius());
  printf("UnplacedParaboloid {%.2f, %.2f, %.2f, %.2f, %.2f}", GetRlo(), GetRhi(), GetDz(), GetA(), GetB());
}

void UnplacedParaboloid::Print(std::ostream &os) const
{
  // os << "UnplacedParaboloid {" << GetRadius() << "}";
  os << "UnplacedParaboloid {" << GetRlo() << ", " << GetRhi() << ", " << GetDz() << ", " << GetA() << ", " << GetB();
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedParaboloid::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedParaboloid<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedParaboloid<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedParaboloid::SpecializedVolume(LogicalVolume const *const volume,
                                                     Transformation3D const *const transformation,
                                                     const TranslationCode trans_code, const RotationCode rot_code,
                                                     VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedParaboloid>(volume, transformation, trans_code, rot_code,
                                                                   placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedParaboloid::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation, const int id,
                                          VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedParaboloid<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedParaboloid<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedParaboloid::SpecializedVolume(LogicalVolume const *const volume,
                                                     Transformation3D const *const transformation,
                                                     const TranslationCode trans_code, const RotationCode rot_code,
                                                     const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedParaboloid>(volume, transformation, trans_code, rot_code, id,
                                                                   placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedParaboloid::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedParaboloid>(in_gpu_ptr, GetRlo(), GetRhi(), GetDz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedParaboloid::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedParaboloid>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedParaboloid>::SizeOf();
template void DevicePtr<cuda::UnplacedParaboloid>::Construct(const Precision rlo, const Precision rhi,
                                                             const Precision dz) const;

} // namespace cxx

#endif
} // namespace vecgeom
