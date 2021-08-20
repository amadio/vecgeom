// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file source/UnplacedOrb.cpp
/// \author Raman Sehgal

#include "VecGeom/volumes/UnplacedOrb.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedOrb.h"
#include "VecGeom/base/RNG.h"
#include <stdio.h>
#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Orb.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedOrb::UnplacedOrb() : fCubicVolume(0), fSurfaceArea(0), fEpsilon(2.e-11), fRTolerance(0.)
{
  // default constructor
  fGlobalConvexity = true;
  SetRadialTolerance();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
UnplacedOrb::UnplacedOrb(const Precision r) : fOrb(r), fEpsilon(2.e-11)
{
  fCubicVolume     = (4 * kPi / 3) * fOrb.fR * fOrb.fR * fOrb.fR;
  fSurfaceArea     = (4 * kPi) * fOrb.fR * fOrb.fR;
  fGlobalConvexity = true;
  SetRadialTolerance();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedOrb::SetRadius(Precision r)
{
  fOrb.fR      = r;
  fCubicVolume = (4 * kPi / 3) * fOrb.fR * fOrb.fR * fOrb.fR;
  fSurfaceArea = (4 * kPi) * fOrb.fR * fOrb.fR;
  SetRadialTolerance();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedOrb::SetRadialTolerance()
{
  fRTolerance = Max(kTolerance, fEpsilon * fOrb.fR);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedOrb::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  aMin.Set(-fOrb.fR);
  aMax.Set(fOrb.fR);
}

Vector3D<Precision> UnplacedOrb::SamplePointOnSurface() const
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

VECCORE_ATT_HOST_DEVICE
void UnplacedOrb::GetParametersList(int, Precision *aArray) const
{
  aArray[0] = GetRadius();
}

VECCORE_ATT_HOST_DEVICE
UnplacedOrb *UnplacedOrb::Clone() const
{
  return new UnplacedOrb(fOrb.fR);
}

// VECCORE_ATT_HOST_DEVICE
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

VECCORE_ATT_HOST_DEVICE
void UnplacedOrb::Print() const
{
  printf("UnplacedOrb {%.2f}", GetRadius());
}

void UnplacedOrb::Print(std::ostream &os) const
{
  os << "UnplacedOrb {" << GetRadius() << "}";
}

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/SolidMesh.h"
SolidMesh *UnplacedOrb::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<Precision> Vec_t;
  SolidMesh *sm = new SolidMesh();

  Vec_t *vertices = new Vec_t[(nSegments + 1) * (nSegments + 1)];

  sm->ResetMesh((nSegments + 1) * (nSegments + 1), nSegments * nSegments);

  Precision phi_step   = 2 * M_PI / nSegments;
  Precision theta_step = M_PI / nSegments;
  Precision phi, theta;

  Precision x, y, z, xy;
  for (size_t i = 0; i <= nSegments; ++i) {
    theta = M_PI / 2 - i * theta_step; // starting from pi/2 to -pi/2
    xy    = GetRadius() * std::cos(theta);
    z     = GetRadius() * std::sin(theta);

    for (size_t j = 0; j <= nSegments; ++j) {
      phi = j * phi_step; // starting from 0 to 2pi

      // vertex position (x, y, z)
      x                                 = xy * std::cos(phi);
      y                                 = xy * std::sin(phi);
      vertices[i * (nSegments + 1) + j] = Vec_t(x, y, z);
    }
  }
  sm->SetVertices(vertices, (nSegments + 1) * (nSegments + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
    for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
      sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
    }
  }

  return sm;
}
#endif

#ifndef VECCORE_CUDA
// conversion functions
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedOrb::ConvertToRoot(char const *label) const
{
  return new TGeoSphere(label, 0., GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedOrb::ConvertToGeant4(char const *label) const
{
  return new G4Orb(label, GetRadius());
}
#endif

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
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedOrb::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, const int id, const int copy_no,
                                   const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedOrb<trans_code, rot_code>(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedOrb<trans_code, rot_code>(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedOrb::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, const int copy_no, const int child_id,
                                              VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedOrb>(volume, transformation, trans_code, rot_code, id, copy_no,
                                                            child_id, placement);
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

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedOrb>::SizeOf();
template void DevicePtr<cuda::UnplacedOrb>::Construct(const Precision r) const;

} // namespace cxx

#endif
} // namespace vecgeom
