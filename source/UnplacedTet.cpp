/// \file UnplacedTet.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/UnplacedTet.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedTet.h"
#include "base/RNG.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedTet::UnplacedTet()
{
  // default constructor
  fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
UnplacedTet::UnplacedTet(const Vector3D<Precision> anchor, const Vector3D<Precision> p2, const Vector3D<Precision> p3,
                         const Vector3D<Precision> p4)
    : fTet(anchor, p2, p3, p4)
{
  fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTet::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  /* Returns the full 3D cartesian extent of the solid.
  ** TODO : Logic to calculate the extent of Tet and assign it to aMin and aMax
  */
  aMin.Set(Min(Min(Min(fTet.fVertex[0].x(), fTet.fVertex[1].x()), fTet.fVertex[2].x()), fTet.fVertex[3].x()),
           Min(Min(Min(fTet.fVertex[0].y(), fTet.fVertex[1].y()), fTet.fVertex[2].y()), fTet.fVertex[3].y()),
           Min(Min(Min(fTet.fVertex[0].z(), fTet.fVertex[1].z()), fTet.fVertex[2].z()), fTet.fVertex[3].z()));
  aMax.Set(Max(Max(Max(fTet.fVertex[0].x(), fTet.fVertex[1].x()), fTet.fVertex[2].x()), fTet.fVertex[3].x()),
           Max(Max(Max(fTet.fVertex[0].y(), fTet.fVertex[1].y()), fTet.fVertex[2].y()), fTet.fVertex[3].y()),
           Max(Max(Max(fTet.fVertex[0].z(), fTet.fVertex[1].z()), fTet.fVertex[2].z()), fTet.fVertex[3].z()));
}

Vector3D<Precision> UnplacedTet::SamplePointOnSurface() const
{
  /* TODO : Logic to sample point on the surface of Tet
  **
  ** Return the sample Vector3D<Precision> point
  **
  ** Vector3D<Precision> point = <Logic to sample point on the surface of Tet>
  **
  ** return point
  */
  // Set areas
  //
  Precision sface[4];
  for (int i = 0; i < 4; ++i) {
    int i0   = (i + 0) % 4;
    int i1   = (i + 1) % 4;
    int i2   = (i + 2) % 4;
    sface[i] = 0.5 * ((fTet.fVertex[i1] - fTet.fVertex[i0]).Cross(fTet.fVertex[i2] - fTet.fVertex[i0])).Mag();
  }
  for (int i = 1; i < 4; ++i) {
    sface[i] += sface[i - 1];
  }

  // Select face
  //
  Precision select          = sface[3] * RNG::Instance().uniform(); // G4UniformRand();
  int k                     = 3;
  if (select <= sface[2]) k = 2;
  if (select <= sface[1]) k = 1;
  if (select <= sface[0]) k = 0;

  // Generate point
  //
  int i0      = (k + 0) % 4;
  int i1      = (k + 1) % 4;
  int i2      = (k + 2) % 4;
  Precision u = RNG::Instance().uniform();
  Precision v = RNG::Instance().uniform();
  if (u + v > 1.) {
    u = 1. - u;
    v = 1. - v;
  }
  return (1. - u - v) * fTet.fVertex[i0] + u * fTet.fVertex[i1] + v * fTet.fVertex[i2];
}

VECCORE_ATT_HOST_DEVICE
bool UnplacedTet::Normal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const
{
  int nsurf = 0;
  norm.Set(0., 0., 0.);

  for (int i = 0; i < 4; ++i) {
    Precision dist = fTet.fPlane[i].n.Dot(p) + fTet.fPlane[i].d;
    if (vecCore::math::Abs(dist) > kHalfTolerance) continue;
    ++nsurf;
    norm += fTet.fPlane[i].n;
  }

  // Return normal
  //
  if (nsurf == 1)
    return true;
  else if (nsurf != 0) {
    norm.Normalize();
    return true; // edge or corner
  } else {
// Point is not on the surface
//
#ifdef G4CSGDEBUG
    std::ostringstream message;
    G4int oldprc = message.precision(16);
    message << "Point p is not on surface (!?) of solid: " << GetName() << G4endl;
    message << "Position:\n";
    message << "   p.x() = " << p.x() / mm << " mm\n";
    message << "   p.y() = " << p.y() / mm << " mm\n";
    message << "   p.z() = " << p.z() / mm << " mm";
    G4cout.precision(oldprc);
    G4Exception("UnplacedTet::SurfaceNormal(p)", "GeomSolids1002", JustWarning, message);
    DumpInfo();
#endif
    ApproxSurfaceNormal(p, norm);
    return false;
  }
}

//////////////////////////////////////////////////////////////////////////+
//
//  Return normal of nearest side

void UnplacedTet::ApproxSurfaceNormal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const
{
  Precision dist = -kInfLength;
  int iside      = 0;
  for (int i = 0; i < 4; ++i) {
    Precision d = fTet.fPlane[i].n.Dot(p) + fTet.fPlane[i].d;
    if (d > dist) {
      dist  = d;
      iside = i;
    }
  }
  norm = fTet.fPlane[iside].n;
}

std::string UnplacedTet::GetEntityType() const
{
  return "Tet\n";
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTet::GetParametersList(int, double *aArray) const
{
  // TODO : Set the aArray elements
}

VECCORE_ATT_HOST_DEVICE
UnplacedTet *UnplacedTet::Clone() const
{
  return new UnplacedTet(fTet.fVertex[0], fTet.fVertex[1], fTet.fVertex[2], fTet.fVertex[3]);
}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedTet::StreamInfo(std::ostream &os) const
// Definition taken from UTet
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     //  << "     *** Dump for solid - " << GetName() << " ***\n"
     //  << "     ===================================================\n"

     << " Solid type: UTet\n"
     << " Parameters: \n"

     //    << "       Anchor Point: " << fTet.fAnchor << " mm \n"
     //    << "       Point P2: " << fTet.fP2 << " mm \n"
     //    << "       Point P3: " << fTet.fP3 << " mm \n"
     //    << "       Point P4: " << fTet.fP4 << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

void UnplacedTet::Print() const
{
}

void UnplacedTet::Print(std::ostream &os) const
{
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedTet::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTet<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedTet<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedTet::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTet>(volume, transformation, trans_code, rot_code, placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTet::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, const int id,
                                   VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTet<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedTet<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTet::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTet>(volume, transformation, trans_code, rot_code, id,
                                                            placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTet::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // Copy vertices on GPU, then create the object
  Precision *xv_gpu_ptr = AllocateOnGpu<Precision>(4 * sizeof(Precision));
  Precision *yv_gpu_ptr = AllocateOnGpu<Precision>(4 * sizeof(Precision));
  Precision *zv_gpu_ptr = AllocateOnGpu<Precision>(4 * sizeof(Precision));

  vecgeom::CopyToGpu(fTet.fVerticesX, xv_gpu_ptr, 4 * sizeof(Precision));
  vecgeom::CopyToGpu(fTet.fVerticesY, yv_gpu_ptr, 4 * sizeof(Precision));
  vecgeom::CopyToGpu(fTet.fVerticesZ, zv_gpu_ptr, 4 * sizeof(Precision));

  DevicePtr<cuda::VUnplacedVolume> gpugentet =
      CopyToGpuImpl<UnplacedTet>(in_gpu_ptr, xv_gpu_ptr, yv_gpu_ptr, zv_gpu_ptr);
  FreeFromGpu(xv_gpu_ptr);
  FreeFromGpu(yv_gpu_ptr);
  FreeFromGpu(zv_gpu_ptr);

  return gpugentet;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTet::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTet>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTet>::SizeOf();
template void DevicePtr<cuda::UnplacedTet>::Construct(Precision *, Precision *, Precision *) const;

} // End cxx namespace

#endif
} // End global namespace
