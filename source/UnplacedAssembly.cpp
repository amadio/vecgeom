// LICENSING INFORMATION TBD

#include "volumes/UnplacedAssembly.h"
#include "volumes/PlacedAssembly.h"
#include "navigation/SimpleLevelLocator.h"
#include "management/ABBoxManager.h" // for Extent == bounding box calculation

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

UnplacedAssembly::UnplacedAssembly() : fLogicalVolume(nullptr), fLowerCorner(-kInfinity), fUpperCorner(kInfinity)
{
  fIsAssembly = true;
}

UnplacedAssembly::~UnplacedAssembly()
{
}

void UnplacedAssembly::AddVolume(VPlacedVolume const *v)
{
  fLogicalVolume->PlaceDaughter(v);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedAssembly::Print() const
{
  printf("UnplacedAssembly ");
}

void UnplacedAssembly::Print(std::ostream &os) const
{
  os << "UnplacedAssembly ";
}

//______________________________________________________________________________
void UnplacedAssembly::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECGEOM_NVCC
  auto &abboxmgr = ABBoxManager::Instance();

  // Returns the full 3D cartesian extent of the solid.
  // Loop nodes and get their extent
  aMin.Set(kInfinity);
  aMax.Set(-kInfinity);
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    Vector3D<Precision> lower, upper;
    abboxmgr.ComputeABBox(pv, &lower, &upper);
    aMin.Set(std::min(lower.x(), aMin.x()), std::min(lower.y(), aMin.y()), std::min(lower.z(), aMin.z()));
    aMax.Set(std::max(upper.x(), aMax.x()), std::max(upper.y(), aMax.y()), std::max(upper.z(), aMax.z()));
  }
#endif
}

Vector3D<Precision> UnplacedAssembly::GetPointOnSurface() const
{
  throw std::runtime_error("GetPointOnSurface for Assembly not yet implemented");
  // this requires some thought:
  // a) we should sample the points taking into the account the surface of
  //    constituents
  // b) we need to make the sure the coordinates of the point returned is
  //    indeed in the reference frame of the UnplacedAssembly !!
  return Vector3D<Precision>(0., 0., 0.);
}

Precision UnplacedAssembly::Capacity() const
{
  Precision capacity = 0.;
  // loop over nodes and sum the capacity of all their unplaced volumes
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    capacity += const_cast<VPlacedVolume *>(pv)->Capacity();
  }
  return capacity;
}

Precision UnplacedAssembly::SurfaceArea() const
{
  Precision area = 0.;
  // loop over nodes and sum the area of all their unplaced volumes
  // (this might be incorrect in case 2 constituents are touching)
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    area += const_cast<VPlacedVolume *>(pv)->SurfaceArea();
  }
  return area;
}

#ifndef VECGEOM_NVCC
VPlacedVolume *UnplacedAssembly::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
                                                   const TranslationCode trans_code, const RotationCode rot_code,
                                                   VPlacedVolume *const placement) const
{
  if (placement) {
    return new (placement) PlacedAssembly("", volume, transformation);
  }
  return new PlacedAssembly("", volume, transformation);
}
#else
__device__ VPlacedVolume *UnplacedAssembly::SpecializedVolume(LogicalVolume const *const volume,
                                                              Transformation3D const *const transformation,
                                                              const TranslationCode trans_code,
                                                              const RotationCode rot_code, const int id,
                                                              VPlacedVolume *const placement) const
{
  if (placement) {
    return new (placement) PlacedAssembly("", volume, transformation, nullptr, id);
  }
  return new PlacedAssembly("", volume, transformation, nullptr, id);
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedAssembly::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedAssembly>(in_gpu_ptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedAssembly::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedAssembly>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedAssembly>::SizeOf();
template void DevicePtr<cuda::UnplacedAssembly>::Construct() const;

} // End cxx namespace

#endif

} // End vecgeom namespace
