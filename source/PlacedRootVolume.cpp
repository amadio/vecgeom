/// \file PlacedRootVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/volumes/PlacedRootVolume.h"
#include "TGeoBBox.h"
#include "VecGeom/base/SOA3D.h"

namespace vecgeom {

PlacedRootVolume::PlacedRootVolume(char const *const label, TGeoShape const *const rootShape,
                                   LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : VPlacedVolume(label, logicalVolume, transformation)
{
}

PlacedRootVolume::PlacedRootVolume(TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : PlacedRootVolume(rootShape->GetName(), rootShape, logicalVolume, transformation)
{
}

void PlacedRootVolume::PrintType() const
{
  printf("PlacedRootVolume");
}

void PlacedRootVolume::PrintType(std::ostream &os) const
{
  os << "PlacedRootVolume";
}

void PlacedRootVolume::PrintImplementationType(std::ostream &os) const
{
  os << "RootVolumeImplementation(does not exist)";
}

void PlacedRootVolume::PrintUnplacedType(std::ostream &os) const
{
  os << "UnplacedRootVolume(does not exist)";
}

Precision PlacedRootVolume::Capacity()
{
  return GetRootShape()->Capacity();
}

void PlacedRootVolume::Contains(SOA3D<Precision> const &points, bool *const output) const
{
  for (size_t i = 0, iMax = points.size(); i < iMax; ++i) {
    output[i] = PlacedRootVolume::Contains(points[i]);
  }
}

void PlacedRootVolume::Inside(SOA3D<Precision> const &points, Inside_t *const output) const
{
  for (size_t i = 0, iMax = points.size(); i < iMax; ++i) {
    output[i] = PlacedRootVolume::Inside(points[i]);
  }
}

void PlacedRootVolume::DistanceToIn(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                                    Precision const *const stepMax, Precision *const output) const
{
  for (int i = 0, iMax = position.size(); i < iMax; ++i) {
    output[i] = PlacedRootVolume::DistanceToIn(position[i], direction[i], stepMax[i]);
  }
}

void PlacedRootVolume::DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                                     Precision const *const stepMax, Precision *const output) const
{
  for (int i = 0, iMax = position.size(); i < iMax; ++i) {
    output[i] = PlacedRootVolume::DistanceToOut(position[i], direction[i], stepMax[i]);
  }
}

void PlacedRootVolume::SafetyToIn(SOA3D<Precision> const &position, Precision *const safeties) const
{
  for (int i = 0, iMax = position.size(); i < iMax; ++i) {
    safeties[i] = PlacedRootVolume::SafetyToIn(position[i]);
  }
}

void PlacedRootVolume::SafetyToOut(SOA3D<Precision> const &position, Precision *const safeties) const
{
  for (int i = 0, iMax = position.size(); i < iMax; ++i) {
    safeties[i] = PlacedRootVolume::SafetyToOut(position[i]);
  }
}

void PlacedRootVolume::DistanceToOut(SOA3D<Precision> const &positions, SOA3D<Precision> const &directions,
                                     Precision const *stepMax, Precision *distance, int *nextNodeIndex) const
{
  for (int i = 0, iMax = positions.size(); i < iMax; ++i) {
    distance[i]      = PlacedRootVolume::DistanceToOut(positions[i], directions[i], stepMax[i]);
    nextNodeIndex[i] = (distance[i] < stepMax[i]) ? -1 : -2;
  }
}

void PlacedRootVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  TGeoBBox const *b = dynamic_cast<TGeoBBox const *>(GetRootShape());
  assert(b != nullptr);
  auto lx = b->GetDX();
  auto ly = b->GetDY();
  auto lz = b->GetDZ();
  auto o  = b->GetOrigin();
  aMin.Set(o[0] - lx, o[1] - ly, o[2] - lz);
  aMax.Set(o[0] + lx, o[1] + ly, o[2] + lz);
}

VPlacedVolume const *PlacedRootVolume::ConvertToUnspecialized() const
{
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
  return NULL;
}
#ifdef VECGEOM_ROOT
TGeoShape const *PlacedRootVolume::ConvertToRoot() const
{
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
  return NULL;
}
#endif
#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedRootVolume::ConvertToGeant4() const
{
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
  return NULL;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VPlacedVolume> PlacedRootVolume::CopyToGpu(DevicePtr<cuda::LogicalVolume> const /* logical_volume */,
                                                           DevicePtr<cuda::Transformation3D> const /* transform */,
                                                           DevicePtr<cuda::VPlacedVolume> const /* in_gpu_ptr */) const
{
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
  return DevicePtr<cuda::VPlacedVolume>(nullptr);
}
DevicePtr<cuda::VPlacedVolume> PlacedRootVolume::CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
    DevicePtr<cuda::Transformation3D> const /* transform */) const
{
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
  return DevicePtr<cuda::VPlacedVolume>(nullptr);
}
#endif

} // End namespace vecgeom
