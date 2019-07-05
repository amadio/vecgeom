/// \file PlacedRootVolume.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedRootVolume.h"

#include "TGeoShape.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedRootVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedRootVolume);

inline namespace cxx {

template <typename T>
class SOA3D;

class PlacedRootVolume : public VPlacedVolume {

private:
  PlacedRootVolume(const PlacedRootVolume &);            // Not implemented
  PlacedRootVolume &operator=(const PlacedRootVolume &); // Not implemented

public:
  PlacedRootVolume(char const *const label, TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  PlacedRootVolume(TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  virtual ~PlacedRootVolume() {}

  TGeoShape const *GetRootShape() const { return ((UnplacedRootVolume *)GetUnplacedVolume())->GetRootShape(); }

  virtual int MemorySize() const override { return sizeof(*this); }

  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &) const override;
  virtual void PrintImplementationType(std::ostream &) const override;
  virtual void PrintUnplacedType(std::ostream &) const override;

  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &point) const override;

  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override;

  virtual void Contains(SOA3D<Precision> const &points, bool *const output) const override;

  VECGEOM_FORCE_INLINE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override;

  VECGEOM_FORCE_INLINE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const override;

  virtual void Inside(SOA3D<Precision> const &points, Inside_t *const output) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max) const override;

  virtual void DistanceToIn(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                            Precision const *const stepMax, Precision *const output) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                  Precision const stepMax) const override;

  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                        Precision const stepMax) const override;

  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output) const override;

  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output,
                             int *const nextnodeindex) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const override;

  virtual void SafetyToOut(SOA3D<Precision> const &position, Precision *const safeties) const override;

  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const override;

  virtual void SafetyToIn(SOA3D<Precision> const &position, Precision *const safeties) const override;

  virtual void SafetyToInMinimize(SOA3D<Precision> const &position, Precision *const safeties) const override;

  // the SIMD vector interfaces (not implemented)
  virtual Real_v DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                 Real_v const step_max = kInfLength) const override
  {
    throw std::runtime_error("unimplemented function called");
    return Real_v(-1.);
  }

  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                  Real_v const step_max = kInfLength) const override
  {
    throw std::runtime_error("unimplemented function called");
    return Real_v(-1.);
  }

  virtual Real_v SafetyToInVec(Vector3D<Real_v> const &position) const override
  {
    throw std::runtime_error("unimplemented function called");
    return Real_v(-1.);
  }

  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const &position) const override
  {
    throw std::runtime_error("unimplemented function called");
    return Real_v(-1.);
  }

  virtual void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  virtual Precision Capacity() override;

  virtual Precision SurfaceArea() const override
  {
    throw std::runtime_error("unimplemented function called");
    return -1.;
  }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return 0; /* return DevicePtr<cuda::PlacedRootVolume>::SizeOf(); */ }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform,
                                                   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const override;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform) const override;
#endif
};

bool PlacedRootVolume::Contains(Vector3D<Precision> const &point) const
{
  const Vector3D<Precision> local = GetTransformation()->Transform(point);
  return UnplacedContains(local);
}

bool PlacedRootVolume::Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const
{
  localPoint = GetTransformation()->Transform(point);
  return UnplacedContains(localPoint);
}

bool PlacedRootVolume::UnplacedContains(Vector3D<Precision> const &point) const
{
  Vector3D<Precision> pointCopy = point; // ROOT expects non const input
  return GetRootShape()->Contains(&pointCopy[0]);
}

EnumInside PlacedRootVolume::Inside(Vector3D<Precision> const &point) const
{
  const Vector3D<Precision> local = GetTransformation()->Transform(point);
  return (UnplacedContains(local)) ? static_cast<EnumInside>(EInside::kInside)
                                   : static_cast<EnumInside>(EInside::kOutside);
}

Precision PlacedRootVolume::DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                         const Precision stepMax) const
{
  Vector3D<Precision> positionLocal  = GetTransformation()->Transform(position);
  Vector3D<Precision> directionLocal = GetTransformation()->TransformDirection(direction);
  return GetRootShape()->DistFromOutside(&positionLocal[0], &directionLocal[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                          const Precision stepMax) const
{
  return GetRootShape()->DistFromInside(&position[0], &direction[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::PlacedDistanceToOut(Vector3D<Precision> const &position,
                                                Vector3D<Precision> const &direction, const Precision stepMax) const
{
  Vector3D<Precision> positionLocal  = GetTransformation()->Transform(position);
  Vector3D<Precision> directionLocal = GetTransformation()->TransformDirection(direction);
  return GetRootShape()->DistFromInside(&positionLocal[0], &directionLocal[0], 3);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::SafetyToOut(Vector3D<Precision> const &position) const
{
  Vector3D<Precision> position_local = GetTransformation()->Transform(position);
  return GetRootShape()->Safety(&position_local[0], true);
}

VECGEOM_FORCE_INLINE
Precision PlacedRootVolume::SafetyToIn(Vector3D<Precision> const &position) const
{
  Vector3D<Precision> position_local = GetTransformation()->Transform(position);
  return GetRootShape()->Safety(&position_local[0], false);
}
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_
