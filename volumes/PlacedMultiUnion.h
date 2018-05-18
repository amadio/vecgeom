/// @file PlacedMultiUnion.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDMULTIUNION_H_
#define VECGEOM_VOLUMES_PLACEDMULTIUNION_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedMultiUnion.h"
#include "volumes/kernel/MultiUnionImplementation.h"
#include "volumes/PlacedVolImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedMultiUnion;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedMultiUnion);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedMultiUnion : public PlacedVolumeImplHelper<UnplacedMultiUnion, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedMultiUnion, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedMultiUnion(char const *const label, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedMultiUnion(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                   vecgeom::PlacedBox const *const boundingBox)
      : PlacedMultiUnion("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  __device__ PlacedMultiUnion(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                              PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedMultiUnion() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedMultiUnion const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedMultiUnion const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#ifndef VECCORE_CUDA
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual Vector3D<Precision> SamplePointOnSurface() const override
  {
    return GetUnplacedVolume()->SamplePointOnSurface();
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDMULTIUNION_H_
