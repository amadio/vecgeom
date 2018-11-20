/// @file PlacedTesselated.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDTESSELLATED_H_
#define VECGEOM_VOLUMES_PLACEDTESSELLATED_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TessellatedImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedTessellated.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTessellated;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTessellated);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated : public PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedTessellated(char const *const label, LogicalVolume const *const logicalVolume,
                    Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                    vecgeom::PlacedBox const *const boundingBox)
      : PlacedTessellated("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  __device__ PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTessellated() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedTessellated const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTessellated const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#ifndef VECCORE_CUDA
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

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
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
