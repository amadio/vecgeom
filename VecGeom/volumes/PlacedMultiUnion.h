/// @file PlacedMultiUnion.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDMULTIUNION_H_
#define VECGEOM_VOLUMES_PLACEDMULTIUNION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/UnplacedMultiUnion.h"
#include "VecGeom/volumes/kernel/MultiUnionImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"

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
                   Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedMultiUnion(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedMultiUnion("", logicalVolume, transformation)
  {
  }
#else
  __device__ PlacedMultiUnion(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                              const int id)
      : Base(logicalVolume, transformation, id)
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
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &s) const override;

#ifndef VECCORE_CUDA
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

#endif // VECGEOM_VOLUMES_PLACEDMULTIUNION_H_
