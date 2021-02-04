#ifndef VECGEOM_VOLUMES_PLACEDSEXTRU_H_
#define VECGEOM_VOLUMES_PLACEDSEXTRU_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedSExtru;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedSExtru);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedSExtru : public PlacedVolumeImplHelper<UnplacedSExtruVolume, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedSExtruVolume, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedSExtru(char const *const label, LogicalVolume const *const logicalVolume,
               Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedSExtru(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedSExtru("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedSExtru(LogicalVolume const *const logicalVolume,
                                  Transformation3D const *const transformation, const int id, const int copy_no,
                                  const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedSExtru() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
