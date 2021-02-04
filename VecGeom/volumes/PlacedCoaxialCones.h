/// @file PlacedCoaxialCones.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDCOAXIALCONES_H_
#define VECGEOM_VOLUMES_PLACEDCOAXIALCONES_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/CoaxialConesImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCoaxialCones.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedCoaxialCones;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedCoaxialCones);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCoaxialCones : public PlacedVolumeImplHelper<UnplacedCoaxialCones, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedCoaxialCones, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedCoaxialCones(char const *const label, LogicalVolume const *const logicalVolume,
                     Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedCoaxialCones(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedCoaxialCones("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedCoaxialCones(LogicalVolume const *const logicalVolume,
                                        Transformation3D const *const transformation, const int id, const int copy_no,
                                        const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedCoaxialCones() {}

  /*
   * Put the Required Getters and Setters here
   */

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
