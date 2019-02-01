/// @file PlacedEllipticalCone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_PLACEDELLIPTICALCONE_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/EllipticalConeImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedEllipticalCone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedEllipticalCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedEllipticalCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalCone : public PlacedVolumeImplHelper<UnplacedEllipticalCone, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedEllipticalCone, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedEllipticalCone(char const *const label, LogicalVolume const *const logicalVolume,
                       Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedEllipticalCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : PlacedEllipticalCone("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedEllipticalCone(LogicalVolume const *const logicalVolume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedEllipticalCone() {}

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
