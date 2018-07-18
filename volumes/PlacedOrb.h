/// 2015: initial version (Raman Sehgal)
/// 2016: cleanup; move to PlacedVolImplHelper (Raman Sehgal)

#ifndef VECGEOM_VOLUMES_PLACEDORB_H_
#define VECGEOM_VOLUMES_PLACEDORB_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/OrbImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedOrb.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedOrb;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedOrb);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedOrb : public PlacedVolumeImplHelper<UnplacedOrb, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedOrb, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedOrb(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingOrb)
      : Base(label, logicalVolume, transformation, boundingOrb)
  {
  }

  PlacedOrb(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingOrb)
      : PlacedOrb("", logicalVolume, transformation, boundingOrb)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedOrb(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingOrb, const int id)
      : Base(logicalVolume, transformation, boundingOrb, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedOrb() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRadius() const { return GetUnplacedVolume()->GetRadius(); }

  VECCORE_ATT_HOST_DEVICE
  void SetRadius(Precision arg) { const_cast<UnplacedOrb *>(GetUnplacedVolume())->SetRadius(arg); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override { return GetUnplacedVolume()->ConvertToGeant4(GetName()); }
#endif
#endif // VECCORE_CUDA
};

} // end inline namespace
} // End global namespace

#endif
