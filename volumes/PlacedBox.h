/// 2014: initial version (J. De Fine Licht + S. Wenzel)
/// Okt 2015: cleanup; move to PlacedVolImplHelper (Sandro Wenzel)

#ifndef VECGEOM_VOLUMES_PLACEDBOX_H_
#define VECGEOM_VOLUMES_PLACEDBOX_H_

#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedBox.h"

//#ifdef VECGEOM_ROOT
//#include "TGeoBBox.h"
//#endif
//#ifdef VECGEOM_USOLIDS
//#include "UBox.hh"
//#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedBox;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedBox);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox : public PlacedVolumeImplHelper<UnplacedBox, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedBox, VPlacedVolume>;

public:
#ifndef VECGEOM_NVCC
  // constructor inheritance;
  using Base::Base;
  PlacedBox(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedBox(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingBox)
      : PlacedBox("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  __device__ PlacedBox(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedBox() {}

  // Accessors -- not sure we need this ever (to be deprecated)
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> const &dimensions() const { return GetUnplacedVolume()->dimensions(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision x() const { return GetUnplacedVolume()->x(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision y() const { return GetUnplacedVolume()->y(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision z() const { return GetUnplacedVolume()->z(); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const *ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECGEOM_NVCC
};

} // end inline namespace
} // End global namespace

#endif
