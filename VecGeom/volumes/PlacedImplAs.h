#ifndef VECGEOM_VOLUMES_PLACEDIMPLAS_H_
#define VECGEOM_VOLUMES_PLACEDIMPLAS_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// the placed analog of UnplacedImplAs
template <typename Unplaced>
class PlacedImplAs : public PlacedVolumeImplHelper<Unplaced, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<Unplaced, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedImplAs(char const *const label, LogicalVolume const *const logicalVolume,
               Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingB)
      : Base(label, logicalVolume, transformation, boundingB)
  {
  }

  PlacedImplAs(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
               vecgeom::PlacedBox const *const boundingB)
      : PlacedImplAs("", logicalVolume, transformation, boundingB)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedImplAs(LogicalVolume const *const logicalVolume,
                                  Transformation3D const *const transformation, PlacedBox const *const boundingB,
                                  const int id)
      : Base(logicalVolume, transformation, boundingB, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedImplAs() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override{};
  virtual void PrintType(std::ostream &os) const override{};

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override
  {
    // at this moment it is hard to reason about unspecializing
    // so just refusing and returning itself
    return this;
  }
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override
  {
    return this->GetUnplacedVolume()->ConvertToRoot(this->GetName());
  }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override
  {
    return this->GetUnplacedVolume()->ConvertToGeant4(this->GetName());
  }
#endif
#endif // VECCORE_CUDA
};

} // end inline namespace
} // End global namespace

#endif
