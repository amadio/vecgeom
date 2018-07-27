/// 2018: initial version (Raman Sehgal, Evgueni Tcherniaev)

#ifndef VECGEOM_VOLUMES_PLACEDTET_H_
#define VECGEOM_VOLUMES_PLACEDTET_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TetImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedTet.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTet;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTet);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTet : public PlacedVolumeImplHelper<UnplacedTet, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedTet, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedTet(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingTet)
      : Base(label, logicalVolume, transformation, boundingTet)
  {
  }

  PlacedTet(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingTet)
      : PlacedTet("", logicalVolume, transformation, boundingTet)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedTet(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingTet, const int id)
      : Base(logicalVolume, transformation, boundingTet, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTet() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertices(Vector3D<Precision> &p0, Vector3D<Precision> &p1, Vector3D<Precision> &p2,
                   Vector3D<Precision> &p3) const
  {
    GetUnplacedVolume()->GetVertices(p0, p1, p2, p3);
  }

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
