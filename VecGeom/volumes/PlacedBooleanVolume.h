#ifndef VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
#define VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/BooleanImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"

#ifdef VECGEOM_ROOT
class TGeoShape;
#endif
#ifdef VECGEOM_GEANT4
class G4VSolid;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <BooleanOperation Op> class PlacedBooleanVolume;);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1v(class, PlacedBooleanVolume, BooleanOperation, Arg1);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <BooleanOperation Op>
class PlacedBooleanVolume : public PlacedVolumeImplHelper<UnplacedBooleanVolume<Op>, VPlacedVolume> {

  using Base          = PlacedVolumeImplHelper<UnplacedBooleanVolume<Op>, VPlacedVolume>;
  using UnplacedVol_t = UnplacedBooleanVolume<Op>;

public:
  using Base::GetLogicalVolume;
#ifndef VECCORE_CUDA
  using Base::Base;
  using Base::Inside;
  PlacedBooleanVolume(char const *const label, LogicalVolume const *const logicalVolume,
                      Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedBooleanVolume(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedBooleanVolume("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedBooleanVolume(LogicalVolume const *const logicalVolume,
                                         Transformation3D const *const transformation, const int id, const int copy_no,
                                         const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedBooleanVolume() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedVol_t const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedVol_t const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  void PrintType() const override{};

  void PrintType(std::ostream &) const override{};

  // CUDA specific
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  // Comparison specific

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override { return this; }
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECGEOM_BENCHMARK

}; // end class declaration

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
