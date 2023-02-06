/// \file PlacedScaledShape.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_
#define VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#include "VecGeom/volumes/kernel/ScaledShapeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedScaledShape;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedScaledShape);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedScaledShape : public PlacedVolumeImplHelper<UnplacedScaledShape, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedScaledShape, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA

  using Base::Base;
  PlacedScaledShape(char const *const label, LogicalVolume const *const logicalVolume,
                    Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedScaledShape(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedScaledShape("", logicalVolume, transformation)
  {
  }

#else

  VECCORE_ATT_DEVICE PlacedScaledShape(LogicalVolume const *const logicalVolume,
                                       Transformation3D const *const transformation, const int id, const int copy_no,
                                       const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }

#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedScaledShape() {}

  // Accessors

  VECCORE_ATT_HOST_DEVICE
  UnplacedScaledShape const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedScaledShape const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;

  /** @brief Print type name */
  void PrintType(std::ostream &os) const override;

  // CUDA specific

  virtual int MemorySize() const override { return sizeof(*this); }

  // Comparison specific

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override
  {
    return nullptr; // No implementation in Geant4
  }
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_
