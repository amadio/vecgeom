/// \file PlacedScaledShape.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_
#define VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedScaledShape.h"
#include "volumes/kernel/ScaledShapeImplementation.h"
#include "volumes/PlacedVolImplHelper.h"

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
                    Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedScaledShape(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                    vecgeom::PlacedBox const *const boundingBox)
      : PlacedScaledShape("", logicalVolume, transformation, boundingBox)
  {
  }

#else

  VECCORE_ATT_DEVICE PlacedScaledShape(LogicalVolume const *const logicalVolume,
                                       Transformation3D const *const transformation, PlacedBox const *const boundingBox,
                                       const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
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

#if !defined(VECCORE_CUDA)
  virtual Precision Capacity() override { return GetUnplacedVolume()->Volume(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }
#endif

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
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDSCALEDSHAPE_H_
