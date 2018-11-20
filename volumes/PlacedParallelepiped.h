/// \file PlacedParallelepiped.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
//#include "volumes/kernel/ParallelepipedImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedParallelepiped.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedParallelepiped;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedParallelepiped);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParallelepiped : public PlacedVolumeImplHelper<UnplacedParallelepiped, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedParallelepiped, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedParallelepiped(char const *const label, LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedParallelepiped(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : PlacedParallelepiped("", logical_volume, transformation, boundingBox)
  {
  }

#else

  VECCORE_ATT_DEVICE PlacedParallelepiped(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingBox, const int id)
      : Base(logical_volume, transformation, boundingBox, id)
  {
  }

#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedParallelepiped() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedParallelepiped const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetDimensions() const { return GetUnplacedVolume()->GetDimensions(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetX() const { return GetUnplacedVolume()->GetX(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetY() const { return GetUnplacedVolume()->GetY(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZ() const { return GetUnplacedVolume()->GetZ(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetAlpha() const { return GetUnplacedVolume()->GetAlpha(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTanAlpha() const { return GetUnplacedVolume()->GetTanAlpha(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaSinPhi() const { return GetUnplacedVolume()->GetTanThetaSinPhi(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaCosPhi() const { return GetUnplacedVolume()->GetTanThetaCosPhi(); }

#ifndef VECCORE_CUDA
  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

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

#endif // VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
