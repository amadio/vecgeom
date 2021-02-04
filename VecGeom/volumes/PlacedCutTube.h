/*
 * PlacedCutTube.h
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */
#ifndef VECGEOM_VOLUMES_PLACEDCUTTUBE_H_
#define VECGEOM_VOLUMES_PLACEDCUTTUBE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/CutTubeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCutTube.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedCutTube;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedCutTube);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCutTube : public PlacedVolumeImplHelper<UnplacedCutTube, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedCutTube, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedCutTube(char const *const label, LogicalVolume const *const logicalVolume,
                Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedCutTube(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedCutTube("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedCutTube(LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation, const int id, const int copy_no,
                                   const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedCutTube() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedCutTube const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedCutTube const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmin() const { return GetUnplacedVolume()->rmin(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmax() const { return GetUnplacedVolume()->rmax(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision z() const { return GetUnplacedVolume()->z(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision sphi() const { return GetUnplacedVolume()->sphi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dphi() const { return GetUnplacedVolume()->dphi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> BottomNormal() const { return GetUnplacedVolume()->BottomNormal(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> TopNormal() const { return GetUnplacedVolume()->TopNormal(); }

#if !defined(VECCORE_CUDA)
  /** @brief Interface method for computing capacity */
  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }
  /** @brief Computes the extent on X/Y/Z of the trapezoid */
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  /** @brief Shortcut for computing the normal */
  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }
#endif

  // CUDA specific
  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

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
