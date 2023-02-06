#pragma once

#include "VecGeom/base/Global.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include <algorithm>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(struct, PlacedVolumeImplHelper, class, Arg1, class, Arg2);

inline namespace VECGEOM_IMPL_NAMESPACE {

// A helper template class to automatically implement (wire)
// interfaces from PlacedVolume using kernel functions and functionality
// from the unplaced shapes
template <class UnplacedShape_t, class BaseVol = VPlacedVolume>
struct PlacedVolumeImplHelper : public BaseVol {

  using Helper_t = PlacedVolumeImplHelper<UnplacedShape_t, BaseVol>;
  using Struct_t = typename UnplacedShape_t::UnplacedStruct_t;

public:
  using BaseVol::BaseVol;
  using BaseVol::GetLogicalVolume;
  using BaseVol::PrintType;

  // destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedVolumeImplHelper() {}

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  UnplacedShape_t const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedShape_t const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  Struct_t const *GetUnplacedStruct() const { return static_cast<Struct_t const *>(&GetUnplacedVolume()->GetStruct()); }

#if !defined(VECCORE_CUDA)
  virtual Precision Capacity() override { return const_cast<UnplacedShape_t *>(GetUnplacedVolume())->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    VPlacedVolume::Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return VPlacedVolume::Normal(point, normal);
  }
#endif

  virtual Precision SurfaceArea() const override { return GetUnplacedVolume()->SurfaceArea(); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::Contains(point);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfLength) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOut(point, direction, stepMax);
  }

  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                  Real_v const step_max = kInfLength) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOutVec(position, direction, step_max);
  }

  // a helper tramponline to dispatch to DistanceToOutVec if type is not scalar
  template <typename T>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T DistanceToOut(Vector3D<T> const &p, Vector3D<T> const &d, T const &step_max) const
  {
    return DistanceToOutVec(p, d, step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::SafetyToOut(point);
  }

  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const &position) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::SafetyToOutVec(position);
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                             Precision const *const stepMax, Precision *const output) const override
  {
    GetUnplacedVolume()->UnplacedShape_t::DistanceToOut(points, directions, stepMax, output);
  }

  virtual void DistanceToOut(SOA3D<Precision> const & /* position */, SOA3D<Precision> const & /* direction */,
                             Precision const *const /* step_max */, Precision *const /* output */,
                             int *const /* nextnodeindex */) const override
  {
    // interface not implemented (depcrecated)
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const override
  {
    GetUnplacedVolume()->UnplacedShape_t::SafetyToOut(points, output);
  }

}; // End class PlacedVolumeImplHelper
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
