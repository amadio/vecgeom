#pragma once

#include "base/Global.h"
#include "base/SOA3D.h"
#include "volumes/PlacedVolume.h"

#include <algorithm>

#ifdef VECGEOM_DISTANCE_DEBUG
#include "volumes/utilities/ResultComparator.h"
#endif

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

  // destructor
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedVolumeImplHelper() {}

  virtual int memory_size() const override { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedShape_t const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedShape_t const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  Struct_t const *GetUnplacedStruct() const { return static_cast<Struct_t const *>(&GetUnplacedVolume()->GetStruct()); }

#if !defined(VECGEOM_NVCC)
  virtual Precision Capacity() override { return const_cast<UnplacedShape_t *>(GetUnplacedVolume())->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
    // careful here: we need to transform result
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    // bool valid;
    // BoxImplementation_Evolution<translation::kIdentity, rotation::kIdentity>::NormalKernel<>(
    //        *GetUnplacedVolume(),
    //        point,
    //        normal, valid);
    // return valid;
    // careful here: we need to transform result
    return GetUnplacedVolume()->Normal(point, normal);
  }

  virtual Vector3D<Precision> GetPointOnSurface() const override
  {
    return GetUnplacedVolume()->GetPointOnSurface();
    // carefull here: we need to transform result
  }

  virtual double SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }

#if defined(VECGEOM_USOLIDS)
//  virtual std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType(); }
#endif
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::Contains(point);
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfLength) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOut(point, direction, stepMax);
  }

  virtual VECGEOM_BACKEND_PRECISION_TYPE DistanceToOutVec(
      Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &position,
      Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &direction,
      VECGEOM_BACKEND_PRECISION_TYPE const step_max = kInfLength) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::DistanceToOutVec(position, direction, step_max);
  }

#ifdef VECGEOM_USOLIDS
  /*
   * WARNING: Trivial implementation for standard USolids interface
   * for DistanceToOut. The value for convex might be wrong
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  Vector3D<Precision> &normal, bool &convex, Precision step = kInfLength) const override
  {
    double d                  = DistanceToOut(point, direction, step);
    Vector3D<double> hitpoint = point + d * direction;
    GetUnplacedVolume()->UnplacedShape_t::Normal(hitpoint, normal);
    // we could make this something like
    // convex = PlacedShape_t::IsConvex;
    convex = false; // the only possible safe choice
    return d;
  }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const override
  {
    return GetUnplacedVolume()->UnplacedShape_t::SafetyToOut(point);
  }

  virtual VECGEOM_BACKEND_PRECISION_TYPE SafetyToOutVec(
      Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &position) const override
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

  virtual void SafetyToOutMinimize(SOA3D<Precision> const & /* points */,
                                   Precision *const /* safeties */) const override
  {
    // interface not implemented (deprecated)
  }

}; // End class PlacedVolumeImplHelper
}
} // End global namespace
