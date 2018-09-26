//@author: initial implementation Sandro Wenzel (July 2018)
#ifndef VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_
#define VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_

#include "base/Global.h"
#include "volumes/UnplacedVolume.h"
#include "management/VolumeFactory.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/kernel/ImplAsImplementation.h"
#include <cassert>
#include <type_traits>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(class, SUnplacedImplAs, typename, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

// A wrapper class which generically implements a specialization of UnplacedBase
// implemented in terms of another **existing** implementing volume ImplementingUnplaced.
template <typename UnplacedBase, typename ImplementingUnplaced>
class SUnplacedImplAs : public UnplacedBase {
  // static assert(make sure UnplacedBase is an UnplacedVolume

public:
  using UnplacedStruct_t = typename ImplementingUnplaced::UnplacedStruct_t;

  template <typename... Args>
  SUnplacedImplAs(Args... args) : UnplacedBase(args...)
  {
    fImplPtr = new ImplementingUnplaced(args...);
  }

  // implement the important unplaced volume interfaces
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Precision> const &p) const override { return fImplPtr->ImplementingUnplaced::Contains(p); }

  VECCORE_ATT_HOST_DEVICE
  EnumInside Inside(Vector3D<Precision> const &p) const override { return fImplPtr->ImplementingUnplaced::Inside(p); }

  // DistanceToOut
  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                          Precision step_max = kInfLength) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOut(p, d, step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d, Vector3D<Precision> &normal,
                          bool &convex, Precision step_max = kInfLength) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOut(p, d, normal, convex, step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  Real_v DistanceToOutVec(Vector3D<Real_v> const &p, Vector3D<Real_v> const &d, Real_v const &step_max) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOutVec(p, d, step_max);
  }

  // the container/basket interface (possibly to be deprecated)
  void DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                     Precision const *const step_max, Precision *const output) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOut(points, directions, step_max, output);
  }

  // SafetyToOut
  VECCORE_ATT_HOST_DEVICE
  Precision SafetyToOut(Vector3D<Precision> const &p) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToOut(p);
  }

  VECCORE_ATT_HOST_DEVICE
  Real_v SafetyToOutVec(Vector3D<Real_v> const &p) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToOutVec(p);
  }

  void SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToOut(points, output);
  }

  // NOTE: This does not work yet (for whatever reason) since the trampoline dispatch is confused
  // in UnplacedVolume.h
  // DistanceToIn
  //  VECCORE_ATT_HOST_DEVICE
  //  Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
  //                         const Precision step_max) const override
  //  {
  //    return fImplPtr->ImplementingUnplaced::DistanceToIn(position, direction, step_max);
  //  }

  //  VECCORE_ATT_HOST_DEVICE
  //  Real_v DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
  //                         const Real_v &step_max) const override
  //  {
  //    return fImplPtr->ImplementingUnplaced::DistanceToInVec(position, direction, step_max);
  //  }

  VECCORE_ATT_HOST_DEVICE
  Precision SafetyToIn(Vector3D<Precision> const &position) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToIn(position);
  }

  VECCORE_ATT_HOST_DEVICE
  Real_v SafetyToInVec(Vector3D<Real_v> const &p) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToInVec(p);
  }

  // ---------------- SamplePointOnSurface ----------------------------------------------------------
  Vector3D<Precision> SamplePointOnSurface() const override
  {
    return fImplPtr->ImplementingUnplaced::SamplePointOnSurface();
  }

  // ----------------- Extent --------------------------------------------------------------------
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    fImplPtr->ImplementingUnplaced::Extent(aMin, aMax);
  }

  using StructType = decltype(std::declval<ImplementingUnplaced>().GetStruct());
  VECCORE_ATT_HOST_DEVICE
  StructType GetStruct() const { return fImplPtr->ImplementingUnplaced::GetStruct(); }

private:
  // select target specialization helpers with SFINAE
  template <typename IUnplaced, typename ImplementingKernel>
  VPlacedVolume *changeTypeKeepTransformation(
      VPlacedVolume const *p, typename std::enable_if<IUnplaced::SIMDHELPER, IUnplaced>::type * = nullptr) const
  {
    // if the implementing helper supports SIMD
    auto c = VolumeFactory::ChangeTypeKeepTransformation<SIMDSpecializedVolImplHelper, ImplementingKernel>(p);
    return c;
  }

  template <typename IUnplaced, typename ImplementingKernel>
  VPlacedVolume *changeTypeKeepTransformation(
      VPlacedVolume const *p, typename std::enable_if<!IUnplaced::SIMDHELPER, IUnplaced>::type * = nullptr) const
  {
    // if the implementing helper does not support SIMD
    auto c = VolumeFactory::ChangeTypeKeepTransformation<LoopSpecializedVolImplHelper, ImplementingKernel>(p);
    return c;
  }

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   const TranslationCode trans_code, const RotationCode rot_code,
                                   VPlacedVolume *const placement = NULL) const override
  {
    auto p = fImplPtr->ImplementingUnplaced::SpecializedVolume(volume, transformation, trans_code, rot_code, placement);

    using ImplementingKernel = IndirectImplementation<SUnplacedImplAs<UnplacedBase, ImplementingUnplaced>,
                                                      typename ImplementingUnplaced::Kernel>;

    // the right function to use will be selected with SFINAE and enable_if
    return changeTypeKeepTransformation<ImplementingUnplaced, ImplementingKernel>(p);
    // TODO: original p is no longer needed in principle: delete and deregister from GeoManager
    // delete p;
  }

  ImplementingUnplaced *fImplPtr;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_ */
