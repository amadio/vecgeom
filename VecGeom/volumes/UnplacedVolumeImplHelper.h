// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief A collection of template helpers to automize implementation
///        of UnplacedVolume interfaces.
/// \file volumes/UnplacedVolumeImplHelper.h
/// \author First version created by Sandro Wenzel

#ifndef VOLUMES_UNPLACEDVOLUMEIMPLHELPER_H_
#define VOLUMES_UNPLACEDVOLUMEIMPLHELPER_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/management/VolumeFactory.h"

#ifndef __clang__
#pragma GCC diagnostic push
// We ignore warnings of this type in this file.
// The warning occurred due to potential overflow of memory address locations output[i]
// where i is an unsigned long long in a loop. It can be safely ignored since such
// memory locations do in fact not exist (~multiple petabyte in memory).
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(class, CommonUnplacedVolumeImplHelper, typename, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

/*!
 * A template class with the aim to automatically implement
 * interfaces of UnplacedVolume by connecting them to the separately
 * available (reusable) kernels.
 *
 * This is an application of the CRT pattern in order to reduce
 * repeating code.
 */
template <class Implementation, class BaseUnplVol = VUnplacedVolume>
class CommonUnplacedVolumeImplHelper : public BaseUnplVol {

public:
  using UnplacedStruct_t = typename Implementation::UnplacedStruct_t;
  using UnplacedVolume_t = typename Implementation::UnplacedVolume_t;

  // bring in constructors
  using BaseUnplVol::BaseUnplVol;
  // bring in some members from base (to avoid name hiding)
  using BaseUnplVol::DistanceToIn;
  using BaseUnplVol::DistanceToOut;
  using BaseUnplVol::SafetyToIn;
  using BaseUnplVol::SafetyToOut;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                  Precision step_max = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(d.IsNormalized() && " direction not normalized in call to  DistanceToOut ");
#endif
    Precision output = kInfLength;
    Implementation::template DistanceToOut(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, d, step_max,
                                           output);

// detect -inf responses which are often an indication for a real bug
#ifndef VECCORE_CUDA
    assert(!((output < 0.) && std::isinf((Precision)output)));
#endif
    return output;
  }

  // the extended DistanceToOut interface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                  Vector3D<Precision> &normal, bool &convex,
                                  Precision step_max = kInfLength) const override
  {
    (void)p;
    (void)d;
    (void)normal;
    (void)convex;
    (void)step_max;
    assert(false);
    return 0.;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &p) const override
  {
    bool output(false);
    Implementation::Contains(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual EnumInside Inside(Vector3D<Precision> const &p) const override
  {
    Inside_t output(0);
    Implementation::Inside(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return (EnumInside)output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                 const Precision step_max = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(d.IsNormalized() && " direction not normalized in call to  DistanceToOut ");
#endif
    Precision output(kInfLength);
    Implementation::DistanceToIn(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, d, step_max, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &p) const override
  {
    Precision output(kInfLength);
    Implementation::SafetyToOut(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &p) const override
  {
    Precision output(kInfLength);
    Implementation::SafetyToIn(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  virtual int MemorySize() const override { return sizeof(*this); }
};

/*! \brief Template implementation helper for the case of unplaced volumes/shapes
 * where the vector interface is implemented in terms of SIMD vector instructions.
 */
template <class Implementation, class BaseUnplVol = VUnplacedVolume>
class SIMDUnplacedVolumeImplHelper : public CommonUnplacedVolumeImplHelper<Implementation, BaseUnplVol> {
public:
  using Real_v           = vecgeom::VectorBackend::Real_v;
  using UnplacedVolume_t = typename Implementation::UnplacedVolume_t;
  using Common_t         = CommonUnplacedVolumeImplHelper<Implementation, BaseUnplVol>;

  static constexpr bool SIMDHELPER = true; // property expressing that this helper provides true external SIMD support
  // bring in constructor
  using Common_t::Common_t;

  using UnplacedStruct_t = typename Implementation::UnplacedStruct_t;
  using Common_t::DistanceToOut;
  using Common_t::SafetyToOut;
};

/*!
 * \brief Template implementation helper for the case of unplaced volumes/shapes
 * where the vector interface is implemented in terms of loops over scalar version.
 */
template <class Implementation, class BaseUnplVol = VUnplacedVolume>
class LoopUnplacedVolumeImplHelper : public CommonUnplacedVolumeImplHelper<Implementation, BaseUnplVol> {
public:
  using Real_v           = vecgeom::VectorBackend::Real_v;
  using Real_s           = vecgeom::ScalarBackend::Real_v;
  using UnplacedVolume_t = typename Implementation::UnplacedVolume_t;
  using Common_t         = CommonUnplacedVolumeImplHelper<Implementation, BaseUnplVol>;

  static constexpr bool SIMDHELPER = false;

  // constructors
  using Common_t::Common_t;
  using Common_t::DistanceToIn;
  using Common_t::DistanceToOut;
  using Common_t::SafetyToIn;
  using Common_t::SafetyToOut;

  using UnplacedStruct_t = typename Implementation::UnplacedStruct_t;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

#endif /* VOLUMES_UnplacedVolumeImplHelper_H_ */
