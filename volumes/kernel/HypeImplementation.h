/// \file HypeImplementation.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_HYPEMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedHype.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct HypeImplementation {

  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedHype const &box,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside) {
    // NYI
  }

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedHype const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {
    // NYI
  }

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedHype const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(
      UnplacedHype const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {
    // NYI
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(
      UnplacedHype const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedHype const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
    // NYI
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedHype const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {
    // NYI
  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_