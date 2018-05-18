//===-- kernel/MultiUnionImplementation.h ---------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file MultiUnionImplementation.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/MultiUnionStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct MultiUnionImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, MultiUnionImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedMultiUnion;
struct MultiUnionStruct;
class UnplacedMultiUnion;

struct MultiUnionImplementation {

  using PlacedShape_t    = PlacedMultiUnion;
  using UnplacedStruct_t = MultiUnionStruct;
  using UnplacedVolume_t = UnplacedMultiUnion;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType() {}

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedMultiUnion<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
  }
  /** @brief Identify the first element containing the point */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int InsideElement(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point)
  {
    int element     = -1;
    auto insidehook = [&](size_t id) {
      auto local = munion.fVolumes[id]->GetTransformation()->Transform(point);
      if (munion.fVolumes[id]->Inside(local) != EInside::kOutside) {
        element = (int)id;
        return true;
      }
      return false;
    };
    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, insidehook);
    return element;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    auto containshook = [&](size_t id) {
      inside = munion.fVolumes[id]->Contains(point);
      return inside;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, containshook);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    inside          = EInside::kOutside;
    auto insidehook = [&](size_t id) {
      auto inside_crt = munion.fVolumes[id]->Inside(point);
      if (inside_crt == EInside::kInside) {
        inside = EInside::kInside;
        return true;
      }
      if (inside_crt == EInside::kSurface) inside = EInside::kSurface;

      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, insidehook);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point, Real_v &safety)
  {
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point, Real_v &safety)
  {
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &munion, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
  }

}; // end MultiUnionImplementation
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_
