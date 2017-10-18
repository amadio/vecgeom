//===-- kernel/ExtrudedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file ExtrudedImplementation.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "TessellatedImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ExtrudedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ExtrudedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedExtruded;
template <typename T>
class TessellatedStruct;
class UnplacedExtruded;

struct ExtrudedImplementation {

  using PlacedShape_t    = PlacedExtruded;
  using UnplacedStruct_t = TessellatedStruct<double>;
  using UnplacedVolume_t = UnplacedExtruded;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //  printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedExtruded<" << transCodeT << "," << rotCodeT << ">";
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

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    TessellatedImplementation::Contains<Real_v, Bool_v>(tessellated, point, inside);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    TessellatedImplementation::Inside<Real_v, Inside_v>(tessellated, point, inside);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    TessellatedImplementation::DistanceToIn<Real_v>(tessellated, point, direction, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    TessellatedImplementation::DistanceToOut<Real_v>(tessellated, point, direction, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    TessellatedImplementation::SafetyToIn<Real_v>(tessellated, point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    TessellatedImplementation::SafetyToOut<Real_v>(tessellated, point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    return TessellatedImplementation::NormalKernel<Real_v>(tessellated, point, valid);
  }

}; // end ExtrudedImplementation
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
