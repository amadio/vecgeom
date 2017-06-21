//===-- kernel/TessellatedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TessellatedImplementation.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/TessellatedStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TessellatedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TessellatedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated;
template <typename T>
class TessellatedStruct;
class UnplacedTessellated;

struct TessellatedImplementation {

  using PlacedShape_t    = PlacedTessellated;
  using UnplacedStruct_t = TessellatedStruct<double>;
  using UnplacedVolume_t = UnplacedTessellated;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //  printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedTessellated<" << transCodeT << "," << rotCodeT << ">";
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
    inside = tessellated.Contains(point);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    inside = tessellated.Inside(point);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf;
    tessellated.DistanceToSolid<true>(point, direction, stepMax, distance, isurf);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf;
    tessellated.DistanceToSolid<false>(point, direction, stepMax, distance, isurf);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    //    tessellated.SafetyToIn(point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &box, Vector3D<Real_v> const &point, Real_v &safety)
  {
    //    tessellated.SafetyToIn(point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &box, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector

    valid = false;
    return Vector3D<Real_v>();
  }

}; // end TessellatedImplementation
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
