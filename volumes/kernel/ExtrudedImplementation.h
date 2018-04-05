//===-- kernel/ExtrudedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file ExtrudedImplementation.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_

#include <cstdio>
#include <VecCore/VecCore>
#include "volumes/kernel/GenericKernels.h"
#include "base/Vector3D.h"

#include "TessellatedImplementation.h"
#include "SExtruImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ExtrudedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ExtrudedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedExtruded;
class ExtrudedStruct;
class UnplacedExtruded;

struct ExtrudedImplementation {

  using PlacedShape_t    = PlacedExtruded;
  using UnplacedStruct_t = ExtrudedStruct;
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
  static void Contains(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::Contains<Real_v, Bool_v>(extruded.fSxtruHelper, point, inside);
    else
      TessellatedImplementation::Contains<Real_v, Bool_v>(extruded.fTslHelper, point, inside);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::Inside<Real_v, Inside_v>(extruded.fSxtruHelper, point, inside);
    else
      TessellatedImplementation::Inside<Real_v, Inside_v>(extruded.fTslHelper, point, inside);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::DistanceToIn<Real_v>(extruded.fSxtruHelper, point, direction, stepMax, distance);
    else
      TessellatedImplementation::DistanceToIn<Real_v>(extruded.fTslHelper, point, direction, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::DistanceToOut<Real_v>(extruded.fSxtruHelper, point, direction, stepMax, distance);
    else
      TessellatedImplementation::DistanceToOut<Real_v>(extruded.fTslHelper, point, direction, stepMax, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::SafetyToIn<Real_v>(extruded.fSxtruHelper, point, safety);
    else
      TessellatedImplementation::SafetyToIn<Real_v>(extruded.fTslHelper, point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::SafetyToOut<Real_v>(extruded.fSxtruHelper, point, safety);
    else
      TessellatedImplementation::SafetyToOut<Real_v>(extruded.fTslHelper, point, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &extruded, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    if (extruded.fIsSxtru) return SExtruImplementation::NormalKernel<Real_v>(extruded.fSxtruHelper, point, valid);
    return TessellatedImplementation::NormalKernel<Real_v>(extruded.fTslHelper, point, valid);
  }

}; // end ExtrudedImplementation

// Scalar specializations
template <>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void ExtrudedImplementation::Contains<double, bool>(UnplacedStruct_t const &extruded, Vector3D<double> const &point,
                                                    bool &inside)
{
  // Scalar specialization for Contains function
  inside = false;
  if (extruded.IsConvexPolygon()) {
    // Find the Z section
    int zIndex = extruded.FindZSegment(point[2]);
    if ((zIndex < 0) || (zIndex >= extruded.GetNPlanes())) return;
    inside = extruded.fTslSections[zIndex]->Contains(point);
    return;
  }

  if (extruded.fIsSxtru)
    SExtruImplementation::Contains<double, bool>(extruded.fSxtruHelper, point, inside);
  else
    TessellatedImplementation::Contains<double, bool>(extruded.fTslHelper, point, inside);
}

template <>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void ExtrudedImplementation::Inside<double, Inside_t>(UnplacedStruct_t const &extruded, Vector3D<double> const &point,
                                                      Inside_t &inside)
{
// Scalar specialization for Inside function
  inside = EInside::kOutside;
  if (extruded.IsConvexPolygon()) {
    int zIndex = extruded.FindZSegment(point[2]);
    if ((zIndex < 0) || (zIndex >= extruded.GetNPlanes())) return;
    inside = extruded.fTslSections[zIndex]->Inside(point);
    if (inside == EInside::kOutside) return;
    if (inside == EInside::kInside) {
      // Need to check if point on Z section
      if (vecCore::math::Abs(point[2] - extruded.fZSections[zIndex]) < kTolerance) {
        inside = EInside::kSurface;
      }
    } else {
      inside = EInside::kSurface;
    }
    return;
  }

  if (extruded.fIsSxtru)
    SExtruImplementation::Inside<double, Inside_t>(extruded.fSxtruHelper, point, inside);
  else
    TessellatedImplementation::Inside<double, Inside_t>(extruded.fTslHelper, point, inside);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
