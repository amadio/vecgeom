/// @file TetImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/TetStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TetImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TetImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTet;
template <typename T>
struct TetStruct;
class UnplacedTet;

struct TetImplementation {

  using PlacedShape_t    = PlacedTet;
  using UnplacedStruct_t = TetStruct<double>;
  using UnplacedVolume_t = UnplacedTet;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //  printf("SpecializedTet<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedTet<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "TetImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    // st << "UnplacedTet";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(tet, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(tet, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void GenericKernelForContainsAndInside(UnplacedStruct_t const &tet, Vector3D<Real_v> const &localPoint,
                                                Bool_v &completelyinside, Bool_v &completelyoutside)
  {
    /* TODO : Logic to check where the point is inside or not.
    **
    ** if ForInside is false then it will only check if the point is outside,
    ** and is used by Contains function
    **
    ** if ForInside is true then it will check whether the point is inside or outside,
    ** and if neither inside nor outside then it is on the surface.
    ** and is used by Inside function
    */

    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = tet.fPlane[i].n.Dot(localPoint) + tet.fPlane[i].d;
    }
    Real_v safety = vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(dist[0], dist[1]), dist[2]), dist[3]);

    completelyoutside               = safety > kHalfTolerance;
    if (ForInside) completelyinside = safety <= -kHalfTolerance;

    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const & /*stepMax*/, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the Tet surface */
    // using Bool_v       = vecCore::Mask_v<Real_v>;
    distance           = -kInfLength;
    Real_v distanceOut = kInfLength;
    Real_v absSafe     = kInfLength;

    Real_v cosa[4];
    Real_v safe[4];
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      cosa[i] = NonZero(Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction));
      safe[i] = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      dist[i] = -safe[i] / cosa[i];
    }

    for (int i = 0; i < 4; ++i) {
      vecCore::MaskedAssign(distance, (cosa[i] < Real_v(0.)), vecCore::math::Max(distance, dist[i]));
      vecCore::MaskedAssign(distanceOut, (cosa[i] > Real_v(0.)), vecCore::math::Min(distanceOut, dist[i]));
      vecCore::MaskedAssign(absSafe, (cosa[i] > Real_v(0.)), vecCore::math::Min(absSafe, vecCore::math::Abs(safe[i])));
    }

    vecCore::MaskedAssign(distance,
                          distance >= distanceOut || distanceOut <= kHalfTolerance || absSafe <= kHalfTolerance,
                          Real_v(kInfLength));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the Tet surface */
    distance      = kInfLength;
    Real_v safety = -kInfLength;

    Real_v cosa[4];
    Real_v safe[4];
    for (int i = 0; i < 4; ++i) {
      cosa[i] = NonZero(Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction));
      safe[i] = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      safety  = vecCore::math::Max(safety, safe[i]);
    }

    for (int i = 0; i < 4; ++i) {
      vecCore::MaskedAssign(distance, (cosa[i] > Real_v(0.)), vecCore::math::Min(distance, -safe[i] / cosa[i]));
    }
    vecCore__MaskedAssignFunc(distance, safety > kHalfTolerance, Real_v(-1.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the Tet surface */

    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = tet.fPlane[i].n.Dot(point) + tet.fPlane[i].d;
    }
    safety = vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(dist[0], dist[1]), dist[2]), dist[3]);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the Tet surface */

    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = tet.fPlane[i].n.Dot(point) + tet.fPlane[i].d;
    }
    safety = -vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(dist[0], dist[1]), dist[2]), dist[3]);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) <= kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    Vector3D<Real_v> normal(0.);
    valid = true;

    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = tet.fPlane[i].n.Dot(point) + tet.fPlane[i].d;
      vecCore__MaskedAssignFunc(normal, vecCore::math::Abs(dist[i]) <= kHalfTolerance, normal + tet.fPlane[i].n)
    }

    vecCore::Mask_v<Real_v> done = normal.Mag2() > 1;
    vecCore__MaskedAssignFunc(normal, done, normal.Unit());
    if (vecCore::MaskFull(done)) return normal;

    // Point is not on the surface - normally, this should never be.
    // Return normal of the nearest face.
    //
    done = normal.Mag2() > 0;
    vecCore__MaskedAssignFunc(valid, !done, false);

    Real_v safety(-kInfLength);
    for (int i = 0; i < 4; ++i) {
      vecCore__MaskedAssignFunc(normal, dist[i] > safety && !done, tet.fPlane[i].n);
      vecCore__MaskedAssignFunc(safety, dist[i] > safety && !done, dist[i]);
    }
    return normal;
  }
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_
