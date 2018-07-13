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

    // Vector3D<Real_v> normal(tet.fPlane[0].n.x(),tet.fPlane[0].n.y(),tet.fPlane[0].n.z());
    Vector3D<Real_v> n0(tet.fPlane[0].n);
    Real_v d0 = n0.Dot(localPoint) + Real_v(tet.fPlane[0].d);

    // normal.Set(tet.fPlane[1].n.x(),tet.fPlane[1].n.y(),tet.fPlane[1].n.z());
    // normal.Set(tet.fPlane[1].n);
    Vector3D<Real_v> n1(tet.fPlane[1].n);
    Real_v d1 = n1.Dot(localPoint) + Real_v(tet.fPlane[1].d);

    // normal.Set(tet.fPlane[2].n.x(),tet.fPlane[2].n.y(),tet.fPlane[2].n.z());
    // normal.Set(tet.fPlane[2].n);
    Vector3D<Real_v> n2(tet.fPlane[2].n);
    Real_v d2 = n2.Dot(localPoint) + Real_v(tet.fPlane[2].d);

    // normal.Set(tet.fPlane[3].n.x(),tet.fPlane[3].n.y(),tet.fPlane[3].n.z());
    // normal.Set(tet.fPlane[3].n);
    Vector3D<Real_v> n3(tet.fPlane[3].n);
    Real_v d3 = n3.Dot(localPoint) + Real_v(tet.fPlane[3].d);

    Real_v dMax                     = vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(d0, d1), d2), d3);
    completelyoutside               = dMax > kHalfTolerance;
    if (ForInside) completelyinside = dMax <= -kHalfTolerance;

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
    Real_v absDist     = kInfLength;
    for (int i = 0; i < 4; ++i) {
      Real_v cosa = NonZero(Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction));
      Real_v dist = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      vecCore::MaskedAssign(distance, (cosa < Real_v(0.)), vecCore::math::Max(distance, -dist / cosa));
      vecCore::MaskedAssign(distanceOut, (cosa > Real_v(0.)), vecCore::math::Min(distanceOut, -dist / cosa));
      vecCore::MaskedAssign(absDist, (cosa > Real_v(0.)), vecCore::math::Min(absDist, vecCore::math::Abs(dist)));
    }

    vecCore::MaskedAssign(distance,
                          distance >= distanceOut || distanceOut <= kHalfTolerance || absDist <= kHalfTolerance,
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
    for (int i = 0; i < 4; ++i) {
      Real_v cosa = NonZero(Vector3D<Real_v>(tet.fPlane[i].n).Dot(direction));
      Real_v dist = Vector3D<Real_v>(tet.fPlane[i].n).Dot(point) + tet.fPlane[i].d;
      vecCore::MaskedAssign(distance, (cosa > Real_v(0.)), vecCore::math::Min(distance, -dist / cosa));
      // vecCore::MaskedAssign(safetyIn, (cosa < Real_v(0.)), vecCore::math::Max(safetyIn, dist));
      safety = vecCore::math::Max(safety, dist);
    }
    vecCore__MaskedAssignFunc(distance, safety > kHalfTolerance, Real_v(-1.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the Tet surface */
    Vector3D<Real_v> normal(tet.fPlane[0].n.x(), tet.fPlane[0].n.y(), tet.fPlane[0].n.z());
    Real_v d0 = normal.Dot(point) + Real_v(tet.fPlane[0].d);

    normal.Set(tet.fPlane[1].n.x(), tet.fPlane[1].n.y(), tet.fPlane[1].n.z());
    Real_v d1 = normal.Dot(point) + Real_v(tet.fPlane[1].d);

    normal.Set(tet.fPlane[2].n.x(), tet.fPlane[2].n.y(), tet.fPlane[2].n.z());
    Real_v d2 = normal.Dot(point) + Real_v(tet.fPlane[2].d);

    normal.Set(tet.fPlane[3].n.x(), tet.fPlane[3].n.y(), tet.fPlane[3].n.z());
    Real_v d3 = normal.Dot(point) + Real_v(tet.fPlane[3].d);

    safety = vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(d0, d1), d2), d3);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < kHalfTolerance, Real_v(0.));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the Tet surface */
    Vector3D<Real_v> normal(tet.fPlane[0].n.x(), tet.fPlane[0].n.y(), tet.fPlane[0].n.z());
    Real_v d0 = normal.Dot(point) + Real_v(tet.fPlane[0].d);

    normal.Set(tet.fPlane[1].n.x(), tet.fPlane[1].n.y(), tet.fPlane[1].n.z());
    Real_v d1 = normal.Dot(point) + Real_v(tet.fPlane[1].d);

    normal.Set(tet.fPlane[2].n.x(), tet.fPlane[2].n.y(), tet.fPlane[2].n.z());
    Real_v d2 = normal.Dot(point) + Real_v(tet.fPlane[2].d);

    normal.Set(tet.fPlane[3].n.x(), tet.fPlane[3].n.y(), tet.fPlane[3].n.z());
    Real_v d3 = normal.Dot(point) + Real_v(tet.fPlane[3].d);

    safety = -vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(d0, d1), d2), d3);
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < kHalfTolerance, Real_v(0.));
  }

  /*
    template <typename Real_v>
    VECGEOM_FORCE_INLINE
    VECCORE_ATT_HOST_DEVICE
    static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &tet, Vector3D<Real_v> const &point,
                                         typename vecCore::Mask_v<Real_v> &valid)
    {
      Vector3D<Real_v> normal(0.,0.,0.) ;

      // Set valid to true if the normal is valid
      //valid = ((rad2 <= tolRMaxO * tolRMaxO) && (rad2 >= tolRMaxI * tolRMaxI)); // means we are on surface

      //Calculate the normal at the point and return that.
      return normal;
    }

    //////////////////////////////////////////////////////////////////////////+
    //
    //  Return normal of nearest side
    template <typename Real_v>
    VECGEOM_FORCE_INLINE
    VECCORE_ATT_HOST_DEVICE
    Vector3D<Real_v> ApproxSurfaceNormal(UnplacedStruct_t const &tet , Vector3D<Real_v> const & p) const
    {
      Real_v dist = -kInfLength;
      //G4int iside = 0;

      for (int i=0; i<4; ++i)
      {
        Real_v d = tet.fPlane[i].n.Dot(p) + tet.fPlane[i].d;
        vecCore::MaskedAssign(dist , d > dist , d);
        //vecCore::MaskedAssign(iside , d > dist , i);

      }
      //return tet.fPlanes[iside].n;
    }
  */
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TETIMPLEMENTATION_H_
