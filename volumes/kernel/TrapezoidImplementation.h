//===-- kernel/TrapezoidImplementation.h ----------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
///
/// \file   kernel/TrapezoidImplementation.h
/// \author Guilherme Lima (lima@fnal.gov)
/// \brief This file implements the algorithms for the trapezoid
///
/// Implementation details: initially based on USolids algorithms and vectorized types.
///
//===--------------------------------------------------------------------------===//
///
/// 140520  G. Lima   Created from USolids' UTrap algorithms
/// 160722  G. Lima   Revision + moving to new backend structure

#ifndef VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/TrapezoidStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TrapezoidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TrapezoidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrapezoid;
class UnplacedTrapezoid;

struct TrapezoidImplementation {

  using PlacedShape_t    = PlacedTrapezoid;
  using UnplacedStruct_t = TrapezoidStruct<double>;
  using UnplacedVolume_t = UnplacedTrapezoid;
#ifdef VECGEOM_PLANESHELL_DISABLE
  using TrapSidePlane = TrapezoidStruct<double>::TrapSidePlane;
#endif

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    // printf("SpecializedTrapezoid<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedTrapezoid<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "TrapezoidImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    st << "UnplacedTrapezoid";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(unplaced, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Inside_t &inside)
  {
    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;

    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(unplaced, point, completelyinside, completelyoutside);
    inside = Inside_t(EInside::kSurface);
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                                Bool_v &completelyInside, Bool_v &completelyOutside)
  {
    constexpr Precision trapSurfaceTolerance = 1.0e-6;
    // z-region
    completelyOutside = Abs(point[2]) > MakePlusTolerant<ForInside>(unplaced.fDz, trapSurfaceTolerance);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(completelyOutside)) {
      return;
    }
    if (ForInside) {
      completelyInside = Abs(point[2]) < MakeMinusTolerant<ForInside>(unplaced.fDz, trapSurfaceTolerance);
    }

#ifndef VECGEOM_PLANESHELL_DISABLE
    unplaced.GetPlanes()->GenericKernelForContainsAndInside<Real_v, ForInside>(point, completelyInside,
                                                                               completelyOutside);
#else
    // here for PLANESHELL=OFF (disabled)
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v dist[4];
    for (unsigned int i = 0; i < 4; ++i) {
      dist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    }

    for (unsigned int i = 0; i < 4; ++i) {
      // is it outside of this side plane?
      completelyOutside = completelyOutside || dist[i] > MakePlusTolerant<ForInside>(0., trapSurfaceTolerance);
      if (ForInside) {
        completelyInside = completelyInside && dist[i] < MakeMinusTolerant<ForInside>(0., trapSurfaceTolerance);
      }
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(completelyOutside)) return;
    }
#endif

    return;
  }

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate distance to shape from outside - return kInfLength if no
  // intersection.
  //
  // ALGORITHM: For each component (z-planes, side planes), calculate
  // pair of minimum (smin) and maximum (smax) intersection values for
  // which the particle is in the extent of the shape.  The point of
  // entrance (exit) is found by the largest smin (smallest smax).
  //
  //  If largest smin > smallest smax, the trajectory does not reach
  //  inside the shape.
  //
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                           Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    distance = kInfLength;

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    // convenience variables for direction pointing to +z or -z.
    // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
    Bool_v posZdir = dir.z() > 0.0;
    Bool_v negZdir = dir.z() < 0.0;
    Real_v zdirSign(1.0); // z-direction
    vecCore::MaskedAssign(zdirSign, negZdir, Real_v(-1.0));

    Real_v max = zdirSign * unplaced.fDz - point.z(); // z-dist to farthest z-plane

    // step 1.a) input particle is moving away --> return infinity

    // check if moving away towards +z
    done = done || (posZdir && max < MakePlusTolerant<true>(0.));

    // check if moving away towards -z
    done = done || (negZdir && max > MakeMinusTolerant<true>(0.));

    // if all particles moving away, we're done
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    // Step 1.b) General case:
    //   smax,smin are range of distances within z-range, taking direction into account.
    //   smin<smax - smax is positive, but smin may be either positive or negative
    Real_v dirFactor = Real_v(1.0) / NonZero(dir.z()); // convert distances from z to dir
    Real_v smax      = max * dirFactor;
    Real_v smin      = (-zdirSign * unplaced.fDz - point.z()) * dirFactor;

    // Step 1.c) special case: if dir is perpendicular to z-axis...
    Bool_v test = (!posZdir) && (!negZdir);

    // ... and out of z-range, then trajectory will not intercept volume
    Bool_v zrange = Abs(point.z()) < MakeMinusTolerant<true>(unplaced.fDz);
    done          = done || (test && !zrange);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    // ... or within z-range, then smin=-1, smax=infinity for now
    vecCore::MaskedAssign(smin, test && zrange, Real_v(-1.0));
    vecCore::MaskedAssign(smax, test && zrange, Real_v(kInfLength));

//
// Step 2: find distances for intersections with side planes.
//

#ifndef VECGEOM_PLANESHELL_DISABLE
    // If disttoplanes is such that smin < dist < smax, then distance=disttoplanes
    Real_v disttoplanes = unplaced.GetPlanes()->DistanceToIn(point, smin, dir);

    //=== special cases
    // 1) point is inside (wrong-side) --> return -1
    Bool_v inside = (smin < -kHalfTolerance && disttoplanes < -kHalfTolerance);
    vecCore::MaskedAssign(distance, !done && inside, Real_v(-1.0)); // wrong-side point
    done = done || inside;

    // 2) point is outside plane-shell and moving away --> return infinity
    done = done || disttoplanes == kInfLength;

    // 3) track misses the trapezoid: smin<0 && disttoplanes>smax --> return infinity
    done = done || (smin < 0 && disttoplanes > smax);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    // at this point we know there is a valid distance - start with the z-plane based one
    vecCore::MaskedAssign(distance, !done, smin);

    // and then take the one from the planar shell, if valid
    Bool_v hitplanarshell = (disttoplanes > smin) && (disttoplanes < smax);
    vecCore::MaskedAssign(distance, !done && hitplanarshell, disttoplanes);

#else
    // here for VECGEOM_PLANESHELL_DISABLE
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();

    // loop over side planes - find pdist,Comp for each side plane
    Real_v pdist[4], comp[4], vdist[4];
    // auto-vectorizable part of loop
    for (unsigned int i = 0; i < 4; ++i) {
      // Note: normal vector is pointing outside the volume (convention), therefore
      // pdist>0 if point is outside  and  pdist<0 means inside
      pdist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;

      // Comp is projection of dir over the normal vector of side plane, hence
      // Comp > 0 if pointing ~same direction as normal and Comp<0 if ~opposite to normal
      comp[i] = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();

      vdist[i] = -pdist[i] / NonZero(comp[i]);
    }

    // wrong-side check: if (inside && smin<0) return -1
    Bool_v inside = smin < MakeMinusTolerant<true>(0.0) && smax > MakePlusTolerant<true>(0.0);

    for (int i = 0; i < 4; ++i) {
      inside = inside && pdist[i] < MakeMinusTolerant<true>(0.0);
    }
    vecCore::MaskedAssign(distance, inside, Real_v(-1.0));
    done = done || inside;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    // this part does not auto-vectorize
    for (unsigned int i = 0; i < 4; ++i) {
      // if outside and moving away, return infinity
      Bool_v posPoint = pdist[i] > MakeMinusTolerant<true>(0.);
      Bool_v posDir   = comp[i] > 0;

      // discard the ones moving away from this plane
      done = done || (posPoint && posDir);

      // check if trajectory will intercept plane within current range (smin,smax)

      Bool_v interceptFromInside = (!posPoint && posDir);
      done                       = done || (interceptFromInside && vdist[i] < smin);

      Bool_v interceptFromOutside = (posPoint && !posDir);
      done                        = done || (interceptFromOutside && vdist[i] > smax);
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

      //.. If dist is such that smin < dist < smax, then adjust either smin or smax.
      Bool_v validVdist = (vdist[i] > smin && vdist[i] < smax);
      vecCore::MaskedAssign(smax, !done && interceptFromInside && validVdist, vdist[i]);
      vecCore::MaskedAssign(smin, !done && interceptFromOutside && validVdist, vdist[i]);
    }

    // Checks in non z plane intersections ensure smin<smax
    vecCore::MaskedAssign(distance, !done, smin);
#endif // end of #ifdef PLANESHELL

    // finally check whether entry point candidate is globally valid
    // --> if it is, entry point should be on the surface.
    //     if entry point candidate is completely outside, track misses the solid --> return infinity
    Vector3D<Real_v> entry = point + distance * dir;
    Bool_v complIn(false), complOut(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(unplaced, entry, complIn, complOut);
    vecCore::MaskedAssign(distance, !done && complOut, Real_v(kInfLength));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &dir, Real_v const &stepMax, Real_v &distance)
  {
    (void)stepMax;
    using Bool_v = vecCore::Mask_v<Real_v>;

    distance = kInfLength;
    Bool_v done(false);

    // step 0: if point is outside any plane --> return -1
    Bool_v outside = Abs(point.z()) > MakePlusTolerant<true>(unplaced.fDz);
    vecCore::MaskedAssign(distance, outside, Real_v(-1.0));
    done = done || outside;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    //
    // Step 1: find range of distances along dir between Z-planes (smin, smax)
    //

    // convenience variables for direction pointing to +z or -z.
    // Note that both posZdir and NegZdir may be false, if dir.z() is zero!
    // Bool_v posZdir = dir.z() > 0.0;
    Bool_v negZdir = dir.z() <= -0.0; // -0.0 is needed, see JIRA-150

    // TODO: consider use of copysign or some other standard function
    Real_v zdirSign(1.0); // z-direction
    vecCore::MaskedAssign(zdirSign, negZdir, Real_v(-1.0));

    Real_v max = zdirSign * unplaced.fDz - point.z(); // z-dist to farthest z-plane

    // // if all particles moving away, we're done
    // MaskedAssign( distance, done, Real_v(0.0) );
    // if ( vecCore::MaskFull(done) ) return;

    // Step 1.b) general case: assign distance to z plane
    Real_v distz = max / NonZero(dir.z());
    vecCore::MaskedAssign(distance, !done && dir.z() != Real_v(0.), distz);

//
// Step 2: find distances for intersections with side planes.
//

#ifndef VECGEOM_PLANESHELL_DISABLE
    Real_v disttoplanes = unplaced.GetPlanes()->DistanceToOut(point, dir);

    // reconcile sides with endcaps
    vecCore::MaskedAssign(distance, disttoplanes < distance, disttoplanes);

#else
    //=== Here for VECGEOM_PLANESHELL_DISABLE
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();

    // loop over side planes - find pdist,Comp for each side plane
    Real_v pdist[4], comp[4], vdist[4];
    for (unsigned int i = 0; i < 4; ++i) {
      // Note: normal vector is pointing outside the volume (convention), therefore
      // pdist>0 if point is outside  and  pdist<0 means inside
      pdist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;

      // Comp is projection of dir over the normal vector of side plane, hence
      // Comp > 0 if pointing ~same direction as normal and Comp<0 if pointing ~opposite to normal
      comp[i] = fPlanes[i].fA * dir.x() + fPlanes[i].fB * dir.y() + fPlanes[i].fC * dir.z();

      vdist[i] = -pdist[i] / NonZero(comp[i]);
    }

    // early return if point is outside of plane
    for (unsigned int i = 0; i < 4; ++i) {
      done = done || (pdist[i] > MakePlusTolerant<true>(0.));
    }
    vecCore::MaskedAssign(distance, done, Real_v(-1.0));
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

    for (unsigned int i = 0; i < 4; ++i) {
      // if point is inside, pointing towards plane and vdist<distance, then distance=vdist
      Bool_v test = (vdist[i] >= MakeMinusTolerant<true>(0.)) && comp[i] > 0.0 && vdist[i] < distance;
      vecCore::MaskedAssign(distance, !done && test, vdist[i]);
    }
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety = Abs(point.z()) - unplaced.fDz;

#ifndef VECGEOM_PLANESHELL_DISABLE
    // Get safety over side planes
    unplaced.GetPlanes()->SafetyToIn(point, safety);
#else
    // Loop over side planes
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();
    Real_v Dist[4];
    for (int i = 0; i < 4; ++i) {
      Dist[i] = fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD;
    }

    for (int i = 0; i < 4; ++i) {
      vecCore::MaskedAssign(safety, Dist[i] > safety, Dist[i]);
    }
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    safety = -1.;

    // If point is outside (wrong-side) --> safety to negative value
    safety = unplaced.fDz - Abs(point.z());
    done   = done || (safety < kHalfTolerance);

    // If all test points are outside, we're done
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return;

#ifndef VECGEOM_PLANESHELL_DISABLE
    // Get safety over side planes
    unplaced.GetPlanes()->SafetyToOut(point, safety);
#else
    // Loop over side planes
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();

    // auto-vectorizable loop
    Real_v Dist[4];
    for (int i = 0; i < 4; ++i) {
      Dist[i] = -(fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD);
    }

    // unvectorizable loop
    for (int i = 0; i < 4; ++i) {
      vecCore::MaskedAssign(safety, !done && Dist[i] < safety, Dist[i]);
    }
#endif
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    using Bool_v  = vecCore::Mask_v<Real_v>;
    using Index_v = vecCore::Index<Real_v>;

    constexpr double delta = 1000. * kTolerance;
    Vector3D<Real_v> normal(0.);
    valid         = true;
    Real_v safety = InfinityLength<Real_v>();
    vecCore::Index<Real_v> iface;

#ifndef VECGEOM_PLANESHELL_DISABLE
    // Get normal from side planes -- PlaneShell case
    // normal = unplaced.GetPlanes()->NormalKernel(point, valid);
    iface = unplaced.GetPlanes()->ClosestFace(point, safety);

#else
    // Loop over side planes
    TrapSidePlane const *fPlanes = unplaced.GetPlanes();

    // vectorizable loop
    Real_v dist[4];
    for (int i = 0; i < 4; ++i) {
      dist[i] = -(fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y() + fPlanes[i].fC * point.z() + fPlanes[i].fD);
    }

    // non-vectorizable part
    for (int i = 0; i < 4; ++i) {
      // valid becomes false if point is outside any plane by more than delta (dist < -delta here means outside)
      valid = valid && dist[i] >= -delta;

      const Bool_v closer = Abs(dist[i]) < safety;
      vecCore::MaskedAssign(safety, closer, Abs(dist[i]));
      vecCore::MaskedAssign(iface, closer, Index_v(i));
      // vecCore::MaskedAssign(normal, Abs(dist[i]) <= delta, normal + unplaced.normals[i]); // averaging
    }
#endif
    // check if normal is valid w.r.t. z-planes
    valid = valid && (unplaced.fDz - Abs(point[2])) >= -delta;

    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid))
      return (normal + Vector3D<Real_v>(1.0)).Normalized();

    // then check z-planes
    // testing -fDZ (still keeping safety for next face)
    Real_v distTmp = Abs(point.z() + unplaced.fDz);
    Bool_v closer  = distTmp < safety;
    vecCore::MaskedAssign(safety, closer, distTmp);
    vecCore::MaskedAssign(iface, closer, Index_v(4));
    // vecCore::MaskedAssign(normal[2], distTmp < delta, normal[2] - Real_v(1.0));  // averaging

    // testing +fDZ (no need to update safety)
    distTmp = Abs(point.z() - unplaced.fDz);
    vecCore::MaskedAssign(iface, distTmp < safety, Index_v(5));
    // vecCore::MaskedAssign(normal[2], distTmp < delta, normal[2] + Real_v(1.0));  // averaging

    // vecCore::MaskedAssign(normal, normal.Mag2() < delta, unplaced.normals[iface]);
    normal = unplaced.normals[iface];

    // returned vector must be normalized
    // vecCore::MaskedAssign(normal, normal.Mag2() > 0., normal.Normalized());

    return normal;
  }
};

} // end of inline namespace
} // end of global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TRAPEZOIDIMPLEMENTATION_H_
