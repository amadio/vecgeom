//===-- kernel/ParaboloidImplementation.h - Instruction class definition -------*- C++ -*-===//
//===----------------------------------------------------------------------===//
///
/// \file
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file implements the Paraboloid shape
///
///
/// _____________________________________________________________________________
/// A paraboloid is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
/// - the surface of revolution of a parabola described by:
/// z = a*(x*x + y*y) + b
/// The parameters a and b are automatically computed from:
/// - rlo is the radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi is the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// -dz = a*rlo^2 + b
/// dz = a*rhi^2 + b      where: rhi>rlo, both >= 0
///
/// note:
/// dd = 1./(rhi^2 - rlo^2);
/// a = 2.*dz*dd;
/// b = - dz * (rlo^2 + rhi^2)*dd;
///
/// in respect with the G4 implementation we have:
/// k1=1/a
/// k2=-b/a
///
/// a=1/k1
/// b=-k2/k1
///
//===----------------------------------------------------------------------===//
///
/// revision + moving to new backend structure : Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARABOLOIDIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/ParaboloidStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ParaboloidImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ParaboloidImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParaboloid;
template <typename T>
struct ParaboloidStruct;
class UnplacedParaboloid;

struct ParaboloidImplementation {

  using PlacedShape_t    = PlacedParaboloid;
  using UnplacedStruct_t = ParaboloidStruct<double>;
  using UnplacedVolume_t = UnplacedParaboloid;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    //  printf("SpecializedParaboloid<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st)
  {
    (void)st;
    // st << "SpecializedParaboloid<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "ParaboloidImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    // st << "UnplacedParaboloid";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(paraboloid, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(paraboloid, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point,
                                                Bool_v &completelyinside, Bool_v &completelyoutside)
  {
    // using Bool_v = vecCore::Mask_v<Real_v>;
    completelyinside  = Bool_v(false);
    completelyoutside = Bool_v(false);

    Real_v rho2       = point.Perp2();
    Real_v paraRho2   = paraboloid.fK1 * point.z() + paraboloid.fK2;
    Real_v diff       = rho2 - paraRho2;
    Real_v absZ       = vecCore::math::Abs(point.z());
    completelyoutside = (absZ > Real_v(paraboloid.fDz + kTolerance)) || (diff > kTolerance);
    if (vecCore::MaskFull(completelyoutside)) return;
    if (ForInside) completelyinside = (absZ < Real_v(paraboloid.fDz - kTolerance)) && (diff < -kTolerance);
  }

  template <typename Real_v, bool ForTopZPlane>
  VECGEOM_CUDA_HEADER_BOTH
  static vecCore::Mask_v<Real_v> IsOnZPlane(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point)
  {
    Real_v rho2 = point.Perp2();
    if (ForTopZPlane) {
      return vecCore::math::Abs(point.z() - paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRhi2 + kHalfTolerance);
    } else {
      return vecCore::math::Abs(point.z() + paraboloid.fDz) < kTolerance && rho2 < (paraboloid.fRlo2 + kHalfTolerance);
    }
  }

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  static vecCore::Mask_v<Real_v> IsOnParabolicSurface(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v value              = paraboloid.fA * point.Perp2() + paraboloid.fB - point.z();
    Bool_v onParabolicSurface = value > -kTolerance && value < kTolerance;
    return onParabolicSurface;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Bool_v done(false);
    distance = vecCore::NumericLimits<Real_v>::Infinity();

    Real_v absZ   = vecCore::math::Abs(point.z());
    Real_v rho2   = point.Perp2(); // point.x()*point.x()+point.y()*point.y();
    Bool_v checkZ = point.z() * direction.z() >= 0;

    // check if the point is distancing in Z
    Bool_v isDistancingInZ = (absZ > paraboloid.fDz && checkZ);
    done |= isDistancingInZ;
    if (vecCore::MaskFull(done)) return;

    Real_v paraRho2 = paraboloid.fK1 * point.z() + paraboloid.fK2;
    Real_v diff     = rho2 - paraRho2;

    vecCore::MaskedAssign(distance, !done, Real_v(-1.));
    Bool_v insideZ                              = absZ < Real_v(paraboloid.fDz - kTolerance);
    Bool_v insideParabolicSurfaceOuterTolerance = (diff < -kTolerance);
    done |= !done && (insideZ && insideParabolicSurfaceOuterTolerance);
    if (vecCore::MaskFull(done)) return;

    Bool_v isOnZPlaneAndMovingInside = (IsOnZPlane<Real_v, true>(paraboloid, point) && direction.z() < 0.) ||
                                       (IsOnZPlane<Real_v, false>(paraboloid, point) && direction.z() > 0.);
    vecCore::MaskedAssign(distance, !done && isOnZPlaneAndMovingInside, Real_v(0.));
    done |= isOnZPlaneAndMovingInside;
    if (vecCore::MaskFull(done)) return;

    Vector3D<Real_v> normal(point.x(), point.y(), -paraboloid.fK1 * 0.5);
    Bool_v isOnParabolicSurfaceAndMovingInside = diff > -kTolerance && diff < kTolerance && direction.Dot(normal) < 0.;
    vecCore::MaskedAssign(distance, !done && isOnParabolicSurfaceAndMovingInside, Real_v(0.));
    done |= isOnParabolicSurfaceAndMovingInside;
    if (vecCore::MaskFull(done)) return;

    vecCore::MaskedAssign(distance, !done, vecCore::NumericLimits<Real_v>::Infinity());

    /* Intersection tests with Z planes are not required if the point is within Z Range
     * In this case it will either intsect with parabolic surface or not intersect at all.
     */
    if (!vecCore::MaskFull(absZ < paraboloid.fDz)) {
      Real_v distZ(vecCore::NumericLimits<Real_v>::Infinity());              // = (absZ - paraboloid.fDz) / absDirZ;
      Bool_v bottomPlane = point.z() < -paraboloid.fDz && direction.z() > 0; //(true);
      Bool_v topPlane    = point.z() > paraboloid.fDz && direction.z() < 0;
      vecCore::MaskedAssign(distZ, topPlane, (paraboloid.fDz - point.z()) / direction.z());
      vecCore::MaskedAssign(distZ, bottomPlane, (-paraboloid.fDz - point.z()) / direction.z());
      Real_v xHit    = point.x() + distZ * direction.x();
      Real_v yHit    = point.y() + distZ * direction.y();
      Real_v rhoHit2 = xHit * xHit + yHit * yHit;

      vecCore::MaskedAssign(distance, !done && topPlane && rhoHit2 <= paraboloid.fRhi2, distZ);
      done |= topPlane && rhoHit2 < paraboloid.fRhi2;
      if (vecCore::MaskFull(done)) return;

      vecCore::MaskedAssign(distance, !done && bottomPlane && rhoHit2 <= paraboloid.fRlo2, distZ);
      done |= (bottomPlane && rhoHit2 <= paraboloid.fRlo2); // || (topPlane && rhoHit2 <= paraboloid.fRhi2);
      if (vecCore::MaskFull(done)) return;
    }

    /* Intersection tests with Parabolic surface are not required if the point is above
     * top Z plane Radius of point is less the Rhi. In this case depending upon the
     * direction it will either intersect with top Z plane or not intersect at all
     */
    if (!vecCore::MaskFull(point.z() > paraboloid.fDz && rho2 < paraboloid.fRhi2)) {
      // Quadratic Solver for Parabolic surface
      Real_v dirRho2 = direction.Perp2();
      Real_v pDotV2D = point.x() * direction.x() + point.y() * direction.y();
      Real_v a       = paraboloid.fA * dirRho2;
      Real_v b       = 0.5 * direction.z() - paraboloid.fA * pDotV2D;
      Real_v c       = (paraboloid.fB + paraboloid.fA * point.Perp2() - point.z());
      Real_v d2      = b * b - a * c;
      done |= d2 < 0.;
      if (vecCore::MaskFull(done)) return;

      Real_v distParab = vecCore::NumericLimits<Real_v>::Infinity();
      vecCore::MaskedAssign(distParab, !done && (b > 0.), (b - Sqrt(d2)) / a);
      vecCore::MaskedAssign(distParab, !done && (b <= 0.), (c / (b + Sqrt(d2))));
      Real_v zHit = point.z() + distParab * direction.z();
      vecCore::MaskedAssign(distance, vecCore::math::Abs(zHit) <= paraboloid.fDz && distParab > 0., distParab);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    // setting distance to -1. for wrong side points
    distance = -1.;
    Bool_v done(false);

    // Outside Z range
    Bool_v outsideZ = vecCore::math::Abs(point.z()) > paraboloid.fDz + kTolerance;
    done |= outsideZ;
    if (vecCore::MaskFull(done)) return;

    // Outside Parabolic surface
    Real_v rho2                                  = point.Perp2();
    Real_v paraRho2                              = paraboloid.fK1 * point.z() + paraboloid.fK2;
    Real_v value                                 = rho2 - paraRho2;
    Bool_v outsideParabolicSurfaceOuterTolerance = (value > kHalfTolerance);
    done |= outsideParabolicSurfaceOuterTolerance;
    if (vecCore::MaskFull(done)) return;

    // On Z Plane and moving outside;
    Bool_v isOnZPlaneAndMovingOutside = (IsOnZPlane<Real_v, true>(paraboloid, point) && direction.z() > 0.) ||
                                        (IsOnZPlane<Real_v, false>(paraboloid, point) && direction.z() < 0.);
    vecCore::MaskedAssign(distance, !done && isOnZPlaneAndMovingOutside, Real_v(0.));
    done |= isOnZPlaneAndMovingOutside;
    if (vecCore::MaskFull(done)) return;

    // On Parabolic Surface and moving outside
    Vector3D<Real_v> normal(point.x(), point.y(), -paraboloid.fK1 * 0.5);
    Bool_v isOnParabolicSurfaceAndMovingInside =
        value > -kTolerance && value < kTolerance && direction.Dot(normal) > 0.;
    vecCore::MaskedAssign(distance, !done && isOnParabolicSurfaceAndMovingInside, Real_v(0.));
    done |= isOnParabolicSurfaceAndMovingInside;
    if (vecCore::MaskFull(done)) return;

    vecCore::MaskedAssign(distance, !done, vecCore::NumericLimits<Real_v>::Infinity());

    Real_v distZ   = vecCore::NumericLimits<Real_v>::Infinity();
    Real_v dirZinv = Real_v(1 / direction.z());

    Bool_v dir_mask = direction.z() < 0;
    vecCore::MaskedAssign(distZ, dir_mask, -(paraboloid.fDz + point.z()) * dirZinv);
    vecCore::MaskedAssign(distZ, !dir_mask, (paraboloid.fDz - point.z()) * dirZinv);

    Real_v dirRho2 = direction.Perp2();
    Real_v pDotV2D = point.x() * direction.x() + point.y() * direction.y();
    Real_v a       = paraboloid.fA * dirRho2;
    Real_v b       = 0.5 * direction.z() - paraboloid.fA * pDotV2D;
    Real_v c       = paraboloid.fB + paraboloid.fA * point.Perp2() - point.z();
    Real_v d2      = b * b - a * c;

    Real_v distParab = vecCore::NumericLimits<Real_v>::Infinity();
    vecCore::MaskedAssign(distParab, d2 >= 0. && (b > 0.), (b + Sqrt(d2)) * (1. / a));
    vecCore::MaskedAssign(distParab, d2 >= 0. && (b <= 0.), (c / (b - Sqrt(d2))));
    distance = vecCore::math::Min(distParab, distZ);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, Real_v &safety)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    Real_v absZ  = vecCore::math::Abs(point.z());
    Real_v safeZ = absZ - paraboloid.fDz;

    safety = -1.;
    Bool_v done(false);
    Bool_v insideZ = absZ < paraboloid.fDz - kTolerance;

    Real_v rho2                                 = point.Perp2();
    Real_v value                                = paraboloid.fA * rho2 + paraboloid.fB - point.z();
    Bool_v insideParabolicSurfaceOuterTolerance = (value < -kHalfTolerance);
    done |= (insideZ && insideParabolicSurfaceOuterTolerance);
    if (vecCore::MaskFull(done)) return;

    Bool_v onZPlane =
        vecCore::math::Abs(vecCore::math::Abs(point.z()) - paraboloid.fDz) < kTolerance &&
        (rho2 < Real_v(paraboloid.fRhi2 + kHalfTolerance) || rho2 < Real_v(paraboloid.fRlo2 + kHalfTolerance));
    vecCore::MaskedAssign(safety, onZPlane, Real_v(0.));
    done |= onZPlane;
    if (vecCore::MaskFull(done)) return;

    Bool_v onParabolicSurface = value > -kTolerance && value < kTolerance;
    vecCore::MaskedAssign(safety, !done && onParabolicSurface, Real_v(0.));
    done |= onParabolicSurface;
    if (vecCore::MaskFull(done)) return;

    vecCore::MaskedAssign(safety, !done, vecCore::NumericLimits<Real_v>::Infinity());

    Real_v r0sq = (point.z() - paraboloid.fB) * paraboloid.fInvA;

    safety = safeZ;

    Bool_v underParaboloid = (r0sq < 0);
    done |= underParaboloid;
    if (vecCore::MaskFull(done)) return;

    Real_v safeR = vecCore::NumericLimits<Real_v>::Infinity();
    Real_v ro2   = point.x() * point.x() + point.y() * point.y();
    Real_v dr    = vecCore::math::Sqrt(ro2) - vecCore::math::Sqrt(r0sq);

    Bool_v drCloseToZero = (dr < 1.E-8);
    done |= drCloseToZero;
    if (vecCore::MaskFull(done)) return;

    // then go for the tangent
    Real_v talf = -2. * paraboloid.fA * vecCore::math::Sqrt(r0sq);
    Real_v salf = talf / vecCore::math::Sqrt(1. + talf * talf);
    safeR       = vecCore::math::Abs(dr * salf);

    Real_v max_safety = vecCore::math::Max(safeR, safeZ);
    vecCore::MaskedAssign(safety, !done, max_safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point, Real_v &safety)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v absZ = vecCore::math::Abs(point.z());
    Real_v safZ = (paraboloid.fDz - absZ);

    safety = -1.;
    Bool_v done(false);
    Bool_v outsideZ = absZ > Real_v(paraboloid.fDz + kTolerance);
    done |= outsideZ;
    if (vecCore::MaskFull(done)) return;

    Real_v rho2                                  = point.Perp2();
    Real_v value                                 = paraboloid.fA * rho2 + paraboloid.fB - point.z();
    Bool_v outsideParabolicSurfaceOuterTolerance = (value > kHalfTolerance);
    done |= outsideParabolicSurfaceOuterTolerance;
    if (vecCore::MaskFull(done)) return;

    Bool_v onZPlane =
        vecCore::math::Abs(vecCore::math::Abs(point.z()) - paraboloid.fDz) < kTolerance &&
        (rho2 < Real_v(paraboloid.fRhi2 + kHalfTolerance) || rho2 < Real_v(paraboloid.fRlo2 + kHalfTolerance));
    vecCore::MaskedAssign(safety, onZPlane, Real_v(0.));
    done |= onZPlane;
    if (vecCore::MaskFull(done)) return;

    Bool_v onSurface = value > -kTolerance && value < kTolerance;
    vecCore::MaskedAssign(safety, !done && onSurface, Real_v(0.));
    done |= onSurface;
    if (vecCore::MaskFull(done)) return;
    Real_v r0sq = (point.z() - paraboloid.fB) * paraboloid.fInvA;

    safety = 0.;

    Bool_v closeToParaboloid = (r0sq < 0);
    done |= closeToParaboloid;
    if (vecCore::MaskFull(done)) return;

    Real_v safR = vecCore::NumericLimits<Real_v>::Infinity();
    Real_v ro2  = point.x() * point.x() + point.y() * point.y();
    Real_v z0   = paraboloid.fA * ro2 + paraboloid.fB;
    Real_v dr   = vecCore::math::Sqrt(ro2) - vecCore::math::Sqrt(r0sq); // avoid square root of a negative number

    Bool_v drCloseToZero = (dr > -1.E-8);
    done |= drCloseToZero;
    if (vecCore::MaskFull(done)) return;

    Real_v dz = vecCore::math::Abs(point.z() - z0);
    safR      = -dr * dz / vecCore::math::Sqrt(dr * dr + dz * dz);

    Real_v min_safety = vecCore::math::Min(safR, safZ);
    vecCore::MaskedAssign(safety, !done, min_safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &paraboloid, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;

    // used to store the normal that needs to be returned
    Vector3D<Real_v> normal(0., 0., 0.);
    Real_v nsurf(0.); // used to store the number of surfaces on which the point lie.
    // in case of paraboloid it can maximum go upto 2

    Real_v r    = point.Perp();
    Real_v talf = -2 * paraboloid.fA * r;
    Real_v calf = 1. / vecCore::math::Sqrt(1. + talf * talf);
    Real_v salf = talf * calf;
    Vector3D<Real_v> normParabolic((salf * point.x() / point.Perp()), (salf * point.y() / point.Perp()), calf);
    normParabolic.Normalize();

    // Logic for Valid Normal i.e. when point is on the surface
    Bool_v isOnZPlane = IsOnZPlane<Real_v, true>(paraboloid, point) || IsOnZPlane<Real_v, false>(paraboloid, point);
    Bool_v isOnParabolicSurface = IsOnParabolicSurface<Real_v>(paraboloid, point);

    vecCore::MaskedAssign(nsurf, isOnZPlane, nsurf + 1);
    vecCore::MaskedAssign(normal[2], IsOnZPlane<Real_v, true>(paraboloid, point), Real_v(1.));
    vecCore::MaskedAssign(normal[2], IsOnZPlane<Real_v, false>(paraboloid, point), Real_v(-1.));

    vecCore::MaskedAssign(nsurf, isOnParabolicSurface, nsurf + 1);
    vecCore::MaskedAssign(normal[0], isOnParabolicSurface, normal[0] - normParabolic[0]);
    vecCore::MaskedAssign(normal[1], isOnParabolicSurface, normal[1] - normParabolic[1]);
    vecCore::MaskedAssign(normal[2], isOnParabolicSurface, normal[2] - normParabolic[2]);

    valid = Bool_v(true);
    valid &= (nsurf > 0);

    if (vecCore::MaskFull(valid)) return normal.Normalized();

    // This block is used to calculate the Approximate normal
    Vector3D<Real_v> norm(0., 0., 0.); // used to store approximate normal
    vecCore::MaskedAssign(norm[2], point.z() > 0., Real_v(1.));
    vecCore::MaskedAssign(norm[2], point.z() < 0, Real_v(-1.));

    Real_v safz = paraboloid.fDz - vecCore::math::Abs(point.z());
    Real_v safr = vecCore::math::Abs(r - vecCore::math::Sqrt((point.z() - paraboloid.fB) * paraboloid.fInvA));
    vecCore::MaskedAssign(norm[0], safz >= 0. && safr < safz, normParabolic.x());
    vecCore::MaskedAssign(norm[1], safz >= 0. && safr < safz, normParabolic.y());
    vecCore::MaskedAssign(norm[2], safz >= 0. && safr < safz, normParabolic.z());

    // If Valid is not set, that means the point is NOT on the surface,
    // So in that case we have to rely on Approximate normal
    vecCore::MaskedAssign(normal[0], !valid, norm.x());
    vecCore::MaskedAssign(normal[1], !valid, norm.y());
    vecCore::MaskedAssign(normal[2], !valid, norm.z());

    return normal.Normalized();
  }
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
