/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch), Sandro Wenzel (sandro.wenzel@cern.ch)

/// History notes:
/// 2013 - 2014: original development (abstracted kernels); Johannes and Sandro
/// Oct 2015: revision + moving to new backend structure (Sandro Wenzel)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/BoxStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct BoxImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, BoxImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
template <typename T>
struct BoxStruct;
class UnplacedBox;

struct BoxImplementation {

  using PlacedShape_t    = PlacedBox;
  using UnplacedStruct_t = BoxStruct<double>;
  using UnplacedVolume_t = UnplacedBox;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    //  printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st)
  {
    (void)st;
    // st << "SpecializedBox<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "BoxImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    // st << "UnplacedBox";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &box, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(box.fDimensions, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &box, Vector3D<Real_v> const &point, Inside_t &inside)
  {
    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(box.fDimensions, point, completelyinside,
                                                            completelyoutside);
    inside = Inside_t(EInside::kSurface);
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(Vector3D<Precision> const &dimensions,
                                                Vector3D<Real_v> const &localPoint, Bool_v &completelyinside,
                                                Bool_v &completelyoutside)
  {
    // here we are explicitely unrolling the loop since  a for statement will likely be a penality
    // check if second call to Abs is compiled away
    // and it can anyway not be vectorized
    /* x */
    completelyoutside = Abs(localPoint[0]) > MakePlusTolerant<ForInside>(dimensions[0]);
    if (ForInside) {
      completelyinside = Abs(localPoint[0]) < MakeMinusTolerant<ForInside>(dimensions[0]);
    }
    if (/*vecCore::EarlyReturnAllowed()*/ true) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }
    /* y */
    completelyoutside |= Abs(localPoint[1]) > MakePlusTolerant<ForInside>(dimensions[1]);
    if (ForInside) {
      completelyinside &= Abs(localPoint[1]) < MakeMinusTolerant<ForInside>(dimensions[1]);
    }
    if (/*vecCore::EarlyReturnAllowed()*/ true) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }
    /* z */
    completelyoutside |= Abs(localPoint[2]) > MakePlusTolerant<ForInside>(dimensions[2]);
    if (ForInside) {
      completelyinside &= Abs(localPoint[2]) < MakeMinusTolerant<ForInside>(dimensions[2]);
    }
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &box, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {

    Vector3D<Real_v> safety;
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    distance = vecCore::NumericLimits<Real_v>::Infinity();

    safety[0] = Abs(point[0]) - box.fDimensions[0];
    safety[1] = Abs(point[1]) - box.fDimensions[1];
    safety[2] = Abs(point[2]) - box.fDimensions[2];

    done |= (safety[0] >= stepMax || safety[1] >= stepMax || safety[2] >= stepMax);
    if (vecCore::MaskFull(done)) return;

    Bool_v inside(false);
    inside = safety[0] < -kHalfTolerance && safety[1] < -kHalfTolerance && safety[2] < -kHalfTolerance;
    vecCore::MaskedAssign(distance, !done && inside, Real_v(-1.));

    done |= inside;
    if (vecCore::MaskFull(done)) return;

    Real_v next, coord1, coord2;
    Bool_v hit;

    // x
    next   = safety[0] / NonZeroAbs(direction[0]);
    coord1 = point[1] + next * direction[1];
    coord2 = point[2] + next * direction[2];
    hit    = safety[0] >= -kHalfTolerance && point[0] * direction[0] < 0. && Abs(coord1) <= box.fDimensions[1] &&
          Abs(coord2) <= box.fDimensions[2];
    vecCore::MaskedAssign(distance, !done && hit, next);
    done |= hit;
    if (vecCore::MaskFull(done)) return;

    // y
    next   = safety[1] / NonZeroAbs(direction[1]);
    coord1 = point[0] + next * direction[0];
    coord2 = point[2] + next * direction[2];
    hit    = safety[1] >= -kHalfTolerance && point[1] * direction[1] < 0 && Abs(coord1) <= box.fDimensions[0] &&
          Abs(coord2) <= box.fDimensions[2];
    vecCore::MaskedAssign(distance, !done && hit, next);
    done |= hit;
    if (vecCore::MaskFull(done)) return;

    // z
    next   = safety[2] / NonZeroAbs(direction[2]);
    coord1 = point[0] + next * direction[0];
    coord2 = point[1] + next * direction[1];
    hit    = safety[2] >= -kHalfTolerance && point[2] * direction[2] < 0 && Abs(coord1) <= box.fDimensions[0] &&
          Abs(coord2) <= box.fDimensions[1];
    vecCore::MaskedAssign(distance, !done && hit, next);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &box, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {
    distance = Real_v(-1.);

    using Bool_v = vecCore::Mask_v<Real_v>;
    // treatment to find out if on wrong side
    Bool_v done = Abs(point[0]) > box.fDimensions[0] + kHalfTolerance;
    done |= Abs(point[1]) > box.fDimensions[1] + kHalfTolerance;
    done |= Abs(point[2]) > box.fDimensions[2] + kHalfTolerance;
    if (vecCore::MaskFull(done)) return;

    const Vector3D<Real_v> moddirection(NonZero(direction[0]), NonZero(direction[1]), NonZero(direction[2]));
    const Vector3D<Real_v> inverseDirection(1. / moddirection[0], 1. / moddirection[1], 1. / moddirection[2]);
    Vector3D<Real_v> distances((box.fDimensions[0] - point[0]) * inverseDirection[0],
                               (box.fDimensions[1] - point[1]) * inverseDirection[1],
                               (box.fDimensions[2] - point[2]) * inverseDirection[2]);

    using vecCore::MaskedAssign;
    MaskedAssign(distances[0], moddirection[0] < 0., (-box.fDimensions[0] - point[0]) * inverseDirection[0]);
    MaskedAssign(distances[1], moddirection[1] < 0., (-box.fDimensions[1] - point[1]) * inverseDirection[1]);
    MaskedAssign(distances[2], moddirection[2] < 0., (-box.fDimensions[2] - point[2]) * inverseDirection[2]);

    MaskedAssign(distance, !done, distances[0]);
    distance = Min(distances[1], distance);
    distance = Min(distances[2], distance);
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &box, Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety               = -box.fDimensions[0] + Abs(point[0]);
    const Real_v safetyY = -box.fDimensions[1] + Abs(point[1]);
    const Real_v safetyZ = -box.fDimensions[2] + Abs(point[2]);
    safety               = Max(safetyY, safety);
    safety               = Max(safetyZ, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &box, Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety               = box.fDimensions[0] - Abs(point[0]);
    const Real_v safetyY = box.fDimensions[1] - Abs(point[1]);
    const Real_v safetyZ = box.fDimensions[2] - Abs(point[2]);
    safety               = Min(safetyY, safety);
    safety               = Min(safetyZ, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &box, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    //   In case a point is further than tolerance_normal from a surface, set validNormal=false
    //   Must return a valid vector. (even if the point is not on the surface.)
    //
    //   On an edge or corner, provide an average normal of all facets within tolerance
    // NOTE: the tolerance value used in here is not yet the global surface
    //     tolerance - we will have to revise this value - TODO
    // this version does not yet consider the case when we are not on the surface

    using Bool_v = vecCore::Mask_v<Real_v>;
    using vecCore::MaskedAssign;

    Vector3D<Precision> dimensions = box.fDimensions;

    constexpr double delta     = 100. * kTolerance;
    constexpr double kInvSqrt2 = 0.7071067811865475; // = 1. / Sqrt(2.);
    constexpr double kInvSqrt3 = 0.5773502691896258; // = 1. / Sqrt(3.);
    Vector3D<Real_v> normal;
    normal.Set(0.);
    Real_v nsurf(0.);
    Real_v safmin(kInfinity);

    // loop here over dimensions
    for (int dim = 0; dim < 3; ++dim) {
      Real_v currentsafe = Abs(Abs(point[dim]) - dimensions[dim]);
      safmin             = Min(currentsafe, safmin);

      // close to this surface
      Bool_v closetoplane = currentsafe < delta;
      if (!vecCore::MaskEmpty(closetoplane)) {
        Real_v nsurftmp = nsurf + 1.;

        Real_v sign(1.);
        // better to use copysign instead of masked assigment?
        MaskedAssign(sign, point[dim] < 0, -Real_v(1.));
        Real_v tmpnormalcomponent = normal[dim] + sign;

        // masked assignment
        MaskedAssign(nsurf, closetoplane, nsurftmp);
        MaskedAssign(normal[dim], closetoplane, tmpnormalcomponent);
      }
    }

    valid = Bool_v(true);
    valid &= nsurf > 0;
    // masked normalization ( a bit ugly since we don't have a masked operator on Vector3D
    MaskedAssign(normal[0], nsurf == 3., normal[0] * kInvSqrt3);
    MaskedAssign(normal[1], nsurf == 3., normal[1] * kInvSqrt3);
    MaskedAssign(normal[2], nsurf == 3., normal[2] * kInvSqrt3);
    MaskedAssign(normal[0], nsurf == 2., normal[0] * kInvSqrt2);
    MaskedAssign(normal[1], nsurf == 2., normal[1] * kInvSqrt2);
    MaskedAssign(normal[2], nsurf == 2., normal[2] * kInvSqrt2);

    // TODO: return normal in case of nonvalid case;
    // need to keep track of minimum safety direction
    return normal;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  // template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static bool Intersect(Vector3D<Precision> const *corners, Vector3D<Precision> const &point,
                        Vector3D<Precision> const &ray, Precision /* t0 */, Precision /* t1 */)
  {
    // intersection algorithm 1 ( Amy Williams )
    Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

    // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES
    double inverserayx = 1. / ray[0];
    double inverserayy = 1. / ray[1];

    // TODO: we should promote this to handle multiple boxes
    int sign[3];
    sign[0] = inverserayx < 0;
    sign[1] = inverserayy < 0;

    tmin  = (corners[sign[0]].x() - point.x()) * inverserayx;
    tmax  = (corners[1 - sign[0]].x() - point.x()) * inverserayx;
    tymin = (corners[sign[1]].y() - point.y()) * inverserayy;
    tymax = (corners[1 - sign[1]].y() - point.y()) * inverserayy;

    if ((tmin > tymax) || (tymin > tmax)) return false;

    double inverserayz = 1. / ray.z();
    sign[2]            = inverserayz < 0;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (corners[sign[2]].z() - point.z()) * inverserayz;
    tzmax = (corners[1 - sign[2]].z() - point.z()) * inverserayz;

    if ((tmin > tzmax) || (tzmin > tmax)) return false;
    if ((tzmin > tmin)) tmin = tzmin;
    if (tzmax < tmax) tmax   = tzmax;
    // return ((tmin < t1) && (tmax > t0));
    // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
    return true;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <int signx, int signy, int signz>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  //__attribute__((noinline))
  static Precision IntersectCached(Vector3D<Precision> const *corners, Vector3D<Precision> const &point,
                                   Vector3D<Precision> const &inverseray, Precision t0, Precision t1)
  {
    // intersection algorithm 1 ( Amy Williams )

    // NOTE THE FASTEST VERSION IS STILL THE ORIGINAL IMPLEMENTATION

    Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

    // TODO: we should promote this to handle multiple boxes
    // observation: we always compute sign and 1-sign; so we could do the assignment
    // to tmin and tmax in a masked assignment thereafter
    tmin  = (corners[signx].x() - point.x()) * inverseray.x();
    tmax  = (corners[1 - signx].x() - point.x()) * inverseray.x();
    tymin = (corners[signy].y() - point.y()) * inverseray.y();
    tymax = (corners[1 - signy].y() - point.y()) * inverseray.y();
    if ((tmin > tymax) || (tymin > tmax)) return vecgeom::kInfinity;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    if ((tmin > tzmax) || (tzmin > tmax)) return vecgeom::kInfinity; // false
    if ((tzmin > tmin)) tmin = tzmin;
    if (tzmax < tmax) tmax   = tzmax;

    if (!((tmin < t1) && (tmax > t0))) return vecgeom::kInfinity;
    return tmin;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <typename Real_v, int signx, int signy, int signz>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Real_v IntersectCachedKernel(Vector3D<Real_v> const *corners, Vector3D<Precision> const &point,
                                      Vector3D<Precision> const &inverseray, Precision t0, Precision t1)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v tmin  = (corners[signx].x() - point.x()) * inverseray.x();
    Real_v tmax  = (corners[1 - signx].x() - point.x()) * inverseray.x();
    Real_v tymin = (corners[signy].y() - point.y()) * inverseray.y();
    Real_v tymax = (corners[1 - signy].y() - point.y()) * inverseray.y();

    // do we need this condition ?
    Bool_v done = (tmin > tymax) || (tymin > tmax);
    if (vecCore::MaskFull(done)) return vecgeom::kInfinity;
    // if((tmin > tymax) || (tymin > tmax))
    //     return vecgeom::kInfinity;

    // Not sure if this has to be maskedassignments
    tmin = Max(tmin, tymin);
    tmax = Min(tmax, tymax);

    Real_v tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    Real_v tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    done |= (tmin > tzmax) || (tzmin > tmax);
    // if((tmin > tzmax) || (tzmin > tmax))
    //     return vecgeom::kInfinity; // false
    if (vecCore::MaskFull(done)) return vecgeom::kInfinity;

    // not sure if this has to be maskedassignments
    tmin = Max(tmin, tzmin);
    tmax = Min(tmax, tzmax);

    done |= !((tmin < t1) && (tmax > t0));
    // if( ! ((tmin < t1) && (tmax > t0)) )
    //     return vecgeom::kInfinity;
    vecCore::MaskedAssign(tmin, done, Real_v(vecgeom::kInfinity));
    return tmin;
  }

  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  template <typename Real_v, typename basep>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Real_v IntersectCachedKernel2(Vector3D<Real_v> const *corners, Vector3D<basep> const &point,
                                       Vector3D<basep> const &inverseray, int signx, int signy, int signz, basep t0,
                                       basep t1)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v tmin  = (corners[signx].x() - Real_v(point.x())) * inverseray.x();
    Real_v tymax = (corners[1 - signy].y() - Real_v(point.y())) * inverseray.y();
    Bool_v done  = tmin > tymax;
    if (vecCore::MaskFull(done)) return Real_v((basep)vecgeom::kInfinity);

    Real_v tmax  = (corners[1 - signx].x() - Real_v(point.x())) * inverseray.x();
    Real_v tymin = (corners[signy].y() - Real_v(point.y())) * inverseray.y();

    // do we need this condition ?
    done |= (tymin > tmax);
    if (vecCore::MaskFull(done)) return Real_v((basep)vecgeom::kInfinity);

    // if((tmin > tymax) || (tymin > tmax))
    //     return vecgeom::kInfinity;

    // Not sure if this has to be maskedassignments
    tmin = Max(tmin, tymin);
    tmax = Min(tmax, tymax);

    Real_v tzmin = (corners[signz].z() - point.z()) * inverseray.z();
    Real_v tzmax = (corners[1 - signz].z() - point.z()) * inverseray.z();

    done |= (Real_v(tmin) > Real_v(tzmax)) || (Real_v(tzmin) > Real_v(tmax));
    // if((tmin > tzmax) || (tzmin > tmax))
    //     return vecgeom::kInfinity; // false
    if (vecCore::MaskFull(done)) return Real_v((basep)vecgeom::kInfinity);

    // not sure if this has to be maskedassignments
    tmin = Max(tmin, tzmin);
    tmax = Min(tmax, tzmax);

    done |= !((tmin < Real_v(t1)) && (tmax > Real_v(t0)));
    // if( ! ((tmin < t1) && (tmax > t0)) )
    //     return vecgeom::kInfinity;
    vecCore::MaskedAssign(tmin, done, Real_v((basep)vecgeom::kInfinity));
    return tmin;
  }

  // an algorithm to test for intersection against many boxes but just one ray;
  // in this case, the inverse ray is cached outside and directly given here as input
  // we could then further specialize this function to the direction of the ray
  // because also the sign[] variables and hence the branches are predefined

  // one could do: template <class Backend, int sign0, int sign1, int sign2>
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Precision IntersectMultiple(Vector3D<Real_v> const lowercorners, Vector3D<Real_v> const uppercorners,
                                     Vector3D<Precision> const &point, Vector3D<Precision> const &inverseray,
                                     Precision t0, Precision t1)
  {
    // intersection algorithm 1 ( Amy Williams )

    typedef Real_v Float_t;

    Float_t tmin, tmax, tymin, tymax, tzmin, tzmax;
    // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES

    // TODO: we should promote this to handle multiple boxes
    // we might need to have an Index type

    // int sign[3];
    Float_t sign[3]; // this also exists
    sign[0] = inverseray.x() < 0;
    sign[1] = inverseray.y() < 0;

    // observation: we always compute sign and 1-sign; so we could do the assignment
    // to tmin and tmax in a masked assignment thereafter

    // tmin =  (corners[(int)sign[0]].x()   -point.x())*inverserayx;
    // tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
    // tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
    // tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

    double x0 = (lowercorners.x() - point.x()) * inverseray.x();
    double x1 = (uppercorners.x() - point.x()) * inverseray.x();
    double y0 = (lowercorners.y() - point.y()) * inverseray.y();
    double y1 = (uppercorners.y() - point.y()) * inverseray.y();
    // could we do this using multiplications?
    //    tmin =   !sign[0] ?  x0 : x1;
    //    tmax =   sign[0] ? x0 : x1;
    //    tymin =  !sign[1] ?  y0 : y1;
    //    tymax =  sign[1] ? y0 : y1;

    // could completely get rid of this ? because the sign is determined by the outside ray

    tmin  = (1 - sign[0]) * x0 + sign[0] * x1;
    tmax  = sign[0] * x0 + (1 - sign[0]) * x1;
    tymin = (1 - sign[1]) * y0 + sign[1] * y1;
    tymax = sign[1] * y0 + (1 - sign[1]) * y1;

    // tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
    // tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
    // tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

    if ((tmin > tymax) || (tymin > tmax)) return vecgeom::kInfinity;

    //  double inverserayz = 1./ray.z();
    sign[2] = inverseray.z() < 0;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    //
    // tzmin = (lowercorners[(int) sign[2]].z()   -point.z())*inverseray.z();
    // tzmax = (uppercorners[(int)(1-sign[2])].z() -point.z())*inverseray.z();

    if ((tmin > tzmax) || (tzmin > tmax)) return vecgeom::kInfinity; // false
    if ((tzmin > tmin)) tmin = tzmin;
    if (tzmax < tmax) tmax   = tzmax;

    if (!((tmin < t1) && (tmax > t0))) return vecgeom::kInfinity;
    // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
    // return true;
    return tmin;
  }
}; // End struct BoxImplementation

struct ABBoxImplementation {

  // a contains kernel to be used with aligned bounding boxes
  // scalar and vector modes (aka backend) for boxes but only single points
  // should be useful to test one point against many bounding boxes
  // TODO: check if this can be unified with the normal generic box kernel
  template <typename Real_v, typename Bool_v = typename vecCore::Mask_v<Real_v>>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ABBoxContainsKernel(Vector3D<Real_v> const &lowercorner, Vector3D<Real_v> const &uppercorner,
                                  Vector3D<Precision> const &point, Bool_v &inside)
  {

    inside = lowercorner.x() < Real_v(point.x());
    inside &= uppercorner.x() > Real_v(point.x());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.y() < Real_v(point.y());
    inside &= uppercorner.y() > Real_v(point.y());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.z() < Real_v(point.z());
    inside &= uppercorner.z() > Real_v(point.z());
  }

  // playing with a kernel that can do multi-box - single particle; multi-box -- multi-particle, single-box --
  // multi-particle
  template <typename T1, typename T2, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void ABBoxContainsKernelGeneric(Vector3D<T1> const &lowercorner, Vector3D<T1> const &uppercorner,
                                         Vector3D<T2> const &point, Bool_v &inside)
  {
    inside = lowercorner.x() < T1(point.x());
    inside &= uppercorner.x() > T1(point.x());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.y() < T1(point.y());
    inside &= uppercorner.y() > T1(point.y());
    if (vecCore::MaskEmpty(inside)) return;

    inside &= lowercorner.z() < T1(point.z());
    inside &= uppercorner.z() > T1(point.z());
  }

  // safety square for Bounding boxes
  // generic kernel treating one track and one or multiple boxes
  // in case a point is inside a box a squared value
  // is returned but given an overall negative sign
  template <typename Real_v, typename Real_s = typename vecCore::TypeTraits<Real_v>::ScalarType>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Real_v ABBoxSafetySqr(Vector3D<Real_v> const &lowercorner, Vector3D<Real_v> const &uppercorner,
                               Vector3D<Real_s> const &point)
  {

    using Vector3D_v = Vector3D<Real_v>;
    using Bool_v     = vecCore::Mask_v<Real_v>;

    const Vector3D_v kHalf(Real_v(static_cast<Real_s>(0.5)));
    const Vector3D_v origin((uppercorner + lowercorner) * kHalf);
    const Vector3D_v delta((uppercorner - lowercorner) * kHalf);
    // promote scalar point to vector point
    Vector3D_v promotedpoint(Real_v(point.x()), Real_v(point.y()), Real_v(point.z()));

    // it would be nicer to have a standalone Abs function taking Vector3D as input
    Vector3D_v safety = ((promotedpoint - origin).Abs()) - delta;
    Bool_v outsidex   = safety.x() > Real_s(0.);
    Bool_v outsidey   = safety.y() > Real_s(0.);
    Bool_v outsidez   = safety.z() > Real_s(0.);

    Real_v runningsafetysqr(0.);            // safety squared from outside
    Real_v runningmax(-vecgeom::kInfinity); // relevant for safety when we are inside

    // loop over dimensions manually unrolled
    // treat x dim
    {
      // this will be much simplified with operator notation
      Real_v tmp(0.);
      vecCore::MaskedAssign(tmp, outsidex, safety.x() * safety.x());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.x());
    }

    // treat y dim
    {
      Real_v tmp(0.);
      vecCore::MaskedAssign(tmp, outsidey, safety.y() * safety.y());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.y());
    }

    // treat z dim
    {
      Real_v tmp(0.);
      vecCore::MaskedAssign(tmp, outsidez, safety.z() * safety.z());
      runningsafetysqr += tmp;
      runningmax = Max(runningmax, safety.z());
    }

    Bool_v inside = !(outsidex || outsidey || outsidez);
    if (!vecCore::MaskEmpty(inside)) vecCore::MaskedAssign(runningsafetysqr, inside, -runningmax * runningmax);
    return runningsafetysqr;
  }

}; // end aligned bounding box struct
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
