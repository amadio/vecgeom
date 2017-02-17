/*
 * ConeImplementation.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedCone.h"
#include "volumes/kernel/shapetypes/ConeTypes.h"
#include <cstdio>
#include <VecCore/VecCore>

#define kConeTolerance 1e-7
#define kHalfConeTolerance 0.5 * kConeTolerance

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(struct, ConeImplementation, TranslationCode, translation::kGeneric,
                                           RotationCode, rotation::kGeneric, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename ConeType>
struct ConeHelper {
  /*
      template <typename VecType, typename OtherType typename BoolType>
      static
      inline
      typename BoolType IsInRightZInterval( VecType const & z, OtherType const & dz )
      {
         return Abs(z) <= dz;
      }

     template<typename ConeType>
     template<typename VectorType, typename MaskType>
     inline
     __attribute__((always_inline))
     typename MaskType determineRHit( UnplacedCone const & unplaced, VectorType const & x, VectorType const & y,
     VectorType const & z,
                                      VectorType const & dirx, VectorType const & diry, VectorType const & dirz,
                                      VectorType const & distanceR ) const
     {
        if( ! checkPhiTreatment<ConeType>( unplaced ) )
        {
           return distanceR > 0 && IsInRightZInterval( z+distanceR*dirz, unplaced->GetDz() );
        }
        else
        {
           // need to have additional look if hitting point on zylinder is not in empty phi range
           VectorType xhit = x + distanceR*dirx;
           VectorType yhit = y + distanceR*diry;
           return distanceR > 0 && IsInRightZInterval( z + distanceR*dirz, unplaced->GetDz() )
                        && ! GeneralPhiUtils::PointIsInPhiSector(
                           unplaced->normalPhi1.x,
                           unplaced->normalPhi1.y,
                           unplaced->normalPhi2.x,
                           unplaced->normalPhi2.y, xhit, yhit );
        }
     }
  */
};

namespace ConeUtilities {

/**
* Returns whether a point is inside a cylindrical sector, as defined
* by the two vectors that go along the endpoints of the sector
*
* The same could be achieved using atan2 to calculate the angle formed
* by the point, the origin and the X-axes, but this is a lot faster,
* using only multiplications and comparisons
*
* (-x*starty + y*startx) >= 0: calculates whether going from the start vector to the point
* we are traveling in the CCW direction (taking the shortest direction, of course)
*
* (-endx*y + endy*x) >= 0: calculates whether going from the point to the end vector
* we are traveling in the CCW direction (taking the shortest direction, of course)
*
* For a sector smaller than pi, we need that BOTH of them hold true - if going from start, to the
* point, and then to the end we are travelling in CCW, it's obvious the point is inside the
* cylindrical sector.
*
* For a sector bigger than pi, only one of the conditions needs to be true. This is less obvious why.
* Since the sector angle is greater than pi, it can be that one of the two vectors might be
* farther than pi away from the point. In that case, the shortest direction will be CW, so even
* if the point is inside, only one of the two conditions need to hold.
*
* If going from start to point is CCW, then certainly the point is inside as the sector
* is larger than pi.
*
* If going from point to end is CCW, again, the point is certainly inside.
*
* This function is a frankensteinian creature that can determine which of the two cases (smaller vs larger than pi)
* to use either at compile time (if it has enough information, saving an if statement) or at runtime.
**/

template <typename Backend, typename ShapeType, typename UnplacedVolumeType, bool onSurfaceT,
          bool includeSurface = true>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static void PointInCyclicalSector(UnplacedVolumeType const &volume, typename Backend::precision_v const &x,
                                  typename Backend::precision_v const &y, typename Backend::bool_v &ret)
{

  using namespace ::vecgeom::ConeTypes;
  // assert(SectorType<ShapeType>::value != kNoAngle && "ShapeType without a sector passed to PointInCyclicalSector");

  typedef typename Backend::precision_v Float_t;

  Float_t startx = volume.GetAlongPhi1X();
  Float_t starty = volume.GetAlongPhi1Y();

  Float_t endx = volume.GetAlongPhi2X();
  Float_t endy = volume.GetAlongPhi2Y();

  bool smallerthanpi;

  if (SectorType<ShapeType>::value == kUnknownAngle)
    smallerthanpi = volume.GetDPhi() <= M_PI;
  else
    smallerthanpi = SectorType<ShapeType>::value == kPi || SectorType<ShapeType>::value == kSmallerThanPi;

  Float_t startCheck = (-x * starty) + (y * startx);
  Float_t endCheck   = (-endx * y) + (endy * x);
  // std::cout<<"StartCheck : "<<startCheck<<" : EndCheck : "<<endCheck<<std::endl;

  if (onSurfaceT) {
    // in this case, includeSurface is irrelevant
    ret = (Abs(startCheck) <= kHalfConeTolerance) | (Abs(endCheck) <= kHalfConeTolerance);
  } else {
    if (smallerthanpi) {
      if (includeSurface)
        ret = (startCheck >= -kHalfConeTolerance) & (endCheck >= -kHalfConeTolerance);
      else
        ret = (startCheck >= kHalfConeTolerance) & (endCheck >= kHalfConeTolerance);
    } else {
      if (includeSurface)
        ret = (startCheck >= -kHalfConeTolerance) | (endCheck >= -kHalfConeTolerance);
      else
        ret = (startCheck >= kHalfConeTolerance) | (endCheck >= kHalfConeTolerance);
    }
  }
}

template <typename Backend, bool ForInnerRadius>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::precision_v GetRadiusOfConeAtPoint(UnplacedCone const &cone,
                                                            typename Backend::precision_v const pointZ)
{

  if (ForInnerRadius) {
    return cone.GetInnerSlope() * pointZ + cone.GetInnerOffset();

  } else {
    return cone.GetOuterSlope() * pointZ + cone.GetOuterOffset();
  }
}

/*
* Check intersection of the trajectory with a phi-plane
* All points of the along-vector of a phi plane lie on
* s * (alongX, alongY)
* All points of the trajectory of the particle lie on
* (x, y) + t * (vx, vy)
* Thefore, it must hold that s * (alongX, alongY) == (x, y) + t * (vx, vy)
* Solving by t we get t = (alongY*x - alongX*y) / (vy*alongX - vx*alongY)
* s = (x + t*vx) / alongX = (newx) / alongX
*
* If we have two non colinear phi-planes, need to make sure
* point falls on its positive direction <=> dot product between
* along vector and hit-point is positive <=> hitx*alongX + hity*alongY > 0
*/

template <typename Backend, typename ConeType, bool PositiveDirectionOfPhiVector, bool insectorCheck>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static void PhiPlaneTrajectoryIntersection(Precision alongX, Precision alongY, Precision normalX, Precision normalY,
                                           UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &pos,
                                           Vector3D<typename Backend::precision_v> const &dir,
                                           typename Backend::precision_v &dist, typename Backend::bool_v &ok)
{

  typedef typename Backend::precision_v Float_t;

  dist = kInfLength;

  // approaching phi plane from the right side?
  // this depends whether we use it for DistanceToIn or DistanceToOut
  // Note: wedge normals poing towards the wedge inside, by convention!
  if (insectorCheck)
    ok = ((dir.x() * normalX) + (dir.y() * normalY) > 0.); // DistToIn  -- require tracks entering volume
  else
    ok = ((dir.x() * normalX) + (dir.y() * normalY) < 0.); // DistToOut -- require tracks leaving volume

  // if( /*Backend::early_returns &&*/ vecCore::MaskEmpty(ok) ) return;

  Float_t dirDotXY = (dir.y() * alongX) - (dir.x() * alongY);
  vecCore::MaskedAssign(dist, dirDotXY != 0, ((alongY * pos.x()) - (alongX * pos.y())) / NonZero(dirDotXY));
  // std::cout<<"Dist From PHI-Plane-Trajectory-Intersection  : "<<dist<<std::endl;
  ok &= dist > -kHalfConeTolerance;
  // if( /*Backend::early_returns &&*/ vecCore::MaskEmpty(ok) ) return;

  if (insectorCheck) {
    Float_t hitx          = pos.x() + dist * dir.x();
    Float_t hity          = pos.y() + dist * dir.y();
    Float_t hitz          = pos.z() + dist * dir.z();
    Float_t r2            = (hitx * hitx) + (hity * hity);
    Float_t innerRadIrTol = GetRadiusOfConeAtPoint<Backend, true>(cone, hitz) + kTolerance;
    Float_t outerRadIrTol = GetRadiusOfConeAtPoint<Backend, false>(cone, hitz) - kTolerance;

    ok &=
        Abs(hitz) <= cone.GetTolIz() && (r2 >= innerRadIrTol * innerRadIrTol) && (r2 <= outerRadIrTol * outerRadIrTol);

    // GL: tested with this if(PosDirPhiVec) around if(insector), so if(insector){} requires PosDirPhiVec==true to run
    //  --> shapeTester still finishes OK (no mismatches) (some cycles saved...)
    if (PositiveDirectionOfPhiVector) {
      ok = ok && ((hitx * alongX) + (hity * alongY)) > 0.;
    }
  } else {
    if (PositiveDirectionOfPhiVector) {
      Float_t hitx = pos.x() + dist * dir.x();
      Float_t hity = pos.y() + dist * dir.y();
      ok           = ok && ((hitx * alongX) + (hity * alongY)) >= 0.;
    }
  }
}

} // End of NS ConeUtilities

class PlacedCone;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
struct ConeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t   = PlacedCone;
  using UnplacedShape_t = UnplacedCone;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() { printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT); }

  template <typename Stream>
  static void PrintType(Stream &s)
  {
    s << "SpecializedCone<" << transCodeT << "," << rotCodeT << ",ConeTypes::" << ConeType::toString() << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &s)
  {
    s << "ConeImplemenation<" << transCodeT << "," << rotCodeT << ",ConeTypes::" << ConeType::toString() << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &s)
  {
    s << "UnplacedCone";
  }

  //------- New Vectorized Definitions -------------------
  template <typename Backend, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedCone const &cone,
                                                Vector3D<typename Backend::precision_v> const &point,
                                                typename Backend::bool_v &completelyinside,
                                                typename Backend::bool_v &completelyoutside)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // very fast check on z-height
    Float_t absz      = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<ForInside>(cone.GetDz(), kConeTolerance);
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<ForInside>(cone.GetDz(), kConeTolerance);
    }
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }

    // check on RMAX
    Float_t r2 = point.x() * point.x() + point.y() * point.y();
    // calculate cone radius at the z-height of position
    Float_t rmax  = cone.GetOuterSlope() * point.z() + cone.GetOuterOffset();
    Float_t rmax2 = rmax * rmax;

    completelyoutside |= r2 > MakePlusTolerantSquare<ForInside>(rmax, rmax2, kConeTolerance);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>(rmax, rmax2, kConeTolerance);
    }
    // if (Backend::early_returns) {
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }
    //}

    // check on RMIN
    if (ConeTypes::checkRminTreatment<ConeType>(cone)) {
      Float_t rmin  = cone.GetInnerSlope() * point.z() + cone.GetInnerOffset();
      Float_t rmin2 = rmin * rmin;

      completelyoutside |= r2 <= MakeMinusTolerantSquare<ForInside>(rmin, rmin2, kConeTolerance);
      if (ForInside) {
        completelyinside &= r2 > MakePlusTolerantSquare<ForInside>(rmin, rmin2, kConeTolerance);
      }
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    if (ConeTypes::checkPhiTreatment<ConeType>(cone)) {
      Bool_t completelyoutsidephi;
      Bool_t completelyinsidephi;
      cone.GetWedge().GenericKernelForContainsAndInside<Backend, ForInside>(point, completelyinsidephi,
                                                                            completelyoutsidephi);
      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::bool_v &contains)
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    GenericKernelForContainsAndInside<Backend, false>(cone, point, unused, outside);
    contains = !outside;
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedCone const &cone, Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint, typename Backend::bool_v &contains)
  {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    return UnplacedContains<Backend>(cone, localPoint, contains);
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedCone const &cone, Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point, typename Backend::int_v &inside)
  {

    Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    /*
        typedef typename Backend::bool_v Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(cone,
          localPoint, completelyinside, completelyoutside);
        inside = EInside::kSurface;
        //vecCore::MaskedAssign(inside,completelyoutside, EInside::kOutside);
        //vecCore::MaskedAssign(inside,completelyinside,  EInside::kInside);
        vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
        vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
        */

    const typename Backend::precision_v in(EInside::kInside);
    const typename Backend::precision_v out(EInside::kOutside);
    typename Backend::bool_v inmask(false), outmask(false);
    typename Backend::precision_v result(EInside::kSurface);

    GenericKernelForContainsAndInside<Backend, true>(cone, localPoint, inmask, outmask);

    vecCore::MaskedAssign(result, inmask, in);
    vecCore::MaskedAssign(result, outmask, out);

    // Manual conversion from double to int here is necessary because int_v and
    // precision_v have different number of elements in SIMD vector, so bool_v
    // (mask for precision_v) cannot be cast to mask for inside, which is a
    // different type and does not exist in the current backend system
    for (size_t i = 0; i < vecCore::VectorSize(result); i++)
      vecCore::Set(inside, i, vecCore::Get(result, i));
  }

  template <class Backend, bool ForInnerSurface>
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<typename Backend::precision_v> GetNormal(UnplacedCone const &cone,
                                                           Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rho = point.Perp();
    Vector3D<Float_t> norm(0., 0., 0.);

    if (ForInnerSurface) {
      // Handling inner conical surface
      Precision rmin1 = cone.GetRmin1();
      Precision rmin2 = cone.GetRmin2();
      if ((rmin1 == rmin2) && (rmin1 != 0.)) {
        // cone act like tube
        norm.Set(-point.x(), -point.y(), 0.);
      } else {
        Precision secRMin = cone.GetSecRmin();
        norm.Set(-point.x(), -point.y(), cone.GetZNormInner() * (rho * secRMin));
      }
    } else {
      Precision rmax1 = cone.GetRmax1();
      Precision rmax2 = cone.GetRmax2();
      if ((rmax1 == rmax2) && (rmax1 != 0.)) {
        // cone act like tube
        norm.Set(point.x(), point.y(), 0.);
      } else {
        Precision secRMax = cone.GetSecRmax();
        norm.Set(point.x(), point.y(), cone.GetZNormOuter() * (rho * secRMax));
      }
    }
    return norm;
  }

  template <class Backend, bool ForTopRing>
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<typename Backend::precision_v> GetNormalAtRing(UnplacedCone const &cone,
                                                                 Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rho = point.Perp();
    Vector3D<Float_t> norm(0., 0., 0.);

    // Logic to calculate normal on Inner conical surface
    // Precision tanRMin = cone.GetTanRmin();
    Precision secRMin = cone.GetSecRmin();
    norm.Set(-point.x(), -point.y(), cone.GetZNormInner() * (rho * secRMin));

    constexpr Precision invSqrt2 = 0.70710678118;
    if (ForTopRing) {
      norm.z() = (norm.z() + 1.) * invSqrt2;
    } else {
      norm.z() = (norm.z() - 1.) * invSqrt2;
    }
    // return norm;

    return norm;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToInKernel(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                                 Vector3D<typename Backend::precision_v> const &dir,
                                 typename Backend::precision_v const & /*stepMax*/,
                                 typename Backend::precision_v &distance)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Bool_t done(false);

    //=== First, for points outside and moving away --> return infinity
    distance = kInfLength;

    // outside of Z range and going away?
    Float_t distz          = Abs(point.z()) - cone.GetDz(); // avoid a division for now
    Bool_t outZAndGoingOut = (distz > kHalfConeTolerance && (point.z() * dir.z()) >= 0) ||
                             (Abs(distz) < kHalfConeTolerance && (point.z() * dir.z()) > 0);
    done |= outZAndGoingOut;
    if (vecCore::MaskFull(done)) return;

    // outside of outer cone and going away?
    Float_t outerRadIrTol  = GetRadiusOfConeAtPoint<Backend, false>(cone, point.z()) - kConeTolerance;
    Float_t outerRadIrTol2 = outerRadIrTol * outerRadIrTol;
    Float_t rsq            = point.Perp2(); // point.x()*point.x() + point.y()*point.y();

    done |= rsq > outerRadIrTol2 && (dir.Dot(GetNormal<Backend, false>(cone, point)) >= 0.);
    if (vecCore::MaskFull(done)) return;

    //=== Next, check all dimensions of the cone, whether points are inside --> return -1

    vecCore::MaskedAssign(distance, !done, Float_t(-1.0));

    // For points inside z-range, return -1
    Bool_t inside = distz < -kHalfConeTolerance;

    inside &= rsq < outerRadIrTol2;

    if (checkRminTreatment<ConeType>(cone)) {
      Float_t innerRadIrTol  = GetRadiusOfConeAtPoint<Backend, true>(cone, point.z()) + kConeTolerance;
      Float_t innerRadIrTol2 = innerRadIrTol * innerRadIrTol;
      inside &= rsq > innerRadIrTol2;
    }
    if (checkPhiTreatment<ConeType>(cone)) { // && !vecCore::MaskEmpty(inside)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false, false>(cone, point.x(), point.y(), insector);
      inside &= insector;
    }
    done |= inside;
    if (vecCore::MaskFull(done)) return;

    //=== Next step: check if z-plane is the right entry point (both r,phi should be valid at z-plane crossing)
    vecCore::MaskedAssign(distance, !done, Float_t(kInfLength));

    distz /= NonZero(Abs(dir.z()));

    Bool_t isOnZPlaneAndMovingInside(false);

    Bool_t isGoingUp          = dir.z() > 0.;
    isOnZPlaneAndMovingInside = !inside && ((isGoingUp && point.z() < 0. && Abs(distz) < kHalfTolerance) ||
                                            (!isGoingUp && point.z() > 0. && Abs(distz) < kHalfTolerance));
    vecCore::MaskedAssign(distz, !done && isOnZPlaneAndMovingInside, Float_t(0.));

#ifdef EDGE_POINTS
    Bool_t newCond = (IsOnRing<Backend, false, true>(cone, point)) || (IsOnRing<Backend, true, true>(cone, point)) ||
                     (IsOnRing<Backend, false, false>(cone, point)) || (IsOnRing<Backend, true, false>(cone, point));

    vecCore::MaskedAssign(distz, newCond, kInfLength);
#endif

    Float_t hitx = point.x() + distz * dir.x();
    Float_t hity = point.y() + distz * dir.y();

    Float_t r2 = (hitx * hitx) + (hity * hity);

    Precision innerZTol         = cone.GetTolIz();
    Bool_t isHittingTopPlane    = (point.z() >= innerZTol) && (r2 <= cone.GetSqRmax2Tol());  // GetSqRmax2());
    Bool_t isHittingBottomPlane = (point.z() <= -innerZTol) && (r2 <= cone.GetSqRmax1Tol()); // GetSqRmax1());
    Bool_t okz                  = (isHittingTopPlane || isHittingBottomPlane);

    if (checkRminTreatment<ConeType>(cone)) {
      isHittingTopPlane &= (r2 >= cone.GetSqRmin2Tol());    // GetSqRmin2());
      isHittingBottomPlane &= (r2 >= cone.GetSqRmin1Tol()); // GetSqRmin1());
      okz &= ((isHittingTopPlane || isHittingBottomPlane));
    }

    if (checkPhiTreatment<ConeType>(cone)) { // && !vecCore::MaskEmpty(okz)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false>(cone, hitx, hity, insector);
      okz &= insector;
    }
    vecCore::MaskedAssign(distance, !done && okz, distz);
    done |= okz;
    if (vecCore::MaskFull(done)) return;

    Float_t dist_rOuter(kInfLength);
    Bool_t ok_outerCone =
        DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, true, false>(cone, point, dir, dist_rOuter);
    ok_outerCone &= dist_rOuter < distance;
    vecCore::MaskedAssign(distance, !done && ok_outerCone, dist_rOuter);
    done |= ok_outerCone;
    if (vecCore::MaskFull(done)) return;

    Float_t dist_rInner(kInfLength);
    if (checkRminTreatment<ConeType>(cone)) {

      Bool_t ok_innerCone =
          DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, true, true>(cone, point, dir, dist_rInner);
      ok_innerCone &= dist_rInner < distance;
      vecCore::MaskedAssign(distance, !done && ok_innerCone, dist_rInner);
    }

    if (checkPhiTreatment<ConeType>(cone)) {

      Wedge const &w = cone.GetWedge();
      // Vector3D<Float_t> normal1 = w.GetNormal1();
      // Vector3D<Float_t> normal2 = w.GetNormal2();

      Float_t dist_phi;
      Bool_t ok_phi;
      PhiPlaneTrajectoryIntersection<Backend, ConeType, SectorType<ConeType>::value != kPi, true>(
          cone.GetAlongPhi1X(), cone.GetAlongPhi1Y(), w.GetNormal1().x(), w.GetNormal1().y(), cone, point, dir,
          dist_phi, ok_phi);
      ok_phi &= dist_phi < distance;
      vecCore::MaskedAssign(distance, !done && ok_phi, dist_phi);
      done |= ok_phi;

      if (SectorType<ConeType>::value != kPi) {

        PhiPlaneTrajectoryIntersection<Backend, ConeType, true, true>(cone.GetAlongPhi2X(), cone.GetAlongPhi2Y(),
                                                                      w.GetNormal2().x(), w.GetNormal2().y(), cone,
                                                                      point, dir, dist_phi, ok_phi);

        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      }
    }

  } // end of DistanceToInKernel()

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedCone const &cone, Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           Vector3D<typename Backend::precision_v> const &direction,
                           typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    DistanceToInKernel<Backend>(cone, transformation.Transform<transCodeT, rotCodeT>(point),
                                transformation.TransformDirection<rotCodeT>(direction), stepMax, distance);
  }

  template <class Backend, bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsMovingOutsideConicalSurface(
      UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction)
  {

    return IsOnConicalSurface<Backend, ForInnerSurface>(cone, point) &&
           (direction.Dot(GetNormal<Backend, ForInnerSurface>(cone, point)) >= 0.);
  }

  template <class Backend, bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsMovingInsideConicalSurface(UnplacedCone const &cone,
                                                               Vector3D<typename Backend::precision_v> const &point,
                                                               Vector3D<typename Backend::precision_v> const &direction)
  {

    return IsOnConicalSurface<Backend, ForInnerSurface>(cone, point) &&
           (direction.Dot(GetNormal<Backend, ForInnerSurface>(cone, point)) <= 0.);
  }

  template <class Backend, bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnConicalSurface(UnplacedCone const &cone,
                                                     Vector3D<typename Backend::precision_v> const &point)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    typedef typename Backend::precision_v Float_t;

    Float_t rho      = point.Perp2();
    Float_t coneRad  = GetRadiusOfConeAtPoint<Backend, ForInnerSurface>(cone, point.z());
    Float_t coneRad2 = coneRad * coneRad;
    return (rho >= (coneRad2 - kConeTolerance * coneRad)) && (rho <= (coneRad2 + kConeTolerance * coneRad)) &&
           (Abs(point.z()) < (cone.GetDz() + kTolerance));
  }

  template <class Backend, bool ForTopPlane>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnZPlaneAndMovingInside(UnplacedCone const &cone,
                                                            Vector3D<typename Backend::precision_v> const &point,
                                                            Vector3D<typename Backend::precision_v> const &direction)
  {

    typedef typename Backend::precision_v Float_t;

    Float_t rho = point.Perp2();
    Float_t fDz = cone.GetDz();

    if (ForTopPlane) {
      return (rho > (cone.GetSqRmin2() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax2() + kHalfConeTolerance)) &&
             (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance)) &&
             (direction.z() < 0.);
    } else {
      return (rho > (cone.GetSqRmin1() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax1() + kHalfConeTolerance)) &&
             (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance)) &&
             (direction.z() > 0.);
    }
  }

  template <class Backend, bool ForTopPlane>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnZPlaneAndMovingOutside(UnplacedCone const &cone,
                                                             Vector3D<typename Backend::precision_v> const &point,
                                                             Vector3D<typename Backend::precision_v> const &direction)
  {

    typedef typename Backend::precision_v Float_t;

    Float_t rho = point.Perp2();
    Float_t fDz = cone.GetDz();

    if (ForTopPlane) {
      return (rho > (cone.GetSqRmin2() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax2() + kHalfConeTolerance)) &&
             (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance)) &&
             (direction.z() > 0.);
    } else {
      return (rho > (cone.GetSqRmin1() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax1() + kHalfConeTolerance)) &&
             (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance)) &&
             (direction.z() < 0.);
    }
  }

  // This function will be useful to detect the point on the circular edge
  template <class Backend, bool ForTopRing, bool ForInnerRing>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnRing(UnplacedCone const &cone,
                                           Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    Float_t rho     = point.Perp2();
    Float_t fDz     = cone.GetDz();
    Precision rmin1 = cone.GetRmin1();
    Precision rmin2 = cone.GetRmin2();
    Precision rmax1 = cone.GetRmax1();
    Precision rmax2 = cone.GetRmax2();

    Vector3D<Float_t> norm(0., 0., 0.);
    Bool_t isOnRing(false);
    Bool_t isOnRingAndMovinInside(false);
    if (ForTopRing) {
      if (ForInnerRing) {
        isOnRing = (rho > MakeMinusTolerantSquare<true>(rmin2, rmin2 * rmin2, kConeTolerance)) &&
                   (rho < MakePlusTolerantSquare<true>(rmin2, rmin2 * rmin2, kConeTolerance)) &&
                   (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance));
      } else {
        isOnRing = (rho > MakeMinusTolerantSquare<true>(rmax2, rmax2 * rmax2, kConeTolerance)) &&
                   (rho < MakePlusTolerantSquare<true>(rmax2, rmax2 * rmax2, kConeTolerance)) &&
                   (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance));
      }

    } else {

      if (ForInnerRing) {
        isOnRing = (rho > MakeMinusTolerantSquare<true>(rmin1, rmin1 * rmin1, kConeTolerance)) &&
                   (rho < MakePlusTolerantSquare<true>(rmin1, rmin1 * rmin1, kConeTolerance)) &&
                   (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance));
      } else {
        isOnRing = (rho > MakeMinusTolerantSquare<true>(rmax1, rmax1 * rmax1, kConeTolerance)) &&
                   (rho < MakePlusTolerantSquare<true>(rmax1, rmax1 * rmax1, kConeTolerance)) &&
                   (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance));
      }
    }

    return isOnRing;
  }

  // This function is the main Quadratic Solver for the Distance functions.
  template <typename Backend, bool ForDistToIn, bool ForInnerSurface>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v DetectIntersectionAndCalculateDistanceToConicalSurface(
      UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v &distance)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Bool_t ok(false);
    Bool_t done(false);
    if (ForDistToIn) {
      Bool_t isOnSurfaceAndMovingInside =
          IsMovingInsideConicalSurface<Backend, ForInnerSurface>(cone, point, direction);

      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingInside, Float_t(0.));
      if (!checkPhiTreatment<ConeType>(cone)) {
        done |= isOnSurfaceAndMovingInside;
        if (vecCore::MaskFull(done)) return done;
      }
    } else {
      Bool_t isOnSurfaceAndMovingOutside =
          IsMovingOutsideConicalSurface<Backend, ForInnerSurface>(cone, point, direction);

      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingOutside, Float_t(0.));
      if (!checkPhiTreatment<ConeType>(cone)) {
        done |= isOnSurfaceAndMovingOutside;
        if (vecCore::MaskFull(done)) return done;
      }
    }

    Float_t pDotV2D = point.x() * direction.x() + point.y() * direction.y();

    if (ForInnerSurface) {
      Precision fDz = cone.GetDz();
      Precision t   = cone.GetTInner();
      Float_t newPz(0.);
      if (cone.GetRmin2() > cone.GetRmin1())
        newPz = (point.z() + fDz + cone.GetInnerConeApex()) * t;
      else
        newPz = (point.z() - fDz - cone.GetInnerConeApex()) * t;

      Float_t dirT = direction.z() * t;
      Float_t a    = (direction.x() * direction.x()) + (direction.y() * direction.y()) - dirT * dirT;

      Float_t b     = pDotV2D - (newPz * dirT);
      Float_t c     = point.Perp2() - (newPz * newPz);
      Float_t d2    = (b * b) - (a * c);
      Float_t delta = Sqrt(vecCore::math::Abs(d2));

      // this block is required only when inner conical surface become surface of simple tube
      Precision rmin1 = cone.GetRmin1();
      Precision rmin2 = cone.GetRmin2();
      if (rmin1 == rmin2) {
        b     = pDotV2D;
        a     = direction.Perp2();
        c     = point.Perp2() - rmin2 * rmin2;
        d2    = b * b - a * c;
        delta = Sqrt(vecCore::math::Abs(d2));
      }
      if (ForDistToIn) {
        vecCore::MaskedAssign(distance, d2 >= 0. && (b >= 0.), (c / NonZero(-b - delta)));
        vecCore::MaskedAssign(distance, d2 >= 0. && (b < 0.), (-b + delta) / NonZero(a));
      } else {
        vecCore::MaskedAssign(distance, d2 >= 0. && (b > 0.), (-b - delta) / NonZero(a));
        vecCore::MaskedAssign(distance, d2 >= 0. && (b < 0.), (c / NonZero(-b + delta)));
      }

      Float_t newZ = point.z() + (direction.z() * distance);
      ok           = (Abs(newZ) < fDz /*cone.GetDz()*/);

    } else {
      Precision fDz = cone.GetDz();
      Precision t   = cone.GetTOuter();
      Float_t newPz(0.);
      if (cone.GetRmax2() > cone.GetRmax1())
        newPz = (point.z() + fDz + cone.GetOuterConeApex()) * t;
      else
        newPz       = (point.z() - fDz - cone.GetOuterConeApex()) * t;
      Float_t dirT  = direction.z() * t;
      Float_t a     = direction.x() * direction.x() + direction.y() * direction.y() - dirT * dirT;
      Float_t b     = pDotV2D - (newPz * dirT);
      Float_t c     = point.Perp2() - (newPz * newPz);
      Float_t d2    = (b * b) - (a * c);
      Float_t delta = Sqrt(vecCore::math::Abs(d2));

      // this block is required only when outer conical surface become surface of simple tube
      Precision rmax1 = cone.GetRmax1();
      Precision rmax2 = cone.GetRmax2();
      if (rmax1 == rmax2) {
        b     = pDotV2D;
        a     = direction.Perp2();
        c     = point.Perp2() - rmax2 * rmax2;
        d2    = b * b - a * c;
        delta = Sqrt(vecCore::math::Abs(d2));
      }

      if (ForDistToIn) {
        vecCore::MaskedAssign(distance, d2 >= 0. && (b > 0.), (-b - delta) / NonZero(a));
        vecCore::MaskedAssign(distance, d2 >= 0. && (b < 0.), (c / NonZero(-b + delta)));
      } else {
        vecCore::MaskedAssign(distance, d2 >= 0. && (b < 0.), (-b + delta) / NonZero(a));
        vecCore::MaskedAssign(distance, d2 >= 0. && (b >= 0.), (c / NonZero(-b - delta)));
      }

      Float_t newZ = point.z() + (direction.z() * distance);
      ok           = (Abs(newZ) < fDz + kHalfConeTolerance);
    }
    vecCore::MaskedAssign(distance, distance < 0., Float_t(kInfLength));

    if (checkPhiTreatment<ConeType>(cone)) {
      Float_t hitx(0), hity(0), hitz(0);
      Bool_t insector = Backend::kFalse;
      vecCore::MaskedAssign(hitx, distance < kInfLength, point.x() + distance * direction.x());
      vecCore::MaskedAssign(hity, distance < kInfLength, point.y() + distance * direction.y());
      vecCore::MaskedAssign(hitz, distance < kInfLength, point.z() + distance * direction.z());

      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false, true>(cone, hitx, hity, insector);
      ok &= ((insector) && (distance < kInfLength));
    }
    return ok;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOutKernel(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                                  Vector3D<typename Backend::precision_v> const &direction,
                                  typename Backend::precision_v const & /*stepMax*/,
                                  typename Backend::precision_v &distance)
  {

    distance = kInfLength;
    using namespace ConeUtilities;
    using namespace ConeTypes;

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Bool_t done(false);

    // Using this logic will improve performance of Scalar code
    Float_t distz          = Abs(point.z()) - cone.GetDz();
    Float_t rsq            = point.Perp2();
    Float_t outerRadOrTol  = GetRadiusOfConeAtPoint<Backend, false>(cone, point.z()) + kConeTolerance;
    Float_t outerRadOrTol2 = outerRadOrTol * outerRadOrTol;

    //=== Next, check all dimensions of the cone, whether points are inside --> return -1
    vecCore::MaskedAssign(distance, !done, Float_t(-1.0));

    // For points inside z-range, return -1
    Bool_t outside = distz > kHalfConeTolerance || rsq > outerRadOrTol2;
    done |= outside;
    if (vecCore::MaskFull(done)) return;

    if (checkRminTreatment<ConeType>(cone) && !vecCore::MaskFull(outside)) {
      Float_t innerRadOrTol  = GetRadiusOfConeAtPoint<Backend, true>(cone, point.z()) - kConeTolerance;
      Float_t innerRadOrTol2 = innerRadOrTol * innerRadOrTol;
      outside |= rsq < innerRadOrTol2;
      done |= outside;
      if (vecCore::MaskFull(done)) return;
    }
    if (checkPhiTreatment<ConeType>(cone) && !vecCore::MaskEmpty(outside)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false, false>(cone, point.x(), point.y(), insector);
      outside |= !insector;
    }
    done |= outside;
    if (vecCore::MaskFull(done)) return;
    Bool_t isGoingUp   = direction.z() > 0.;
    Bool_t isGoingDown = direction.z() < 0.;
    Bool_t isOnZPlaneAndMovingOutside(false);
    isOnZPlaneAndMovingOutside = !outside && ((isGoingUp && point.z() > 0. && Abs(distz) < kHalfTolerance) ||
                                              (!isGoingUp && point.z() < 0. && Abs(distz) < kHalfTolerance));
    vecCore::MaskedAssign(distance, !done && isOnZPlaneAndMovingOutside, Float_t(0.));
    done |= isOnZPlaneAndMovingOutside;
    if (vecCore::MaskFull(done)) return;

    //=== Next step: check if z-plane is the right entry point (both r,phi should be valid at z-plane crossing)
    vecCore::MaskedAssign(distance, !done, Float_t(kInfLength));

    Precision fDz   = cone.GetDz();
    Float_t dirZInv = 1. / NonZero(direction.z());
    vecCore::MaskedAssign(distance, isGoingUp, (fDz - point.z()) * dirZInv);
    vecCore::MaskedAssign(distance, isGoingDown, (-fDz - point.z()) * dirZInv);

    Float_t dist_rOuter(kInfLength);
    Bool_t ok_outerCone = DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, false, false>(
        cone, point, direction, dist_rOuter);

    vecCore::MaskedAssign(distance, !done && ok_outerCone && dist_rOuter < distance, dist_rOuter);

    Float_t dist_rInner(kInfLength);
    if (checkRminTreatment<ConeType>(cone)) {
      Bool_t ok_innerCone = DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, false, true>(
          cone, point, direction, dist_rInner);
      vecCore::MaskedAssign(distance, !done && ok_innerCone && dist_rInner < distance, dist_rInner);
    }

    if (checkPhiTreatment<ConeType>(cone)) {

      Bool_t isOnStartPhi       = IsOnStartPhi<Backend>(cone, point);
      Bool_t isOnEndPhi         = IsOnEndPhi<Backend>(cone, point);
      Vector3D<Float_t> normal1 = cone.GetWedge().GetNormal1();
      Vector3D<Float_t> normal2 = cone.GetWedge().GetNormal2();
      Bool_t cond = (isOnStartPhi && direction.Dot(-normal1) > 0.) || (isOnEndPhi && direction.Dot(-normal2) > 0.);
      vecCore::MaskedAssign(distance, !done && cond, Float_t(0.));
      done |= cond;
      if (vecCore::MaskFull(done)) return;

      Float_t dist_phi;
      Bool_t ok_phi;
      Wedge const &w = cone.GetWedge();
      PhiPlaneTrajectoryIntersection<Backend, ConeType, SectorType<ConeType>::value != kPi, false>(
          cone.GetAlongPhi1X(), cone.GetAlongPhi1Y(), w.GetNormal1().x(), w.GetNormal1().y(), cone, point, direction,
          dist_phi, ok_phi);
      ok_phi &= dist_phi < distance;
      vecCore::MaskedAssign(distance, !done && ok_phi, dist_phi);
      done |= ok_phi;

      if (SectorType<ConeType>::value != kPi) {
        PhiPlaneTrajectoryIntersection<Backend, ConeType, true, false>(cone.GetAlongPhi2X(), cone.GetAlongPhi2Y(),
                                                                       w.GetNormal2().x(), w.GetNormal2().y(), cone,
                                                                       point, direction, dist_phi, ok_phi);
        vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
      }
    }
    vecCore::MaskedAssign(distance, distance < 0. && Abs(distance) < kTolerance, Float_t(0.));
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> point,
                            Vector3D<typename Backend::precision_v> direction,
                            typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    DistanceToOutKernel<Backend>(cone, point, direction, stepMax, distance);
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToInKernel(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::precision_v &safety)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    safety = -kInfLength;
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;

    Bool_t done(false);
    Precision fDz = cone.GetDz();
    Float_t distz = Abs(point.z()) - fDz;

    // Next, check all dimensions of the cone, whether points are inside --> return -1
    vecCore::MaskedAssign(safety, !done, Float_t(-1.0));

    // For points inside z-range, return -1
    Bool_t inside = distz < -kHalfConeTolerance;

    // This logic to check if the point is inside is far better then
    // using GenericKernel and will improve performance.
    Float_t outerRadIrTol  = GetRadiusOfConeAtPoint<Backend, false>(cone, point.z()) - kTolerance;
    Float_t outerRadIrTol2 = outerRadIrTol * outerRadIrTol;
    Float_t rsq            = point.Perp2();
    inside &= rsq < outerRadIrTol2;

    if (checkRminTreatment<ConeType>(cone)) {
      Float_t innerRadIrTol  = GetRadiusOfConeAtPoint<Backend, true>(cone, point.z()) + kTolerance;
      Float_t innerRadIrTol2 = innerRadIrTol * innerRadIrTol;
      inside &= rsq > innerRadIrTol2;
    }
    if (checkPhiTreatment<ConeType>(cone) && !vecCore::MaskEmpty(inside)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false, false>(cone, point.x(), point.y(), insector);
      inside &= insector;
    }
    done |= inside;
    if (vecCore::MaskFull(done)) return;

    // Once it is checked that the point is inside or not, safety can be set to 0.
    // This will serve the case that the point is on the surface. So no need to check
    // that the point is really on surface.
    vecCore::MaskedAssign(safety, !done, Float_t(0.));

    // Now if the point is neither inside nor on surface, then it should be outside
    // and the safety should be set to some finite value, which is done by below logic

    Float_t safeZ                = Abs(point.z()) - fDz;
    Float_t safeDistOuterSurface = -SafeDistanceToConicalSurface<Backend, false>(cone, point);

    Float_t safeDistInnerSurface(-kInfLength);
    if (checkRminTreatment<ConeType>(cone)) {
      safeDistInnerSurface = -SafeDistanceToConicalSurface<Backend, true>(cone, point);
    }

    vecCore::MaskedAssign(safety, !done, Max(safeZ, Max(safeDistOuterSurface, safeDistInnerSurface)));

    if (checkPhiTreatment<ConeType>(cone)) {
      Float_t safetyPhi = cone.GetWedge().SafetyToIn<Backend>(point);
      vecCore::MaskedAssign(safety, !done, Max(safetyPhi, safety));
    }

    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < kTolerance, Float_t(0.));
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedCone const &cone, Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety)
  {

    SafetyToInKernel<Backend>(cone, transformation.Transform<transCodeT, rotCodeT>(point), safety);
  }

  template <typename Backend, bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v SafeDistanceToConicalSurface(
      UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rho = point.Perp();
    if (ForInnerSurface) {
      Float_t pRMin = cone.GetTanRmin() * point.z() + cone.GetRminAv();
      return (rho - pRMin) * cone.GetInvSecRMin();
    } else {
      Float_t pRMax = cone.GetTanRmax() * point.z() + cone.GetRmaxAv();
      return (pRMax - rho) * cone.GetInvSecRMax();
    }
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOutKernel(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                                typename Backend::precision_v &safety)
  {

    using namespace ConeUtilities;
    using namespace ConeTypes;
    safety = kInfLength;
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;

    Bool_t done(false);

    Float_t distz = Abs(point.z()) - cone.GetDz();
    Float_t rsq   = point.Perp2();

    // This logic to check if the point is outside is far better then
    // using GenericKernel and will improve performance.
    Float_t outerRadOrTol  = GetRadiusOfConeAtPoint<Backend, false>(cone, point.z()) + kTolerance;
    Float_t outerRadOrTol2 = outerRadOrTol * outerRadOrTol;

    //=== Next, check all dimensions of the cone, whether points are inside --> return -1
    vecCore::MaskedAssign(safety, !done, Float_t(-1.0));

    // For points outside z-range, return -1
    Bool_t outside = distz > kHalfConeTolerance;

    outside |= rsq > outerRadOrTol2;

    if (checkRminTreatment<ConeType>(cone)) {
      Float_t innerRadOrTol  = GetRadiusOfConeAtPoint<Backend, true>(cone, point.z()) - kTolerance;
      Float_t innerRadOrTol2 = innerRadOrTol * innerRadOrTol;
      outside |= rsq < innerRadOrTol2;
    }
    if (checkPhiTreatment<ConeType>(cone) && !vecCore::MaskEmpty(outside)) {
      Bool_t insector;
      PointInCyclicalSector<Backend, ConeType, UnplacedCone, false, false>(cone, point.x(), point.y(), insector);
      outside |= !insector;
    }
    done |= outside;
    if (vecCore::MaskFull(done)) return;

    // Once it is checked that the point is inside or not, safety can be set to 0.
    // This will serve the case that the point is on the surface. So no need to check
    // that the point is really on surface.
    vecCore::MaskedAssign(safety, !done, Float_t(0.));

    // Now if the point is neither outside nor on surface, then it should be inside
    // and the safety should be set to some finite value, which is done by below logic

    Precision fDz = cone.GetDz();
    Float_t safeZ = fDz - Abs(point.z());

    Float_t safeDistOuterSurface = SafeDistanceToConicalSurface<Backend, false>(cone, point);
    Float_t safeDistInnerSurface(kInfLength);
    if (checkRminTreatment<ConeType>(cone)) {
      safeDistInnerSurface = SafeDistanceToConicalSurface<Backend, true>(cone, point);
    } else {
    }

    vecCore::MaskedAssign(safety, !done, Min(safeZ, Min(safeDistOuterSurface, safeDistInnerSurface)));

    if (checkPhiTreatment<ConeType>(cone)) {
      Float_t safetyPhi = cone.GetWedge().SafetyToOut<Backend>(point);
      vecCore::MaskedAssign(safety, !done, Min(safetyPhi, safety));
    }
    vecCore::MaskedAssign(safety, vecCore::math::Abs(safety) < kTolerance, Float_t(0.));
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety)
  {

    SafetyToOutKernel<Backend>(cone, point, safety);
  }

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void Normal(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
  {

    NormalKernel<Backend>(cone, point, normal, valid);
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnTopZPlane(UnplacedCone const &cone,
                                                Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t fDz = cone.GetDz();
    Float_t rho = point.Perp2();

    return (rho > (cone.GetSqRmin2() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax2() + kHalfConeTolerance)) &&
           (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance));
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnBottomZPlane(UnplacedCone const &cone,
                                                   Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t fDz = cone.GetDz();
    Float_t rho = point.Perp2();

    return (rho > (cone.GetSqRmin1() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax1() + kHalfConeTolerance)) &&
           (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance));
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnInnerConicalSurface(UnplacedCone const &cone,
                                                          Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rmin  = cone.GetInnerSlope() * point.z() + cone.GetInnerOffset();
    Float_t rmin2 = rmin * rmin;
    Float_t rho   = point.Perp2();
    return (rho > MakeMinusTolerantSquare<true>(rmin, rmin2)) && (rho < MakePlusTolerantSquare<true>(rmin, rmin2));
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnOuterConicalSurface(UnplacedCone const &cone,
                                                          Vector3D<typename Backend::precision_v> const &point)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rmax  = cone.GetOuterSlope() * point.z() + cone.GetOuterOffset();
    Float_t rmax2 = rmax * rmax;
    Float_t rho   = point.Perp2();
    return (rho > MakeMinusTolerantSquare<true>(rmax, rmax2)) && (rho < MakePlusTolerantSquare<true>(rmax, rmax2));
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnStartPhi(UnplacedCone const &cone,
                                               Vector3D<typename Backend::precision_v> const &point)
  {

    return cone.GetWedge().IsOnSurfaceGeneric<Backend>(cone.GetWedge().GetAlong1(), cone.GetWedge().GetNormal1(),
                                                       point);
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v IsOnEndPhi(UnplacedCone const &cone,
                                             Vector3D<typename Backend::precision_v> const &point)
  {

    return cone.GetWedge().IsOnSurfaceGeneric<Backend>(cone.GetWedge().GetAlong2(), cone.GetWedge().GetNormal2(),
                                                       point);
  }

  // This function will be used to calculate normal to the closest surface when the point is
  // not on the surface
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<typename Backend::precision_v> ApproxSurfaceNormalKernel(
      UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &p)
  {

    typedef typename Backend::precision_v Float_t;
    Vector3D<Float_t> norm(0., 0., 0.);
    Float_t distZ, distRMin, distRMax;
    Float_t distSphi(kInfLength), distEPhi(kInfLength);
    distZ       = Abs(p.z()) - cone.GetDz();
    Float_t rho = p.Perp();

    //--------------------------------------
    /*  pRMin = rho - p.z() * fTanRMin;
    widRMin = fRmin2 - fDz * fTanRMin;
    distRMin = (pRMin - widRMin) / fSecRMin;

    pRMax = rho - p.z() * fTanRMax;
    widRMax = fRmax2 - fDz * fTanRMax;
    distRMax = (pRMax - widRMax) / fSecRMax;*/
    //------------------------------------------------

    Float_t pRMin     = rho - p.z() * cone.GetTanRmin();                    // fTanRMin;
    Precision widRMin = cone.GetRmin2() - cone.GetDz() * cone.GetTanRmin(); // fTanRMin;
    distRMin          = (pRMin - widRMin) * cone.GetInvSecRMin();           // fInvSecRMin;

    Float_t pRMax     = rho - p.z() * cone.GetTanRmax();                    // fTanRMax;
    Precision widRMax = cone.GetRmax2() - cone.GetDz() * cone.GetTanRmax(); // fTanRMax;
    distRMax          = (pRMax - widRMax) * cone.GetInvSecRMax();           // fInvSecRMax;

    distZ           = Abs(distZ);
    distRMin        = Abs(distRMin);
    distRMax        = Abs(distRMax);
    Float_t distMin = Min(distRMin, distRMax);

    Float_t distPhi1(kInfLength);
    Float_t distPhi2(kInfLength);
    // Changing signs of normal component to negative, because Wedge gave inward normal
    if (ConeTypes::checkPhiTreatment<ConeType>(cone)) {
      distPhi1 = -p.x() * cone.GetWedge().GetNormal1().x() - p.y() * cone.GetWedge().GetNormal1().y();
      distPhi2 = -p.x() * cone.GetWedge().GetNormal2().x() - p.y() * cone.GetWedge().GetNormal2().y();
      vecCore::MaskedAssign(distPhi1, distPhi1 < 0., kInfLength);
      vecCore::MaskedAssign(distPhi2, distPhi2 < 0., kInfLength);

      distMin = Min(distZ, Min(distMin, Min(distPhi1, distPhi2)));
    }

    // Chosing the correct surface
    vecCore::MaskedAssign(norm.z(), (distMin == distZ) && (p.z() > 0.), 1.);
    vecCore::MaskedAssign(norm.z(), (distMin == distZ) && (p.z() < 0.), -1.);

    vecCore::MaskedAssign(norm, (distMin == distRMin), GetNormal<Backend, true>(cone, p));
    vecCore::MaskedAssign(norm, (distMin == distRMax), GetNormal<Backend, false>(cone, p));

    Vector3D<Float_t> normal1 = cone.GetWedge().GetNormal1();
    Vector3D<Float_t> normal2 = cone.GetWedge().GetNormal2();
    vecCore::MaskedAssign(norm, distMin == distPhi1, -normal1);
    vecCore::MaskedAssign(norm, distMin == distPhi2, -normal2);

    norm.Normalize();
    return norm;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void NormalKernel(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &p,
                           Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
  {

    // normal.Set(NonZero(0.), NonZero(0.), NonZero(0.));
    normal.Set(0., 0., 0.);
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    Bool_t compleltelyinside(false), completelyoutside(false);

    GenericKernelForContainsAndInside<Backend, true>(cone, p, compleltelyinside, completelyoutside);
    Bool_t isOnSurface = !compleltelyinside && !completelyoutside;

    Float_t noSurfaces(0.);
    Bool_t isOnTopZPlane           = IsOnTopZPlane<Backend>(cone, p);
    Bool_t isOnBottomZPlane        = IsOnBottomZPlane<Backend>(cone, p);
    Bool_t isOnInnerConicalSurface = IsOnInnerConicalSurface<Backend>(cone, p);
    Bool_t isOnOuterConicalSurface = IsOnOuterConicalSurface<Backend>(cone, p);
    Bool_t isOnStartPhi            = IsOnStartPhi<Backend>(cone, p);
    Bool_t isOnEndPhi              = IsOnEndPhi<Backend>(cone, p);

    Vector3D<Float_t> innerConicalNormal = GetNormal<Backend, true>(cone, p);
    Vector3D<Float_t> outerConicalNormal = GetNormal<Backend, false>(cone, p);

    Vector3D<Float_t> startPhiNormal = cone.GetWedge().GetNormal1();
    Vector3D<Float_t> endPhiNormal   = cone.GetWedge().GetNormal2();

    Vector3D<Float_t> approxNorm = ApproxSurfaceNormalKernel<Backend>(cone, p);
    // vecCore::MaskedAssign(normal,!isOnSurface, ApproxSurfaceNormalKernel<Backend>(cone, p));
    vecCore::MaskedAssign(normal.x(), !isOnSurface, approxNorm.x());
    vecCore::MaskedAssign(normal.y(), !isOnSurface, approxNorm.y());
    vecCore::MaskedAssign(normal.z(), !isOnSurface, approxNorm.z());

    // Counting number of surfaces
    vecCore::MaskedAssign(noSurfaces, isOnSurface && (isOnTopZPlane || isOnBottomZPlane), (noSurfaces + 1));
    vecCore::MaskedAssign(noSurfaces, isOnSurface && (isOnInnerConicalSurface || isOnOuterConicalSurface),
                          (noSurfaces + 1));
    vecCore::MaskedAssign(noSurfaces, isOnSurface && (isOnStartPhi || isOnEndPhi), (noSurfaces + 1));

    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnTopZPlane, (normal.z() + 1.));
    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnBottomZPlane, (normal.z() - 1.));
    normal.Normalize();

    innerConicalNormal.Normalize();
    vecCore::MaskedAssign(normal.x(), isOnSurface && isOnInnerConicalSurface, normal.x() + innerConicalNormal.x());
    vecCore::MaskedAssign(normal.y(), isOnSurface && isOnInnerConicalSurface, normal.y() + innerConicalNormal.y());
    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnInnerConicalSurface, normal.z() + innerConicalNormal.z());

    outerConicalNormal.Normalize();
    vecCore::MaskedAssign(normal.x(), isOnSurface && isOnOuterConicalSurface, normal.x() + outerConicalNormal.x());
    vecCore::MaskedAssign(normal.y(), isOnSurface && isOnOuterConicalSurface, normal.y() + outerConicalNormal.y());
    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnOuterConicalSurface, normal.z() + outerConicalNormal.z());

    // Handling start phi
    vecCore::MaskedAssign(normal.x(), isOnSurface && isOnStartPhi, normal.x() - startPhiNormal.x());
    vecCore::MaskedAssign(normal.y(), isOnSurface && isOnStartPhi, normal.y() - startPhiNormal.y());
    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnStartPhi, normal.z() - startPhiNormal.z());

    // Handling End phi
    vecCore::MaskedAssign(normal.x(), isOnSurface && isOnEndPhi, normal.x() - endPhiNormal.x());
    vecCore::MaskedAssign(normal.y(), isOnSurface && isOnEndPhi, normal.y() - endPhiNormal.y());
    vecCore::MaskedAssign(normal.z(), isOnSurface && isOnEndPhi, normal.z() - endPhiNormal.z());

    normal.Normalize();
    valid = (noSurfaces > 0.);
  }

//--------------------------------------
// Previous Definitions

#if (0)
  /////GenericKernel Contains/Inside implementation
  template <typename Backend, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedCone const &cone,
                                                Vector3D<typename Backend::precision_v> const &point,
                                                typename Backend::bool_v &completelyinside,
                                                typename Backend::bool_v &completelyoutside)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // very fast check on z-height
    Float_t absz      = Abs(point[2]);
    completelyoutside = absz > MakePlusTolerant<ForInside>(cone.GetDz());
    if (ForInside) {
      completelyinside = absz < MakeMinusTolerant<ForInside>(cone.GetDz());
    }
    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    // check on RMAX
    Float_t r2 = point.x() * point.x() + point.y() * point.y();
    // calculate cone radius at the z-height of position
    Float_t rmax  = cone.GetOuterSlope() * point.z() + cone.GetOuterOffset();
    Float_t rmax2 = rmax * rmax;

    completelyoutside |= r2 > MakePlusTolerantSquare<ForInside>(rmax, rmax2);
    if (ForInside) {
      completelyinside &= r2 < MakeMinusTolerantSquare<ForInside>(rmax, rmax2);
    }
    if (vecCore::EarlyReturnAllowed()) {
      if (vecCore::MaskFull(completelyoutside)) {
        return;
      }
    }

    // check on RMIN
    if (ConeTypes::checkRminTreatment<ConeType>(cone)) {
      Float_t rmin  = cone.GetInnerSlope() * point.z() + cone.GetInnerOffset();
      Float_t rmin2 = rmin * rmin;

      completelyoutside |= r2 <= MakeMinusTolerantSquare<ForInside>(rmin, rmin2);
      if (ForInside) {
        completelyinside &= r2 > MakePlusTolerantSquare<ForInside>(rmin, rmin2);
      }
      if (vecCore::EarlyReturnAllowed()) {
        if (vecCore::MaskFull(completelyoutside)) {
          return;
        }
      }
    }

    if (ConeTypes::checkPhiTreatment<ConeType>(cone)) {
      /* phi */
      //        if(( cone.dphi() < kTwoPi ) ){
      Bool_t completelyoutsidephi;
      Bool_t completelyinsidephi;
      cone.GetWedge().GenericKernelForContainsAndInside<Backend, ForInside>(point, completelyinsidephi,
                                                                            completelyoutsidephi);

      completelyoutside |= completelyoutsidephi;
      if (ForInside) completelyinside &= completelyinsidephi;
    }
    // }
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::bool_v &contains)
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    GenericKernelForContainsAndInside<Backend, false>(unplaced, point, unused, outside);
    contains = !outside;
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedCone const &unplaced, Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint, typename Backend::bool_v &contains)
  {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    return UnplacedContains<Backend>(unplaced, localPoint, contains);
  }

#if 0 // removed, as it was producing warnings in clang-3.6 -- passes 'make test' at 100%
  // TODO: do we need both interfaces?
  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedCone const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &localPoint,
                     typename Backend::int_v &inside) {
 //   localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
 //   UnplacedInside<Backend>(unplaced, localPoint, inside);
  }
#endif

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedCone const &unplaced, Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point, typename Backend::int_v &inside)
  {
    // typename Backend::bool_v contains;
    Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);

    typedef typename Backend::bool_v Bool_t;
    Bool_t completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Backend, true>(unplaced, localPoint, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, completelyoutside, EInside::kOutside);
    vecCore::MaskedAssign(inside, completelyinside, EInside::kInside);
  }

  VECGEOM_CLASS_GLOBAL double kHalfCarTolerance = VECGEOM_NAMESPACE::kTolerance * 0.5;
  // static constexpr double kHalfCarTolerance = VECGEOM_NAMESPACE::kHalfTolerance;
  VECGEOM_CLASS_GLOBAL double kHalfRadTolerance = kRadTolerance * 0.5;
  VECGEOM_CLASS_GLOBAL double kHalfAngTolerance = kAngTolerance * 0.5;

  // the fall-back version from USolids
  // to take a short cut towards full functionality
  // this really only makes sense for Scalar and CUDA backend and is copied here until
  //  a generic and fully optimized VecGeom version is available
  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInUSolids(UnplacedCone const &unplaced,
                                                           Transformation3D const &transformation,
                                                           Vector3D<typename Backend::precision_v> const &point,
                                                           Vector3D<typename Backend::precision_v> const &direction,
                                                           typename Backend::precision_v const & /*stepMax*/)
  {

    // first of all: transform points and directions
    Vector3D<typename Backend::precision_v> p = transformation.Transform<transCodeT, rotCodeT>(point);
    Vector3D<typename Backend::precision_v> v = transformation.TransformDirection<rotCodeT>(direction);

    double snxt        = VECGEOM_NAMESPACE::kInfLength;
    const double dRmax = 100 * Max(unplaced.GetRmax1(), unplaced.GetRmax2());
    // const double halfCarTolerance = VECGEOM_NAMESPACE::kTolerance * 0.5;
    // const double halfRadTolerance = kRadTolerance * 0.5;

    double rMaxAv, rMaxOAv; // Data for cones
    double rMinAv, rMinOAv;
    double rout, rin;

    double tolORMin, tolORMin2, tolIRMin, tolIRMin2; // `generous' radii squared
    double tolORMax2, tolIRMax, tolIRMax2;
    double tolODz, tolIDz;

    double Dist, sd, xi, yi, zi, ri = 0., risec, rhoi2, cosPsi; // Intersection point vars

    double t1, t2, t3, b, c, d; // Quadratic solver variables
    double nt1, nt2, nt3;
    double Comp;

    Vector3D<Precision> norm;

    // Cone Precalcs
    rMinAv = (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;

    if (rMinAv > kHalfRadTolerance) {
      rMinOAv = rMinAv - kHalfRadTolerance;
    } else {
      rMinOAv = 0.0;
    }
    rMaxAv  = (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
    rMaxOAv = rMaxAv + kHalfRadTolerance;

    // Intersection with z-surfaces

    tolIDz = unplaced.GetDz() - kHalfCarTolerance;
    tolODz = unplaced.GetDz() + kHalfCarTolerance;

    if (std::fabs(p.z()) >= tolIDz) {
      if (p.z() * v.z() < 0) // at +Z going in -Z or visa versa
      {
        sd = (std::fabs(p.z()) - unplaced.GetDz()) / std::fabs(v.z()); // Z intersect distance

        if (sd < 0.0) {
          sd = 0.0; // negative dist -> zero
        }

        xi    = p.x() + sd * v.x(); // Intersection coords
        yi    = p.y() + sd * v.y();
        rhoi2 = xi * xi + yi * yi;

        // Check validity of intersection
        // Calculate (outer) tolerant radi^2 at intersecion

        if (v.z() > 0) {
          tolORMin  = unplaced.GetRmin1() - kHalfRadTolerance * unplaced.fSecRMin;
          tolIRMin  = unplaced.GetRmin1() + kHalfRadTolerance * unplaced.fSecRMin;
          tolIRMax  = unplaced.GetRmax1() - kHalfRadTolerance * unplaced.fSecRMin;
          tolORMax2 = (unplaced.GetRmax1() + kHalfRadTolerance * unplaced.fSecRMax) *
                      (unplaced.GetRmax1() + kHalfRadTolerance * unplaced.fSecRMax);
        } else {
          tolORMin  = unplaced.GetRmin2() - kHalfRadTolerance * unplaced.fSecRMin;
          tolIRMin  = unplaced.GetRmin2() + kHalfRadTolerance * unplaced.fSecRMin;
          tolIRMax  = unplaced.GetRmax2() - kHalfRadTolerance * unplaced.fSecRMin;
          tolORMax2 = (unplaced.GetRmax2() + kHalfRadTolerance * unplaced.fSecRMax) *
                      (unplaced.GetRmax2() + kHalfRadTolerance * unplaced.fSecRMax);
        }
        if (tolORMin > 0) {
          tolORMin2 = tolORMin * tolORMin;
          tolIRMin2 = tolIRMin * tolIRMin;
        } else {
          tolORMin2 = 0.0;
          tolIRMin2 = 0.0;
        }
        if (tolIRMax > 0) {
          tolIRMax2 = tolIRMax * tolIRMax;
        } else {
          tolIRMax2 = 0.0;
        }

        if ((tolIRMin2 <= rhoi2) && (rhoi2 <= tolIRMax2)) {
          if (!unplaced.IsFullPhi() && rhoi2) {
            // Psi = angle made with central (average) phi of shape

            cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / std::sqrt(rhoi2);

            if (cosPsi >= unplaced.fCosHDPhiIT) {
              return sd;
            }
          } else {
            return sd;
          }
        }
      } else // On/outside extent, and heading away  -> cannot intersect
      {
        return snxt;
      }
    }

    // ----> Can not intersect z surfaces

    // Intersection with outer cone (possible return) and
    //                   inner cone (must also check phi)
    //
    // Intersection point (xi,yi,zi) on line x=p.x()+t*v.x() etc.
    //
    // Intersects with x^2+y^2=(a*z+b)^2
    //
    // where a=unplaced.fTanRMax or unplaced.fTanRMin
    //       b=rMaxAv or rMinAv
    //
    // (vx^2+vy^2-(a*vz)^2)t^2+2t(pxvx+pyvy-a*vz(a*pz+b))+px^2+py^2-(a*pz+b)^2=0;
    //     t1                       t2                      t3
    //
    //  \--------u-------/       \-----------v----------/ \---------w--------/
    //

    t1   = 1.0 - v.z() * v.z();
    t2   = p.x() * v.x() + p.y() * v.y();
    t3   = p.x() * p.x() + p.y() * p.y();
    rin  = unplaced.fTanRMin * p.z() + rMinAv;
    rout = unplaced.fTanRMax * p.z() + rMaxAv;

    // Outer Cone Intersection
    // Must be outside/on outer cone for valid intersection

    nt1 = t1 - (unplaced.fTanRMax * v.z()) * (unplaced.fTanRMax * v.z());
    nt2 = t2 - unplaced.fTanRMax * v.z() * rout;
    nt3 = t3 - rout * rout;

    if (std::fabs(nt1) > kRadTolerance) // Equation quadratic => 2 roots
    {
      b = nt2 / nt1;
      c = nt3 / nt1;
      d = b * b - c;
      if ((nt3 > rout * rout * kRadTolerance * kRadTolerance * unplaced.fSecRMax * unplaced.fSecRMax) || (rout < 0)) {
        // If outside real cone (should be rho-rout>kRadTolerance*0.5
        // NOT rho^2 etc) saves a std::sqrt() at expense of accuracy

        if (d >= 0) {

          if ((rout < 0) && (nt3 <= 0)) {
            // Inside `shadow cone' with -ve radius
            // -> 2nd root could be on real cone

            if (b > 0) {
              sd = c / (-b - std::sqrt(d));
            } else {
              sd = -b + std::sqrt(d);
            }
          } else {
            if ((b <= 0) && (c >= 0)) // both >=0, try smaller root
            {
              sd = c / (-b + std::sqrt(d));
            } else {
              if (c <= 0) // second >=0
              {
                sd = -b + std::sqrt(d);
              } else // both negative, travel away
              {
                return VECGEOM_NAMESPACE::kInfLength;
              }
            }
          }
          if (sd > 0) // If 'forwards'. Check z intersection
          {
            if (sd > dRmax) // Avoid rounding errors due to precision issues on
            {
              // 64 bits systems. Split long distances and recompute
              //   double fTerm = sd - std::fmod(sd, dRmax);
              //   sd = fTerm + DistanceToIn(p + fTerm * v, v);
            }
            zi = p.z() + sd * v.z();

            if (std::fabs(zi) <= tolODz) {
              // Z ok. Check phi intersection if reqd

              if (unplaced.IsFullPhi()) {
                return sd;
              } else {
                xi     = p.x() + sd * v.x();
                yi     = p.y() + sd * v.y();
                ri     = rMaxAv + zi * unplaced.fTanRMax;
                cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                if (cosPsi >= unplaced.fCosHDPhiIT) {
                  return sd;
                }
              }
            }
          } // end if (sd>0)
        }
      } else {
        // Inside outer cone
        // check not inside, and heading through UCons (-> 0 to in)

        if ((t3 > (rin + kHalfRadTolerance * unplaced.fSecRMin) * (rin + kHalfRadTolerance * unplaced.fSecRMin)) &&
            (nt2 < 0) && (d >= 0) && (std::fabs(p.z()) <= tolIDz)) {
          // Inside cones, delta r -ve, inside z extent
          // Point is on the Surface => check Direction using Normal.Dot(v)

          xi    = p.x();
          yi    = p.y();
          risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
          norm  = Vector3D<Precision>(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
          if (!unplaced.IsFullPhi()) {
            cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi) / std::sqrt(t3);
            if (cosPsi >= unplaced.fCosHDPhiIT) {
              if (norm.Dot(v) <= 0) {
                return 0.0;
              }
            }
          } else {
            if (norm.Dot(v) <= 0) {
              return 0.0;
            }
          }
        }
      }
    } else //  Single root case
    {
      if (std::fabs(nt2) > kRadTolerance) {
        sd = -0.5 * nt3 / nt2;

        if (sd < 0) {
          return VECGEOM_NAMESPACE::kInfLength; // travel away
        } else                                  // sd >= 0, If 'forwards'. Check z intersection
        {
          zi = p.z() + sd * v.z();

          if ((std::fabs(zi) <= tolODz) && (nt2 < 0)) {
            // Z ok. Check phi intersection if reqd

            if (unplaced.IsFullPhi()) {
              return sd;
            } else {
              xi     = p.x() + sd * v.x();
              yi     = p.y() + sd * v.y();
              ri     = rMaxAv + zi * unplaced.fTanRMax;
              cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

              if (cosPsi >= unplaced.fCosHDPhiIT) {
                return sd;
              }
            }
          }
        }
      } else //    travel || cone surface from its origin
      {
        sd = VECGEOM_NAMESPACE::kInfLength;
      }
    }

    // Inner Cone Intersection
    // o Space is divided into 3 areas:
    //   1) Radius greater than real inner cone & imaginary cone & outside
    //      tolerance
    //   2) Radius less than inner or imaginary cone & outside tolarance
    //   3) Within tolerance of real or imaginary cones
    //      - Extra checks needed for 3's intersections
    //        => lots of duplicated code

    if (rMinAv) {
      nt1 = t1 - (unplaced.fTanRMin * v.z()) * (unplaced.fTanRMin * v.z());
      nt2 = t2 - unplaced.fTanRMin * v.z() * rin;
      nt3 = t3 - rin * rin;

      if (nt1) {
        if (nt3 > rin * kRadTolerance * unplaced.fSecRMin) {
          // At radius greater than real & imaginary cones
          // -> 2nd root, with zi check

          b = nt2 / nt1;
          c = nt3 / nt1;
          d = b * b - c;
          if (d >= 0) // > 0
          {
            if (b > 0) {
              sd = c / (-b - std::sqrt(d));
            } else {
              sd = -b + std::sqrt(d);
            }

            if (sd >= 0) // > 0
            {
              if (sd > dRmax) // Avoid rounding errors due to precision issues on
              {
                // 64 bits systems. Split long distance and recompute
                //   double fTerm = sd - std::fmod(sd, dRmax);
                //   sd = fTerm + DistanceToIn(p + fTerm * v, v);
              }
              zi = p.z() + sd * v.z();

              if (std::fabs(zi) <= tolODz) {
                if (!unplaced.IsFullPhi()) {
                  xi     = p.x() + sd * v.x();
                  yi     = p.y() + sd * v.y();
                  ri     = rMinAv + zi * unplaced.fTanRMin;
                  cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                  if (cosPsi >= unplaced.fCosHDPhiIT) {
                    if (sd > kHalfRadTolerance) {
                      snxt = sd;
                    } else {
                      // Calculate a normal vector in order to check Direction

                      risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                      norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                      if (norm.Dot(v) <= 0) {
                        snxt = sd;
                      }
                    }
                  }
                } else {
                  if (sd > kHalfRadTolerance) {
                    return sd;
                  } else {
                    // Calculate a normal vector in order to check Direction

                    xi    = p.x() + sd * v.x();
                    yi    = p.y() + sd * v.y();
                    risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                    norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                    if (norm.Dot(v) <= 0) {
                      return sd;
                    }
                  }
                }
              }
            }
          }
        } else if (nt3 < -rin * kRadTolerance * unplaced.fSecRMin) {
          // Within radius of inner cone (real or imaginary)
          // -> Try 2nd root, with checking intersection is with real cone
          // -> If check fails, try 1st root, also checking intersection is
          //    on real cone

          b = nt2 / nt1;
          c = nt3 / nt1;
          d = b * b - c;

          if (d >= 0) // > 0
          {
            if (b > 0) {
              sd = c / (-b - std::sqrt(d));
            } else {
              sd = -b + std::sqrt(d);
            }
            zi = p.z() + sd * v.z();
            ri = rMinAv + zi * unplaced.fTanRMin;

            if (ri > 0) {
              if ((sd >= 0) && (std::fabs(zi) <= tolODz)) // sd > 0
              {
                if (sd > dRmax) // Avoid rounding errors due to precision issues
                {
                  // seen on 64 bits systems. Split and recompute
                  //   double fTerm = sd - std::fmod(sd, dRmax);
                  //   sd = fTerm + DistanceToIn(p + fTerm * v, v);
                }
                if (!unplaced.IsFullPhi()) {
                  xi     = p.x() + sd * v.x();
                  yi     = p.y() + sd * v.y();
                  cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                  if (cosPsi >= unplaced.fCosHDPhiOT) {
                    if (sd > kHalfRadTolerance) {
                      snxt = sd;
                    } else {
                      // Calculate a normal vector in order to check Direction

                      risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                      norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                      if (norm.Dot(v) <= 0) {
                        snxt = sd;
                      }
                    }
                  }
                } else {
                  if (sd > kHalfRadTolerance) {
                    return sd;
                  } else {
                    // Calculate a normal vector in order to check Direction

                    xi    = p.x() + sd * v.x();
                    yi    = p.y() + sd * v.y();
                    risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                    norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                    if (norm.Dot(v) <= 0) {
                      return sd;
                    }
                  }
                }
              }
            } else {
              if (b > 0) {
                sd = -b - std::sqrt(d);
              } else {
                sd = c / (-b + std::sqrt(d));
              }
              zi = p.z() + sd * v.z();
              ri = rMinAv + zi * unplaced.fTanRMin;

              if ((sd >= 0) && (ri > 0) && (std::fabs(zi) <= tolODz)) // sd>0
              {
                if (sd > dRmax) // Avoid rounding errors due to precision issues
                {
                  // seen on 64 bits systems. Split and recompute
                  //  double fTerm = sd - std::fmod(sd, dRmax);
                  //  sd = fTerm + DistanceToIn(p + fTerm * v, v);
                }
                if (!unplaced.IsFullPhi()) {
                  xi     = p.x() + sd * v.x();
                  yi     = p.y() + sd * v.y();
                  cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                  if (cosPsi >= unplaced.fCosHDPhiIT) {
                    if (sd > kHalfRadTolerance) {
                      snxt = sd;
                    } else {
                      // Calculate a normal vector in order to check Direction

                      risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                      norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                      if (norm.Dot(v) <= 0) {
                        snxt = sd;
                      }
                    }
                  }
                } else {
                  if (sd > kHalfRadTolerance) {
                    return sd;
                  } else {
                    // Calculate a normal vector in order to check Direction

                    xi    = p.x() + sd * v.x();
                    yi    = p.y() + sd * v.y();
                    risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                    norm  = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                    if (norm.Dot(v) <= 0) {
                      return sd;
                    }
                  }
                }
              }
            }
          }
        } else {
          // Within kRadTol*0.5 of inner cone (real OR imaginary)
          // ----> Check not travelling through (=>0 to in)
          // ----> if not:
          //    -2nd root with validity check

          if (std::fabs(p.z()) <= tolODz) {
            if (nt2 > 0) {
              // Inside inner real cone, heading outwards, inside z range

              if (!unplaced.IsFullPhi()) {
                cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi) / std::sqrt(t3);

                if (cosPsi >= unplaced.fCosHDPhiIT) {
                  return 0.0;
                }
              } else {
                return 0.0;
              }
            } else {
              // Within z extent, but not travelling through
              // -> 2nd root or VECGEOM_NAMESPACE::kInfLength if 1st root on imaginary cone

              b = nt2 / nt1;
              c = nt3 / nt1;
              d = b * b - c;

              if (d >= 0) // > 0
              {
                if (b > 0) {
                  sd = -b - std::sqrt(d);
                } else {
                  sd = c / (-b + std::sqrt(d));
                }
                zi = p.z() + sd * v.z();
                ri = rMinAv + zi * unplaced.fTanRMin;

                if (ri > 0) // 2nd root
                {
                  if (b > 0) {
                    sd = c / (-b - std::sqrt(d));
                  } else {
                    sd = -b + std::sqrt(d);
                  }

                  zi = p.z() + sd * v.z();

                  if ((sd >= 0) && (std::fabs(zi) <= tolODz)) // sd>0
                  {
                    if (sd > dRmax) // Avoid rounding errors due to precision issue
                    {
                      // seen on 64 bits systems. Split and recompute
                      // double fTerm = sd - std::fmod(sd, dRmax);
                      // sd = fTerm + DistanceToIn(p + fTerm * v, v);
                    }
                    if (!unplaced.IsFullPhi()) {
                      xi     = p.x() + sd * v.x();
                      yi     = p.y() + sd * v.y();
                      ri     = rMinAv + zi * unplaced.fTanRMin;
                      cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                      if (cosPsi >= unplaced.fCosHDPhiIT) {
                        snxt = sd;
                      }
                    } else {
                      return sd;
                    }
                  }
                } else {
                  return VECGEOM_NAMESPACE::kInfLength;
                }
              }
            }
          } else // 2nd root
          {
            b = nt2 / nt1;
            c = nt3 / nt1;
            d = b * b - c;

            if (d > 0) {
              if (b > 0) {
                sd = c / (-b - std::sqrt(d));
              } else {
                sd = -b + std::sqrt(d);
              }
              zi = p.z() + sd * v.z();

              if ((sd >= 0) && (std::fabs(zi) <= tolODz)) // sd>0
              {
                if (sd > dRmax) // Avoid rounding errors due to precision issues
                {
                  // seen on 64 bits systems. Split and recompute
                  //  double fTerm = sd - std::fmod(sd, dRmax);
                  //  sd = fTerm + DistanceToIn(p + fTerm * v, v);
                }
                if (!unplaced.IsFullPhi()) {
                  xi     = p.x() + sd * v.x();
                  yi     = p.y() + sd * v.y();
                  ri     = rMinAv + zi * unplaced.fTanRMin;
                  cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                  if (cosPsi >= unplaced.fCosHDPhiIT) {
                    snxt = sd;
                  }
                } else {
                  return sd;
                }
              }
            }
          }
        }
      }
    }

    // Phi segment intersection
    //
    // o Tolerant of points inside phi planes by up to VUSolid::Tolerance()*0.5
    //
    // o NOTE: Large duplication of code between sphi & ephi checks
    //         -> only diffs: sphi -> ephi, Comp -> -Comp and half-plane
    //            intersection check <=0 -> >=0
    //         -> Should use some form of loop Construct

    if (!unplaced.IsFullPhi()) {
      // First phi surface (starting phi)

      Comp = v.x() * unplaced.fSinSPhi - v.y() * unplaced.fCosSPhi;

      if (Comp < 0) // Component in outwards normal dirn
      {
        Dist = (p.y() * unplaced.fCosSPhi - p.x() * unplaced.fSinSPhi);

        if (Dist < kHalfCarTolerance) {
          sd = Dist / Comp;

          if (sd < snxt) {
            if (sd < 0) {
              sd = 0.0;
            }

            zi = p.z() + sd * v.z();

            if (std::fabs(zi) <= tolODz) {
              xi        = p.x() + sd * v.x();
              yi        = p.y() + sd * v.y();
              rhoi2     = xi * xi + yi * yi;
              tolORMin2 = (rMinOAv + zi * unplaced.fTanRMin) * (rMinOAv + zi * unplaced.fTanRMin);
              tolORMax2 = (rMaxOAv + zi * unplaced.fTanRMax) * (rMaxOAv + zi * unplaced.fTanRMax);

              if ((rhoi2 >= tolORMin2) && (rhoi2 <= tolORMax2)) {
                // z and r intersections good - check intersecting with
                // correct half-plane

                if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) <= 0) {
                  snxt = sd;
                }
              }
            }
          }
        }
      }

      // Second phi surface (Ending phi)

      Comp = -(v.x() * unplaced.fSinEPhi - v.y() * unplaced.fCosEPhi);

      if (Comp < 0) // Component in outwards normal dirn
      {
        Dist = -(p.y() * unplaced.fCosEPhi - p.x() * unplaced.fSinEPhi);
        if (Dist < kHalfCarTolerance) {
          sd = Dist / Comp;

          if (sd < snxt) {
            if (sd < 0) {
              sd = 0.0;
            }

            zi = p.z() + sd * v.z();

            if (std::fabs(zi) <= tolODz) {
              xi        = p.x() + sd * v.x();
              yi        = p.y() + sd * v.y();
              rhoi2     = xi * xi + yi * yi;
              tolORMin2 = (rMinOAv + zi * unplaced.fTanRMin) * (rMinOAv + zi * unplaced.fTanRMin);
              tolORMax2 = (rMaxOAv + zi * unplaced.fTanRMax) * (rMaxOAv + zi * unplaced.fTanRMax);

              if ((rhoi2 >= tolORMin2) && (rhoi2 <= tolORMax2)) {
                // z and r intersections good - check intersecting with
                // correct half-plane

                if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) >= 0.0) {
                  snxt = sd;
                }
              }
            }
          }
        }
      }
    }
    if (snxt < kHalfCarTolerance) {
      snxt = 0.;
    }

    return snxt;
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedCone const &unplaced, Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           Vector3D<typename Backend::precision_v> const &direction,
                           typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    // TOBEIMPLEMENTED
    distance = DistanceToInUSolids<Backend>(unplaced, transformation, point, direction, stepMax);

    //    typedef typename Backend::bool_v MaskType;
    //    typedef typename Backend::precision_v VectorType;
    //    typedef typename Vector3D<typename Backend::precision_v> Vector3D;
    //
    //    MaskType done_m(false); // which particles in the vector are ready to be returned == aka have been treated
    //    distance = kInfLength; // initialize distance to infinity
    //
    //    // TODO: check that compiler is doing same thing
    //    // as if we used a combined point + direction transformation
    //    Vector3D localpoint;
    //    localpoint=transformation.Transform<transCodeT,rotCodeT>(point);
    //    Vector3D localdir; // VectorType dirx, diry, dirz;
    //    localdir=transformation.TransformDirection<transCodeT,rotCodeT>(localdir);
    //
    //    // do some inside checks
    //    // if safez is > 0 it means that particle is within z range
    //    // if safez is < 0 it means that particle is outside z range
    //
    //    VectorType z = localpoint.z();
    //    VectorType safez = unplaced.GetDz() - Abs(z);
    //    MaskType inz_m = safez > Utils::fgToleranceVc;
    //    VectorType dirz = localdir.z();
    //    done_m = !inz_m && ( z*dirz >= 0 ); // particle outside the z-range and moving away
    //
    //    VectorType x=localpoint.x();
    //    VectorType y=localpoint.y();
    //    VectorType r2 = x*x + y*y; // use of Perp2
    //    VectorType n2 = VectorType(1)-(unplaced.GetOuterSlopeSquare() + 1) *dirz*dirz; // dirx_v*dirx_v +
    //    diry_v*diry_v; ( dir is normalized !! )
    //    VectorType dirx = localdir.x();
    //    VectorType diry = localdir.y();
    //    VectorType rdotnplanar = x*dirx + y*diry; // use of PerpDot
    //
    //    // T a = 1 - dir.z*dir.z*(1+m*m);
    //    // T b = 2 * ( pos.x*dir.x + pos.y*dir.y - m*m*pos.z*dir.z - m*n*dir.z);
    //    // T c = ( pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z - n*n );
    //
    //    // QUICK CHECK IF OUTER RADIUS CAN BE HIT AT ALL
    //    // BELOW WE WILL SOLVE A QUADRATIC EQUATION OF THE TYPE
    //    // a * t^2 + b * t + c = 0
    //    // if this equation has a solution at all ( == hit )
    //    // the following condition needs to be satisfied
    //    // DISCRIMINANT = b^2 -  4 a*c > 0
    //    //
    //    // THIS CONDITION DOES NOT NEED ANY EXPENSIVE OPERATION !!
    //    //
    //    // then the solutions will be given by
    //    //
    //    // t = (-b +- SQRT(DISCRIMINANT)) / (2a)
    //    //
    //    // b = 2*(dirx*x + diry*y)  -- independent of shape
    //    // a = dirx*dirx + diry*diry -- independent of shape
    //    // c = x*x + y*y - R^2 = r2 - R^2 -- dependent on shape
    //    VectorType c = r2 - unplaced.GetOuterSlopeSquare()*z*z -
    //    2*unplaced.GetOuterSlope()*unplaced.GetOuterOffset() * z - unplaced.GetOuterOffsetSquare();
    //
    //    VectorType a = n2;
    //
    //    VectorType b = 2*(rdotnplanar - z*dirz*unplaced.GetOuterSlopeSquare() -
    //    unplaced.GetOuterSlope()*unplaced.GetOuterOffset()*dirz);
    //    VectorType discriminant = b*b-4*a*c;
    //    MaskType   canhitrmax = ( discriminant >= 0 );
    //
    //    done_m |= ! canhitrmax;
    //
    //    // this might be optional
    //    if( done_m.vecCore::MaskFull() )
    //    {
    //           // joint z-away or no chance to hit Rmax condition
    //     #ifdef LOG_EARLYRETURNS
    //           std::cerr << " RETURN1 IN DISTANCETOIN " << std::endl;
    //     #endif
    //           return;
    //    }
    //
    //    // Check outer cylinder (only r>rmax has to be considered)
    //    // this IS ALWAYS the MINUS (-) solution
    //    VectorType distanceRmax( Utils::kInfLengthVc );
    //    distanceRmax( canhitrmax ) = (-b - Sqrt( discriminant ))/(2.*a);
    //
    //    // this determines which vectors are done here already
    //    MaskType Rdone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmax );
    //    distanceRmax( ! Rdone ) = Utils::kInfLengthVc;
    //    MaskType rmindone;
    //    // **** inner tube ***** only compiled in for tubes having inner hollow cone -- or in case of a universal
    //    runtime shape ******/
    //    if ( checkRminTreatment<ConeType>(unplaced) )
    //    {
    //       // in case of the Cone, generally all coefficients a, b, and c change
    //       a = 1.-(unplaced.GetInnerSlopeSquare() + 1) *dirz*dirz;
    //       c = r2 - unplaced.GetInnerSlopeSquare()*z*z - 2*unplaced.GetInnerSlope()*unplaced.GetInnerOffset() * z
    //               - unplaced.GetInnerOffsetSquare();
    //       b = 2*(rdotnplanar - dirz*(z*unplaced.GetInnerSlopeSquare + unplaced.GetInnerOffset*
    //       unplaced.GetInnerSlope()));
    //       discriminant =  b*b-4*a*c;
    //       MaskType canhitrmin = ( discriminant >= Vc::Zero );
    //       VectorType distanceRmin ( Utils::kInfLengthVc );
    //       // this is always + solution
    //       distanceRmin ( canhitrmin ) = (-b + Vc::sqrt( discriminant ))/(2*a);
    //       rmindone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmin );
    //       distanceRmin ( ! rmindone ) = Utils::kInfLength;
    //
    //       // reduction of distances
    //       distanceRmax = Vc::min( distanceRmax, distanceRmin );
    //       Rdone |= rmindone;
    //     }
    //        //distance( ! done_m && Rdone ) = distanceRmax;
    //        //done_m |= Rdone;
    //
    //        /* might check early here */
    //
    //        // now do Z-Face
    //        VectorType distancez = -safez/Vc::abs(dirz);
    //        MaskType zdone = determineZHit(x,y,z,dirx,diry,dirz,distancez);
    //        distance ( ! done_m && zdone ) = distancez;
    //        distance ( ! done_m && ! zdone && Rdone ) = distanceRmax;
    //        done_m |= ( zdone ) || (!zdone && (Rdone));
    //
    //        // now PHI
    //
    //        // **** PHI TREATMENT FOR CASE OF HAVING RMAX ONLY ***** only compiled in for cones having phi sektion
    //        ***** //
    //        if ( ConeTraits::NeedsPhiTreatment<ConeType>::value )
    //        {
    //           // all particles not done until here have the potential to hit a phi surface
    //           // phi surfaces require divisions so it might be useful to check before continuing
    //
    //           if( ConeTraits::NeedsRminTreatment<ConeType>::value || ! done_m.vecCore::MaskFull() )
    //           {
    //              VectorType distphi;
    //              ConeUtils::DistanceToPhiPlanes<ValueType,ConeTraits::IsPhiEqualsPiCase<ConeType>::value,ConeTraits::NeedsRminTreatment<ConeType>::value>(coneparams->dZ,
    //                    coneparams->outerslope, coneparams->outeroffset,
    //                    coneparams->innerslope, coneparams->inneroffset,
    //                    coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x,
    //                    coneparams->normalPhi2.y,
    //                    coneparams->alongPhi1, coneparams->alongPhi2,
    //                    x, y, z, dirx, diry, dirz, distphi);
    //              if(ConeTraits::NeedsRminTreatment<ConeType>::value)
    //              {
    //                 // distance(! done_m || (rmindone && ! inrmin_m ) || (rmaxdone && ) ) = distphi;
    //                 // distance ( ! done_m ) = distphi;
    //                 distance = Vc::min(distance, distphi);
    //              }
    //              else
    //              {
    //                 distance ( ! done_m ) = distphi;
    //              }
    //           }
    //        }
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Precision DistanceToOutUSOLIDS(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> p,
                                        Vector3D<typename Backend::precision_v> v,
                                        typename Backend::precision_v const & /*stepMax*/
                                        )
  {

    double snxt, srd, sphi, pdist;

    double rMaxAv; // Data for outer cone
    double rMinAv; // Data for inner cone

    double t1, t2, t3, rout, rin, nt1, nt2, nt3;
    double b, c, d, sr2, sr3;

    // Vars for intersection within tolerance

    //  ESide   sidetol = kNull;
    //  double slentol = VECGEOM_NAMESPACE::kInfLength;

    // Vars for phi intersection:

    double pDistS, compS, pDistE, compE, sphi2, xi, yi, /*risec,*/ vphi;
    double zi, ri, deltaRoi2;

    // Z plane intersection

    if (v.z() > 0.0) {
      pdist = unplaced.GetDz() - p.z();

      if (pdist > kHalfCarTolerance) {
        snxt = pdist / v.z();
        //   side = kPZ;
      } else {
        //  aNormalVector        = Vector3D<Precision>(0, 0, 1);
        //  aConvex = true;
        return snxt = 0.0;
      }
    } else if (v.z() < 0.0) {
      pdist = unplaced.GetDz() + p.z();

      if (pdist > kHalfCarTolerance) {
        snxt = -pdist / v.z();
        // side = kMZ;
      } else {
        // aNormalVector        = Vector3D<Precision>(0, 0, -1);
        // aConvex = true;
        return snxt = 0.0;
      }
    } else // Travel perpendicular to z axis
    {
      snxt = VECGEOM_NAMESPACE::kInfLength;
      // side = kNull;
    }

    // Radial Intersections
    //
    // Intersection with outer cone (possible return) and
    //                   inner cone (must also check phi)
    //
    // Intersection point (xi,yi,zi) on line x=p.x()+t*v.x() etc.
    //
    // Intersects with x^2+y^2=(a*z+b)^2
    //
    // where a=unplaced.fTanRMax or unplaced.fTanRMin
    //       b=rMaxAv or rMinAv
    //
    // (vx^2+vy^2-(a*vz)^2)t^2+2t(pxvx+pyvy-a*vz(a*pz+b))+px^2+py^2-(a*pz+b)^2=0;
    //     t1                       t2                      t3
    //
    //  \--------u-------/       \-----------v----------/ \---------w--------/

    rMaxAv = (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;

    t1   = 1.0 - v.z() * v.z(); // since v normalised
    t2   = p.x() * v.x() + p.y() * v.y();
    t3   = p.x() * p.x() + p.y() * p.y();
    rout = unplaced.fTanRMax * p.z() + rMaxAv;

    nt1 = t1 - (unplaced.fTanRMax * v.z()) * (unplaced.fTanRMax * v.z());
    nt2 = t2 - unplaced.fTanRMax * v.z() * rout;
    nt3 = t3 - rout * rout;

    if (v.z() > 0.0) {
      deltaRoi2 = snxt * snxt * t1 + 2 * snxt * t2 + t3 -
                  unplaced.GetRmax2() * (unplaced.GetRmax2() + kRadTolerance * unplaced.fSecRMax);
    } else if (v.z() < 0.0) {
      deltaRoi2 = snxt * snxt * t1 + 2 * snxt * t2 + t3 -
                  unplaced.GetRmax1() * (unplaced.GetRmax1() + kRadTolerance * unplaced.fSecRMax);
    } else {
      deltaRoi2 = 1.0;
    }

    if (nt1 && (deltaRoi2 > 0.0)) {
      // Equation quadratic => 2 roots : second root must be leaving

      b = nt2 / nt1;
      c = nt3 / nt1;
      d = b * b - c;

      if (d >= 0) {
        // Check if on outer cone & heading outwards
        // NOTE: Should use rho-rout>-kRadTolerance*0.5

        if (nt3 > -kHalfRadTolerance && nt2 >= 0) {
          //              risec     = Sqrt(t3) * unplaced.fSecRMax;
          // aConvex = true;
          // aNormalVector        = Vector3D<Precision>(p.x() / risec, p.y() / risec, -unplaced.fTanRMax /
          // unplaced.fSecRMax);
          return snxt = 0;
        } else {
          // sider = kRMax ;
          if (b > 0) {
            srd = -b - Sqrt(d);
          } else {
            srd = c / (-b + Sqrt(d));
          }

          zi = p.z() + srd * v.z();
          ri = unplaced.fTanRMax * zi + rMaxAv;

          if ((ri >= 0) && (-kHalfRadTolerance <= srd) && (srd <= kHalfRadTolerance)) {
            // An intersection within the tolerance
            //   we will Store it in case it is good -
            //
            // slentol = srd;
            // sidetol = kRMax;
          }
          if ((ri < 0) || (srd < kHalfRadTolerance)) {
            // Safety: if both roots -ve ensure that srd cannot `win'
            //         distance to out

            if (b > 0) {
              sr2 = c / (-b - Sqrt(d));
            } else {
              sr2 = -b + Sqrt(d);
            }
            zi = p.z() + sr2 * v.z();
            ri = unplaced.fTanRMax * zi + rMaxAv;

            if ((ri >= 0) && (sr2 > kHalfRadTolerance)) {
              srd = sr2;
            } else {
              srd = VECGEOM_NAMESPACE::kInfLength;

              if ((-kHalfRadTolerance <= sr2) && (sr2 <= kHalfRadTolerance)) {
                // An intersection within the tolerance.
                // Storing it in case it is good.

                //   slentol = sr2;
                //  sidetol = kRMax;
              }
            }
          }
        }
      } else {
        // No intersection with outer cone & not parallel
        // -> already outside, no intersection

        // risec     = Sqrt(t3) * unplaced.fSecRMax;
        // aConvex = true;
        // aNormalVector        = Vector3D<Precision>(p.x() / risec, p.y() / risec, -unplaced.fTanRMax /
        // unplaced.fSecRMax);
        return snxt = 0.0;
      }
    } else if (nt2 && (deltaRoi2 > 0.0)) {
      // Linear case (only one intersection) => point outside outer cone

      //          risec     = Sqrt(t3) * unplaced.fSecRMax;
      // aConvex = true;
      // aNormalVector        = Vector3D<Precision>(p.x() / risec, p.y() / risec, -unplaced.fTanRMax /
      // unplaced.fSecRMax);
      return snxt = 0.0;
    } else {
      // No intersection -> parallel to outer cone
      // => Z or inner cone intersection

      srd = VECGEOM_NAMESPACE::kInfLength;
    }

    // Check possible intersection within tolerance

    /*
    if (slentol <= kHalfCarTolerance)
    {
      // An intersection within the tolerance was found.
      // We must accept it only if the momentum points outwards.
      //
      // Vector3D<Precision> ptTol;  // The point of the intersection
      // ptTol= p + slentol*v;
      // ri=unplaced.fTanRMax*zi+rMaxAv;
      //
      // Calculate a normal vector, as below

      xi    = p.x() + slentol * v.x();
      yi    = p.y() + slentol * v.y();
      risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
      Vector3D<Precision> norm = Vector3D<Precision>(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);

      if (norm.Dot(v) > 0)     // We will leave the Cone immediatelly
      {
        aNormalVector        = norm.Unit();
        aConvex = true;
        return snxt = 0.0;
      }
      else // On the surface, but not heading out so we ignore this intersection
      {
        //                                        (as it is within tolerance).
        slentol = VECGEOM_NAMESPACE::kInfLength;
      }

    }
*/

    // Inner Cone intersection

    if (unplaced.GetRmin1() || unplaced.GetRmin2()) {
      nt1 = t1 - (unplaced.fTanRMin * v.z()) * (unplaced.fTanRMin * v.z());

      if (nt1) {
        rMinAv = (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
        rin    = unplaced.fTanRMin * p.z() + rMinAv;
        nt2    = t2 - unplaced.fTanRMin * v.z() * rin;
        nt3    = t3 - rin * rin;

        // Equation quadratic => 2 roots : first root must be leaving

        b = nt2 / nt1;
        c = nt3 / nt1;
        d = b * b - c;

        if (d >= 0.0) {
          // NOTE: should be rho-rin<kRadTolerance*0.5,
          //       but using squared versions for efficiency

          if (nt3 < kRadTolerance * (rin + kRadTolerance * 0.25)) {
            if (nt2 < 0.0) {
              // aConvex = false;
              // risec = Sqrt(p.x() * p.x() + p.y() * p.y()) * unplaced.fSecRMin;
              // aNormalVector = UVector3(-p.x() / risec, -p.y() / risec, unplaced.fTanRMin / unplaced.fSecRMin);
              return snxt = 0.0;
            }
          } else {
            if (b > 0) {
              sr2 = -b - Sqrt(d);
            } else {
              sr2 = c / (-b + Sqrt(d));
            }
            zi = p.z() + sr2 * v.z();
            ri = unplaced.fTanRMin * zi + rMinAv;

            if ((ri >= 0.0) && (-kHalfRadTolerance <= sr2) && (sr2 <= kHalfRadTolerance)) {
              // An intersection within the tolerance
              // storing it in case it is good.

              // slentol = sr2;
              // sidetol = kRMax;
            }
            if ((ri < 0) || (sr2 < kHalfRadTolerance)) {
              if (b > 0) {
                sr3 = c / (-b - Sqrt(d));
              } else {
                sr3 = -b + Sqrt(d);
              }

              // Safety: if both roots -ve ensure that srd cannot `win'
              //         distancetoout

              if (sr3 > kHalfRadTolerance) {
                if (sr3 < srd) {
                  zi = p.z() + sr3 * v.z();
                  ri = unplaced.fTanRMin * zi + rMinAv;

                  if (ri >= 0.0) {
                    srd = sr3;
                    //     sider = kRMin;
                  }
                }
              } else if (sr3 > -kHalfRadTolerance) {
                // Intersection in tolerance. Store to check if it's good

                //     slentol = sr3;
                //    sidetol = kRMin;
              }
            } else if ((sr2 < srd) && (sr2 > kHalfCarTolerance)) {
              srd = sr2;
              //  sider = kRMin;
            } else if (sr2 > -kHalfCarTolerance) {
              // Intersection in tolerance. Store to check if it's good

              //  slentol = sr2;
              //  sidetol = kRMin;
            }

            /*
            if (slentol <= kHalfCarTolerance)
            {
              // An intersection within the tolerance was found.
              // We must accept it only if  the momentum points outwards.

              UVector3 norm;

              // Calculate a normal vector, as below

              xi     = p.x() + slentol * v.x();
              yi     = p.y() + slentol * v.y();
              if (sidetol == kRMax)
              {
                risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
                norm = UVector3(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
              }
              else
              {
                risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                norm = UVector3(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
              }
              if (norm.Dot(v) > 0)
              {
                // We will leave the cone immediately

                aNormalVector        = norm.Unit();
                aConvex = true;
                return snxt = 0.0;
              }
              else
              {
                // On the surface, but not heading out so we ignore this
                // intersection (as it is within tolerance).

                slentol = VECGEOM_NAMESPACE::kInfLength;
              }
            }*/
          }
        }
      }
    }

    // Linear case => point outside inner cone ---> outer cone intersect
    //
    // Phi Intersection

    if (!unplaced.IsFullPhi()) {
      // add angle calculation with correction
      // of the difference in domain of atan2 and Sphi

      vphi = ATan2(v.y(), v.x());

      if (vphi < unplaced.GetSPhi() - kHalfAngTolerance) {
        vphi += 2 * VECGEOM_NAMESPACE::kPi;
      } else if (vphi > unplaced.GetSPhi() + unplaced.GetDPhi() + kHalfAngTolerance) {
        vphi -= 2 * VECGEOM_NAMESPACE::kPi;
      }

      if (p.x() || p.y()) // Check if on z axis (rho not needed later)
      {
        // pDist -ve when inside

        pDistS = p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi;
        pDistE = -p.x() * unplaced.fSinEPhi + p.y() * unplaced.fCosEPhi;

        // Comp -ve when in direction of outwards normal

        compS = -unplaced.fSinSPhi * v.x() + unplaced.fCosSPhi * v.y();
        compE = unplaced.fSinEPhi * v.x() - unplaced.fCosEPhi * v.y();

        //    sidephi = kNull;

        if (((unplaced.GetDPhi() <= VECGEOM_NAMESPACE::kPi) &&
             ((pDistS <= kHalfCarTolerance) && (pDistE <= kHalfCarTolerance))) ||
            ((unplaced.GetDPhi() > VECGEOM_NAMESPACE::kPi) &&
             !((pDistS > kHalfCarTolerance) && (pDistE > kHalfCarTolerance)))) {
          // Inside both phi *full* planes
          if (compS < 0) {
            sphi = pDistS / compS;
            if (sphi >= -kHalfCarTolerance) {
              xi = p.x() + sphi * v.x();
              yi = p.y() + sphi * v.y();

              // Check intersecting with correct kHalf-plane
              // (if not -> no intersect)
              //
              if ((Abs(xi) <= VECGEOM_NAMESPACE::kTolerance) && (Abs(yi) <= VECGEOM_NAMESPACE::kTolerance)) {
                // sidephi = kSPhi;
                if ((unplaced.GetSPhi() - kHalfAngTolerance <= vphi) &&
                    (unplaced.GetSPhi() + unplaced.GetDPhi() + kHalfAngTolerance >= vphi)) {
                  sphi = VECGEOM_NAMESPACE::kInfLength;
                }
              } else if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) >= 0) {
                sphi = VECGEOM_NAMESPACE::kInfLength;
              } else {
                // sidephi = kSPhi;
                if (pDistS > -kHalfCarTolerance) {
                  sphi = 0.0; // Leave by sphi immediately
                }
              }
            } else {
              sphi = VECGEOM_NAMESPACE::kInfLength;
            }
          } else {
            sphi = VECGEOM_NAMESPACE::kInfLength;
          }

          if (compE < 0) {
            sphi2 = pDistE / compE;

            // Only check further if < starting phi intersection
            //
            if ((sphi2 > -kHalfCarTolerance) && (sphi2 < sphi)) {
              xi = p.x() + sphi2 * v.x();
              yi = p.y() + sphi2 * v.y();

              // Check intersecting with correct kHalf-plane

              if ((Abs(xi) <= VECGEOM_NAMESPACE::kTolerance) && (Abs(yi) <= VECGEOM_NAMESPACE::kTolerance)) {
                // Leaving via ending phi

                if (!((unplaced.GetSPhi() - kHalfAngTolerance <= vphi) &&
                      (unplaced.GetSPhi() + unplaced.GetDPhi() + kHalfAngTolerance >= vphi))) {
                  //  sidephi = kEPhi;
                  if (pDistE <= -kHalfCarTolerance) {
                    sphi = sphi2;
                  } else {
                    sphi = 0.0;
                  }
                }
              } else // Check intersecting with correct half-plane
                  if (yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi >= 0) {
                // Leaving via ending phi

                //   sidephi = kEPhi;
                if (pDistE <= -kHalfCarTolerance) {
                  sphi = sphi2;
                } else {
                  sphi = 0.0;
                }
              }
            }
          }
        } else {
          sphi = VECGEOM_NAMESPACE::kInfLength;
        }
      } else {
        // On z axis + travel not || to z axis -> if phi of vector direction
        // within phi of shape, Step limited by rmax, else Step =0

        if ((unplaced.GetSPhi() - kHalfAngTolerance <= vphi) &&
            (vphi <= unplaced.GetSPhi() + unplaced.GetDPhi() + kHalfAngTolerance)) {
          sphi = VECGEOM_NAMESPACE::kInfLength;
        } else {
          // sidephi = kSPhi ;  // arbitrary
          sphi = 0.0;
        }
      }
      if (sphi < snxt) // Order intersecttions
      {
        snxt = sphi;
        //    side = sidephi;
      }
    }
    if (srd < snxt) // Order intersections
    {
      snxt = srd;
      // side = sider;
    }
    if (snxt < kHalfCarTolerance) {
      snxt = 0.;
    }

    return snxt;
  }

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> point,
                            Vector3D<typename Backend::precision_v> direction,
                            typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    // TOBEIMPLEMENTED

    // has to be implemented in a way
    // as to be tolerant when particle is outside
    distance = DistanceToOutUSOLIDS<Backend>(unplaced, point, direction, stepMax);
  }

  template <class Backend, bool ForPolycone>
  VECGEOM_CUDA_HEADER_BOTH
  static Precision SafetyToInUSOLIDS(UnplacedCone const &unplaced, Transformation3D const &transformation,
                                     Vector3D<typename Backend::precision_v> const &point)
  {
    double safe = 0.0, rho, safeR1, safeR2, safeZ, safePhi, cosPsi;
    double pRMin, pRMax;

    // need a transformation
    Vector3D<Precision> p = transformation.Transform<transCodeT, rotCodeT>(point);

    rho    = Sqrt(p.x() * p.x() + p.y() * p.y());
    safeZ  = Abs(p.z()) - unplaced.GetDz();
    safeR1 = 0;
    safeR2 = 0;

    if (unplaced.GetRmin1() || unplaced.GetRmin2()) {
      pRMin  = unplaced.fTanRMin * p.z() + (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
      safeR1 = (pRMin - rho) * unplaced.fInvSecRMin;

      pRMax  = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
      safeR2 = (rho - pRMax) * unplaced.fInvSecRMax;

      if (safeR1 > safeR2) {
        safe = safeR1;
      } else {
        safe = safeR2;
      }
    } else {
      pRMax = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
      safe  = (rho - pRMax) * unplaced.fInvSecRMax;
    }
    if (!ForPolycone) // For Polycone only safety in R and Phi is needed
    {
      if (safeZ > safe) {
        safe = safeZ;
      }
    }
    if (!unplaced.IsFullPhi() && rho) {
      // Psi=angle from central phi to point

      cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi);

      if (cosPsi < unplaced.fCosHDPhi * rho) // Point lies outside phi range
      {
        if ((p.y() * unplaced.fCosCPhi - p.x() * unplaced.fSinCPhi) <= 0.0) {
          safePhi = Abs(p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi);
        } else {
          safePhi = Abs(p.x() * unplaced.fSinEPhi - p.y() * unplaced.fCosEPhi);
        }
        if (safePhi > safe) {
          safe = safePhi;
        }
      }
    }
    if (safe < 0.0) {
      safe = 0.0;
      return safe; // point is Inside
    }
    return safe;
    /*
    if (!aAccurate) return safe;

    double safsq = 0.0;
    int count = 0;
    if (safeR1 > 0)
    {
      safsq += safeR1 * safeR1;
      count++;
    }
    if (safeR2 > 0)
    {
      safsq += safeR2 * safeR2;
      count++;
    }
    if (safeZ > 0)
    {
      safsq += safeZ * safeZ;
      count++;
    }
    if (count == 1) return safe;
    return Sqrt(safsq);
*/
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedCone const &unplaced, Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety)
  {

    // TOBEIMPLEMENTED -- momentarily dispatching to USolids
    safety = SafetyToInUSOLIDS<Backend, false>(unplaced, transformation, point);
  }

  template <class Backend, bool ForPolycone>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Precision SafetyToOutUSOLIDS(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> p)
  {
    double safe = 0.0, rho, safeR1, safeR2, safeZ, safePhi;
    double pRMin;
    double pRMax;

    rho   = Sqrt(p.x() * p.x() + p.y() * p.y());
    safeZ = unplaced.GetDz() - Abs(p.z());

    if (unplaced.GetRmin1() || unplaced.GetRmin2()) {
      pRMin  = unplaced.fTanRMin * p.z() + (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
      safeR1 = (rho - pRMin) * unplaced.fInvSecRMin;
    } else {
      safeR1 = VECGEOM_NAMESPACE::kInfLength;
    }

    pRMax  = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
    safeR2 = (pRMax - rho) * unplaced.fInvSecRMax;

    if (safeR1 < safeR2) {
      safe = safeR1;
    } else {
      safe = safeR2;
    }
    if (!ForPolycone) // For Polycone only safety in R and Phi is needed
    {
      if (safeZ < safe) {
        safe = safeZ;
      }
    }
    // Check if phi divided, Calc distances closest phi plane

    if (!unplaced.IsFullPhi()) {
      // Above/below central phi of UCons?

      if ((p.y() * unplaced.fCosCPhi - p.x() * unplaced.fSinCPhi) <= 0) {
        safePhi = -(p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi);
      } else {
        safePhi = (p.x() * unplaced.fSinEPhi - p.y() * unplaced.fCosEPhi);
      }
      if (safePhi < safe) {
        safe = safePhi;
      }
    }
    if (safe < 0) {
      safe = 0;
    }

    return safe;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety)
  {

    safety = SafetyToOutUSOLIDS<Backend, false>(unplaced, point);
  }

  // normal kernel
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void NormalKernel(UnplacedCone const &unplaced, Vector3D<typename Backend::precision_v> const &p,
                           Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
  {
    // TODO: provide generic vectorized implementation
    // TODO: transform point p to local coordinates (GL: code assumes input point is in local coords already)

    int noSurfaces = 0;
    double rho, pPhi;
    double distZ, distRMin, distRMax;
    double distSPhi = kInfLength, distEPhi = kInfLength;
    double pRMin, widRMin;
    double pRMax, widRMax;

    const double delta = 0.5 * kTolerance;
    // const double dAngle = 0.5 * kAngTolerance;
    typedef Vector3D<typename Backend::precision_v> Vec3D_t;
    Vec3D_t sumnorm(0., 0., 0.), nZ(0., 0., 1.);
    Vec3D_t nR, nr(0., 0., 0.), nPs, nPe;

    // distZ<0 for between z-planes, distZ>0 for outside
    distZ = Abs(p.z()) - unplaced.GetDz();
    rho   = Sqrt(p.x() * p.x() + p.y() * p.y());

    pRMin    = rho - p.z() * unplaced.fTanRMin;
    widRMin  = unplaced.GetRmin2() - unplaced.GetDz() * unplaced.fTanRMin;
    distRMin = (pRMin - widRMin) * unplaced.fInvSecRMin;

    pRMax    = rho - p.z() * unplaced.fTanRMax;
    widRMax  = unplaced.GetRmax2() - unplaced.GetDz() * unplaced.fTanRMax;
    distRMax = (pRMax - widRMax) * unplaced.fInvSecRMax;

    bool inside = distZ < kTolerance && distRMax < kTolerance && (unplaced.GetRmin1() || unplaced.GetRmin2()) &&
                  distRMin > -kTolerance;

    distZ    = Abs(distZ);
    distRMin = Abs(distRMin);
    distRMax = Abs(distRMax);

    // keep track of which surface is nearest point P, needed in case point is not on a surface
    double distNearest              = distZ;
    Vector3D<Precision> normNearest = nZ;
    if (p.z() < 0.) normNearest.Set(0, 0, -1.);
    // std::cout<<"ConeImpl: spot 1: p="<< p <<", normNearest="<< normNearest <<", distZ="<< distZ <<"\n";

    if (inside && distZ <= delta) {
      noSurfaces++;
      if (p.z() >= 0.)
        sumnorm += nZ;
      else
        sumnorm.Set(0, 0, -1.);
    }

    if (!unplaced.IsFullPhi()) {
      // Protect against (0,0,z)
      if (rho) {
        pPhi = ATan2(p.y(), p.x());
        if (pPhi < unplaced.GetSPhi() - delta)
          pPhi += kTwoPi;
        else if (pPhi > unplaced.GetSPhi() + unplaced.GetDPhi() + delta)
          pPhi -= kTwoPi;

        distSPhi = rho * (pPhi - unplaced.GetSPhi());
        distEPhi = rho * (pPhi - unplaced.GetSPhi() - unplaced.GetDPhi());
        inside   = inside && distSPhi > -delta && distEPhi < delta;
        distSPhi = Abs(distSPhi);
        distEPhi = Abs(distEPhi);
      } else if (!(unplaced.GetRmin1()) || !(unplaced.GetRmin2())) {
        distSPhi = 0.;
        distEPhi = 0.;
      }
      nPs = Vec3D_t(unplaced.fSinSPhi, -unplaced.fCosSPhi, 0);
      nPe = Vec3D_t(-unplaced.fSinEPhi, unplaced.fCosEPhi, 0);
    }

    if (rho > delta) {
      nR = Vec3D_t(p.x() / rho * unplaced.fInvSecRMax, p.y() / rho * unplaced.fInvSecRMax,
                   -unplaced.fTanRMax * unplaced.fInvSecRMax);
      if (unplaced.GetRmin1() || unplaced.GetRmin2()) {
        nr = Vec3D_t(-p.x() / rho * unplaced.fInvSecRMin, -p.y() / rho * unplaced.fInvSecRMin,
                     unplaced.fTanRMin * unplaced.fInvSecRMin);
      }
    }

    if (inside && distRMax <= delta) {
      noSurfaces++;
      sumnorm += nR;
    } else if (noSurfaces == 0 && distRMax < distNearest) {
      distNearest = distRMax;
      normNearest = nR;
    }

    if (unplaced.GetRmin1() || unplaced.GetRmin2()) {
      if (inside && distRMin <= delta) {
        noSurfaces++;
        sumnorm += nr;
      } else if (noSurfaces == 0 && distRMin < distNearest) {
        distNearest = distRMin;
        normNearest = nr;
      }
    }

    if (!unplaced.IsFullPhi()) {
      if (inside && distSPhi <= delta) {
        noSurfaces++;
        sumnorm += nPs;
      } else if (noSurfaces == 0 && distSPhi < distNearest) {
        distNearest = distSPhi;
        normNearest = nPs;
      }

      if (inside && distEPhi <= delta) {
        noSurfaces++;
        sumnorm += nPe;
      } else if (noSurfaces == 0 && distEPhi < distNearest) {
        distNearest = distEPhi;
        normNearest = nPe;
      }
    }

    // Final checks
    if (noSurfaces == 0)
      normal = normNearest;
    else if (noSurfaces == 1)
      normal = sumnorm;
    else
      normal = sumnorm.Unit();

    valid = (bool)noSurfaces;
  }

#endif

}; // end struct
}
} // End global namespace

#endif /* VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_ */
