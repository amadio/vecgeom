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
#include <VecCore/VecCore>
#include <cstdio>

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
      typename BoolType IsInRightZInterval( VecType const & z, OtherType const &
     dz )
      {
         return Abs(z) <= dz;
      }

     template<typename ConeType>
     template<typename VectorType, typename MaskType>
     inline
     __attribute__((always_inline))
     typename MaskType determineRHit( UnplacedCone const & unplaced, VectorType
     const & x, VectorType const & y,
     VectorType const & z,
                                      VectorType const & dirx, VectorType const
     & diry, VectorType const & dirz,
                                      VectorType const & distanceR ) const
     {
        if( ! checkPhiTreatment<ConeType>( unplaced ) )
        {
           return distanceR > 0 && IsInRightZInterval( z+distanceR*dirz,
     unplaced->GetDz() );
        }
        else
        {
           // need to have additional look if hitting point on zylinder is not
     in empty phi range
           VectorType xhit = x + distanceR*dirx;
           VectorType yhit = y + distanceR*diry;
           return distanceR > 0 && IsInRightZInterval( z + distanceR*dirz,
     unplaced->GetDz() )
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
* (-x*starty + y*startx) >= 0: calculates whether going from the start vector to
*the point
* we are traveling in the CCW direction (taking the shortest direction, of
*course)
*
* (-endx*y + endy*x) >= 0: calculates whether going from the point to the end
*vector
* we are traveling in the CCW direction (taking the shortest direction, of
*course)
*
* For a sector smaller than pi, we need that BOTH of them hold true - if going
*from start, to the
* point, and then to the end we are travelling in CCW, it's obvious the point is
*inside the
* cylindrical sector.
*
* For a sector bigger than pi, only one of the conditions needs to be true. This
*is less obvious why.
* Since the sector angle is greater than pi, it can be that one of the two
*vectors might be
* farther than pi away from the point. In that case, the shortest direction will
*be CW, so even
* if the point is inside, only one of the two conditions need to hold.
*
* If going from start to point is CCW, then certainly the point is inside as the
*sector
* is larger than pi.
*
* If going from point to end is CCW, again, the point is certainly inside.
*
* This function is a frankensteinian creature that can determine which of the
*two cases (smaller vs larger than pi)
* to use either at compile time (if it has enough information, saving an if
*statement) or at runtime.
**/

template <typename Backend, typename ShapeType, typename UnplacedVolumeType, bool onSurfaceT,
          bool includeSurface = true>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static void PointInCyclicalSector(UnplacedVolumeType const &volume, typename Backend::precision_v const &x,
                                  typename Backend::precision_v const &y, typename Backend::bool_v &ret)
{

  using namespace ::vecgeom::ConeTypes;
  // assert(SectorType<ShapeType>::value != kNoAngle && "ShapeType without a
  // sector passed to PointInCyclicalSector");

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

    // GL: tested with this if(PosDirPhiVec) around if(insector), so
    // if(insector){} requires PosDirPhiVec==true to run
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
  //  std::cout<<"Normal : "<< norm << " : " << __LINE__ << std::endl;
  return norm;
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

  Float_t rho = point.Perp2();
  // std::cout<<"ForInnerSurface : " <<ForInnerSurface <<" : " << __LINE__ << std::endl;
  Float_t coneRad  = GetRadiusOfConeAtPoint<Backend, ForInnerSurface>(cone, point.z());
  Float_t coneRad2 = coneRad * coneRad;
  // std::cout<<"rho2 : "<< rho <<" : ConeRad2 : "<< coneRad2 << std::endl;
  return (rho >= (coneRad2 - kConeTolerance * coneRad)) && (rho <= (coneRad2 + kConeTolerance * coneRad)) &&
         (Abs(point.z()) < (cone.GetDz() + kTolerance));
}

template <class Backend, bool ForInnerSurface>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsMovingOutsideConicalSurface(UnplacedCone const &cone,
                                                              Vector3D<typename Backend::precision_v> const &point,
                                                              Vector3D<typename Backend::precision_v> const &direction)
{

  // std::cout<<" ==== Entered IsOnConicalSurface ==== " << std::endl;
  // typedef typename Backend::bool_v Bool_t;
  // Bool_t isOnConicalSurface = IsOnConicalSurface<Backend, ForInnerSurface>(cone, point);
  // Vector3D<typename Backend::precision_v> norm = GetNormal<Backend, ForInnerSurface>(cone, point);
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

  //  std::cout << "IsOnConicalSurface : "<< IsOnConicalSurface<Backend, ForInnerSurface>(cone, point) << " : " <<
  //  __LINE__ << std::endl;
  return IsOnConicalSurface<Backend, ForInnerSurface>(cone, point) &&
         (direction.Dot(GetNormal<Backend, ForInnerSurface>(cone, point)) <= 0.);
}

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
    Bool_t isOnSurfaceAndMovingInside = IsMovingInsideConicalSurface<Backend, ForInnerSurface>(cone, point, direction);

    if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingInside, Float_t(0.));
      done |= isOnSurfaceAndMovingInside;
      if (vecCore::MaskFull(done)) return done;
    } else {
      Bool_t insector(false);
      PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, point.x(), point.y(),
                                                                                          insector);
      vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingInside, Float_t(0.));
      done |= (insector && isOnSurfaceAndMovingInside);
      if (vecCore::MaskFull(done)) return done;
    }

  } else {
    Bool_t isOnSurfaceAndMovingOutside =
        IsMovingOutsideConicalSurface<Backend, ForInnerSurface>(cone, point, direction);

    if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingOutside, Float_t(0.));
      done |= isOnSurfaceAndMovingOutside;
      if (vecCore::MaskFull(done)) return done;
    } else {
      Bool_t insector(false);
      PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, point.x(), point.y(),
                                                                                          insector);
      vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingOutside, Float_t(0.));
      done |= (insector && isOnSurfaceAndMovingOutside);
      if (vecCore::MaskFull(done)) return done;
    }
  }

  Float_t pDotV2D = point.x() * direction.x() + point.y() * direction.y();

  Float_t a(0.), b(0.), c(0.);
  Precision fDz = cone.GetDz();
  if (ForInnerSurface) {

    Precision rmin1 = cone.GetRmin1();
    Precision rmin2 = cone.GetRmin2();
    if (rmin1 == rmin2) {
      b = pDotV2D;
      a = direction.Perp2();
      c = point.Perp2() - rmin2 * rmin2;
    } else {

      Precision t = cone.GetTInner();
      Float_t newPz(0.);
      if (cone.GetRmin2() > cone.GetRmin1())
        newPz = (point.z() + fDz + cone.GetInnerConeApex()) * t;
      else
        newPz = (point.z() - fDz - cone.GetInnerConeApex()) * t;

      Float_t dirT = direction.z() * t;
      a            = (direction.x() * direction.x()) + (direction.y() * direction.y()) - dirT * dirT;

      b = pDotV2D - (newPz * dirT);
      c = point.Perp2() - (newPz * newPz);
    }

    Float_t b2 = b * b;
    Float_t ac = a * c;
    if (vecCore::MaskFull(b2 < ac)) return Bool_t(false);
    Float_t d2 = b2 - ac;

    Float_t delta = Sqrt(vecCore::math::Abs(d2));
    if (ForDistToIn) {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b >= 0.), (c / (-b - delta)));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.), (-b + delta) / NonZero(a));
    } else {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b >= 0.), (-b - delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.), (c / (-b + delta)));
    }

    if (vecCore::MaskFull(distance < 0.)) return Bool_t(false);
    Float_t newZ = point.z() + (direction.z() * distance);
    ok           = (Abs(newZ) < fDz);

  } else {

    Precision rmax1 = cone.GetRmax1();
    Precision rmax2 = cone.GetRmax2();
    if (rmax1 == rmax2) {
      b = pDotV2D;
      a = direction.Perp2();
      c = point.Perp2() - rmax2 * rmax2;
    } else {

      Precision t = cone.GetTOuter();
      Float_t newPz(0.);
      if (cone.GetRmax2() > cone.GetRmax1())
        newPz = (point.z() + fDz + cone.GetOuterConeApex()) * t;
      else
        newPz      = (point.z() - fDz - cone.GetOuterConeApex()) * t;
      Float_t dirT = direction.z() * t;
      a            = direction.x() * direction.x() + direction.y() * direction.y() - dirT * dirT;
      b            = pDotV2D - (newPz * dirT);
      c            = point.Perp2() - (newPz * newPz);
    }
    Float_t b2 = b * b;
    Float_t ac = a * c;
    if (vecCore::MaskFull(b2 < ac)) return Bool_t(false);
    Float_t d2    = b2 - ac;
    Float_t delta = Sqrt(vecCore::math::Abs(d2));

    if (ForDistToIn) {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.), (-b - delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.), (c / (-b + delta)));
    } else {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.), (-b + delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b >= 0.), (c / (-b - delta)));
    }

    if (vecCore::MaskFull(distance < 0.)) return Bool_t(false);
    Float_t newZ = point.z() + (direction.z() * distance);
    ok           = (Abs(newZ) < fDz + kHalfConeTolerance);
  }
  vecCore::MaskedAssign(distance, distance < 0., Float_t(kInfLength));

  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    Float_t hitx(0), hity(0), hitz(0);
    Bool_t insector = Backend::kFalse;
    vecCore::MaskedAssign(hitx, distance < kInfLength, point.x() + distance * direction.x());
    vecCore::MaskedAssign(hity, distance < kInfLength, point.y() + distance * direction.y());
    vecCore::MaskedAssign(hitz, distance < kInfLength, point.z() + distance * direction.z());

    PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, hitx, hity, insector);
    ok &= ((insector) && (distance < kInfLength));
  }
  return ok;
}

#if (0)
template <> // typename Backend, bool ForDistToIn, bool ForInnerSurface>
VECGEOM_CUDA_HEADER_BOTH
bool DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, true>(UnplacedCone const &cone,
                                                                                  Vector3D<Precision> const &point,
                                                                                  Vector3D<Precision> const &direction,
                                                                                  Precision &distance)
{

  using namespace ConeUtilities;
  using namespace ConeTypes;
  // typedef typename Backend::precision_v Float_t;
  // typedef typename Backend::bool_v Bool_t;
  using Float_t = Precision;
  using Bool_t  = bool;

  Bool_t ok(false);
  Bool_t done(false);
  /*
  if (ForDistToIn) {
    Bool_t isOnSurfaceAndMovingInside =
        IsMovingInsideConicalSurface<Backend, ForInnerSurface>(cone, point,
                                                               direction);

    if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingInside,
                            Float_t(0.));
      done |= isOnSurfaceAndMovingInside;
      if (vecCore::MaskFull(done))
        return done;
    } else {
      Bool_t insector(false);
      PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, true>(
          cone, point.x(), point.y(), insector);
      vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingInside,
                            Float_t(0.));
      done |= (insector && isOnSurfaceAndMovingInside);
      if (vecCore::MaskFull(done))
        return done;
    }

  } else {

  */
  Bool_t isOnSurfaceAndMovingOutside = IsMovingOutsideConicalSurface<kScalar, true>(cone, point, direction);

  if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    if (isOnSurfaceAndMovingOutside) {
      distance = 0.;
      done     = isOnSurfaceAndMovingOutside;
      return done;
    }
    // vecCore::MaskedAssign(distance, isOnSurfaceAndMovingOutside,
    //                    Float_t(0.));
    // done |= isOnSurfaceAndMovingOutside;
    // if (vecCore::MaskFull(done))
    // return done;
  } else {
    Bool_t insector(false);
    PointInCyclicalSector<kScalar, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, point.x(), point.y(),
                                                                                        insector);
    if (insector && isOnSurfaceAndMovingOutside) {
      distance = 0.;
      done     = (insector && isOnSurfaceAndMovingOutside);
      return done;
    }
    // vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingOutside,
    //                    Float_t(0.));
    // done |= (insector && isOnSurfaceAndMovingOutside);
    // if (vecCore::MaskFull(done))
    // return done;
  }
  //  }

  Float_t pDotV2D = point.x() * direction.x() + point.y() * direction.y();

  Float_t a(0.), b(0.), c(0.);
  Precision fDz = cone.GetDz();
  // if (ForInnerSurface) {

  Precision rmin1 = cone.GetRmin1();
  Precision rmin2 = cone.GetRmin2();
  if (rmin1 == rmin2) {
    b = pDotV2D;
    a = direction.Perp2();
    c = point.Perp2() - rmin2 * rmin2;
  } else {

    Precision t = cone.GetTInner();
    Float_t newPz(0.);
    if (cone.GetRmin2() > cone.GetRmin1())
      newPz = (point.z() + fDz + cone.GetInnerConeApex()) * t;
    else
      newPz = (point.z() - fDz - cone.GetInnerConeApex()) * t;

    Float_t dirT = direction.z() * t;
    a            = (direction.x() * direction.x()) + (direction.y() * direction.y()) - dirT * dirT;

    b = pDotV2D - (newPz * dirT);
    c = point.Perp2() - (newPz * newPz);
  }

  Float_t b2 = b * b;
  Float_t ac = a * c;
  Float_t delta(0.);
  // if (vecCore::MaskFull(b2 < ac))
  // return Bool_t(false);
  if (b2 < ac) {
    return false;
  } else {
    Float_t d2 = b2 - ac;
    delta      = Sqrt(vecCore::math::Abs(d2));
    if (b >= 0.)
      distance = (-b - delta) / NonZero(a);
    else
      distance = (c / (-b + delta));

    // {
    // vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.),
    //                     (-b - delta) / NonZero(a));
    // vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
    //                    (c / (-b + delta)));
    //}
  }
  // if (vecCore::MaskFull(distance < 0.))
  // return Bool_t(false);
  Float_t newZ = point.z() + (direction.z() * distance);
  ok           = (Abs(newZ) < fDz);

  // }
  /*
  else {

    Precision rmax1 = cone.GetRmax1();
    Precision rmax2 = cone.GetRmax2();
    if (rmax1 == rmax2) {
      b = pDotV2D;
      a = direction.Perp2();
      c = point.Perp2() - rmax2 * rmax2;
    } else {

      Precision t = cone.GetTOuter();
      Float_t newPz(0.);
      if (cone.GetRmax2() > cone.GetRmax1())
        newPz = (point.z() + fDz + cone.GetOuterConeApex()) * t;
      else
        newPz = (point.z() - fDz - cone.GetOuterConeApex()) * t;
      Float_t dirT = direction.z() * t;
      a = direction.x() * direction.x() + direction.y() * direction.y() -
          dirT * dirT;
      b = pDotV2D - (newPz * dirT);
      c = point.Perp2() - (newPz * newPz);
    }
    Float_t b2 = b * b;
    Float_t ac = a * c;
    if (vecCore::MaskFull(b2 < ac))
      return Bool_t(false);
    Float_t d2 = b2 - ac;
    Float_t delta = Sqrt(vecCore::math::Abs(d2));

    if (ForDistToIn) {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.),
                            (-b - delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
                            (c / (-b + delta)));
    } else {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
                            (-b + delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b >= 0.),
                            (c / (-b - delta)));
    }

    if (vecCore::MaskFull(distance < 0.))
      return Bool_t(false);
    Float_t newZ = point.z() + (direction.z() * distance);
    ok = (Abs(newZ) < fDz + kHalfConeTolerance);
  }
*/
  // vecCore::MaskedAssign(distance, distance < 0., Float_t(kInfLength));

  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    Float_t hitx(0), hity(0); //, hitz(0);
    Bool_t insector = false;  // = Backend::kFalse;
    // vecCore::MaskedAssign(hitx, distance < kInfLength,
    //                    point.x() + distance * direction.x());
    // vecCore::MaskedAssign(hity, distance < kInfLength,
    //                    point.y() + distance * direction.y());
    // vecCore::MaskedAssign(hitz, distance < kInfLength,
    //   point.z() + distance * direction.z());
    hitx = point.x() + distance * direction.x();
    hity = point.y() + distance * direction.y();
    PointInCyclicalSector<kScalar, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, hitx, hity, insector);
    ok &= ((insector) && (distance < kInfLength));
  }
  return ok;
}

#endif

template <> // typename Backend, bool ForDistToIn, bool ForInnerSurface>
VECGEOM_CUDA_HEADER_BOTH
bool DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, false>(UnplacedCone const &cone,
                                                                                   Vector3D<Precision> const &point,
                                                                                   Vector3D<Precision> const &direction,
                                                                                   Precision &distance)
{

  // std::cout<<"********** Entered at Expected Place ****************" << std::endl;
  using namespace ConeUtilities;
  using namespace ConeTypes;
  // typedef typename Backend::precision_v Float_t;
  // typedef typename Backend::bool_v Bool_t;
  using Float_t = Precision;
  using Bool_t  = bool;

  Bool_t ok(false);
  Bool_t done(false);
  /*
  if (ForDistToIn) {
    Bool_t isOnSurfaceAndMovingInside =
        IsMovingInsideConicalSurface<Backend, ForInnerSurface>(cone, point,
                                                               direction);

    if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
      vecCore::MaskedAssign(distance, isOnSurfaceAndMovingInside,
                            Float_t(0.));
      done |= isOnSurfaceAndMovingInside;
      if (vecCore::MaskFull(done))
        return done;
    } else {
      Bool_t insector(false);
      PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, true>(
          cone, point.x(), point.y(), insector);
      vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingInside,
                            Float_t(0.));
      done |= (insector && isOnSurfaceAndMovingInside);
      if (vecCore::MaskFull(done))
        return done;
    }

  } else {

  */
  Bool_t isOnSurfaceAndMovingOutside = IsMovingOutsideConicalSurface<kScalar, false>(cone, point, direction);

  if (!checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    // vecCore::MaskedAssign(distance, isOnSurfaceAndMovingOutside,Float_t(0.));
    if (isOnSurfaceAndMovingOutside) {
      distance = 0.;
      done     = isOnSurfaceAndMovingOutside;
      return done;
    }
    // done |= isOnSurfaceAndMovingOutside;
    // if (vecCore::MaskFull(done))
    // return done;
  } else {
    Bool_t insector(false);
    PointInCyclicalSector<kScalar, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, point.x(), point.y(),
                                                                                        insector);
    // vecCore::MaskedAssign(distance, insector && isOnSurfaceAndMovingOutside,Float_t(0.));
    if (insector && isOnSurfaceAndMovingOutside) {
      distance = 0.;
      done |= insector && isOnSurfaceAndMovingOutside;
      return done;
    }
    // done |= (insector && isOnSurfaceAndMovingOutside);
    // if (vecCore::MaskFull(done))
    //  return done;
  }
  //}

  Float_t pDotV2D = point.x() * direction.x() + point.y() * direction.y();

  Float_t a(0.), b(0.), c(0.);
  Precision fDz = cone.GetDz();
  /*
  if (ForInnerSurface) {

    Precision rmin1 = cone.GetRmin1();
    Precision rmin2 = cone.GetRmin2();
    if (rmin1 == rmin2) {
      b = pDotV2D;
      a = direction.Perp2();
      c = point.Perp2() - rmin2 * rmin2;
    } else {

      Precision t = cone.GetTInner();
      Float_t newPz(0.);
      if (cone.GetRmin2() > cone.GetRmin1())
        newPz = (point.z() + fDz + cone.GetInnerConeApex()) * t;
      else
        newPz = (point.z() - fDz - cone.GetInnerConeApex()) * t;

      Float_t dirT = direction.z() * t;
      a = (direction.x() * direction.x()) + (direction.y() * direction.y()) -
          dirT * dirT;

      b = pDotV2D - (newPz * dirT);
      c = point.Perp2() - (newPz * newPz);
    }

    Float_t b2 = b * b;
    Float_t ac = a * c;
    if (vecCore::MaskFull(b2 < ac))
      return Bool_t(false);
    Float_t d2 = b2 - ac;

    Float_t delta = Sqrt(vecCore::math::Abs(d2));
    if (ForDistToIn) {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.),
                            (c / (-b - delta)));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
                            (-b + delta) / NonZero(a));
    } else {
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.),
                            (-b - delta) / NonZero(a));
      vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
                            (c / (-b + delta)));
    }

    if (vecCore::MaskFull(distance < 0.))
      return Bool_t(false);
    Float_t newZ = point.z() + (direction.z() * distance);
    ok = (Abs(newZ) < fDz);

  } else {

  */

  Precision rmax1 = cone.GetRmax1();
  Precision rmax2 = cone.GetRmax2();
  if (rmax1 == rmax2) {
    b = pDotV2D;
    a = direction.Perp2();
    c = point.Perp2() - rmax2 * rmax2;
  } else {

    Precision t = cone.GetTOuter();
    Float_t newPz(0.);
    if (cone.GetRmax2() > cone.GetRmax1())
      newPz = (point.z() + fDz + cone.GetOuterConeApex()) * t;
    else
      newPz      = (point.z() - fDz - cone.GetOuterConeApex()) * t;
    Float_t dirT = direction.z() * t;
    a            = direction.x() * direction.x() + direction.y() * direction.y() - dirT * dirT;
    b            = pDotV2D - (newPz * dirT);
    c            = point.Perp2() - (newPz * newPz);
  }
  Float_t b2 = b * b;
  Float_t ac = a * c;
  // std::cout<<"A : "<< a <<" : B : " << b <<" : C : "<< c << " : B2 : "<< b2 << " : AC : "<< ac << std::endl;
  if (vecCore::MaskFull(b2 < ac)) return Bool_t(false);
  Float_t d2    = b2 - ac;
  Float_t delta = 0.;
  if (d2 >= 0.) {
    delta = Sqrt(vecCore::math::Abs(d2));
    if (b < 0.)
      distance = (-b + delta) / NonZero(a);
    else {
      // std::cout<< "*** HERE ****" << std::endl;
      // std::cout<<"Test CAlc : " << (c / (-b - delta)) << std::endl;
      distance = (c / (-b - delta));
    }
  } else {
    return false;
  }
  // Float_t delta = Sqrt(vecCore::math::Abs(d2));
  /*
  if (ForDistToIn) {
    vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b > 0.),
                          (-b - delta) / NonZero(a));
    vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
                          (c / (-b + delta)));
  } else {

  */

  // vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b < 0.),
  //                     (-b + delta) / NonZero(a));
  // vecCore::MaskedAssign(distance, !done && d2 >= 0. && (b >= 0.),
  //                    (c / (-b - delta)));

  //}

  // if (vecCore::MaskFull(distance < 0.))
  // return Bool_t(false);

  Float_t newZ = point.z() + (direction.z() * distance);
  ok           = (Abs(newZ) < fDz + kHalfConeTolerance);

  //}
  // vecCore::MaskedAssign(distance, distance < 0., Float_t(kInfLength));

  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    Float_t hitx(0), hity(0); //, hitz(0);
    Bool_t insector = false;  // = Backend::kFalse;
                              // vecCore::MaskedAssign(hitx, distance < kInfLength,
                              //                     point.x() + distance * direction.x());
    // vecCore::MaskedAssign(hity, distance < kInfLength,
    //                    point.y() + distance * direction.y());
    // vecCore::MaskedAssign(hitz, distance < kInfLength,
    //                    point.z() + distance * direction.z());
    hitx = point.x() + distance * direction.x();
    hity = point.y() + distance * direction.y();
    PointInCyclicalSector<kScalar, ConeTypes::UniversalCone, UnplacedCone, false, true>(cone, hitx, hity, insector);
    ok &= ((insector) && (distance < kInfLength));
  }
  return ok;
}

template <class Backend>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsOnStartPhi(UnplacedCone const &cone,
                                             Vector3D<typename Backend::precision_v> const &point)
{

  return cone.GetWedge().IsOnSurfaceGeneric<Backend>(cone.GetWedge().GetAlong1(), cone.GetWedge().GetNormal1(), point);
}

template <class Backend>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsOnEndPhi(UnplacedCone const &cone,
                                           Vector3D<typename Backend::precision_v> const &point)
{

  return cone.GetWedge().IsOnSurfaceGeneric<Backend>(cone.GetWedge().GetAlong2(), cone.GetWedge().GetNormal2(), point);
}

template <class Backend>
VECGEOM_FORCE_INLINE
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

  //=== Next, check all dimensions of the cone, whether points are inside -->
  // return -1
  vecCore::MaskedAssign(distance, !done, Float_t(-1.0));

  // For points inside z-range, return -1
  Bool_t outside = distz > kHalfConeTolerance || rsq > outerRadOrTol2;
  done |= outside;
  if (vecCore::MaskFull(done)) return;

  if (checkRminTreatment<ConeTypes::UniversalCone>(cone) && !vecCore::MaskFull(outside)) {
    Float_t innerRadOrTol  = GetRadiusOfConeAtPoint<Backend, true>(cone, point.z()) - kConeTolerance;
    Float_t innerRadOrTol2 = innerRadOrTol * innerRadOrTol;
    outside |= rsq < innerRadOrTol2;
    done |= outside;
    if (vecCore::MaskFull(done)) return;
  }
  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone) && !vecCore::MaskEmpty(outside)) {
    Bool_t insector;
    PointInCyclicalSector<Backend, ConeTypes::UniversalCone, UnplacedCone, false, false>(cone, point.x(), point.y(),
                                                                                         insector);
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

  //=== Next step: check if z-plane is the right entry point (both r,phi
  // should be valid at z-plane crossing)
  vecCore::MaskedAssign(distance, !done, Float_t(kInfLength));

  Precision fDz   = cone.GetDz();
  Float_t dirZInv = 1. / NonZero(direction.z());
  vecCore::MaskedAssign(distance, isGoingUp, (fDz - point.z()) * dirZInv);
  vecCore::MaskedAssign(distance, isGoingDown, (-fDz - point.z()) * dirZInv);
  /*
      if(vecCore::MaskFull(isGoingUp))
        vecCore::MaskedAssign(distance, isGoingUp, (fDz - point.z()) * dirZInv);
      else
          vecCore::MaskedAssign(distance, isGoingDown, (-fDz - point.z()) * dirZInv);
  */
  /*
  //if (!checkPhiTreatment<ConeType>(cone) && !checkRminTreatment<ConeType>(cone)){
  if(true){
      Vector3D<Float_t> newPt = point + distance*direction;
      Float_t rho2 = newPt.Perp2();
      Bool_t cond = (rho2 < cone.GetRmax2() && direction.z() > 0.) ||
                    (rho2 < cone.GetRmax1() && direction.z() < 0.) ;
      if(vecCore::MaskFull(cond)) return;
  }
  */

  Float_t dist_rOuter(kInfLength);
  Bool_t ok_outerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, false, false>(
      cone, point, direction, dist_rOuter);

  vecCore::MaskedAssign(distance, !done && ok_outerCone && dist_rOuter < distance, dist_rOuter);

  Float_t dist_rInner(kInfLength);
  if (checkRminTreatment<ConeTypes::UniversalCone>(cone)) {
    Bool_t ok_innerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<Backend, false, true>(
        cone, point, direction, dist_rInner);
    vecCore::MaskedAssign(distance, !done && ok_innerCone && dist_rInner < distance, dist_rInner);
  }

  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {

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
    PhiPlaneTrajectoryIntersection<Backend, ConeTypes::UniversalCone,
                                   SectorType<ConeTypes::UniversalCone>::value != kPi, false>(
        cone.GetAlongPhi1X(), cone.GetAlongPhi1Y(), w.GetNormal1().x(), w.GetNormal1().y(), cone, point, direction,
        dist_phi, ok_phi);
    ok_phi &= dist_phi < distance;
    vecCore::MaskedAssign(distance, !done && ok_phi, dist_phi);
    done |= ok_phi;

    if (SectorType<ConeTypes::UniversalCone>::value != kPi) {
      PhiPlaneTrajectoryIntersection<Backend, ConeTypes::UniversalCone, true, false>(
          cone.GetAlongPhi2X(), cone.GetAlongPhi2Y(), w.GetNormal2().x(), w.GetNormal2().y(), cone, point, direction,
          dist_phi, ok_phi);
      vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
    }
  }
  vecCore::MaskedAssign(distance, distance < 0. && Abs(distance) < kTolerance, Float_t(0.));
}

// Specialized verison of DistanceToOut
template <>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void DistanceToOutKernel<kScalar>(UnplacedCone const &cone, Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction, Precision const & /*stepMax*/,
                                  Precision &distance)
{

  distance = kInfLength;
  using namespace ConeUtilities;
  using namespace ConeTypes;

  using Float_t = Precision;
  using Bool_t  = bool;

  Bool_t done(false);

  // Using this logic will improve performance of Scalar code
  Float_t distz          = Abs(point.z()) - cone.GetDz();
  Float_t rsq            = point.Perp2();
  Float_t outerRadOrTol  = GetRadiusOfConeAtPoint<kScalar, false>(cone, point.z()) + kConeTolerance;
  Float_t outerRadOrTol2 = outerRadOrTol * outerRadOrTol;

  //=== Next, check all dimensions of the cone, whether points are inside -->
  // return -1

  distance = -1.;

  // For points inside z-range, return -1
  Bool_t outside = distz > kHalfConeTolerance || rsq > outerRadOrTol2;
  done |= outside;
  if (vecCore::MaskFull(done)) return;

  if (checkRminTreatment<ConeTypes::UniversalCone>(cone) && !vecCore::MaskFull(outside)) {
    Float_t innerRadOrTol  = GetRadiusOfConeAtPoint<kScalar, true>(cone, point.z()) - kConeTolerance;
    Float_t innerRadOrTol2 = innerRadOrTol * innerRadOrTol;
    outside |= rsq < innerRadOrTol2;
    done |= outside;
    if (vecCore::MaskFull(done)) return;
  }
  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone) && !vecCore::MaskEmpty(outside)) {
    Bool_t insector;
    PointInCyclicalSector<kScalar, ConeTypes::UniversalCone, UnplacedCone, false, false>(cone, point.x(), point.y(),
                                                                                         insector);
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

  //=== Next step: check if z-plane is the right entry point (both r,phi
  // should be valid at z-plane crossing)
  distance = kInfLength;

  Precision fDz   = cone.GetDz();
  Float_t dirZInv = 1. / NonZero(direction.z());

  if (isGoingUp) distance   = (fDz - point.z()) * dirZInv;
  if (isGoingDown) distance = (-fDz - point.z()) * dirZInv;

  /*
  //if (!checkPhiTreatment<ConeType>(cone) && !checkRminTreatment<ConeType>(cone)){
  if(true){
      Vector3D<Float_t> newPt = point + distance*direction;
      Float_t rho2 = newPt.Perp2();
      Bool_t cond = (rho2 < cone.GetRmax2() && direction.z() > 0.) ||
                    (rho2 < cone.GetRmax1() && direction.z() < 0.) ;
      if(vecCore::MaskFull(cond)) return;
  }
  */

  /* Float_t dist_rOuter(kInfLength);
   Bool_t ok_outerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, false>(
       cone, point, direction, dist_rOuter);

   vecCore::MaskedAssign(distance, !done && ok_outerCone && dist_rOuter < distance, dist_rOuter);

   Float_t dist_rInner(kInfLength);
   if (checkRminTreatment<ConeTypes::UniversalCone>(cone)) {
     Bool_t ok_innerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, true>(
         cone, point, direction, dist_rInner);
     vecCore::MaskedAssign(distance, !done && ok_innerCone && dist_rInner < distance, dist_rInner);
   }*/

  Bool_t ok_innerCone(false);
  Float_t dist_rInner(kInfLength);
  if (checkRminTreatment<ConeTypes::UniversalCone>(cone)) {
    ok_innerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, true>(
        cone, point, direction, dist_rInner);

    // vecCore::MaskedAssign(distance, !done && ok_innerCone && dist_rInner < distance, dist_rInner);
  }

  if (ok_innerCone && dist_rInner < distance) {
    distance = dist_rInner;
  } else {

    Float_t dist_rOuter(kInfLength);
    Bool_t ok_outerCone = ConeUtilities::DetectIntersectionAndCalculateDistanceToConicalSurface<kScalar, false, false>(
        cone, point, direction, dist_rOuter);

    vecCore::MaskedAssign(distance, !done && ok_outerCone && dist_rOuter < distance, dist_rOuter);
  }

  if (checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {

    Bool_t isOnStartPhi       = IsOnStartPhi<kScalar>(cone, point);
    Bool_t isOnEndPhi         = IsOnEndPhi<kScalar>(cone, point);
    Vector3D<Float_t> normal1 = cone.GetWedge().GetNormal1();
    Vector3D<Float_t> normal2 = cone.GetWedge().GetNormal2();
    Bool_t cond = (isOnStartPhi && direction.Dot(-normal1) > 0.) || (isOnEndPhi && direction.Dot(-normal2) > 0.);
    vecCore::MaskedAssign(distance, !done && cond, Float_t(0.));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    Float_t dist_phi;
    Bool_t ok_phi;
    Wedge const &w = cone.GetWedge();
    PhiPlaneTrajectoryIntersection<kScalar, ConeTypes::UniversalCone,
                                   SectorType<ConeTypes::UniversalCone>::value != kPi, false>(
        cone.GetAlongPhi1X(), cone.GetAlongPhi1Y(), w.GetNormal1().x(), w.GetNormal1().y(), cone, point, direction,
        dist_phi, ok_phi);
    ok_phi &= dist_phi < distance;
    vecCore::MaskedAssign(distance, !done && ok_phi, dist_phi);
    done |= ok_phi;

    if (SectorType<ConeTypes::UniversalCone>::value != kPi) {
      PhiPlaneTrajectoryIntersection<kScalar, ConeTypes::UniversalCone, true, false>(
          cone.GetAlongPhi2X(), cone.GetAlongPhi2Y(), w.GetNormal2().x(), w.GetNormal2().y(), cone, point, direction,
          dist_phi, ok_phi);
      vecCore::MaskedAssign(distance, ok_phi && dist_phi < distance, dist_phi);
    }
  }
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
           (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance)) && (direction.z() < 0.);
  } else {
    return (rho > (cone.GetSqRmin1() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax1() + kHalfConeTolerance)) &&
           (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance)) &&
           (direction.z() > 0.);
  }
}

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
  if (vecCore::MaskFull(completelyoutside)) {
    return;
  }

  // check on RMIN
  if (ConeTypes::checkRminTreatment<ConeTypes::UniversalCone>(cone)) {
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

  if (ConeTypes::checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {
    Bool_t completelyoutsidephi;
    Bool_t completelyinsidephi;
    cone.GetWedge().GenericKernelForContainsAndInside<Backend, ForInside>(point, completelyinsidephi,
                                                                          completelyoutsidephi);
    completelyoutside |= completelyoutsidephi;
    if (ForInside) completelyinside &= completelyinsidephi;
  }
}

// Specialized Version for Scalar backend
template <>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenericKernelForContainsAndInside<kScalar, true>(UnplacedCone const &cone, Vector3D<Precision> const &point,
                                                      bool &completelyinside, bool &completelyoutside)
{

  using Float_t = Precision;
  using Bool_t  = bool;

  // very fast check on z-height
  Float_t absz      = Abs(point[2]);
  completelyoutside = absz > MakePlusTolerant<true>(cone.GetDz(), kConeTolerance);
  if (true) {
    completelyinside = absz < MakeMinusTolerant<true>(cone.GetDz(), kConeTolerance);
  }
  if (vecCore::MaskFull(completelyoutside)) {
    return;
  }

  // check on RMAX
  Float_t r2 = point.x() * point.x() + point.y() * point.y();
  // calculate cone radius at the z-height of position
  Float_t rmax  = cone.GetOuterSlope() * point.z() + cone.GetOuterOffset();
  Float_t rmax2 = rmax * rmax;

  completelyoutside |= r2 > MakePlusTolerantSquare<true>(rmax, rmax2, kConeTolerance);
  if (true) {
    completelyinside &= r2 < MakeMinusTolerantSquare<true>(rmax, rmax2, kConeTolerance);
  }
  if (vecCore::MaskFull(completelyoutside)) {
    return;
  }

  // check on RMIN
  if (ConeTypes::checkRminTreatment<ConeTypes::UniversalCone>(cone)) {
    Float_t rmin  = cone.GetInnerSlope() * point.z() + cone.GetInnerOffset();
    Float_t rmin2 = rmin * rmin;

    completelyoutside |= r2 <= MakeMinusTolerantSquare<true>(rmin, rmin2, kConeTolerance);
    if (true) {
      completelyinside &= r2 > MakePlusTolerantSquare<true>(rmin, rmin2, kConeTolerance);
    }
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }
  }

  if (ConeTypes::checkPhiTreatment<ConeTypes::UniversalCone>(cone)) {

    Bool_t completelyoutsidephi;
    Bool_t completelyinsidephi;
    cone.GetWedge().GenericKernelForContainsAndInside<kScalar, true>(point, completelyinsidephi, completelyoutsidephi);
    completelyoutside |= completelyoutsidephi;
    if (true) completelyinside &= completelyinsidephi;
  }
}

template <class Backend>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static void Inside(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                   typename Backend::int_v &inside)
{

  const typename Backend::precision_v in(EInside::kInside);
  const typename Backend::precision_v out(EInside::kOutside);
  typename Backend::bool_v inmask(false), outmask(false);
  typename Backend::precision_v result(EInside::kSurface);

  GenericKernelForContainsAndInside<Backend, true>(cone, point, inmask, outmask);

  vecCore::MaskedAssign(result, inmask, in);
  vecCore::MaskedAssign(result, outmask, out);

  // Manual conversion from double to int here is necessary because int_v and
  // precision_v have different number of elements in SIMD vector, so bool_v
  // (mask for precision_v) cannot be cast to mask for inside, which is a
  // different type and does not exist in the current backend system
  for (size_t i = 0; i < vecCore::VectorSize(result); i++)
    vecCore::Set(inside, i, vecCore::Get(result, i));
}

// Speicalized version of Inside, required to make DistanceToOut of Polycone, even more
// faster, coz that call Inside and Contains functions.
template <>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Inside<kScalar>(UnplacedCone const &cone, Vector3D<Precision> const &point, int &inside)
{

  bool completelyinside = false, completelyoutside = false;
  GenericKernelForContainsAndInside<kScalar, true>(cone, point, completelyinside, completelyoutside);
  inside = EInside::kSurface;
  if (completelyinside)
    inside = EInside::kInside;
  else if (completelyoutside)
    inside = EInside::kOutside;
  return;
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
           (point.z() < (fDz + kHalfConeTolerance)) && (point.z() > (fDz - kHalfConeTolerance)) && (direction.z() > 0.);
  } else {
    return (rho > (cone.GetSqRmin1() - kHalfConeTolerance)) && (rho < (cone.GetSqRmax1() + kHalfConeTolerance)) &&
           (point.z() < (-fDz + kHalfConeTolerance)) && (point.z() > (-fDz - kHalfConeTolerance)) &&
           (direction.z() < 0.);
  }
}

// This function will be useful to detect the point on the circular edge
template <class Backend, bool ForTopRing, bool ForInnerRing>
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsOnRing(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point)
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

  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::bool_v &contains)
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    ConeUtilities::GenericKernelForContainsAndInside<Backend, false>(cone, point, unused, outside);
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

  template <typename Backend, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedCone const &cone,
                                                Vector3D<typename Backend::precision_v> const &point,
                                                typename Backend::bool_v &completelyinside,
                                                typename Backend::bool_v &completelyoutside)
  {

    ConeUtilities::GenericKernelForContainsAndInside<Backend, ForInside>(cone, point, completelyinside,
                                                                         completelyoutside);
  }
  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedCone const &cone, Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point, typename Backend::int_v &inside)
  {

    Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    ConeUtilities::Inside<Backend>(cone, localPoint, inside);
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

    //=== Next, check all dimensions of the cone, whether points are inside -->
    // return -1

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

    //=== Next step: check if z-plane is the right entry point (both r,phi
    // should be valid at z-plane crossing)
    vecCore::MaskedAssign(distance, !done, Float_t(kInfLength));

    distz /= NonZero(Abs(dir.z()));

    Bool_t isOnZPlaneAndMovingInside(false);

    Bool_t isGoingUp = dir.z() > 0.;
    // isOnZPlaneAndMovingInside = ((isGoingUp && point.z() < 0. && Abs(distz) <
    // kHalfTolerance) ||
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
    Bool_t isHittingTopPlane    = (point.z() >= innerZTol) && (r2 <= cone.GetSqRmax2Tol());
    Bool_t isHittingBottomPlane = (point.z() <= -innerZTol) && (r2 <= cone.GetSqRmax1Tol());
    Bool_t okz                  = (isHittingTopPlane || isHittingBottomPlane);

    if (checkRminTreatment<ConeType>(cone)) {
      isHittingTopPlane &= (r2 >= cone.GetSqRmin2Tol());
      isHittingBottomPlane &= (r2 >= cone.GetSqRmin1Tol());
      okz &= ((isHittingTopPlane || isHittingBottomPlane));
    }

    if (checkPhiTreatment<ConeType>(cone)) {
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
  template <class Backend>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedCone const &cone, Vector3D<typename Backend::precision_v> point,
                            Vector3D<typename Backend::precision_v> direction,
                            typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    ConeUtilities::DistanceToOutKernel<Backend>(cone, point, direction, stepMax, distance);
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

    // Next, check all dimensions of the cone, whether points are inside -->
    // return -1
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

    // Once it is checked that the point is inside or not, safety can be set to
    // 0.
    // This will serve the case that the point is on the surface. So no need to
    // check
    // that the point is really on surface.
    vecCore::MaskedAssign(safety, !done, Float_t(0.));

    // Now if the point is neither inside nor on surface, then it should be
    // outside
    // and the safety should be set to some finite value, which is done by below
    // logic

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

    //=== Next, check all dimensions of the cone, whether points are inside -->
    // return -1
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

    // Once it is checked that the point is inside or not, safety can be set to
    // 0.
    // This will serve the case that the point is on the surface. So no need to
    // check
    // that the point is really on surface.
    vecCore::MaskedAssign(safety, !done, Float_t(0.));

    // Now if the point is neither outside nor on surface, then it should be
    // inside
    // and the safety should be set to some finite value, which is done by below
    // logic

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

  // This function will be used to calculate normal to the closest surface when
  // the point is
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
    // Changing signs of normal component to negative, because Wedge gave inward
    // normal
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

    vecCore::MaskedAssign(norm, (distMin == distRMin), ConeUtilities::GetNormal<Backend, true>(cone, p));
    vecCore::MaskedAssign(norm, (distMin == distRMax), ConeUtilities::GetNormal<Backend, false>(cone, p));

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

    ConeUtilities::GenericKernelForContainsAndInside<Backend, true>(cone, p, compleltelyinside, completelyoutside);
    Bool_t isOnSurface = !compleltelyinside && !completelyoutside;

    Float_t noSurfaces(0.);
    Bool_t isOnTopZPlane           = ConeUtilities::IsOnTopZPlane<Backend>(cone, p);
    Bool_t isOnBottomZPlane        = ConeUtilities::IsOnBottomZPlane<Backend>(cone, p);
    Bool_t isOnInnerConicalSurface = ConeUtilities::IsOnInnerConicalSurface<Backend>(cone, p);
    Bool_t isOnOuterConicalSurface = ConeUtilities::IsOnOuterConicalSurface<Backend>(cone, p);
    Bool_t isOnStartPhi            = ConeUtilities::IsOnStartPhi<Backend>(cone, p);
    Bool_t isOnEndPhi              = ConeUtilities::IsOnEndPhi<Backend>(cone, p);

    Vector3D<Float_t> innerConicalNormal = ConeUtilities::GetNormal<Backend, true>(cone, p);
    Vector3D<Float_t> outerConicalNormal = ConeUtilities::GetNormal<Backend, false>(cone, p);

    Vector3D<Float_t> startPhiNormal = cone.GetWedge().GetNormal1();
    Vector3D<Float_t> endPhiNormal   = cone.GetWedge().GetNormal2();

    Vector3D<Float_t> approxNorm = ApproxSurfaceNormalKernel<Backend>(cone, p);
    // vecCore::MaskedAssign(normal,!isOnSurface,
    // ApproxSurfaceNormalKernel<Backend>(cone, p));
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

}; // end struct
}
} // End global namespace

#endif /* VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_ */
