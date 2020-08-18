/*
 * HypeUtilities.h
 *
 *  Created on: Jun 19, 2017
 *      Author: rsehgal
 */

#ifndef VOLUMES_HYPEUTILITIES_H_
#define VOLUMES_HYPEUTILITIES_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/shapetypes/HypeTypes.h"
#include <cstdio>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedHype;
template <typename T>
struct HypeStruct;

namespace HypeUtilities {
using UnplacedStruct_t = HypeStruct<Precision>;
template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsCompletelyOutside(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point)
{
  using namespace ::vecgeom::HypeTypes;
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Real_v r2    = point.Perp2();
  Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());

  Bool_v completelyoutside = (Abs(point.z()) > (hype.fDz + hype.zToleranceLevel));
  if (vecCore::MaskFull(completelyoutside)) return completelyoutside;
  completelyoutside |= (r2 > oRad2 + hype.outerRadToleranceLevel);
  if (vecCore::MaskFull(completelyoutside)) return completelyoutside;

  // if (hype.InnerSurfaceExists()) {
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    Real_v iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
    completelyoutside |= (r2 < (iRad2 - hype.innerRadToleranceLevel));
  }
  return completelyoutside;
}

template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsCompletelyInside(UnplacedStruct_t const &hype, Vector3D<Real_v> const &point)
{
  using namespace ::vecgeom::HypeTypes;
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Real_v r2    = point.Perp2();
  Real_v oRad2 = (hype.fRmax2 + hype.fTOut2 * point.z() * point.z());

  Bool_v completelyinside =
      (Abs(point.z()) < (hype.fDz - hype.zToleranceLevel)) && (r2 < oRad2 - hype.outerRadToleranceLevel);
  // if (hype.InnerSurfaceExists()) {
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    Real_v iRad2 = (hype.fRmin2 + hype.fTIn2 * point.z() * point.z());
    completelyinside &= (r2 > (iRad2 + hype.innerRadToleranceLevel));
  }
  return completelyinside;
}

template <typename Real_v, bool ForInnerRad>
VECCORE_ATT_HOST_DEVICE
Real_v RadiusHypeSq(UnplacedStruct_t const &hype, Real_v z)
{

  if (ForInnerRad)
    return (hype.fRmin2 + hype.fTIn2 * z * z);
  else
    return (hype.fRmax2 + hype.fTOut2 * z * z);
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointMovingInsideOuterSurface(UnplacedStruct_t const &hype,
                                                                 Vector3D<Real_v> const &point,
                                                                 Vector3D<Real_v> const &direction)
{
  Real_v pz = point.z();
  Real_v vz = direction.z();
  vecCore__MaskedAssignFunc(vz, pz < Real_v(0.), -vz);
  vecCore__MaskedAssignFunc(pz, pz < Real_v(0.), -pz);
  return ((point.x() * direction.x() + point.y() * direction.y() - pz * hype.fTOut2 * vz) < Real_v(0.));
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointMovingInsideInnerSurface(UnplacedStruct_t const &hype,
                                                                 Vector3D<Real_v> const &point,
                                                                 Vector3D<Real_v> const &direction)
{
  Real_v pz = point.z();
  Real_v vz = direction.z();

  vecCore__MaskedAssignFunc(vz, pz < Real_v(0.), -vz);
  vecCore__MaskedAssignFunc(pz, pz < Real_v(0.), -pz);

  // Precision tanInnerStereo2 = hype.GetTIn2();
  return ((point.x() * direction.x() + point.y() * direction.y() - pz * hype.fTIn2 * vz) > Real_v(0.));
}

template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnSurfaceAndMovingInside(UnplacedStruct_t const &hype,
                                                                 Vector3D<Real_v> const &point,
                                                                 Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Bool_v innerHypeSurf(false), outerHypeSurf(false), zSurf(false);
  Bool_v done(false);
  Real_v rho2  = point.Perp2();
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  Real_v radO2 = RadiusHypeSq<Real_v, false>(hype, point.z());

  Bool_v in(false);
  zSurf = ((rho2 - hype.fEndOuterRadius2) < kTolerance) && ((hype.fEndInnerRadius2 - rho2) < kTolerance) &&
          (Abs(Abs(point.z()) - hype.fDz) < kTolerance);
  in |= (zSurf && (point.z() * direction.z() < Real_v(0.)));

  done |= zSurf;
  if (vecCore::MaskFull(done)) return in;

  outerHypeSurf |= (!zSurf && (Abs((radO2) - (rho2)) < hype.outerRadToleranceLevel));
  in |= (!done && outerHypeSurf && IsPointMovingInsideOuterSurface<Real_v>(hype, point, direction));

  // if (hype.InnerSurfaceExists()) {
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    done |= (!zSurf && outerHypeSurf);
    if (vecCore::MaskFull(done)) return in;

    innerHypeSurf |= (!zSurf && !outerHypeSurf && (Abs((radI2) - (rho2)) < hype.innerRadToleranceLevel));
    in |= (!done && !zSurf && innerHypeSurf && IsPointMovingInsideInnerSurface<Real_v>(hype, point, direction));
    done |= (!zSurf && innerHypeSurf);
    if (vecCore::MaskFull(done)) return in;
  }
  return in;
}

template <typename Real_v, typename hypeType, bool ForDistToIn>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> GetPointOfIntersectionWithZPlane(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point,
                                                                  Vector3D<Real_v> const &direction, Real_v &zDist)
{
  using namespace ::vecgeom::HypeTypes;
  zDist = (Sign(ForDistToIn ? point.z() : direction.z()) * hype.fDz - point.z()) / NonZero(direction.z());

  auto r2 = (point + zDist * direction).Perp2();
  // if (!hype.InnerSurfaceExists())
  if (!checkInnerSurfaceTreatment<hypeType>(hype))
    return (r2 < hype.fEndOuterRadius2);
  else
    return ((r2 < hype.fEndOuterRadius2) && (r2 > hype.fEndInnerRadius2));
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointMovingOutsideOuterSurface(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point,
                                                                  Vector3D<Real_v> const &direction)
{

  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Bool_v out(false);

  Real_v pz = point.z();
  Real_v vz = direction.z();
  vecCore__MaskedAssignFunc(pz, vz < Real_v(0.), -pz);
  vecCore__MaskedAssignFunc(vz, vz < Real_v(0.), -vz);
  Vector3D<Real_v> normHere(point.x(), point.y(), -point.z() * hype.fTOut2);
  out = (normHere.Dot(direction) > Real_v(0.));
  return out;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointMovingOutsideInnerSurface(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point,
                                                                  Vector3D<Real_v> const &direction)
{

  Real_v pz = point.z();
  Real_v vz = direction.z();
  vecCore__MaskedAssignFunc(pz, vz < Real_v(0.), -pz);
  vecCore__MaskedAssignFunc(vz, vz < Real_v(0.), -vz);
  Vector3D<Real_v> normHere(-point.x(), -point.y(), point.z() * hype.fTIn2);
  return (normHere.Dot(direction) > Real_v(0.));
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnOuterSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                                       Vector3D<Real_v> const &point,
                                                                       Vector3D<Real_v> const &direction)
{

  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Real_v rho2  = point.x() * point.x() + point.y() * point.y();
  Real_v absZ  = Abs(point.z());
  Real_v radO2 = RadiusHypeSq<Real_v, false>(hype, point.z());
  Bool_v out(false), outerHypeSurf(false);
  outerHypeSurf = (Abs((radO2) - (rho2)) < hype.outerRadToleranceLevel) && (absZ >= Real_v(0.)) && (absZ < hype.fDz);
  out           = outerHypeSurf && IsPointMovingOutsideOuterSurface<Real_v>(hype, point, direction);
  return out;
}

template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnInnerSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                                       Vector3D<Real_v> const &point,
                                                                       Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Real_v rho2  = point.x() * point.x() + point.y() * point.y();
  Real_v absZ  = Abs(point.z());
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  Bool_v out(false), innerHypeSurf(false);
  // if (hype.InnerSurfaceExists()) {
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    innerHypeSurf = (Abs((radI2) - (rho2)) < hype.innerRadToleranceLevel) && (absZ >= Real_v(0.)) && (absZ < hype.fDz);
    out           = innerHypeSurf && HypeUtilities::IsPointMovingOutsideInnerSurface<Real_v>(hype, point, direction);
  }
  return out;
}

template <typename Real_v, typename hypeType>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnSurfaceAndMovingOutside(UnplacedStruct_t const &hype,
                                                                  Vector3D<Real_v> const &point,
                                                                  Vector3D<Real_v> const &direction)
{
  using namespace ::vecgeom::HypeTypes;
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Bool_v innerHypeSurf(false), outerHypeSurf(false), zSurf(false);
  Bool_v done(false);

  Real_v rho2  = point.x() * point.x() + point.y() * point.y();
  Real_v radI2 = RadiusHypeSq<Real_v, true>(hype, point.z());
  Real_v radO2 = RadiusHypeSq<Real_v, false>(hype, point.z());

  Bool_v out(false);
  //  zSurf = ((hype.fEndOuterRadius2 - rho2) < kTolerance) && ((rho2 - hype.fEndInnerRadius2) < kTolerance) &&
  //          (Abs(Abs(point.z()) - hype.fDz) < kTolerance);
  zSurf = ((rho2 - hype.fEndOuterRadius2) < kTolerance) && ((hype.fEndInnerRadius2 - rho2) < kTolerance) &&
          (Abs(Abs(point.z()) - hype.fDz) < kTolerance);

  out |= (zSurf && (point.z() * direction.z() > Real_v(0.)));
  // done |= zSurf;
  done = out;
  if (vecCore::MaskFull(done)) return out;

  outerHypeSurf |= !done && (Abs(radO2 - rho2) < hype.outerRadToleranceLevel);
  // out |= (!done && !zSurf && outerHypeSurf &&
  out |= (outerHypeSurf && HypeUtilities::IsPointMovingOutsideOuterSurface<Real_v>(hype, point, direction));

  // done |= (!zSurf && outerHypeSurf);
  done |= out;
  if (vecCore::MaskFull(done)) return out;

  // if (hype.InnerSurfaceExists()) {
  if (checkInnerSurfaceTreatment<hypeType>(hype)) {
    // innerHypeSurf |= (!done && !zSurf && !outerHypeSurf && (Abs((radI2) - (rho2)) < hype.innerRadToleranceLevel));
    innerHypeSurf |= (!done && (Abs(radI2 - rho2) < hype.innerRadToleranceLevel));
    // out |= (!done && !zSurf && innerHypeSurf &&
    out |= (innerHypeSurf && HypeUtilities::IsPointMovingOutsideInnerSurface<Real_v>(hype, point, direction));
    // done |= (!zSurf && innerHypeSurf);
    done |= out;
    if (vecCore::MaskFull(done)) return out;
  }

  return out;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
Real_v ApproxDistOutside(Real_v pr, Real_v pz, Precision r0, Precision tanPhi)
{
  Real_v r1 = Sqrt(r0 * r0 + tanPhi * tanPhi * pz * pz);
  Real_v z1 = pz;
  Real_v r2 = pr;
  Real_v z2 = Sqrt((pr * pr - r0 * r0) / (tanPhi * tanPhi));
  Real_v dz = z2 - z1;
  Real_v dr = r2 - r1;
  Real_v r3 = Sqrt(dr * dr + dz * dz);
  auto mask = r3 < vecCore::NumericLimits<Real_v>::Min();
  return vecCore::Blend(mask, (r2 - r1), (r2 - r1) * dz / r3);
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE
Real_v ApproxDistInside(Real_v pr, Real_v pz, Precision r0, Precision tan2Phi)
{
  using Bool_v = typename vecCore::Mask_v<Real_v>;
  Bool_v done(false);
  Real_v ret(0.);
  Real_v tan2Phi_v(tan2Phi);
  vecCore__MaskedAssignFunc(ret, (tan2Phi_v < vecCore::NumericLimits<Real_v>::Min()), r0 - pr);
  done |= (tan2Phi_v < vecCore::NumericLimits<Real_v>::Min());
  if (vecCore::MaskFull(done)) return ret;

  Real_v rh  = Sqrt(r0 * r0 + pz * pz * tan2Phi_v);
  Real_v dr  = -rh;
  Real_v dz  = pz * tan2Phi_v;
  Real_v len = Sqrt(dr * dr + dz * dz);

  vecCore__MaskedAssignFunc(ret, !done, Abs((pr - rh) * dr) / len);
  return ret;
}

} // namespace HypeUtilities

/* This class  is basically constructed to allow partial specialization
 * for Scalar Backend.
 */
template <class Real_v, bool ForDistToIn, bool ForInnerSurface>
class HypeHelpers {

public:
  HypeHelpers() {}
  ~HypeHelpers() {}

  VECCORE_ATT_HOST_DEVICE
  static typename vecCore::Mask_v<Real_v> GetPointOfIntersectionWithHyperbolicSurface(HypeStruct<Precision> const &hype,
                                                                                      Vector3D<Real_v> const &point,
                                                                                      Vector3D<Real_v> const &direction,
                                                                                      Real_v &dist)
  {

    using Bool_v = typename vecCore::Mask_v<Real_v>;

    if (ForInnerSurface) {
      Real_v a     = direction.Perp2() - hype.fTIn2 * direction.z() * direction.z();
      Real_v b     = (direction.x() * point.x() + direction.y() * point.y() - hype.fTIn2 * direction.z() * point.z());
      Real_v c     = point.Perp2() - hype.fTIn2 * point.z() * point.z() - hype.fRmin2;
      Bool_v exist = (b * b - a * c > Real_v(0.));
      if (ForDistToIn) {
        vecCore__MaskedAssignFunc(dist, exist && b < Real_v(0.), ((-b + Sqrt(b * b - a * c)) / (a)));
        vecCore__MaskedAssignFunc(dist, exist && b >= Real_v(0.), ((c) / (-b - Sqrt(b * b - a * c))));

      } else {
        vecCore__MaskedAssignFunc(dist, exist && b > Real_v(0.), ((-b - Sqrt(b * b - a * c)) / (a)));
        vecCore__MaskedAssignFunc(dist, exist && b <= Real_v(0.), ((c) / (-b + Sqrt(b * b - a * c))));
      }

    } else {
      Real_v a     = direction.Perp2() - hype.fTOut2 * direction.z() * direction.z();
      Real_v b     = (direction.x() * point.x() + direction.y() * point.y() - hype.fTOut2 * direction.z() * point.z());
      Real_v c     = point.Perp2() - hype.fTOut2 * point.z() * point.z() - hype.fRmax2;
      Bool_v exist = (b * b - a * c > Real_v(0.));
      if (ForDistToIn) {
        vecCore__MaskedAssignFunc(dist, exist && b >= Real_v(0.), ((-b - Sqrt(b * b - a * c)) / (a)));
        vecCore__MaskedAssignFunc(dist, exist && b < Real_v(0.), ((c) / (-b + Sqrt(b * b - a * c))));
      } else {
        vecCore__MaskedAssignFunc(dist, exist && b < Real_v(0.), ((-b + Sqrt(b * b - a * c)) / (a)));
        vecCore__MaskedAssignFunc(dist, exist && b >= Real_v(0.), ((c) / (-b - Sqrt(b * b - a * c))));
      }
    }

    vecCore__MaskedAssignFunc(dist, dist < Real_v(0.), InfinityLength<Real_v>());

    Real_v newPtZ = point.z() + dist * direction.z();

    return (Abs(newPtZ) <= hype.fDz);
  }
};

template <bool ForDistToIn, bool ForInnerSurface>
class HypeHelpers<Precision, ForDistToIn, ForInnerSurface> {
public:
  HypeHelpers() {}
  ~HypeHelpers() {}

  VECCORE_ATT_HOST_DEVICE
  static bool GetPointOfIntersectionWithHyperbolicSurface(HypeStruct<Precision> const &hype,
                                                          Vector3D<Precision> const &point,
                                                          Vector3D<Precision> const &direction, Precision &dist)
  {
    if (ForInnerSurface) {
      Precision a = direction.Perp2() - hype.fTIn2 * direction.z() * direction.z();
      Precision b = (direction.x() * point.x() + direction.y() * point.y() - hype.fTIn2 * direction.z() * point.z());
      Precision c = point.Perp2() - hype.fTIn2 * point.z() * point.z() - hype.fRmin2;
      bool exist  = (b * b - a * c > 0.);
      if (exist) {
        if (ForDistToIn) {
          if (b < 0.)
            dist = ((-b + Sqrt(b * b - a * c)) / (a));
          else
            dist = ((c) / (-b - Sqrt(b * b - a * c)));
        } else {

          if (b > 0.)
            dist = ((-b - Sqrt(b * b - a * c)) / (a));
          else
            dist = ((c) / (-b + Sqrt(b * b - a * c)));
        }
      } else
        return false;

    } else {
      Precision a = direction.Perp2() - hype.fTOut2 * direction.z() * direction.z();
      Precision b = (direction.x() * point.x() + direction.y() * point.y() - hype.fTOut2 * direction.z() * point.z());
      Precision c = point.Perp2() - hype.fTOut2 * point.z() * point.z() - hype.fRmax2;
      bool exist  = (b * b - a * c > 0.);

      if (exist) {
        if (ForDistToIn) {
          if (b >= 0.)
            dist = ((-b - Sqrt(b * b - a * c)) / (a));
          else
            dist = ((c) / (-b + Sqrt(b * b - a * c)));
        } else {

          if (b < 0.)
            dist = ((-b + Sqrt(b * b - a * c)) / a);
          else
            dist = (c / (-b - Sqrt(b * b - a * c)));
        }
      } else
        return false;
    }
    if (dist < 0.) dist = kInfLength;
    // vecCore__MaskedAssignFunc(dist, dist < 0.0, InfinityLength<Real_v>());

    Precision newPtZ = point.z() + dist * direction.z();

    return (Abs(newPtZ) <= hype.fDz);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_HYPEUTILITIES_H_ */
