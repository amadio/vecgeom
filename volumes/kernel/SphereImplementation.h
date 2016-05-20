
/// @file SphereImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/UnplacedSphere.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(SphereImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {


class PlacedSphere;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct SphereImplementation {


  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

using PlacedShape_t = PlacedSphere;
using UnplacedShape_t = UnplacedSphere;

VECGEOM_CUDA_HEADER_BOTH
static void PrintType() {
   printf("SpecializedSphere<%i, %i>", transCodeT, rotCodeT);
}

template <typename Stream>
static void PrintType(Stream &s) {
   s << "SpecializedSphere<" << transCodeT << "," << rotCodeT << ","
     << ">";
 }

template <typename Stream>
static void PrintImplementationType(Stream &s) {
  s << "SphereImplementation<" << transCodeT << "," << rotCodeT << ">";
}

template <typename Stream>
static void PrintUnplacedType(Stream &s) {
  s << "UnplacedSphere";
}

// Some New Helper functions
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsPointOnInnerRadius(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  Precision innerRad2 = unplaced.GetInnerRadius() * unplaced.GetInnerRadius();
  return ((point.Mag2() >= (innerRad2 - kTolerance)) && (point.Mag2() <= (innerRad2 + kTolerance)));
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsPointOnOuterRadius(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  Precision outerRad2 = unplaced.GetOuterRadius() * unplaced.GetOuterRadius();
  return ((point.Mag2() >= (outerRad2 - kTolerance)) && (point.Mag2() <= (outerRad2 + kTolerance)));
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsPointOnStartPhi(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  return unplaced.GetWedge().IsOnSurfaceGeneric<Backend>(unplaced.GetWedge().GetAlong1(),
                                                         unplaced.GetWedge().GetNormal1(), point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsPointOnEndPhi(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  return unplaced.GetWedge().IsOnSurfaceGeneric<Backend>(unplaced.GetWedge().GetAlong2(),
                                                         unplaced.GetWedge().GetNormal2(), point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsPointOnStartTheta(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  return unplaced.GetThetaCone().IsOnSurfaceGeneric<Backend, true>(point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsPointOnEndTheta(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  return unplaced.GetThetaCone().IsOnSurfaceGeneric<Backend, false>(point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void UnplacedContains(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::bool_v &inside);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void Contains(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void Inside(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside);

template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GenericKernelForContainsAndInside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void DistanceToIn(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void DistanceToOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void DistanceToOutKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void SafetyToIn(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void SafetyToOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety);

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void ContainsKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside);


template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void InsideKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void DistanceToInKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

template <class Backend, bool DistToIn>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void GetMinDistFromPhi(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    Vector3D<typename Backend::precision_v> const &localDir,
    typename Backend::bool_v &done,
    typename Backend::precision_v &distance);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void SafetyToInKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void SafetyToOutKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void Normal(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal,
    typename Backend::bool_v &valid);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static void NormalKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal,
    typename Backend::bool_v &valid);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static Vector3D<typename Backend::precision_v> ApproxSurfaceNormalKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point);

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsCompletelyOutside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  Float_t rad = localPoint.Mag();
  Precision fRmax = unplaced.GetOuterRadius();
  Precision fRmin = unplaced.GetInnerRadius();
  Bool_t outsideRadiusRange = (rad > (fRmax + kTolerance)) || (rad < (fRmin - kTolerance));
  Bool_t outsidePhiRange(false), insidePhiRange(false);
  unplaced.GetWedge().GenericKernelForContainsAndInside<Backend, true>(localPoint, insidePhiRange, outsidePhiRange);
  Bool_t outsideThetaRange = unplaced.GetThetaCone().IsCompletelyOutside<Backend>(localPoint);
  Bool_t completelyoutside = outsideRadiusRange || outsidePhiRange || outsideThetaRange;
  return completelyoutside;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsCompletelyInside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  Float_t rad = localPoint.Mag();
  Precision fRmax = unplaced.GetOuterRadius();
  Precision fRmin = unplaced.GetInnerRadius();
  Bool_t insideRadiusRange = (rad < (fRmax - kTolerance)) && (rad > (fRmin + kTolerance));
  Bool_t outsidePhiRange(false), insidePhiRange(false);
  unplaced.GetWedge().GenericKernelForContainsAndInside<Backend, true>(localPoint, insidePhiRange, outsidePhiRange);
  Bool_t insideThetaRange = unplaced.GetThetaCone().IsCompletelyInside<Backend>(localPoint);
  Bool_t completelyinside = insideRadiusRange && insidePhiRange && insideThetaRange;
  return completelyinside;
  }

template <class Backend, bool ForInnerRadius, bool MovingOut>
VECGEOM_CUDA_HEADER_BOTH
static typename Backend::bool_v IsPointOnRadialSurfaceAndMovingOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &dir) {

  if (MovingOut) {
    if (ForInnerRadius) {
      return IsPointOnInnerRadius<Backend>(unplaced, point) && (dir.Dot(-point) > 0.);
    } else {
      return IsPointOnOuterRadius<Backend>(unplaced, point) && (dir.Dot(point) > 0.);
    }
  } else {
    if (ForInnerRadius) {
      return IsPointOnInnerRadius<Backend>(unplaced, point) && (dir.Dot(-point) < 0.);
    } else
      return IsPointOnOuterRadius<Backend>(unplaced, point) && (dir.Dot(point) < 0.);
  }
}

template <class Backend, bool MovingOut>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
static typename Backend::bool_v IsPointOnSurfaceAndMovingOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &dir) {

  typedef typename Backend::bool_v Bool_t;

  Bool_t tempOuterRad = IsPointOnRadialSurfaceAndMovingOut<Backend, false, MovingOut>(unplaced, point, dir);
  Bool_t tempInnerRad(false), tempStartPhi(false), tempEndPhi(false), tempStartTheta(false), tempEndTheta(false);
  if (unplaced.GetInnerRadius())
    tempInnerRad = IsPointOnRadialSurfaceAndMovingOut<Backend, true, MovingOut>(unplaced, point, dir);
  if (unplaced.GetDeltaPhiAngle() < (kTwoPi - kHalfTolerance)) {
    tempStartPhi = unplaced.GetWedge().IsPointOnSurfaceAndMovingOut<Backend, true, MovingOut>(point, dir);
    tempEndPhi = unplaced.GetWedge().IsPointOnSurfaceAndMovingOut<Backend, false, MovingOut>(point, dir);
  }
  if (unplaced.GetDeltaThetaAngle() < (kPi - kHalfTolerance)) {
    tempStartTheta = unplaced.GetThetaCone().IsPointOnSurfaceAndMovingOut<Backend, true, MovingOut>(point, dir);
    tempEndTheta = unplaced.GetThetaCone().IsPointOnSurfaceAndMovingOut<Backend, false, MovingOut>(point, dir);
  }

  Bool_t isPointOnSurfaceAndMovingOut =
      ((tempOuterRad || tempInnerRad) && unplaced.GetWedge().Contains<Backend>(point) &&
       unplaced.GetThetaCone().Contains<Backend>(point)) ||
      ((tempStartPhi || tempEndPhi) && (point.Mag2() >= unplaced.GetInnerRadius() * unplaced.GetInnerRadius()) &&
       (point.Mag2() <= unplaced.GetOuterRadius() * unplaced.GetOuterRadius()) &&
       unplaced.GetThetaCone().Contains<Backend>(point)) ||
      ((tempStartTheta || tempEndTheta) && (point.Mag2() >= unplaced.GetInnerRadius() * unplaced.GetInnerRadius()) &&
       (point.Mag2() <= unplaced.GetOuterRadius() * unplaced.GetOuterRadius()) &&
       unplaced.GetWedge().Contains<Backend>(point));

  return isPointOnSurfaceAndMovingOut;
}
};

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH void SphereImplementation<transCodeT, rotCodeT>::Normal(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal,
    typename Backend::bool_v &valid) {

    NormalKernel<Backend>(unplaced, point, normal, valid);
}

/* This function should be called from NormalKernel, only for the
 * cases when the point is not on the surface and one want to calculate
 * the SurfaceNormal.
 *
 * Algo : Find the boundary which is closest to the point,
 * and return the normal to that boundary.
 *
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<typename Backend::precision_v> SphereImplementation<transCodeT, rotCodeT>::ApproxSurfaceNormalKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;

  Vector3D<Float_t> norm(0., 0., 0.);
  Float_t radius = point.Mag();
  Float_t distRMax = Abs(radius - unplaced.GetOuterRadius());
  Float_t distRMin = Abs(unplaced.GetInnerRadius() - radius);
  MaskedAssign(distRMax < 0., kInfinity, &distRMax);
  MaskedAssign(distRMin < 0., kInfinity, &distRMin);
  Float_t distMin = Min(distRMin, distRMax);

  Float_t distPhi1 =
      point.x() * unplaced.GetWedge().GetNormal1().x() + point.y() * unplaced.GetWedge().GetNormal1().y();
  Float_t distPhi2 =
      point.x() * unplaced.GetWedge().GetNormal2().x() + point.y() * unplaced.GetWedge().GetNormal2().y();
  MaskedAssign(distPhi1 < 0., kInfinity, &distPhi1);
  MaskedAssign(distPhi2 < 0., kInfinity, &distPhi2);
  distMin = Min(distMin, Min(distPhi1, distPhi2));

  Float_t rho = point.Perp();
  Float_t distTheta1 =
      unplaced.GetThetaCone().DistanceToLine<Backend>(unplaced.GetThetaCone().GetSlope1(), rho, point.z());
  Float_t distTheta2 =
      unplaced.GetThetaCone().DistanceToLine<Backend>(unplaced.GetThetaCone().GetSlope2(), rho, point.z());
  MaskedAssign(distTheta1 < 0., kInfinity, &distTheta1);
  MaskedAssign(distTheta2 < 0., kInfinity, &distTheta2);
  distMin = Min(distMin, Min(distTheta1, distTheta2));

  MaskedAssign(distMin == distRMax, point.Unit(), &norm);
  MaskedAssign(distMin == distRMin, -point.Unit(), &norm);

  Vector3D<Float_t> normal1 = unplaced.GetWedge().GetNormal1();
  Vector3D<Float_t> normal2 = unplaced.GetWedge().GetNormal2();
  MaskedAssign(distMin == distPhi1, -normal1, &norm);
  MaskedAssign(distMin == distPhi2, -normal2, &norm);

  MaskedAssign(distMin == distTheta1, norm + unplaced.GetThetaCone().GetNormal1<Backend>(point), &norm);
  MaskedAssign(distMin == distTheta2, norm + unplaced.GetThetaCone().GetNormal2<Backend>(point), &norm);

  return norm;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::NormalKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal,
    typename Backend::bool_v &valid) {

  normal.Set(0.);
  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  /* Assumption : This function assumes that the point is on the surface.
   *
   * Algorithm :
   * Detect all those surfaces on which the point is at, and count the
   * numOfSurfaces. if(numOfSurfaces == 1) then normal corresponds to the
   * normal for that particular case.
   *
   * if(numOfSurfaces > 1 ), then add the normals corresponds to different
   * cases, and finally normalize it and return.
   *
   * We need following function
   * IsPointOnInnerRadius()
   * IsPointOnOuterRadius()
   * IsPointOnStartPhi()
   * IsPointOnEndPhi()
   * IsPointOnStartTheta()
   * IsPointOnEndTheta()
   *
   * set valid=true if numOfSurface > 0
   *
   * if above mentioned assumption not followed , ie.
   * In case the given point is outside, then find the closest boundary,
   * the required normal will be the normal to that boundary.
   * This logic is implemented in "ApproxSurfaceNormalKernel" function
   */

  Bool_t isPointOutside(false);

  //May be required Later
  /*
  if (!ForDistanceToOut) {
    Bool_t unused(false);
    GenericKernelForContainsAndInside<Backend, true>(unplaced, point, unused, isPointOutside);
    MaskedAssign(unused || isPointOutside, ApproxSurfaceNormalKernel<Backend>(unplaced, point), &normal);
  }
  */

  Bool_t isPointInside(false);
  GenericKernelForContainsAndInside<Backend, true>(unplaced, point, isPointInside, isPointOutside);
  MaskedAssign(isPointInside || isPointOutside, ApproxSurfaceNormalKernel<Backend>(unplaced, point), &normal);

  valid = Bool_t(false);

  Float_t noSurfaces(0.);
  Bool_t isPointOnOuterRadius = IsPointOnOuterRadius<Backend>(unplaced, point);

  MaskedAssign(isPointOnOuterRadius, noSurfaces + 1, &noSurfaces);
  MaskedAssign(!isPointOutside && isPointOnOuterRadius, normal + (point.Unit()), &normal);

  if (unplaced.GetInnerRadius()) {
    Bool_t isPointOnInnerRadius = IsPointOnInnerRadius<Backend>(unplaced, point);
    MaskedAssign(isPointOnInnerRadius, noSurfaces + 1, &noSurfaces);
    MaskedAssign(!isPointOutside && isPointOnInnerRadius, normal - point.Unit(), &normal);
  }

  if (!unplaced.IsFullPhiSphere()) {
    Bool_t isPointOnStartPhi = IsPointOnStartPhi<Backend>(unplaced, point);
    Bool_t isPointOnEndPhi = IsPointOnEndPhi<Backend>(unplaced, point);
    MaskedAssign(isPointOnStartPhi, noSurfaces + 1, &noSurfaces);
    MaskedAssign(!isPointOutside && isPointOnStartPhi, normal - unplaced.GetWedge().GetNormal1(), &normal);
    MaskedAssign(isPointOnEndPhi, noSurfaces + 1, &noSurfaces);
    MaskedAssign(!isPointOutside && isPointOnEndPhi, normal - unplaced.GetWedge().GetNormal2(), &normal);
  }

  if (!unplaced.IsFullThetaSphere()) {
    Bool_t isPointOnStartTheta = IsPointOnStartTheta<Backend>(unplaced, point);
    Bool_t isPointOnEndTheta = IsPointOnEndTheta<Backend>(unplaced, point);

    MaskedAssign(isPointOnStartTheta, noSurfaces + 1, &noSurfaces);
    MaskedAssign(!isPointOutside && isPointOnStartTheta, normal + unplaced.GetThetaCone().GetNormal1<Backend>(point),
                 &normal);

    MaskedAssign(isPointOnEndTheta, noSurfaces + 1, &noSurfaces);
    MaskedAssign(!isPointOutside && isPointOnEndTheta, normal + unplaced.GetThetaCone().GetNormal2<Backend>(point),
                 &normal);

    Vector3D<Float_t> tempNormal(0., 0., -1.);
    MaskedAssign(!isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (unplaced.GetETheta() <= kPi / 2.),
                 tempNormal, &normal);
    Vector3D<Float_t> tempNormal2(0., 0., 1.);
    MaskedAssign(!isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (unplaced.GetSTheta() >= kPi / 2.),
                 tempNormal2, &normal);
  }

  normal = normal.Unit();

  valid = (noSurfaces > 0.);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside){

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedSphere const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>(unplaced, point, inside);
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::ContainsKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  typedef typename Backend::bool_v Bool_t;
  Bool_t unused;
  Bool_t outside;
  GenericKernelForContainsAndInside<Backend, false >(unplaced,
    localPoint, unused, outside);
  inside=!outside;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Precision fRmin = unplaced.GetInnerRadius();
  Precision fRminTolerance = unplaced.GetFRminTolerance();
  Precision fRmax = unplaced.GetOuterRadius();

  Float_t rad2 = localPoint.Mag2();
  Float_t tolRMin(fRmin + (fRminTolerance * 10. * 2));
  Float_t tolRMax(fRmax - (unplaced.GetMKTolerance() * 10. * 2));

  // Check radial surfaces
  // Radial check for GenericKernel Start
  if (unplaced.GetInnerRadius())
    completelyinside = (rad2 <= tolRMax * tolRMax) && (rad2 >= tolRMin * tolRMin);
  else
    completelyinside = (rad2 <= tolRMax * tolRMax);
  // std::cout<<"Comp In - Rad : "<<completelyinside<<std::endl;

  tolRMin = fRmin - (0.5 * fRminTolerance * 10 * 2);
  tolRMax = fRmax + (0.5 * unplaced.GetMKTolerance() * 10 * 2);
  if (unplaced.GetInnerRadius())
    completelyoutside = (rad2 <= tolRMin * tolRMin) || (rad2 >= tolRMax * tolRMax);
  else
    completelyoutside = (rad2 >= tolRMax * tolRMax);

  // Phi boundaries  : Do not check if it has no phi boundary!
  if (!unplaced.IsFullPhiSphere()) {

    Bool_t completelyoutsidephi;
    Bool_t completelyinsidephi;
    unplaced.GetWedge().GenericKernelForContainsAndInside<Backend, ForInside>(localPoint, completelyinsidephi,
                                                                              completelyoutsidephi);
    completelyoutside |= completelyoutsidephi;

    if (ForInside)
      completelyinside &= completelyinsidephi;
  }
  // Phi Check for GenericKernel Over

  // Theta bondaries
  if (!unplaced.IsFullThetaSphere()) {

    Bool_t completelyoutsidetheta(false);
    Bool_t completelyinsidetheta(false);
    unplaced.GetThetaCone().GenericKernelForContainsAndInside<Backend, ForInside>(localPoint, completelyinsidetheta,
                                                                                  completelyoutsidetheta);
    completelyoutside |= completelyoutsidetheta;

    if (ForInside)
      completelyinside &= completelyinsidetheta;
  }
  return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside){

  InsideKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::InsideKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v      Bool_t;
  Bool_t completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      unplaced, point, completelyinside, completelyoutside);
  inside=EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
}



template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SphereImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety){

  SafetyToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), safety);
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety){

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Bool_t done(false);

  // General Precalcs
  Float_t rad = point.Mag();

  // Distance to r shells
  Precision fRmin = unplaced.GetInnerRadius();
  Float_t fRminV(fRmin);
  Precision fRmax = unplaced.GetOuterRadius();
  Float_t fRmaxV(fRmax);
  Float_t safeRMin(0.);
  Float_t safeRMax(0.);

  Bool_t completelyinside(false), completelyoutside(false);
  GenericKernelForContainsAndInside<Backend, true>(unplaced, point, completelyinside, completelyoutside);

  MaskedAssign(completelyinside, -1., &safety);
  done |= completelyinside;
  if (IsFull(done))
    return;

  Bool_t isOnSurface = !completelyinside && !completelyoutside;
  MaskedAssign(!done && isOnSurface, 0., &safety);
  done |= isOnSurface;
  if (IsFull(done))
    return;

  if (fRmin) {
    safeRMin = fRminV - rad;
    safeRMax = rad - fRmaxV;
    CondAssign(!done && (safeRMin > safeRMax), safeRMin, safeRMax, &safety);
  } else {
    MaskedAssign(!done, (rad - fRmaxV), &safety);
  }
  // Distance to r shells over

  // Distance to phi extent
  if (!unplaced.IsFullPhiSphere()) {
    Float_t safetyPhi = unplaced.GetWedge().SafetyToIn<Backend>(point);
    MaskedAssign(!done, Max(safetyPhi, safety), &safety);
  }

  // Distance to Theta extent
  if (!unplaced.IsFullThetaSphere()) {
    Float_t safetyTheta = unplaced.GetThetaCone().SafetyToIn<Backend>(point);
    MaskedAssign(!done, Max(safetyTheta, safety), &safety);
  }
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void SphereImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety){
  SafetyToOutKernel<Backend>(unplaced, point, safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety){

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t rad = point.Mag();

  Bool_t done(false);

  // Distance to r shells
  Precision fRmin = unplaced.GetInnerRadius();
  Float_t fRminV(fRmin);
  Precision fRmax = unplaced.GetOuterRadius();
  Float_t fRmaxV(fRmax);

  Bool_t completelyinside(false), completelyoutside(false);
  GenericKernelForContainsAndInside<Backend, true>(unplaced, point, completelyinside, completelyoutside);
  MaskedAssign(completelyoutside, -1., &safety);
  done |= completelyoutside;
  if (IsFull(done))
    return;

  Bool_t isOnSurface = !completelyinside && !completelyoutside;
  MaskedAssign(!done && isOnSurface, 0., &safety);
  done |= isOnSurface;
  if (IsFull(done))
    return;

  // Distance to r shells
  if (fRmin) {
    Float_t safeRMin = (rad - fRminV);
    Float_t safeRMax = (fRmaxV - rad);
    CondAssign((!done && (safeRMin < safeRMax)), safeRMin, safeRMax, &safety);
  } else {
    MaskedAssign(!done, (fRmaxV - rad), &safety);
  }

  // Distance to phi extent
  if (!unplaced.IsFullPhiSphere()) {
    Float_t safetyPhi = unplaced.GetWedge().SafetyToOut<Backend>(point);
    MaskedAssign(!done, Min(safetyPhi, safety), &safety);
  }

  // Distance to Theta extent
  Float_t safeTheta(0.);
  if (!unplaced.IsFullThetaSphere()) {
    safeTheta = unplaced.GetThetaCone().SafetyToOut<Backend>(point);
    MaskedAssign(!done, Min(safeTheta, safety), &safety);
  }
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedSphere const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point),
                              transformation.TransformDirection<rotCodeT>(direction), stepMax, distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const & /*stepMax*/,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  distance = kInfinity;

  Bool_t done(false);

  Float_t fRmax(unplaced.GetOuterRadius());
  Float_t fRmin(unplaced.GetInnerRadius());

  bool fullPhiSphere = unplaced.IsFullPhiSphere();
  bool fullThetaSphere = unplaced.IsFullThetaSphere();

  Vector3D<Float_t> tmpPt;
  // General Precalcs
  Float_t rad2 = point.Mag2();
  Float_t pDotV3d = point.Dot(direction);

  Float_t c = rad2 - fRmax * fRmax;

  Bool_t cond = IsCompletelyInside<Backend>(unplaced, point);
  MaskedAssign(cond, -1., &distance);
  done |= cond;
  if (IsFull(done))
    return;

  cond = IsPointOnSurfaceAndMovingOut<Backend, false>(unplaced, point, direction);
  MaskedAssign(!done && cond, 0., &distance);
  done |= cond;
  if (IsFull(done))
    return;

  Float_t sd1(kInfinity);
  Float_t sd2(kInfinity);
  Float_t d2 = (pDotV3d * pDotV3d - c);
  cond = (d2 < 0. || ((c > 0.0) && (pDotV3d > 0.0)));
  done |= cond;
  if (IsFull(done))
    return; // Returning in case of no intersection with outer shell

  // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
  MaskedAssign( d2 >= 0., (-pDotV3d - Sqrt(Abs(d2))), &sd1);

  Float_t outerDist(kInfinity);
  Float_t innerDist(kInfinity);

  if (unplaced.IsFullSphere()) {
    MaskedAssign(!done && (sd1 >= 0.), sd1, &outerDist);
  } else {
    tmpPt = point + sd1 * direction;
    MaskedAssign(!done && unplaced.GetWedge().Contains<Backend>(tmpPt) &&
                     unplaced.GetThetaCone().Contains<Backend>(tmpPt) && (sd1 >= 0.),
                 sd1, &outerDist);
  }

  if (unplaced.GetInnerRadius()) {
    c = rad2 - fRmin * fRmin;
    d2 = pDotV3d * pDotV3d - c;
    // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
    MaskedAssign( d2 >= 0., (-pDotV3d + Sqrt(Abs(d2))), &sd2);

    if (unplaced.IsFullSphere()) {
      MaskedAssign(!done && (sd2 >= 0.), sd2, &innerDist);
    } else {
      tmpPt = point + sd2 * direction;
      MaskedAssign(!done && (sd2 >= 0.) && unplaced.GetWedge().Contains<Backend>(tmpPt) &&
                       unplaced.GetThetaCone().Contains<Backend>(tmpPt),
                   sd2, &innerDist);
    }
  }

  distance = Min(outerDist, innerDist);

  if (!fullPhiSphere) {
    GetMinDistFromPhi<Backend, true>(unplaced, point, direction, done, distance);
  }

  Float_t distThetaMin(kInfinity);

  if (!fullThetaSphere) {
    Bool_t intsect1(false);
    Bool_t intsect2(false);
    Float_t distTheta1(kInfinity);
    Float_t distTheta2(kInfinity);

    unplaced.GetThetaCone().DistanceToIn<Backend>(point, direction, distTheta1, distTheta2, intsect1,
                                                  intsect2); //,cone1IntSecPt, cone2IntSecPt);
    Vector3D<Float_t> coneIntSecPt1 = point + distTheta1 * direction;
    Float_t distCone1 = coneIntSecPt1.Mag2();

    Vector3D<Float_t> coneIntSecPt2 = point + distTheta2 * direction;
    Float_t distCone2 = coneIntSecPt2.Mag2();

    Bool_t isValidCone1 = (distCone1 >= fRmin * fRmin && distCone1 <= fRmax * fRmax) && intsect1;
    Bool_t isValidCone2 = (distCone2 >= fRmin * fRmin && distCone2 <= fRmax * fRmax) && intsect2;

    if (!fullPhiSphere) {
      isValidCone1 &= unplaced.GetWedge().Contains<Backend>(coneIntSecPt1);
      isValidCone2 &= unplaced.GetWedge().Contains<Backend>(coneIntSecPt2);
    }
    MaskedAssign((!done && isValidCone2 && !isValidCone1), distTheta2, &distThetaMin);

    MaskedAssign((!done && isValidCone1 && !isValidCone2), distTheta1, &distThetaMin);

    MaskedAssign((!done && isValidCone1 && isValidCone2), Min(distTheta1, distTheta2), &distThetaMin);
  }

  distance = Min(distThetaMin, distance);

  Vector3D<Float_t> directDir = (Vector3D<Float_t>(0., 0., 0.) - point);
  Float_t newDist = directDir.Mag();
  MaskedAssign(Bool_t(unplaced.GetSTheta() > kHalfTolerance || unplaced.GetETheta() < (kPi - kHalfTolerance)) &&
                   (Abs(directDir.Unit().x() - direction.x()) < kHalfTolerance) &&
                   (Abs(directDir.Unit().y() - direction.y()) < kHalfTolerance) &&
                   (Abs(directDir.Unit().z() - direction.z()) < kHalfTolerance),
               Min(distance, newDist), &distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend, bool DistToIn>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::GetMinDistFromPhi(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    Vector3D<typename Backend::precision_v> const &localDir,
    typename Backend::bool_v &done,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  Float_t distPhi1(kInfinity);
  Float_t distPhi2(kInfinity);
  Float_t dist(kInfinity);

  if (DistToIn)
    unplaced.GetWedge().DistanceToIn<Backend>(localPoint, localDir, distPhi1, distPhi2);
  else
    unplaced.GetWedge().DistanceToOut<Backend>(localPoint, localDir, distPhi1, distPhi2);

  Vector3D<Float_t> tmpPt;
  Bool_t containsCond1(false), containsCond2(false);
  // Min Face
  dist = Min(distPhi1, distPhi2);
  tmpPt = localPoint + dist * localDir;

  Precision fRmax = unplaced.GetOuterRadius();
  Precision fRmin = unplaced.GetInnerRadius();
  Float_t rad2 = tmpPt.Mag2();

  Bool_t tempCond(false);
  tempCond = ((dist == distPhi1) && unplaced.GetWedge().IsOnSurfaceGeneric<Backend, true>(tmpPt)) ||
             ((dist == distPhi2) && unplaced.GetWedge().IsOnSurfaceGeneric<Backend, false>(tmpPt));

  containsCond1 =
      tempCond && (rad2 > fRmin * fRmin) && (rad2 < fRmax * fRmax) && unplaced.GetThetaCone().Contains<Backend>(tmpPt);

  MaskedAssign(!done && containsCond1, Min(dist, distance), &distance);

  // Max Face
  dist = Max(distPhi1, distPhi2);
  tmpPt = localPoint + dist * localDir;

  rad2 = tmpPt.Mag2();
  tempCond = Bool_t(false);
  tempCond = ((dist == distPhi1) && unplaced.GetWedge().IsOnSurfaceGeneric<Backend, true>(tmpPt)) ||
             ((dist == distPhi2) && unplaced.GetWedge().IsOnSurfaceGeneric<Backend, false>(tmpPt));

  containsCond2 =
      tempCond && (rad2 > fRmin * fRmin) && (rad2 < fRmax * fRmax) && unplaced.GetThetaCone().Contains<Backend>(tmpPt);
  MaskedAssign(((!done) && (!containsCond1) && containsCond2), Min(dist, distance), &distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH void SphereImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend>(unplaced, point, direction, stepMax, distance);
}

//V3
template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void SphereImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedSphere const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &/*stepMax*/,
    typename Backend::precision_v &distance){

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  distance = kInfinity;

  Bool_t done(false);
  Float_t snxt(kInfinity);

  Float_t fRmax(unplaced.GetOuterRadius());
  Float_t fRmin(unplaced.GetInnerRadius());

  // Intersection point
  Vector3D<Float_t> intSecPt;
  Float_t d2(0.);

  Float_t pDotV3d = point.Dot(direction);

  Float_t rad2 = point.Mag2();
  Float_t c = rad2 - fRmax * fRmax;

  Float_t sd1(kInfinity);
  Float_t sd2(kInfinity);

  Bool_t cond = IsCompletelyOutside<Backend>(unplaced, point);
  MaskedAssign(cond, -1., &distance);

  done |= cond;
  if (IsFull(done))
    return;

  cond = IsPointOnSurfaceAndMovingOut<Backend, true>(unplaced, point, direction);
  MaskedAssign(!done && cond, 0., &distance);
  done |= cond;
  if (IsFull(done))
    return;

  // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
  d2 = (pDotV3d * pDotV3d - c);
  MaskedAssign((!done && (d2 >= 0.)), (-pDotV3d + Sqrt(Abs(d2))), &sd1);

  if (unplaced.GetInnerRadius()) {
    c = rad2 - fRmin * fRmin;
    d2 = (pDotV3d * pDotV3d - c);
    MaskedAssign((!done && (d2 >= 0.) && (pDotV3d < 0.)), (-pDotV3d - Sqrt(Abs(d2))), &sd2);
  }

  snxt = Min(sd1, sd2);
  Float_t distThetaMin(kInfinity);
  Float_t distPhiMin(kInfinity);

  if (!unplaced.IsFullThetaSphere()) {
    Bool_t intsect1(false);
    Bool_t intsect2(false);
    Float_t distTheta1(0.);
    Float_t distTheta2(0.);
    unplaced.GetThetaCone().DistanceToOut<Backend>(point, direction, distTheta1, distTheta2, intsect1, intsect2);
    MaskedAssign((intsect2 && !intsect1), distTheta2, &distThetaMin);
    MaskedAssign((!intsect2 && intsect1), distTheta1, &distThetaMin);
    MaskedAssign((intsect2 && intsect1) /*|| (!intsect2 && !intsect1)*/, Min(distTheta1, distTheta2), &distThetaMin);
  }

  distance = Min(distThetaMin, snxt);

  if (!unplaced.IsFullPhiSphere()) {
    if (unplaced.GetDeltaPhiAngle() <= kPi) {
      Float_t distPhi1;
      Float_t distPhi2;
      unplaced.GetWedge().DistanceToOut<Backend>(point, direction, distPhi1, distPhi2);
      distPhiMin = Min(distPhi1, distPhi2);
      distance = Min(distPhiMin, distance);
    } else {
      GetMinDistFromPhi<Backend, false>(unplaced, point, direction, done, distance);
    }
  }
}
} } // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
