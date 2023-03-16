/*
 * TBooleanMinusImplementation.h
 *
 *  Created on: Aug 13, 2014
 *      Author: swenzel
 */

#ifndef TBOOLEANMINUSIMPLEMENTATION_H_
#define TBOOLEANMINUSIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TUnplacedBooleanMinusVolume.h"

namespace VECGEOM_NAMESPACE {

template <typename LeftPlacedType_t, typename RightPlacedType_t>
struct TBooleanMinusImplementation {

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void UnplacedContains(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(
      TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(TUnplacedBooleanMinusVolume const &unplaced,
                                                                  Transformation3D const &transformation,
                                                                  Vector3D<typename Backend::precision_v> const &point,
                                                                  typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(
      TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(
      TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ContainsKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOutKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToInKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOutKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(
      TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid);

}; // End struct TBooleanMinusImplementation

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::UnplacedContains(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside)
{

  ContainsKernel<Backend>(unplaced, localPoint, inside);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::Contains(
    TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside)
{

  localPoint = transformation.Transform(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::Inside(
    TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside)
{

  InsideKernel<Backend>(unplaced, transformation.Transform(point), inside);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::DistanceToIn(
    TUnplacedBooleanMinusVolume const &unplaced, Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
{

  DistanceToInKernel<Backend>(unplaced, transformation.Transform(point), transformation.TransformDirection(direction),
                              stepMax, distance);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::DistanceToOut(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{

  DistanceToOutKernel<Backend>(unplaced, point, direction, stepMax, distance);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<
    LeftPlacedType_t, RightPlacedType_t>::SafetyToIn(TUnplacedBooleanMinusVolume const &unplaced,
                                                     Transformation3D const &transformation,
                                                     Vector3D<typename Backend::precision_v> const &point,
                                                     typename Backend::precision_v &safety)
{

  SafetyToInKernel<Backend>(unplaced, transformation.Transform(point), safety);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <class Backend>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<
    LeftPlacedType_t, RightPlacedType_t>::SafetyToOut(TUnplacedBooleanMinusVolume const &unplaced,
                                                      Vector3D<typename Backend::precision_v> const &point,
                                                      typename Backend::precision_v &safety)
{

  SafetyToOutKernel<Backend>(unplaced, point, safety);
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::ContainsKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside)
{

  // now just use the Contains functionality
  // of Unplaced and its left and right components
  // Find if a subtraction of two shapes contains a given point

  // have to figure this out
  Vector3D<typename Backend::precision_v> tmp;
  LeftPlacedType_t::Implementation::template Contains<Backend>(
      *((LeftPlacedType_t *)unplaced.fLeftVolume)->GetUnplacedVolume(), *unplaced.fLeftVolume->transformation(),
      localPoint, tmp, inside);

  //   TUnplacedBooleanMinusVolume const &unplaced,
  //        Transformation3D const &transformation,
  //        Vector3D<typename Backend::precision_v> const &point,
  //        Vector3D<typename Backend::precision_v> &localPoint,
  //        typename Backend::bool_v &inside
  //

  if (vecCore::MaskEmpty(inside)) return;

  typename Backend::bool_v rightInside;
  RightPlacedType_t::Implementation::template Contains<Backend>(
      *((RightPlacedType_t *)unplaced.fRightVolume)->GetUnplacedVolume(), *unplaced.fRightVolume->transformation(),
      localPoint, tmp, rightInside);

  inside &= !rightInside;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::InsideKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside)
{

  // now use the Inside functionality of left and right components

  // going to be a bit more complicated due to Surface states
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::DistanceToInKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
}
#include <iostream>
template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::DistanceToOutKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  // how can we force to inline this ?
  // Left is an unplaced shape; but it could be a complicated one

  // what happens if LeftType is VPlacedVolume itself?
  // For the SOA case: is it better to push SOA down or do ONE loop around?
  // distance = unplaced.fLeftVolume->Unplaced_t::LeftType::DistanceToOut( point, direction, stepMax );
  // Float_t dinright = unplaced.fRightVolume->Unplaced_t::RightType::DistanceToIn( point, direction, stepMax );

  // we need a template specialization for this in case we have LeftType or RightType equals to
  // VPlacedVolume

  LeftPlacedType_t::Implementation::template DistanceToOut<Backend>(
      *((LeftPlacedType_t *)unplaced.fLeftVolume)->GetUnplacedVolume(), point, direction, stepMax, distance);
  Float_t dinright(kInfLength);
  RightPlacedType_t::Implementation::template DistanceToIn<Backend>(
      *((RightPlacedType_t *)unplaced.fRightVolume)->GetUnplacedVolume(), *unplaced.fRightVolume->transformation(),
      point, direction, stepMax, dinright);
  distance = Min(distance, dinright);
  return;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::SafetyToInKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety)
{
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::SafetyToOutKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety)
{

  typedef typename Backend::bool_v Bool_t;
  typedef typename Backend::precision_v Float_t;

  LeftPlacedType_t::Implementation::template SafetyToOut<Backend>(
      *((LeftPlacedType_t *)unplaced.fLeftVolume)->GetUnplacedVolume(), point, safety);
  Float_t safetyright(kInfLength);
  RightPlacedType_t::Implementation::template SafetyToIn<Backend>(
      *((RightPlacedType_t *)unplaced.fRightVolume)->GetUnplacedVolume(), *unplaced.fRightVolume->transformation(),
      point, safetyright);
  safety = Min(safety, safetyright);
  return;
}

template <typename LeftPlacedType_t, typename RightPlacedType_t>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE void TBooleanMinusImplementation<LeftPlacedType_t, RightPlacedType_t>::NormalKernel(
    TUnplacedBooleanMinusVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
{
  // TBDONE
}

} // namespace VECGEOM_NAMESPACE

#endif /* TBOOLEANMINUSIMPLEMENTATION_H_ */
