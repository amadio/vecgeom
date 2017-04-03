/*
 * BooleanImplementation.h
 */

#ifndef BOOLEANUNIONIMPLEMENTATION_H_
#define BOOLEANUNIONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBooleanVolume.h"

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * partial template specialization for UNION implementation
 */
template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BooleanImplementation<kUnion, transCodeT, rotCodeT> {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t   = PlacedBooleanVolume;
  using UnplacedShape_t = UnplacedBooleanVolume;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType() { printf("SpecializedBooleanVolume<%i, %i, %i>", kUnion, transCodeT, rotCodeT); }

  template <typename Stream>
  static void PrintType(Stream &s)
  {
    s << "SpecializedBooleanVolume<kUnion"
      << "," << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &s)
  {
    s << "BooleanImplementation<kUnion"
      << "," << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &s)
  {
    s << "UnplacedBooleanVolume";
  }

  //
  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void UnplacedContains(UnplacedBooleanVolume const &unplaced,
                               Vector3D<typename Backend::precision_v> const &localPoint,
                               typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint, typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           Vector3D<typename Backend::precision_v> const &direction,
                           typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                            Vector3D<typename Backend::precision_v> const &direction,
                            typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void ContainsKernel(UnplacedBooleanVolume const &unplaced,
                             Vector3D<typename Backend::precision_v> const &point, typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void InsideKernel(UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                           typename Backend::inside_v &inside);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToInKernel(UnplacedBooleanVolume const &unplaced,
                                 Vector3D<typename Backend::precision_v> const &point,
                                 Vector3D<typename Backend::precision_v> const &direction,
                                 typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOutKernel(UnplacedBooleanVolume const &unplaced,
                                  Vector3D<typename Backend::precision_v> const &point,
                                  Vector3D<typename Backend::precision_v> const &direction,
                                  typename Backend::precision_v const &stepMax,
                                  typename Backend::precision_v &distance);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToInKernel(UnplacedBooleanVolume const &unplaced,
                               Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOutKernel(UnplacedBooleanVolume const &unplaced,
                                Vector3D<typename Backend::precision_v> const &point,
                                typename Backend::precision_v &safety);

  template <typename Backend>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void NormalKernel(UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                           Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid);

}; // End struct BooleanImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside)
{

  ContainsKernel<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::Contains(UnplacedBooleanVolume const &unplaced,
                                                                   Transformation3D const &transformation,
                                                                   Vector3D<typename Backend::precision_v> const &point,
                                                                   Vector3D<typename Backend::precision_v> &localPoint,
                                                                   typename Backend::bool_v &inside)
{

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::Inside(UnplacedBooleanVolume const &unplaced,
                                                                 Transformation3D const &transformation,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::inside_v &inside)
{

  InsideKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToIn(
    UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
{

  DistanceToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point),
                              transformation.TransformDirection<rotCodeT>(direction), stepMax, distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToOut(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{

  DistanceToOutKernel<Backend>(unplaced, point, direction, stepMax, distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToIn(
    UnplacedBooleanVolume const &unplaced, Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety)
{

  SafetyToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToOut(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety)
{

  SafetyToOutKernel<Backend>(unplaced, point, safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::ContainsKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside)
{

  inside = unplaced.fLeftVolume->Contains(localPoint);
  if (vecCore::MaskFull(inside)) return;
  inside |= unplaced.fRightVolume->Contains(localPoint);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::InsideKernel(UnplacedBooleanVolume const &unplaced,
                                                                       Vector3D<typename Backend::precision_v> const &p,
                                                                       typename Backend::inside_v &inside)
{

  // now use the Inside functionality of left and right components
  // algorithm taken from Geant4 implementation
  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

  typename Backend::inside_v positionA = fPtrSolidA->Inside(p);
  if (positionA == EInside::kInside) {
    inside = EInside::kInside;
    return;
  }

  typename Backend::inside_v positionB = fPtrSolidB->Inside(p);
  if (positionB == EInside::kInside
      /* leaving away this part of the condition for the moment until SurfaceNormal implemented
      ||
    ( positionA == EInside::kSurface && positionB == EInside::kSurface )


    &&
        ( fPtrSolidA->SurfaceNormal(p) +
          fPtrSolidB->SurfaceNormal(p) ).mag2() <
          1000*G4GeometryTolerance::GetInstance()->GetRadialTolerance() )
      */) {
    inside = EInside::kInside;
    return;
  } else {
    if ((positionB == EInside::kSurface) || (positionA == EInside::kSurface)) {
      inside = EInside::kSurface;
      return;
    } else {
      inside = EInside::kOutside;
      return;
    }
  }
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{

  typedef typename Backend::precision_v Float_t;

  Float_t d1 = unplaced.fLeftVolume->DistanceToIn(p, v, stepMax);
  Float_t d2 = unplaced.fRightVolume->DistanceToIn(p, v, stepMax);
  distance   = Min(d1, d2);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &p,
    Vector3D<typename Backend::precision_v> const &v, typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance)
{
  typedef typename Backend::precision_v Float_t;
  VPlacedVolume const *const ptrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const ptrSolidB = unplaced.fRightVolume;

  Float_t dist = 0., disTmp = 0.;
  Float_t pushdist(1E-6);
  size_t push                        = 0;
  typename Backend::Bool_t positionA = ptrSolidA->Contains(p);
  Vector3D<Precision> nextp(p);
  bool connectingstep(false);

  // reusable kernel as lambda
  auto kernel = [&](VPlacedVolume const *A, VPlacedVolume const *B) {
    do {
      connectingstep = false;
      disTmp         = A->PlacedDistanceToOut(nextp, v);
      dist += (disTmp >= 0. && disTmp < kInfLength) ? disTmp : 0;
      // give a push
      dist += pushdist;
      push++;
      nextp = p + dist * v;
      // B could be overlapping with A -- and/or connecting A to another part of A
      // if (B->Contains(nextp)) {
      if (B->Inside(nextp) != vecgeom::kOutside) {
        disTmp = B->PlacedDistanceToOut(nextp, v);
        dist += (disTmp >= 0. && disTmp < kInfLength) ? disTmp : 0;
        dist += pushdist;
        push++;
        nextp          = p + dist * v;
        connectingstep = true;
      }
    } while (connectingstep && A->Contains(nextp));
  };

  if (positionA) { // initially in A
    kernel(ptrSolidA, ptrSolidB);
  }
  // if( positionB != kOutside )
  else {
    kernel(ptrSolidB, ptrSolidA);
  }
  distance = dist - push * pushdist;
  return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety)
{

  typedef typename Backend::precision_v Float_t;

  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;
  Float_t distA                         = fPtrSolidA->SafetyToIn(p);
  Float_t distB                         = fPtrSolidB->SafetyToIn(p);
  safety                                = Min(distA, distB);
  vecCore::MaskedAssign(safety, safety < 0.0, 0.0);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &p,
    typename Backend::precision_v &safety)
{

  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

  typedef typename Backend::bool_v Bool_t;

  Bool_t containedA = fPtrSolidA->Contains(p);
  Bool_t containedB = fPtrSolidB->Contains(p);

  if (containedA && containedB) /* in both */
  {
    safety = Max(fPtrSolidA->SafetyToOut(p),
                 fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(p))); // is max correct ??
  } else {
    if (containedB) /* only contained in B */
    {
      safety = fPtrSolidB->SafetyToOut(fPtrSolidB->GetTransformation()->Transform(p));
    } else {
      safety = fPtrSolidA->SafetyToOut(p);
    }
  }
  return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECCORE_ATT_HOST_DEVICE
void BooleanImplementation<kUnion, transCodeT, rotCodeT>::NormalKernel(
    UnplacedBooleanVolume const &unplaced, Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
{
  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  Vector3D<Float_t> localNorm;
  Vector3D<Float_t> localPoint;
  valid = Backend::kFalse;

  VPlacedVolume const *const fPtrSolidA = unplaced.fLeftVolume;
  VPlacedVolume const *const fPtrSolidB = unplaced.fRightVolume;

  // If point is inside A, then it must be on a surface of A (points on the
  // intersection between A and B cannot be on surface, or if they are they
  // are on a common surface and the normal can be computer for A or B)
  if (fPtrSolidA->Contains(point)) {
    fPtrSolidA->GetTransformation()->Transform(point, localPoint);
    valid = fPtrSolidA->Normal(localPoint, localNorm);
    fPtrSolidA->GetTransformation()->InverseTransformDirection(localNorm, normal);
    return;
  }
  // Same for points inside B
  if (fPtrSolidB->Contains(point)) {
    fPtrSolidB->GetTransformation()->Transform(point, localPoint);
    valid = fPtrSolidB->Normal(localPoint, localNorm);
    fPtrSolidB->GetTransformation()->InverseTransformDirection(localNorm, normal);
    return;
  }
  // Points outside both A and B can be on any surface. We use the safety.
  Float_t safetyA = fPtrSolidA->SafetyToIn(point);
  Float_t safetyB = fPtrSolidB->SafetyToIn(point);
  Bool_t onA      = safetyA < safetyB;
  if (vecCore::MaskFull(onA)) {
    fPtrSolidA->GetTransformation()->Transform(point, localPoint);
    valid = fPtrSolidA->Normal(localPoint, localNorm);
    fPtrSolidA->GetTransformation()->InverseTransformDirection(localNorm, normal);
    return;
  } else {
    //  if (vecCore::MaskEmpty(onA)) {  // to use real mask operation when supporting vectors
    fPtrSolidB->GetTransformation()->Transform(point, localPoint);
    valid = fPtrSolidB->Normal(localPoint, localNorm);
    fPtrSolidB->GetTransformation()->InverseTransformDirection(localNorm, normal);
    return;
  }
  // Some particles are on A, some on B. We never arrive here in the scalar case
  // If the interface to Normal will support the vector case, we have to write code here.
  return;
}

} // End impl namespace

} // End global namespace

#endif /* BooleanImplementation_H_ */
