/// @file ScaledShapeImplementation.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedScaledShape.h"
#include "volumes/kernel/GenericKernels.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(ScaledShapeImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)


inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedScaledShape;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct ScaledShapeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedScaledShape;
  using UnplacedShape_t = UnplacedScaledShape;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedScaledShape<%i, %i>", transCodeT, rotCodeT);
  }

  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedScaledShape const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedScaledShape const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedScaledShape const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedScaledShape const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedScaledShape const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);



  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      UnplacedScaledShape const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

 // template <class Backend>
 // static void Normal( Vector3D<Precision> const &dimensions,
 //         Vector3D<typename Backend::precision_v> const &point,
 //        Vector3D<typename Backend::precision_v> &normal,
 //         Vector3D<typename Backend::precision_v> &valid )

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedScaledShape const &,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );


}; // End struct ScaledShapeImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>(unplaced, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedScaledShape const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedScaledShape const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Backend>(unplaced,
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedScaledShape const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    transformation.TransformDirection<rotCodeT>(direction),
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend>(
    unplaced,
    point,
    direction,
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void ScaledShapeImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedScaledShape const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Backend>(
    unplaced,
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void ScaledShapeImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Backend>(
    unplaced,
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::ContainsKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  // Transform local point to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(localPoint, ulocalPoint);

  // Now call Contains for the unscaled shape
  inside = unplaced.fPlaced->Contains(ulocalPoint);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::InsideKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::inside_v &inside) {

  // Transform local point to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(localPoint, ulocalPoint);

  // Now call Inside for the unscaled shape
  inside = unplaced.fPlaced->Inside(ulocalPoint);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  // Transform point, direction and stepMax to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Direction is un-normalized after scale transformation
  Vector3D<typename Backend::precision_v> ulocalDir;
  unplaced.fScale.Transform(direction, ulocalDir);
  ulocalDir.Normalize();

  auto ustepMax = unplaced.fScale.TransformDistance(stepMax, direction);

  // Compute distance in unscaled system
  distance = unplaced.fPlaced->DistanceToIn(ulocalPoint, ulocalDir, ustepMax);

  // Convert distance back to master
  distance = unplaced.fScale.InverseTransformDistance(distance, ulocalDir);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  // Transform point, direction and stepMax to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Direction is un-normalized after scale transformation
  Vector3D<typename Backend::precision_v> ulocalDir;
  unplaced.fScale.Transform(direction, ulocalDir);
  ulocalDir.Normalize();

  auto ustepMax = unplaced.fScale.TransformDistance(stepMax, direction);

  // Compute distance in unscaled system
  distance = unplaced.fPlaced->DistanceToOut(ulocalPoint, ulocalDir, ustepMax);

  // Convert distance back to master
  distance = unplaced.fScale.InverseTransformDistance(distance, ulocalDir);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  // Transform point to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute unscaled safety, then scale it.
  safety = unplaced.fPlaced->SafetyToIn(ulocalPoint);
  safety = unplaced.fScale.InverseTransformSafety(safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedScaledShape const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  // Transform point to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute unscaled safety, then scale it.
  safety = unplaced.fPlaced->SafetyToOut(ulocalPoint);
  safety = unplaced.fScale.InverseTransformSafety(safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void ScaledShapeImplementation<transCodeT, rotCodeT>::NormalKernel(
     UnplacedScaledShape const &unplaced,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid) {

  // Transform point to unscaled shape frame
  Vector3D<typename Backend::precision_v> ulocalPoint;
  unplaced.fScale.Transform(point, ulocalPoint);

  // Compute normal in unscaled frame
  Vector3D<typename Backend::precision_v> ulocalNorm;
  unplaced.fPlaced->Normal(ulocalPoint, ulocalNorm/*, valid*/);

  // Convert normal to scaled frame
  unplaced.fScale.InverseTransform(ulocalNorm, normal);
  normal.Normalize();
}


} } // End global namespace


#endif // VECGEOM_VOLUMES_KERNEL_SCALEDSHAPEIMPLEMENTATION_H_
