/// @file ParallelepipedImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_

#include "base/Global.h"

#include "volumes/kernel/BoxImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedParallelepiped.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ParallelepipedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ParallelepipedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedParallelepiped;
template <typename T>
struct ParallelepipedStruct;
class UnplacedParallelepiped;

struct ParallelepipedImplementation {

  using PlacedShape_t    = PlacedParallelepiped;
  using UnplacedStruct_t = ParallelepipedStruct<double>;
  using UnplacedVolume_t = UnplacedParallelepiped;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    // printf("SpecializedParallelepiped<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream & /*s*/)
  {
    // s << "SpecializedParallelepiped<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream & /*s*/)
  {
    // s << "ParallelepipedImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream & /*s*/)
  {
    // s << "UnplacedParallelepiped";
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Transform(UnplacedStruct_t const &unplaced, Vector3D<Real_v> &point);

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside);

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety);
};

template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::Transform(UnplacedStruct_t const &unplaced, Vector3D<Real_v> &point)
{
  point[1] -= unplaced.fTanThetaSinPhi * point[2];
  point[0] -= unplaced.fTanThetaCosPhi * point[2] + unplaced.fTanAlpha * point[1];
}

template <typename Real_v, typename Bool_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                            Bool_v &inside)
{
  Vector3D<Real_v> localPoint = point;
  Transform<Real_v>(unplaced, localPoint);
  // Run unplaced box kernel
  BoxImplementation::Contains<Real_v, Bool_v>(BoxStruct<double>(unplaced.fDimensions), localPoint, inside);
}

template <typename Real_v, typename Inside_t>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                          Inside_t &inside)
{

  Vector3D<Real_v> localPoint = point;
  Transform<Real_v>(unplaced, localPoint);
  BoxImplementation::Inside<Real_v, Inside_t>(BoxStruct<double>(unplaced.fDimensions), localPoint, inside);
}

template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::DistanceToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                                Vector3D<Real_v> const &direction, Real_v const &stepMax,
                                                Real_v &distance)
{

  Vector3D<Real_v> localPoint     = point;
  Vector3D<Real_v> localDirection = direction;

  Transform<Real_v>(unplaced, localPoint);
  Transform<Real_v>(unplaced, localDirection);

  // Run unplaced box kernel
  BoxImplementation::DistanceToIn<Real_v>(BoxStruct<double>(unplaced.fDimensions), localPoint, localDirection, stepMax,
                                          distance);
  // The check below has to be added because the box does not follow the boundary
  // convention yet. To be removed when this will be the case.
  vecCore::MaskedAssign(distance, Abs(distance) < kHalfTolerance, Real_v(0.));
}

template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::DistanceToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                                 Vector3D<Real_v> const &direction, Real_v const & /* stepMax */,
                                                 Real_v &distance)
{

  using vecCore::MaskedAssign;
  using Bool_v = vecCore::Mask_v<Real_v>;
  Real_v max;
  Bool_v inPoint, inDirection, goingAway;
  Bool_v done(false);
  distance = kInfinity;

  // Z intersection
  // Outside Z range
  Bool_v outside = (Abs(point[2]) - unplaced.fDimensions[2]) > kHalfTolerance;
  done |= outside;
  MaskedAssign(distance, outside, Real_v(-1.));
  inDirection = direction[2] > 0;
  max         = unplaced.fDimensions[2] - point[2];
  inPoint     = max > kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection, max / direction[2]);
  done |= goingAway;
  if (vecCore::MaskFull(done)) return;

  inDirection = direction[2] < 0;
  max         = -unplaced.fDimensions[2] - point[2];
  inPoint     = max < -kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection, max / direction[2]);
  done |= goingAway;
  if (vecCore::MaskFull(done)) return;

  // Y plane intersection

  Real_v localPointY, localDirectionY;

  localPointY     = point[1] - unplaced.fTanThetaSinPhi * point[2];
  localDirectionY = direction[1] - unplaced.fTanThetaSinPhi * direction[2];

  // Outside Y range
  outside = (Abs(localPointY) - unplaced.fDimensions[1]) > kHalfTolerance;
  done |= outside;
  MaskedAssign(distance, outside, Real_v(-1.));
  inDirection = localDirectionY > 0;
  max         = unplaced.fDimensions[1] - localPointY;
  inPoint     = max > kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  max /= localDirectionY;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection && max < distance, max);
  done |= goingAway;
  if (vecCore::MaskFull(done)) return;

  inDirection = localDirectionY < 0;
  max         = -unplaced.fDimensions[1] - localPointY;
  inPoint     = max < -kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  max /= localDirectionY;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection && max < distance, max);
  done |= goingAway;
  if (IsFull(done)) return;

  // X plane intersection

  Real_v localPointX, localDirectionX;

  localPointX     = point[0] - unplaced.fTanThetaCosPhi * point[2] - unplaced.fTanAlpha * localPointY;
  localDirectionX = direction[0] - unplaced.fTanThetaCosPhi * direction[2] - unplaced.fTanAlpha * localDirectionY;

  // Outside X range
  outside = (Abs(localPointX) - unplaced.fDimensions[0]) > kHalfTolerance;
  done |= outside;
  MaskedAssign(distance, outside, Real_v(-1.));
  inDirection = localDirectionX > 0;
  max         = unplaced.fDimensions[0] - localPointX;
  inPoint     = max > kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  max /= localDirectionX;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection && max < distance, max);
  done |= goingAway;
  if (vecCore::MaskFull(done)) return;

  inDirection = localDirectionX < 0;
  max         = -unplaced.fDimensions[0] - localPointX;
  inPoint     = max < -kHalfTolerance;
  goingAway   = inDirection && !inPoint;
  max /= localDirectionX;
  MaskedAssign(distance, !done && goingAway, Real_v(0.));
  MaskedAssign(distance, !done && inPoint && inDirection && max < distance, max);
}

template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::SafetyToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                              Real_v &safety)
{

  Vector3D<Real_v> safetyVector;
  using vecCore::MaskedAssign;

  Vector3D<Real_v> localPoint = point;
  Transform<Real_v>(unplaced, localPoint);

  safetyVector[0] = Abs(localPoint[0]) - unplaced.fDimensions[0];
  safetyVector[1] = Abs(localPoint[1]) - unplaced.fDimensions[1];
  safetyVector[2] = Abs(localPoint[2]) - unplaced.fDimensions[2];

  safetyVector[0] *= unplaced.fCtx;
  safetyVector[1] *= unplaced.fCty;

  safety = safetyVector[0];
  MaskedAssign(safety, safetyVector[1] > safety, safetyVector[1]);
  MaskedAssign(safety, safetyVector[2] > safety, safetyVector[2]);
  MaskedAssign(safety, Abs(safety) < kTolerance, Real_v(0.));
}

template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void ParallelepipedImplementation::SafetyToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                               Real_v &safety)
{

  using vecCore::MaskedAssign;

  Vector3D<Real_v> safetyVector;
  Vector3D<Real_v> localPoint = point;
  Transform<Real_v>(unplaced, localPoint);

  safetyVector[0] = unplaced.fDimensions[0] - Abs(localPoint[0]);
  safetyVector[1] = unplaced.fDimensions[1] - Abs(localPoint[1]);
  safetyVector[2] = unplaced.fDimensions[2] - Abs(localPoint[2]);

  safetyVector[0] *= unplaced.fCtx;
  safetyVector[1] *= unplaced.fCty;

  safety = safetyVector[0];
  MaskedAssign(safety, safetyVector[1] < safety, safetyVector[1]);
  MaskedAssign(safety, safetyVector[2] < safety, safetyVector[2]);
  MaskedAssign(safety, Abs(safety) < kTolerance, Real_v(0.));
}
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDIMPLEMENTATION_H_
