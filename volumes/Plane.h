/// \file Plane.h
/// \author mgheata

#ifndef VECGEOM_VOLUMES_PLANE_H_
#define VECGEOM_VOLUMES_PLANE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/kernel/GenericKernels.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Plane;);
VECGEOM_DEVICE_DECLARE_CONV(class, Plane);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// A plane defined by the distance to origin and a normal vector. The normal
/// direction points to the outside halfspace.

class Plane : public AlignedBase {

private:
  Vector3D<Precision> fNormal; ///< Normalized normal of the plane.
  Precision fDistance;         ///< Distance from plane to origin (0, 0, 0).

public:
  VECGEOM_CUDA_HEADER_BOTH
  Plane() : fNormal(), fDistance(0.) {}

  VECGEOM_CUDA_HEADER_BOTH
  ~Plane() = default;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> const &GetNormal() const { return fNormal; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDistance() const { return fDistance; }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(Vector3D<Precision> const &normal, Vector3D<Precision> const &origin)
  {
    Vector3D<Precision> fixedNormal(normal);
    fixedNormal.FixZeroes();
    Precision inverseLength = 1. / fixedNormal.Mag();
    fNormal                 = inverseLength * fixedNormal;
    fDistance               = inverseLength * -fixedNormal.Dot(origin);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(Vector3D<Precision> const &normal, Precision distance)
  {
    fNormal   = normal;
    fDistance = distance;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistPlane(Vector3D<Real_v> const &point) const;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Contains(Vector3D<Real_v> const &point, Bool_v &inside) const;

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Inside(Vector3D<Real_v> const &point, Inside_v &inside) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SafetyToIn(Vector3D<Real_v> const &point, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SafetyToOut(Vector3D<Real_v> const &point, Real_v &distance) const;

}; // class Plane

std::ostream &operator<<(std::ostream &os, Plane const &plane);

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
Real_v Plane::DistPlane(Vector3D<Real_v> const &point) const
{
  // Returns distance from point to plane. This is positive if the point is on
  // the outside halfspace, negative otherwise.
  return (point.Dot(fNormal) + fDistance);
}

template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::Contains(Vector3D<Real_v> const &point, Bool_v &inside) const
{
  // Returns true if the point is in the halfspace behind the normal.
  inside = DistPlane(point) < Real_v(0.);
}

template <typename Real_v, typename Inside_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::Inside(Vector3D<Real_v> const &point, Inside_v &inside) const
{
  Real_v dplane = DistPlane(point);
  inside        = vecCore::Blend(dplane < Real_v(0.0), Inside_v(EInside::kInside), Inside_v(EInside::kOutside));
  vecCore::MaskedAssign(inside, vecCore::math::Abs(dplane) < Real_v(kTolerance), Inside_v(EInside::kSurface));
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  // The function returns a negative distance for points already inside or
  // direction going outwards (along the normal)
  using Bool_v = vecCore::Mask_v<Real_v>;
  distance     = -InfinityLength<Real_v>();

  Real_v ndd   = NonZero(direction.Dot(fNormal));
  Real_v saf   = DistPlane(point);
  Bool_v valid = ndd < Real_v(0.) && saf > Real_v(-kTolerance);

  if (vecCore::EarlyReturnAllowed()) {
    if (vecCore::MaskEmpty(valid)) return;
  }
  // If competing with other planes, the maximum distance is winning
  vecCore__MaskedAssignFunc(distance, valid, -saf / ndd);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  // The function returns infinity if the plane is not hit from inside, negative
  // if the point is outside
  using Bool_v = vecCore::Mask_v<Real_v>;
  distance     = InfinityLength<Real_v>();

  Real_v ndd   = NonZero(direction.Dot(fNormal));
  Real_v saf   = DistPlane(point);
  Bool_v valid = ndd > Real_v(0.) && saf < Real_v(kTolerance);

  vecCore__MaskedAssignFunc(distance, saf > Real_v(kTolerance), -InfinityLength<Real_v>());

  if (vecCore::EarlyReturnAllowed()) {
    if (vecCore::MaskEmpty(valid)) return;
  }

  // If competing with other planes, the minimum distance is winning
  vecCore__MaskedAssignFunc(distance, valid, -saf / ndd);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::SafetyToIn(Vector3D<Real_v> const &point, Real_v &distance) const
{
  // The safety contains the sign, i.e. if point is inside it is negative.
  // If competing with other planes, the maximum distance is winning
  distance = DistPlane(point);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void Plane::SafetyToOut(Vector3D<Real_v> const &point, Real_v &distance) const
{
  // The safety contains the sign, i.e. if point is outside it is negative.
  // If competing with other planes, the minimum distance is winning
  distance = -DistPlane(point);
}

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANE_H_
