/// \file CutPlanes.h
/// \author mgheata

#ifndef VECGEOM_VOLUMES_CUTPLANES_H_
#define VECGEOM_VOLUMES_CUTPLANES_H_

#include "volumes/Plane.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class CutPlanes;);
VECGEOM_DEVICE_DECLARE_CONV(class, CutPlanes);

inline namespace VECGEOM_IMPL_NAMESPACE {

class CutPlanes : public AlignedBase {

private:
  Plane fCutPlanes[2]; ///< Two cut planes

public:
  VECGEOM_CUDA_HEADER_BOTH
  CutPlanes() {}

  VECGEOM_CUDA_HEADER_BOTH
  ~CutPlanes() = default;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Plane const &GetCutPlane(int i) const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDistance(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  void Set(int index, Vector3D<Precision> const &normal, Vector3D<Precision> const &origin);

  VECGEOM_CUDA_HEADER_BOTH
  void Set(int index, Vector3D<Precision> const &normal, Precision distance);

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

}; // class CutPlanes

std::ostream &operator<<(std::ostream &os, CutPlanes const &planes);

VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
Plane const &CutPlanes::GetCutPlane(int i) const
{
  return fCutPlanes[i];
}

VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> CutPlanes::GetNormal(int i) const
{
  return fCutPlanes[i].GetNormal();
}

VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
Precision CutPlanes::GetDistance(int i) const
{
  return fCutPlanes[i].GetDistance();
}

template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::Contains(Vector3D<Real_v> const &point, Bool_v &inside) const
{
  inside = fCutPlanes[0].DistPlane(point) < Real_v(0.) && fCutPlanes[1].DistPlane(point) < Real_v(0.);
}

template <typename Real_v, typename Inside_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::Inside(Vector3D<Real_v> const &point, Inside_v &inside) const
{
  Real_v d0 = fCutPlanes[0].DistPlane(point);
  Real_v d1 = fCutPlanes[1].DistPlane(point);

  inside =
      vecCore::Blend(d0 < Real_v(0.0) && d1 < Real_v(0.0), Inside_v(EInside::kInside), Inside_v(EInside::kOutside));
  vecCore::MaskedAssign(inside,
                        vecCore::math::Abs(d0) < Real_v(kTolerance) || vecCore::math::Abs(d1) < Real_v(kTolerance),
                        Inside_v(EInside::kSurface));
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  // The function returns a negative distance for points coming from the
  // region in between the two planes, or from points outside going outwards
  // If a particle has valid crossings withe the 2 planes, the maximum distance
  // has to be taken
  Real_v d0, d1;
  fCutPlanes[0].DistanceToIn(point, direction, d0);
  fCutPlanes[1].DistanceToIn(point, direction, d1);
  distance = vecCore::math::Max(d0, d1);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].DistanceToOut(point, direction, d0);
  fCutPlanes[1].DistanceToOut(point, direction, d1);
  distance = vecCore::math::Min(d0, d1);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::SafetyToIn(Vector3D<Real_v> const &point, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].SafetyToIn(point, d0);
  fCutPlanes[1].SafetyToIn(point, d1);
  distance = vecCore::math::Max(d0, d1);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CutPlanes::SafetyToOut(Vector3D<Real_v> const &point, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].SafetyToOut(point, d0);
  fCutPlanes[1].SafetyToOut(point, d1);
  distance = vecCore::math::Min(d0, d1);
}

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_CUTPLANES_H_
