#ifndef VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct SExtruImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, SExtruImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedSExtru;
class PolygonalShell;
class UnplacedSExtruVolume;

struct SExtruImplementation {

  using PlacedShape_t    = PlacedSExtru;
  using UnplacedStruct_t = PolygonalShell;
  using UnplacedVolume_t = UnplacedSExtruVolume;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //
  }

  template <typename Stream>
  static void PrintType(Stream &)
  {
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &)
  {
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &)
  {
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &p, Bool_v &inside)
  {
    inside    = Bool_v(false);
    auto done = p.z() > Real_v(unplaced.fUpperZ);
    done |= p.z() < Real_v(unplaced.fLowerZ);
    if (vecCore::MaskFull(done)) return;
    if (unplaced.fPolygon.IsConvex())
      inside = !done && unplaced.fPolygon.ContainsConvex(p);
    else
      inside = !done && unplaced.fPolygon.Contains(p);
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    // this is a quick / non-optimized ans scalar only implementation:
    if (point.z() > unplaced.fUpperZ + kTolerance) {
      inside = vecgeom::kOutside;
      return;
    }
    if (point.z() < unplaced.fLowerZ - kTolerance) {
      inside = vecgeom::kOutside;
      return;
    }

    // check conditions for surface first
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v onZ   = Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    onZ |= Abs(point.z() - unplaced.fLowerZ) < kTolerance;

    if (unplaced.fPolygon.IsConvex()) {
      inside = unplaced.fPolygon.InsideConvex(point);
      if (onZ && inside != vecgeom::kOutside) inside = vecgeom::kSurface;
      return;
    }

    if (onZ) {
      if (unplaced.fPolygon.Contains(point)) {
        inside = vecgeom::kSurface;
        return;
      }
    }

    // not on z-surface --> check other surface with safety for moment
    if (unplaced.fLowerZ <= point.z() && point.z() <= unplaced.fUpperZ) {
      int unused;
      auto s = unplaced.fPolygon.SafetySqr(point, unused);
      if (s < kTolerance * kTolerance) {
        inside = vecgeom::kSurface;
        return;
      }
    }

    Bool_v c;
    Contains(unplaced, point, c);

    if (c)
      inside = vecgeom::kInside;
    else
      inside = vecgeom::kOutside;
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &polyshell,
                                                                        Vector3D<Real_v> const &p,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    if (polyshell.fPolygon.IsConvex()) {
      distance = polyshell.DistanceToInConvex(p, dir);
      return;
    }
    distance = Real_v(kInfLength);
    // consider adding bounding box check

    // check collision with +z or -z
    const auto s = vecCore::Blend(dir.z() > Real_v(0.), p.z() - polyshell.fLowerZ, polyshell.fUpperZ - p.z());

    const auto canhit = s < Real_v(kTolerance);
    if (!vecCore::MaskEmpty(canhit)) {
      const auto dist = -s / Abs(dir.z());
      // propagate
      const auto xInters = p.x() + dist * dir.x();
      const auto yInters = p.y() + dist * dir.y();

      const auto hits = polyshell.fPolygon.Contains(Vector3D<Real_v>(xInters, yInters, Real_v(0.)));

      vecCore::MaskedAssign(distance, hits, dist);
      if (vecCore::MaskFull(hits)) {
        return;
      }
    }

    // check collision with polyshell
    vecCore__MaskedAssignFunc(distance, distance == Real_v(kInfLength), polyshell.DistanceToIn(p, dir));
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &polyshell,
                                                                         Vector3D<Real_v> const &p,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    if (polyshell.fPolygon.IsConvex()) {
      distance = polyshell.DistanceToOutConvex(p, dir);
      return;
    }
    distance = Real_v(-1.);
    // or do a hit check; if not then it has to be the z planes
    const auto dshell   = polyshell.DistanceToOut(p, dir);
    const auto hitshell = dshell < Real_v(kInfLength);
    if (vecCore::MaskFull(hitshell)) {
      distance = dshell;
      return;
    }
    const auto correctZ = vecCore::Blend(dir.z() > Real_v(0.), Real_v(polyshell.fUpperZ), Real_v(polyshell.fLowerZ));
    distance            = (correctZ - p.z()) / dir.z();
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &polyshell,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (polyshell.fPolygon.IsConvex()) {
      Real_v safeZ = vecCore::math::Max(polyshell.fLowerZ - point.z(), point.z() - polyshell.fUpperZ);
      safety       = vecCore::math::Max(safeZ, polyshell.fPolygon.SafetyConvex(point, false));
      return;
    }

    Vector3D<Precision> aMin, aMax;
    polyshell.Extent(aMin, aMax);

    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v isInExtent;
    ABBoxImplementation::ABBoxContainsKernelGeneric(aMin, aMax, point, isInExtent);

    // no one is in --> return precise safety to box
    if (vecCore::MaskEmpty(isInExtent)) {
      const auto ssqr = ABBoxImplementation::ABBoxSafetySqr(aMin, aMax, point);
      if (ssqr <= 0.) {
        safety = 0.;
        return;
      }
      safety = std::sqrt(ssqr);
      return;
    }

    const auto zSafety1 = polyshell.fLowerZ - point.z();
    const auto zSafety2 = polyshell.fUpperZ - point.z();
    if (Abs(zSafety1) < kTolerance || Abs(zSafety2) < kTolerance) {
      // on the z - entering surface:
      // need more careful treatment
      bool c;
      Contains(polyshell, point, c);
      if (c) {
        safety = 0.;
        return;
      }
    }
    int unused;
    safety = std::sqrt(polyshell.fPolygon.SafetySqr(point, unused));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &polyshell,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    int unused;
    if (polyshell.fPolygon.IsConvex()) {
      Real_v safeZ = vecCore::math::Min(point.z() - polyshell.fLowerZ, polyshell.fUpperZ - point.z());
      safety       = vecCore::math::Min(safeZ, polyshell.fPolygon.SafetyConvex(point, true));
      return;
    }
    safety = std::sqrt(polyshell.fPolygon.SafetySqr(point, unused));
    safety = Min(safety, polyshell.fUpperZ - point.z());
    safety = Min(safety, point.z() - polyshell.fLowerZ);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // very rough implementation
    // not doing any sort of vector addition for normals on corners etc.
    valid = false;
    Vector3D<Real_v> normal(0., 0., 0.);

    // check conditions for surface first
    using Bool_v    = vecCore::Mask_v<Real_v>;
    Bool_v onUpperZ = Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    Bool_v onLowerZ = Abs(point.z() - unplaced.fLowerZ) < kTolerance;

    if (onUpperZ || onLowerZ) {
      if (unplaced.fPolygon.Contains(point)) {
        valid = true;
        if (onUpperZ)
          normal = Vector3D<Real_v>(0., 0., 1);
        else {
          normal = Vector3D<Real_v>(0., 0., -1.);
        }
        return normal;
      }
    }

    // not on z-surface --> check other surface with safety for moment
    if (unplaced.fLowerZ <= point.z() && point.z() <= unplaced.fUpperZ) {
      int surfaceindex;
      auto s = unplaced.fPolygon.SafetySqr(point, surfaceindex);
      normal = Vector3D<Real_v>(-unplaced.fPolygon.fA[surfaceindex], -unplaced.fPolygon.fB[surfaceindex], 0.);
      if (s < kTolerance * kTolerance) {
        valid = true;
      }
    }
    return normal;
  }

}; // End struct SExtruImplementation
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
