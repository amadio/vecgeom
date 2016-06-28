#ifndef VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/PolygonalShell.h"
#include "volumes/kernel/BoxImplementation.h"

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

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    //
  }

  template <typename Stream>
  static void PrintType(Stream &st)
  {
    (void)st;
    //...
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "SExtruImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    // st << "UnplacedSExtruVolume";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &p, Bool_v &inside)
  {
    inside    = Bool_v(false);
    auto done = p.z() > Real_v(unplaced.fUpperZ);
    done |= p.z() < Real_v(unplaced.fLowerZ);
    if (vecCore::MaskFull(done)) return;
    inside = !done && unplaced.fPolygon.Contains(p);
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Inside_t &inside)
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
    Bool_v onZ   = vecCore::math::Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    onZ |= vecCore::math::Abs(point.z() - unplaced.fLowerZ) < kTolerance;

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
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &polyshell, Vector3D<Real_v> const &p, Vector3D<Real_v> const &dir,
                           Real_v const &stepMax, Real_v &distance)
  {
    distance = Real_v(kInfinity);
    // consider adding bounding box check

    // check collision with +z or -z
    const auto s = vecCore::Blend(dir.z() > Real_v(0.), p.z() - polyshell.fLowerZ, polyshell.fUpperZ - p.z());

    const auto canhit = s < Real_v(kTolerance);
    if (!vecCore::MaskEmpty(canhit)) {
      const auto dist = -s / vecCore::math::Abs(dir.z());
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
    vecCore::MaskedAssign(distance, distance == Real_v(kInfinity), polyshell.DistanceToIn(p, dir));
    return;
  }

  template <typename Real_v>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &polyshell, Vector3D<Real_v> const &p, Vector3D<Real_v> const &dir,
                            Real_v const & /* stepMax */, Real_v &distance)
  {
    distance = Real_v(-1.);
    // or do a hit check; if not then it has to be the z planes
    const auto dshell   = polyshell.DistanceToOut(p, dir);
    const auto hitshell = dshell < Real_v(kInfinity);
    if (vecCore::MaskFull(hitshell)) {
      distance = dshell;
      return;
    }
    const auto correctZ = vecCore::Blend(dir.z() > Real_v(0.), Real_v(polyshell.fUpperZ), Real_v(polyshell.fLowerZ));
    distance            = (correctZ - p.z()) / dir.z();
    return;
  }

  template <typename Real_v>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &polyshell, Vector3D<Real_v> const &point, Real_v &safety)
  {
    Vector3D<Precision> aMin, aMax;
    polyshell.Extent(aMin, aMax);

    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v isInExtent;
    ABBoxImplementation::ABBoxContainsKernelGeneric(aMin, aMax, point, isInExtent);

    // no one is in --> return precise safety to box
    if (vecCore::MaskEmpty(isInExtent)) {
      safety = std::sqrt(ABBoxImplementation::ABBoxSafetySqr(aMin, aMax, point));
      return;
    }

    const auto zSafety1 = polyshell.fLowerZ - point.z();
    const auto zSafety2 = polyshell.fUpperZ - point.z();
    if (vecCore::math::Abs(zSafety1) < kTolerance || vecCore::math::Abs(zSafety2) < kTolerance) {
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
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &polyshell, Vector3D<Real_v> const &point, Real_v &safety)
  {
    int unused;
    safety = std::sqrt(polyshell.fPolygon.SafetySqr(point, unused));
    safety = vecCore::math::Min(safety, polyshell.fUpperZ - point.z());
    safety = vecCore::math::Min(safety, point.z() - polyshell.fLowerZ);
  }

  template <typename Real_v>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // very rough implementation
    // not doing any sort of vector addition for normals on corners etc.
    valid = false;
    Vector3D<Real_v> normal(0., 0., 0.);

    // check conditions for surface first
    using Bool_v    = vecCore::Mask_v<Real_v>;
    Bool_v onUpperZ = vecCore::math::Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    Bool_v onLowerZ = vecCore::math::Abs(point.z() - unplaced.fLowerZ) < kTolerance;

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
}
} // end namespaces

#endif
