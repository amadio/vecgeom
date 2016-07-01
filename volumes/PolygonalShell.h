#ifndef VECGEOM_POLYGONAL_SHELL_H
#define VECGEOM_POLYGONAL_SHELL_H

#include "base/Global.h"
#include "volumes/PlanarPolygon.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PolygonalShell;);
VECGEOM_DEVICE_DECLARE_CONV(class, PolygonalShell);

inline namespace VECGEOM_IMPL_NAMESPACE {

// a set of z-axis aligned rectangles
// looking from the z - direction the rectangles form a convex or concave polygon
class PolygonalShell : AlignedBase {

private:
  // the polygon (with friend access)
  PlanarPolygon fPolygon;
  Precision fLowerZ; // lower z plane
  Precision fUpperZ; // upper z plane

  friend class SimpleExtruPolygon;
  friend struct SExtruImplementation;
  friend class UnplacedSExtruVolume;

public:
  VECGEOM_CUDA_HEADER_BOTH
  PolygonalShell(int nvertices, double *x, double *y, Precision lowerz, Precision upperz)
      : fPolygon(nvertices, x, y), fLowerZ(lowerz), fUpperZ(upperz)
  {
  }

  // the area of the shell ( does not include the area of the planar polygon )
  VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const
  {
    const auto kS = fPolygon.fVertices.size();
    Precision area(0.);
    for (size_t i = 0; i < kS; ++i) {
      // vertex lengh x (fUpperZ - fLowerZ)
      area += fPolygon.fLengthSqr[i];
    }
    return std::sqrt(area) * (fUpperZ - fLowerZ);
  }

  VECGEOM_CUDA_HEADER_BOTH
  PlanarPolygon const &GetPolygon() const { return fPolygon; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetLowerZ() const { return fLowerZ; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetUpperZ() const { return fUpperZ; }

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Real_v> &aMin, Vector3D<Real_v> &aMax) const
  {
    aMin[0] = Real_v(fPolygon.GetMinX());
    aMin[1] = Real_v(fPolygon.GetMinY());
    aMin[2] = Real_v(fLowerZ);

    aMax[0] = Real_v(fPolygon.GetMaxX());
    aMax[1] = Real_v(fPolygon.GetMaxY());
    aMax[2] = Real_v(fUpperZ);
  }

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    return fPolygon.IsConvex() ? DistanceToInConvex(point, dir) : DistanceToInConcave(point, dir);
  }

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToInConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfinity);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj >= -kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist > kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ);
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters);

        vecCore::MaskedAssign(result, !done && intersects, dist);
        done |= intersects;
      }
      if (vecCore::MaskFull(done)) {
        return result;
      }
    }
    return result;
  }

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToInConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfinity);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj >= -kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist > kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ);
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters);

        vecCore::MaskedAssign(result, intersects, vecCore::math::Min(dist, result));
      }
      // if (vecCore::MaskFull(done)) {
      //        return result;
      //      }
    }
    return result;
  }

  // -- DistanceToOut --

  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    return fPolygon.IsConvex() ? DistanceToOutConvex(point, dir) : DistanceToOutConcave(point, dir);
  }

  // convex distance to out; checks for hits and aborts loop if hit found
  // NOTE: this kernel is the same as DistanceToIn apart from the comparisons for early return
  // these could become a template parameter
  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToOutConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfinity);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj <= kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist < -kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters) && zRangeOk &&
                                  (dist >= -Real_v(kTolerance));

        vecCore::MaskedAssign(result, !done && intersects, dist);
        done |= intersects;
      }
      if (vecCore::MaskFull(done)) {
        return result;
      }
    }
    return result;
  }

  // DistanceToOut for the concave case
  // we should ideally combine this with the other kernel
  template <typename Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToOutConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfinity);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj <= kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist < -kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters) && zRangeOk &&
                                  (dist >= -Real_v(kTolerance));

        vecCore::MaskedAssign(result, intersects, vecCore::math::Min(dist, result));
        // done |= intersects;
      }
      // if (vecCore::MaskFull(done)) {
      //  return result;
      //}
    }
    return result;
  }

}; // end class

#define SPECIALIZATION
#ifdef SPECIALIZATION
// template specialization for Distance functions
template <>
inline Precision PolygonalShell::DistanceToOutConvex(Vector3D<Precision> const &point,
                                                     Vector3D<Precision> const &dir) const
{
  // this template specialization provides internal vectorization
  // of the loop over i via vecCore vector types

  using Real_v = vecgeom::VectorBackend::Real_v;
  using Bool_v = vecCore::Mask_v<Real_v>;

  const auto S = fPolygon.fVertices.size();

  // the vector stride
  const auto kVS = vecCore::VectorSize<Real_v>();

  using vecCore::FromPtr;
  using vecCore::MaskEmpty;
  using vecCore::MaskFull;
  for (size_t i = 0; i < S; i += kVS) { // side/rectangle index
    // load parameters into vector
    const auto A(FromPtr<Real_v>(&fPolygon.fA[i]));
    const auto B(FromPtr<Real_v>(&fPolygon.fB[i]));
    const auto D(FromPtr<Real_v>(&fPolygon.fD[i]));

    // approaching from right side?
    // under the assumption that surface normals points "inwards"
    const auto proj          = A * dir.x() + B * dir.y();
    const Bool_v sidecorrect = proj <= Real_v(kTolerance);
    if (MaskEmpty(sidecorrect)) {
      continue;
    }

    // the distance to the plane (specialized for fNormalsZ == 0)
    const auto pdist         = A * point.x() + B * point.y() + D;
    const Bool_v moving_away = pdist < -Real_v(kTolerance);
    if (MaskFull(moving_away)) {
      continue;
    }

    const auto dist = -pdist / NonZero(proj);

    // propagate to plane (first just z)
    const auto zInters(point.z() + dist * dir.z());
    const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
    if (!MaskEmpty(zRangeOk)) {
      // check intersection with rest of rectangle
      const auto xInters(point.x() + dist * dir.x());
      const auto yInters(point.y() + dist * dir.y());

      // we could already check if intersection within the known extent

      const Bool_v intersects =
          fPolygon.OnSegment<Real_v, Real_v, Bool_v>(i, xInters, yInters) && zRangeOk && (dist >= -Real_v(kTolerance));

      // TODO: ignore "zero" results when going wrong direction
      // we are done if we have any hit
      if (!MaskEmpty(intersects)) {
        // need to find out correct lane
        for (size_t lane = 0; lane < kVS; ++lane) {
          if (vecCore::MaskLaneAt(intersects, lane)) {
            return vecCore::LaneAt(dist, lane);
          }
        }
      }
    }
  }
  return kInfinity;
}

// template specialization for Distance functions
template <>
inline Precision PolygonalShell::DistanceToInConvex(Vector3D<Precision> const &point,
                                                    Vector3D<Precision> const &dir) const
{
  // this template specialization provides internal vectorization
  // of the loop over i via vecCore vector types

  using Real_v = vecgeom::VectorBackend::Real_v;
  using Bool_v = vecCore::Mask_v<Real_v>;

  const auto S = fPolygon.fVertices.size();

  // the vector stride
  const auto kVS = vecCore::VectorSize<Real_v>();

  using vecCore::FromPtr;
  using vecCore::MaskEmpty;
  using vecCore::MaskFull;
  for (size_t i = 0; i < S; i += kVS) { // side/rectangle index
    // load parameters into vector
    const auto A(FromPtr<Real_v>(&fPolygon.fA[i]));
    const auto B(FromPtr<Real_v>(&fPolygon.fB[i]));
    const auto D(FromPtr<Real_v>(&fPolygon.fD[i]));

    // approaching from right side?
    // under the assumption that surface normals points "inwards"
    const auto proj          = A * dir.x() + B * dir.y();
    const Bool_v sidecorrect = proj >= -Real_v(kTolerance);
    if (MaskEmpty(sidecorrect)) {
      continue;
    }

    // the distance to the plane (specialized for fNormalsZ == 0)
    const auto pdist         = A * point.x() + B * point.y() + D;
    const Bool_v moving_away = pdist > Real_v(kTolerance);
    if (MaskFull(moving_away)) {
      continue;
    }

    const auto dist = -pdist / NonZero(proj);

    // propagate to plane (first just z)
    const auto zInters(point.z() + dist * dir.z());
    const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
    if (!MaskEmpty(zRangeOk)) {
      // check intersection with rest of rectangle
      const auto xInters(point.x() + dist * dir.x());
      const auto yInters(point.y() + dist * dir.y());

      // we could already check if intersection within the known extent

      const Bool_v intersects =
          fPolygon.OnSegment<Real_v, Real_v, Bool_v>(i, xInters, yInters) && zRangeOk && (dist >= -Real_v(kTolerance));

      // TODO: ignore "zero" results when going wrong direction
      // we are done if we have any hit
      if (!MaskEmpty(intersects)) {
        // need to find out correct lane
        for (size_t lane = 0; lane < kVS; ++lane) {
          if (vecCore::MaskLaneAt(intersects, lane)) {
            return vecCore::LaneAt(dist, lane);
          }
        }
      }
    }
  }
  return kInfinity;
}

#endif

} // end inline namespace
} // end vecgeom namespace

#endif
