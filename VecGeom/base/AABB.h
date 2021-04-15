/// \file AABB.h
/// \author Guilherme Amadio

#ifndef VECGEOM_BASE_AABB_H_
#define VECGEOM_BASE_AABB_H_

#include "VecGeom/base/Vector3D.h"

#include <algorithm>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Simple class to represent Axis-Aligned Bounding Boxes (AABB).
 * @details The AABB is represented internally using the minimum and maximum corners.
 */

class AABB {
public:
  /** Default constructor. Required to use AABBs as elements in standard containers. */
  AABB() = default;
  /** Constructor. */
  AABB(Vector3D<Precision> Min, Vector3D<Precision> Max) : fMin(Min), fMax(Max) {}

  /** Returns the minimum coordinates of the AABB. */
  Vector3D<Precision> Min() const { return fMin; }

  /** Returns the maximum coordinates of the AABB. */
  Vector3D<Precision> Max() const { return fMax; }

  /** Returns the center of the AABB. */
  Vector3D<Precision> Center() const { return 0.5 * (fMax + fMin); }

  /** Returns the extents of the AABB along each axis. */
  Vector3D<Precision> Size() const { return fMax - fMin; }

  /** Expand AABB. @param s Amount by which to expand in each direction. */
  void Expand(Precision s)
  {
    s *= 0.5;
    fMin -= s;
    fMax += s;
  }

  /** Check whether a point is contained by the AABB. */
  bool Contains(Vector3D<Precision> p) const
  {
    return p[0] >= fMin[0] && p[0] < fMax[0] && p[1] >= fMin[1] && p[1] < fMax[1] && p[2] >= fMin[2] && p[2] < fMax[2];
  }

  /**
   * Compute a safety margin from a point to AABB's surface.
   * The AABB is guaranteed to be further than the safety.
   * @param[in] point Input point.
   * @remark Returns a negative value if point is inside AABB.
   */
  Precision Safety(Vector3D<Precision> point) const { return ((point - Center()).Abs() - 0.5 * Size()).Max(); }

  /**
   * Compute distance from a point to AABB's surface along the given direction.
   * @param[in] point Starting point for input ray.
   * @param[in] direction Direction of the input ray.
   * @param[in] step Maximum distance for which an intersection should be reported.
   * @remark Returns a negative value if starting point is already inside AABB.
   */
  Precision Distance(Vector3D<Precision> point, Vector3D<Precision> direction) const
  {
    Precision tmin, tmax;
    ComputeIntersection(point, direction, tmin, tmax);
    return (tmin < tmax && tmax > 0.0) ? tmin : kInfLength;
  }

  /**
   * Compute distance from a point to AABB's surface along the given direction.
   * @param[in] point Starting point for input ray.
   * @param[in] invdir Inverse of direction vector of the input ray.
   * @param[in] step Maximum distance for which an intersection should be reported.
   * @remark Returns a negative value if starting point is already inside AABB.
   */
  Precision DistanceInvDir(Vector3D<Precision> point, Vector3D<Precision> invdir) const
  {
    Precision tmin, tmax;
    ComputeIntersectionInvDir(point, invdir, tmin, tmax);
    return (tmin < tmax && tmax > 0.0) ? tmin : kInfLength;
  }

  /**
   * Compute intersection interval with a line, given a point and a direction defining it.
   * @param[in] point Starting point on the line.
   * @param[in] direction Direction of the line.
   * @param tmin[out] Minimum `t` such that `point + t * direction` intersects the AABB.
   * @param tmax[out] Maximum `t` such that `point + t * direction` intersects the AABB.
   */
  void ComputeIntersection(Vector3D<Precision> point, Vector3D<Precision> direction, Precision &tmin,
                           Precision &tmax) const
  {
    Vector3D<Precision> invdir(1.0 / NonZero(direction[0]), 1.0 / NonZero(direction[1]), 1.0 / NonZero(direction[2]));
    ComputeIntersectionInvDir(point, invdir, tmin, tmax);
  }

  /**
   * Compute intersection interval with a line, given a point and the inverse of the direction vector defining it.
   * @param[in] point Starting point on the line.
   * @param[in] invdir Inverse of direction vector of the input ray.
   * @param tmin[out] Minimum `t` such that `point + t * direction` intersects the AABB.
   * @param tmax[out] Maximum `t` such that `point + t * direction` intersects the AABB.
   */
  void ComputeIntersectionInvDir(Vector3D<Precision> point, Vector3D<Precision> invdir, Precision &tmin,
                                 Precision &tmax) const
  {
    using std::swap;

    Vector3D<Precision> t0 = (fMin - point) * invdir;
    Vector3D<Precision> t1 = (fMax - point) * invdir;

    if (t0[0] > t1[0]) swap(t0[0], t1[0]);
    if (t0[1] > t1[1]) swap(t0[1], t1[1]);
    if (t0[2] > t1[2]) swap(t0[2], t1[2]);

    tmin = t0.Max();
    tmax = t1.Min();
  }

  /**
   * Check whether the line intersects AABB.
   * @param[in] point Starting point on the line.
   * @param[in] direction Direction of the line.
   */
  bool Intersect(Vector3D<Precision> point, Vector3D<Precision> direction) const
  {
    Precision tmin, tmax;
    ComputeIntersection(point, direction, tmin, tmax);
    return tmin <= tmax && tmax >= 0.0;
  }

  /**
   * Check whether the line intersects AABB.
   * @param[in] point Starting point on the line.
   * @param[in] invdir Inverse of direction vector of the input ray.
   */
  bool IntersectInvDir(Vector3D<Precision> point, Vector3D<Precision> invdir) const
  {
    Precision tmin, tmax;
    ComputeIntersectionInvDir(point, invdir, tmin, tmax);
    return tmin <= tmax && tmax >= 0.0;
  }

  /**
   * Check whether the ray intersects AABB within given step length.
   * @param[in] point Starting point for input ray.
   * @param[in] direction Direction of the input ray.
   * @param[in] step Maximum distance for which an intersection should be reported.
   * @remark Does not report an intersection if the AABB lies fully behind the ray.
   */
  bool Intersect(Vector3D<Precision> point, Vector3D<Precision> direction, Precision step) const
  {
    Precision tmin, tmax;
    ComputeIntersection(point, direction, tmin, tmax);
    return tmin <= tmax && tmax >= 0.0 && tmin < step;
  }

  /**
   * Check whether the ray intersects AABB within given step length.
   * @param[in] point Starting point for input ray.
   * @param[in] invdir Inverse of direction vector of the input ray.
   * @param[in] step Maximum distance for which an intersection should be reported.
   * @remark Does not report an intersection if the AABB lies fully behind the ray.
   */
  bool IntersectInvDir(Vector3D<Precision> point, Vector3D<Precision> invdir, Precision step) const
  {
    Precision tmin, tmax;
    ComputeIntersectionInvDir(point, invdir, tmin, tmax);
    return tmin <= tmax && tmax >= 0.0 && tmin < step;
  }

  /**
   * Compute minimum AABB that encloses the two input AABBs, A and B.
   */
  static AABB Union(AABB const &A, AABB const &B)
  {
    using std::max;
    using std::min;
    Vector3D<Precision> MinC(min(A.fMin[0], B.fMin[0]), min(A.fMin[1], B.fMin[1]), min(A.fMin[2], B.fMin[2]));
    Vector3D<Precision> MaxC(max(A.fMax[0], B.fMax[0]), max(A.fMax[1], B.fMax[1]), max(A.fMax[2], B.fMax[2]));
    return {MinC, MaxC};
  }

private:
  Vector3D<Precision> fMin; ///< Minimum coordinates of the AABB.
  Vector3D<Precision> fMax; ///< Maximum coordinates of the AABB.
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
