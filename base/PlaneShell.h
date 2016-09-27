//===-- base/PlaneShell.h ----------------------------*- C++ -*-===//
/// \file PlaneShell.h
/// \author Guilherme Lima (lima at fnal dot gov)

#ifndef VECGEOM_BASE_SIDEPLANES_H_
#define VECGEOM_BASE_SIDEPLANES_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"

// namespace vecgeom::cuda { template <typename Real_v, int N> class PlaneShell; }
#include <VecCore/VecCore>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Uses SoA layout to store arrays of N (plane) parameters,
 *        representing a set of planes defining a volume or shape.
 *
 * For some volumes, e.g. trapezoid, when two of the planes are
 * parallel, they should be set perpendicular to the Z-axis, and then
 * the inside/outside calculations become trivial.  Therefore those planes
 * should NOT be included in this class.
 *
 * @details If vector acceleration is enabled, the scalar template
 *        instantiation will use vector instructions for operations
 *        when possible.
 */

template <int N, typename Type>
struct PlaneShell {

  // Using a SOA-like data structure for vectorization
  Precision fA[N];
  Precision fB[N];
  Precision fC[N];
  Precision fD[N];

public:
  /**
   * Initializes the SOA with existing data arrays, performing no allocation.
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell(Precision *const a, Precision *const b, Precision *const c, Precision *const d)
  {
    memcpy(&(this->fA), a, N * sizeof(Type));
    memcpy(&(this->fB), b, N * sizeof(Type));
    memcpy(&(this->fC), c, N * sizeof(Type));
    memcpy(&(this->fD), d, N * sizeof(Type));
  }

  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell()
  {
    memset(&(this->fA), 0, N * sizeof(Type));
    memset(&(this->fB), 0, N * sizeof(Type));
    memset(&(this->fC), 0, N * sizeof(Type));
    memset(&(this->fD), 0, N * sizeof(Type));
  }

  /**
   * Copy constructor
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell(PlaneShell const &other)
  {
    memcpy(&(this->fA), &(other->fA), N * sizeof(Type));
    memcpy(&(this->fB), &(other->fB), N * sizeof(Type));
    memcpy(&(this->fC), &(other->fC), N * sizeof(Type));
    memcpy(&(this->fD), &(other->fD), N * sizeof(Type));
  }

  /**
   * assignment operator
   */
  VECGEOM_CUDA_HEADER_BOTH
  PlaneShell &operator=(PlaneShell const &other)
  {
    memcpy(this->fA, other.fA, N * sizeof(Type));
    memcpy(this->fB, other.fB, N * sizeof(Type));
    memcpy(this->fC, other.fC, N * sizeof(Type));
    memcpy(this->fD, other.fD, N * sizeof(Type));
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(int i, Precision a, Precision b, Precision c, Precision d)
  {
    fA[i] = a;
    fB[i] = b;
    fC[i] = c;
    fD[i] = d;
  }

  VECGEOM_CUDA_HEADER_BOTH
  unsigned int size() { return N; }

  VECGEOM_CUDA_HEADER_BOTH
  ~PlaneShell() {}

  /// \return the distance from point to each plane.  The type returned is float, double, or various SIMD vector types.
  /// Distances are negative (positive) for points in same (opposite) side from plane as the normal vector.
  template <typename Type2>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void DistanceToPoint(Vector3D<Type2> const &point, Type2 *distances) const
  {
    for (int i = 0; i < N; ++i) {
      distances[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }
  }

  /// \return the projection of a (Vector3D) direction into each plane's normal vector.
  /// The type returned is float, double, or various SIMD vector types.
  template <typename Type2>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void ProjectionToNormal(Vector3D<Type2> const &dir, Type2 *projection) const
  {
    for (int i = 0; i < N; ++i) {
      projection[i] = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();
    }
  }

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void GenericKernelForContainsAndInside(Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyInside,
                                         vecCore::Mask_v<Real_v> &completelyOutside) const
  {
    // auto-vectorizable loop for Backend==scalar
    Real_v dist[N];
    for (unsigned int i = 0; i < N; ++i) {
      dist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }

    // analysis loop - not auto-vectorizable
    constexpr Precision trapSurfaceTolerance = 1.0e-6;
    for (unsigned int i = 0; i < N; ++i) {
      // is it outside of this side plane?
      completelyOutside = completelyOutside || (dist[i] > MakePlusTolerant<ForInside>(0., trapSurfaceTolerance));
      if (ForInside) {
        completelyInside = completelyInside && (dist[i] < MakeMinusTolerant<ForInside>(0., trapSurfaceTolerance));
      }
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(completelyOutside)) return;
    }
  }

  /// \return the distance to the planar shell when the point is located outside.
  /// The type returned is the type corresponding to the backend given.
  /// For some special cases, the value returned is:
  ///     (1) +inf, if point+dir is outside & moving AWAY FROM OR PARALLEL TO any plane,
  ///     (2) -1, if point+dir crosses out a plane BEFORE crossing in ALL other planes (wrong-side)
  ///
  /// Note: smin0 parameter is needed here, otherwise smax can become smaller than smin0,
  ///   which means condition (2) happens and +inf must be returned.  Without smin0, this
  ///   condition is sometimes missed.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToIn(Vector3D<Real_v> const &point, Real_v const &smin0, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);

    Real_v distIn(kInfLength); // set for earlier returns
    Real_v smax(kInfLength);
    Real_v smin(smin0);

    // hope for a vectorization of this part for Backend==scalar!!
    Real_v pdist[N];
    Real_v proj[N];
    Real_v vdist[N];
    // vectorizable part
    for (int i = 0; i < N; ++i) {
      pdist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
      proj[i]  = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();

      // note(SW): on my machine it was better to keep vdist[N] instead of a local variable vdist below
      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }

    // wrong-side check: if (inside && smin<0) return -1
    Bool_v inside(smin < MakeMinusTolerant<true>(0.0));
    for (int i = 0; i < N; ++i) {
      inside = inside && pdist[i] < MakeMinusTolerant<true>(0.0);
    }
    vecCore::MaskedAssign(distIn, inside, Real_v(-1.0));
    done = done || inside;
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return distIn;

    // analysis loop
    for (int i = 0; i < N; ++i) {
      // if outside and moving away, return infinity
      Bool_v posPoint = pdist[i] > MakeMinusTolerant<true>(0.0);
      Bool_v posDir   = proj[i] > 0;
      done            = done || (posPoint && posDir);

      // check if trajectory will intercept plane within current range (smin,smax)
      Bool_v interceptFromInside = (!posPoint && posDir);
      done                       = done || (interceptFromInside && vdist[i] < smin);

      Bool_v interceptFromOutside = (posPoint && !posDir);
      done                        = done || (interceptFromOutside && vdist[i] > smax);
      if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return distIn;

      // update smin,smax
      Bool_v validVdist = (smin < vdist[i] && vdist[i] < smax);
      vecCore::MaskedAssign(smin, !done && interceptFromOutside && validVdist, vdist[i]);
      vecCore::MaskedAssign(smax, !done && interceptFromInside && validVdist, vdist[i]);
    }

    // Survivors will return smin, which is the maximum distance in an interceptFromOutside situation
    // (SW: not sure this is true since smin is initialized from outside and can have any arbitrary value)
    vecCore::MaskedAssign(distIn, !done && smin >= -kTolerance, smin);
    return distIn;
  }

  /// \return the distance to the planar shell when the point is located within the shell itself.
  /// The type returned is the type corresponding to the backend given.
  /// For some special cases, the value returned is:
  ///     (1) -1, if point is outside (wrong-side)
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v distOut(kInfLength);

    // hope for a vectorization of this part for Backend==scalar !!
    // the idea is to put vectorizable things into this loop
    // and separate the analysis into a separate loop if need be
    Real_v pdist[N];
    Real_v proj[N];
    Real_v vdist[N];
    for (int i = 0; i < N; ++i) {
      pdist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
      proj[i]  = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();
      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }

    // early return if point is outside of plane
    for (int i = 0; i < N; ++i) {
      done = done || (pdist[i] > kHalfTolerance);
    }
    vecCore::MaskedAssign(distOut, done, Real_v(-1.0));
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskFull(done)) return distOut;

    // add = in vdist[i]>=0  and "proj[i]>0" in order to pass unit tests, but this will slow down DistToOut()!!! check
    // effect!
    for (int i = 0; i < N; ++i) {
      Bool_v test = (vdist[i] >= -kHalfTolerance && proj[i] > 0 && vdist[i] < distOut);
      vecCore::MaskedAssign(distOut, test, vdist[i]);
    }

    return distOut;
  }

  /// \return the safety distance to the planar shell when the point is located within the shell itself.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SafetyToIn(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }

    // non-vectorizable part
    for (int i = 0; i < N; ++i) {
      vecCore::MaskedAssign(safety, dist[i] > safety, dist[i]);
    }
  }

  /// \return the distance to the planar shell when the point is located within the shell itself.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SafetyToOut(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = -(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    for (int i = 0; i < N; ++i) {
      vecCore::MaskedAssign(safety, dist[i] < safety, dist[i]);
    }

    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  size_t ClosestFace(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = Abs(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    using Bool_v    = vecCore::Mask_v<Real_v>;
    using Index_v   = vecCore::Index<Real_v>;
    Index_v closest = static_cast<Index_v>(-1);
    for (size_t i = 0; i < N; ++i) {
      Bool_v closer = dist[i] < safety;
      vecCore::MaskedAssign(safety, closer, dist[i]);
      vecCore::MaskedAssign(closest, closer, i);
    }

    return closest;
  }

  /// \return a *non-normalized* vector normal to the plane containing (or really close) to point.
  /// In most cases the resulting normal is either (0,0,0) when point is not close to any plane,
  /// or the normal of that single plane really close to the point (in this case, it *is* normalized).
  ///
  /// Note: If the point is really close to more than one plane, those planes' normals are added,
  /// and in this case the vector returned *is not* normalized.  This can be used as a flag, so that
  /// the callee knows that nsurf==2 (the maximum value possible).
  ///
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Real_v> NormalKernel(Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid) const
  {
    Vector3D<Real_v> normal(0., 0., 0.);

    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = -(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    constexpr double delta = 1000. * kTolerance;
    for (int i = 0; i < N; ++i) {
      vecCore::MaskedAssign(normal[0], dist[i] <= delta, normal[0] + this->fA[i]);
      vecCore::MaskedAssign(normal[1], dist[i] <= delta, normal[1] + this->fB[i]);
      vecCore::MaskedAssign(normal[2], dist[i] <= delta, normal[2] + this->fC[i]);
      // valid becomes false if any point is away from surface by more than delta
      valid = valid && dist[i] >= -delta;
    }

    // Note: this could be (rarely) a non-normalized normal vector (when point is close to 2 planes)
    return normal;
  }
};

} // end inline namespace
} // end global namespace

#endif // VECGEOM_BASE_SIDEPLANES_H_
