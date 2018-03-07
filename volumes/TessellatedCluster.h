/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_
#define VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_

#include <VecCore/VecCore>

#include "base/AlignedBase.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Vector.h"
#include "volumes/kernel/GenericKernels.h"
#include "Tile.h"
//#include "TriangleFacet.h"

namespace vecgeom {

// VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(class, TessellatedCluster, size_t, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

constexpr size_t kVecSize = vecCore::VectorSize<vecgeom::VectorBackend::Real_v>();

//______________________________________________________________________________
// Structure used for vectorizing queries on groups of triangles. Stores Real_v
// types matched with the compiled SIMD vectors of double.
template <size_t NVERT, typename Real_v>
class TessellatedCluster : public AlignedBase {
public:
  using T       = typename vecCore::ScalarType<Real_v>::Type;
  using Facet_t = Tile<NVERT, T>;

  Vector3D<Real_v> fNormals;            ///< Normals to facet components
  Real_v fDistances;                    ///< Distances from origin to facets
  Vector3D<Real_v> fSideVectors[NVERT]; ///< Side vectors of the triangular facets
  Vector3D<Real_v> fVertices[NVERT];    ///< Vertices stored in SIMD format
  size_t fIfacets[kVecSize]  = {};      ///< Real indices of facets
  Facet_t *fFacets[kVecSize] = {};      ///< Array of scalar facets matching the ones in the cluster
  Vector3D<T> fMinExtent;               ///< Minimum extent
  Vector3D<T> fMaxExtent;               ///< Maximum extent
  bool fConvex = false;                 ///< Convexity of the cluster with respect to the solid it belongs to

  VECCORE_ATT_HOST_DEVICE
  TessellatedCluster()
  {
    fMinExtent.Set(InfinityLength<T>());
    fMaxExtent.Set(-InfinityLength<T>());
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsConvex() const { return fConvex; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool CalculateConvexity()
  {
    bool convex = true;
    for (size_t i = 0; i < kVecSize; ++i)
      convex &= fFacets[i]->fConvex;
    return convex;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void GetVertex(size_t ifacet, size_t ivert, Vector3D<T> &vertex) const
  {
    vertex[0] = vecCore::Get(fVertices[ivert].x(), ifacet);
    vertex[1] = vecCore::Get(fVertices[ivert].y(), ifacet);
    vertex[2] = vecCore::Get(fVertices[ivert].z(), ifacet);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Facet_t *GetFacet(size_t ifacet) const { return fFacets[ifacet]; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T ComputeSparsity(int &nblobs, int &nfacets)
  {
    // Calculate cluster sparsity (number of separate blobs)
    // Number of facets represent the number of non-replicated facets
    // Returns ratio between maximum facet distance and maximum facet size

    // Find cluster center
    Vector3D<T> clcenter;
    for (unsigned ifacet = 0; ifacet < kVecSize; ++ifacet) {
      clcenter += fFacets[ifacet]->fCenter;
    }
    clcenter /= kVecSize;

    // Compute dispersion
    T maxsize = 0, lensq = 0, dmax = 0;
    for (unsigned ifacet = 0; ifacet < kVecSize; ++ifacet) {
      T facetsizesq = 0;
      for (int i = 0; i < 3; ++i) {
        lensq = (fFacets[ifacet]->fVertices[i] - fFacets[ifacet]->fVertices[(i + 1) % 3]).Mag2();
        if (lensq > facetsizesq) facetsizesq = lensq;
      }
      if (facetsizesq > maxsize) maxsize = facetsizesq;
      lensq                              = (fFacets[ifacet]->fCenter - clcenter).Mag2();
      if (lensq > dmax) dmax             = lensq;
    }
    T dispersion = vecCore::math::Sqrt(dmax / maxsize);

    // Compute number of distinct facets
    nfacets = 0;
    for (unsigned ifacet = 0; ifacet < kVecSize; ++ifacet) {
      bool duplicate = false;
      for (unsigned jfacet = ifacet + 1; jfacet < kVecSize; ++jfacet) {
        if (fFacets[jfacet] == fFacets[ifacet]) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) nfacets++;
    }

    // Compute number of blobs
    nblobs = 0;
    int cluster[kVecSize];
    int ncl             = 0;
    bool used[kVecSize] = {false};
    for (unsigned ifacet = 0; ifacet < kVecSize; ++ifacet) {
      ncl = 0;
      if (used[ifacet]) break;
      cluster[ncl++] = ifacet;
      used[ifacet]   = true;
      nblobs++;
      // loop remaining facets
      for (unsigned jfacet = ifacet + 1; jfacet < kVecSize; ++jfacet) {
        if (used[jfacet]) break;
        // loop facets already in sub-cluster
        int nneighbors = 0;
        for (int incl = 0; incl < ncl; ++incl) {
          nneighbors += fFacets[jfacet]->IsNeighbor(*fFacets[cluster[incl]]);
        }
        if (nneighbors > ncl) {
          cluster[ncl++] = jfacet;
          used[jfacet]   = true;
        }
      }
    }
    return dispersion;
  }

  /** @brief Fill the components 'i' of the cluster with facet data
   * @param index Triangle index, equivalent to SIMD lane index
   * @param facet Triangle facet data
   */
  VECCORE_ATT_HOST_DEVICE
  void AddFacet(size_t index, Facet_t *facet, size_t ifacet)
  {
    // Fill the facet normal by accessing individual SIMD lanes
    assert(index < kVecSize);
    vecCore::Set(fNormals.x(), index, facet->fNormal.x());
    vecCore::Set(fNormals.y(), index, facet->fNormal.y());
    vecCore::Set(fNormals.z(), index, facet->fNormal.z());
    // Fill the distance to the plane
    vecCore::Set(fDistances, index, facet->fDistance);
    // Compute side vectors and fill them using the store operation per SIMD lane
    for (size_t ivert = 0; ivert < NVERT; ++ivert) {
      Vector3D<T> c0                            = facet->fVertices[ivert];
      if (c0.x() < fMinExtent[0]) fMinExtent[0] = c0.x();
      if (c0.y() < fMinExtent[1]) fMinExtent[1] = c0.y();
      if (c0.z() < fMinExtent[2]) fMinExtent[2] = c0.z();
      if (c0.x() > fMaxExtent[0]) fMaxExtent[0] = c0.x();
      if (c0.y() > fMaxExtent[1]) fMaxExtent[1] = c0.y();
      if (c0.z() > fMaxExtent[2]) fMaxExtent[2] = c0.z();
      Vector3D<T> c1                            = facet->fVertices[(ivert + 1) % NVERT];
      Vector3D<T> sideVector                    = facet->fNormal.Cross(c1 - c0).Normalized();
      vecCore::Set(fSideVectors[ivert].x(), index, sideVector.x());
      vecCore::Set(fSideVectors[ivert].y(), index, sideVector.y());
      vecCore::Set(fSideVectors[ivert].z(), index, sideVector.z());
      vecCore::Set(fVertices[ivert].x(), index, c0.x());
      vecCore::Set(fVertices[ivert].y(), index, c0.y());
      vecCore::Set(fVertices[ivert].z(), index, c0.z());
    }
    fFacets[index]  = facet;
    fIfacets[index] = ifacet;
    if (index == kVecSize - 1) CalculateConvexity();
  }

  // === Navigation functionality === //

  /** @brief Check if a scalar point is inside any of the cluster tiles */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<T> point)
  {
    using Bool_v = vecCore::Mask<Real_v>;

    Bool_v inside;
    // Implicit conversion of point to Real_v
    InsideCluster(point, inside);
    return (!vecCore::MaskEmpty(inside));
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void InsideCluster(Vector3D<Real_v> const &point, typename vecCore::Mask<Real_v> &inside) const
  {
    // Check if the points are inside some of the triangles. The points are assumed
    // to be already propagated on the triangle planes.
    using Bool_v = vecCore::Mask<Real_v>;

    inside = Bool_v(true);
    for (size_t i = 0; i < NVERT; ++i) {
      Real_v saf = (point - fVertices[i]).Dot(fSideVectors[i]);
      inside &= saf > Real_v(-kTolerance);
    }
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v DistPlanes(Vector3D<Real_v> const &point) const
  {
    // Returns distance from point to plane. This is positive if the point is on
    // the outside halfspace, negative otherwise.
    return (point.Dot(fNormals) + fDistances);
  }

  VECCORE_ATT_HOST_DEVICE
  void DistanceToCluster(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T &distanceToIn,
                         T &distanceToOut, int &isurfToIn, int &isurfToOut) const
  {
    // Computes both distance to in and distance to out for the cluster
    using Bool_v = vecCore::Mask<Real_v>;

    distanceToIn  = InfinityLength<T>();
    distanceToOut = InfinityLength<T>();
    //    Real_v distToIn   = InfinityLength<Real_v>();
    //    Real_v distToOut  = InfinityLength<Real_v>();
    isurfToIn  = -1;
    isurfToOut = -1;

    // Vector3D<Real_v> pointv(point);
    // Vector3D<Real_v> dirv(direction);
    Real_v ndd = NonZero(direction.Dot(fNormals));
    Real_v saf = DistPlanes(point);

    Bool_v validToIn  = ndd < Real_v(0.) && saf > Real_v(-kTolerance);
    Bool_v validToOut = ndd > Real_v(0.) && saf < Real_v(kTolerance);

    Real_v dist             = -saf / ndd;
    Vector3D<Real_v> pointv = point + dist * direction;
    // Check if propagated points hit the triangles
    Bool_v hit;
    InsideCluster(pointv, hit);

    validToIn &= hit;
    validToOut &= hit;

    // Now we need to return the minimum distance for the hit facets
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(validToIn || validToOut)) return;

    // Since we can make no assumptions on convexity, we need to actually check
    // which surface is actually crossed. First propagate the point with the
    // distance to each plane.
    for (size_t i = 0; i < kVecSize; ++i) {
      if (vecCore::Get(validToIn, i)) {
        T dlane = vecCore::Get(dist, i);
        if (dlane < distanceToIn) {
          distanceToIn = dlane;
          isurfToIn    = fIfacets[i];
        }
      } else {
        if (vecCore::Get(validToOut, i)) {
          T dlane = vecCore::Get(dist, i);
          if (dlane < distanceToOut) {
            distanceToOut = dlane;
            isurfToOut    = fIfacets[i];
          }
        }
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  void DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T const & /*stepMax*/,
                    T &distance, int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    distance    = InfinityLength<T>();
    Real_v dist = InfinityLength<Real_v>();
    isurf       = -1;
    //    Vector3D<Real_v> pointv(point);
    //    Vector3D<Real_v> dirv(direction);
    Real_v ndd   = NonZero(direction.Dot(fNormals));
    Real_v saf   = DistPlanes(point);
    Bool_v valid = ndd < Real_v(0.) && saf > Real_v(-kTolerance);
    //    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid)) return;

    vecCore__MaskedAssignFunc(dist, valid, -saf / ndd);
    // Since we can make no assumptions on convexity, we need to actually check
    // which surface is actually crossed. First propagate the point with the
    // distance to each plane.
    Vector3D<Real_v> pointv = point + dist * direction;
    // Check if propagated points hit the triangles
    Bool_v hit;
    InsideCluster(pointv, hit);
    valid &= hit;
    // Now we need to return the minimum distance for the hit facets
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid)) return;

    for (size_t i = 0; i < kVecSize; ++i) {
      if (vecCore::Get(valid, i) && (vecCore::Get(dist, i) < distance)) {
        distance = vecCore::Get(dist, i);
        isurf    = fIfacets[i];
      }
    }
  }

  /** @brief Compute distance from point outside within limit. Returns
             validity of the computed distance.
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool DistanceToInConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T &distance, T &limit) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    // Check if track is moving away from facets
    const Real_v proj        = NonZero(direction.Dot(fNormals));
    const Bool_v moving_away = proj > Real_v(-kTolerance);
    // Check if track is on the correct side of of the planes
    const Real_v pdist        = DistPlanes(point);
    const Bool_v side_correct = pdist > Real_v(-kTolerance);

    if (!vecCore::MaskEmpty(side_correct && moving_away)) return false;

    // These facets can be hit from outside
    const Bool_v from_outside = side_correct && !moving_away;

    // These facets can be hit from inside
    const Bool_v from_inside = !side_correct && moving_away;

    Real_v dmin = -InfinityLength<Real_v>();
    Real_v dmax = InfinityLength<Real_v>();
    // Distances to facets
    const Real_v dist = -pdist / NonZero(proj);
    vecCore__MaskedAssignFunc(dmin, from_outside, dist);
    vecCore__MaskedAssignFunc(dmax, from_inside, dist);
    distance = vecCore::math::Max(distance, vecCore::ReduceMax(dmin));
    limit    = vecCore::math::Min(limit, vecCore::ReduceMin(dmax));

    // if (distance < limit - kTolerance) return true;
    // distance = InfinityLength<T>();
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  void DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T const & /*stepMax*/,
                     T &distance, int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    distance    = 0.;
    Real_v dist = InfinityLength<Real_v>();
    isurf       = -1;
    // Transform scalar point and direction into Real_v types
    //    Vector3D<Real_v> pointv(point);
    //    Vector3D<Real_v> dirv(direction);

    // Dot product between direction and facet normals should be positive
    // for valid crossings
    Real_v ndd = NonZero(direction.Dot(fNormals));

    // Distances to facet planes should be negative for valid crossing ("behind" normals)
    Real_v saf = DistPlanes(point);

    Bool_v valid = ndd > Real_v(0.) && saf < Real_v(kTolerance);
    // In case no crossing is valid, the point is outside and returns 0 distance
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid)) return;

    vecCore__MaskedAssignFunc(dist, valid, -saf / ndd);
    // Since we can make no assumptions on convexity, we need to actually check
    // which surface is actually crossed. First propagate the point with the
    // distance to each plane.
    Vector3D<Real_v> pointv = point + dist * direction;
    // Check if propagated points hit the triangles
    Bool_v hit;
    InsideCluster(pointv, hit);
    valid &= hit;
    if (vecCore::MaskEmpty(valid)) return;

    // Now we need to return the minimum distance for the hit facets
    distance = InfinityLength<T>();
    for (size_t i = 0; i < kVecSize; ++i) {
      if (vecCore::Get(valid, i) && vecCore::Get(dist, i) < distance) {
        distance = vecCore::Get(dist, i);
        isurf    = fIfacets[i];
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  bool DistanceToOutConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T &distance) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    distance    = -kTolerance;
    Real_v dist = InfinityLength<Real_v>();

    // Distances to facet planes should be negative for valid crossing ("behind" normals)
    Real_v saf   = DistPlanes(point);
    Bool_v valid = saf < Real_v(kTolerance);
    if (vecCore::EarlyReturnAllowed() && !vecCore::MaskFull(valid)) return false;

    // Dot product between direction and facet normals should be positive
    // for valid crossings
    Real_v ndd = NonZero(direction.Dot(fNormals));
    valid      = ndd > Real_v(0.);

    vecCore__MaskedAssignFunc(dist, valid, -saf / ndd);
    distance = vecCore::ReduceMin(dist);
    return true;
  }

  /* @brief Compute distance to in/out for the convex case. */
  VECCORE_ATT_HOST_DEVICE
  bool DistanceToInOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, T &dtoin, T &dtoout) const
  {
    using Bool_v = vecCore::Mask<Real_v>;
    using vecCore::math::Min;
    using vecCore::math::Max;
    using vecCore::ReduceMin;
    using vecCore::ReduceMax;

    // Direction projected to all facets
    Real_v projdir_v   = NonZero(direction.Dot(fNormals));
    Bool_v moving_away = projdir_v > Real_v(0.);
    // Signed projected distances to facets
    Real_v projdist_v = DistPlanes(point);
    Bool_v outside    = projdist_v > Real_v(kTolerance);
    // If outside and mowing away any facet, no hit possible (convexity)
    if (!vecCore::MaskEmpty(outside && moving_away)) return false;
    // Facets that can be hit from inside
    Bool_v from_inside = !outside && moving_away;
    // Facets that can be hit from outside
    Bool_v from_outside = outside && !moving_away;
    // Distances to all facets
    const Real_v dist_v = -projdist_v / NonZero(projdir_v);
    Real_v dtoin_v      = -InfinityLength<Real_v>();
    Real_v dtoout_v     = InfinityLength<Real_v>();
    vecCore__MaskedAssignFunc(dtoin_v, from_outside, dist_v);
    dtoin = Max(dtoin, ReduceMax(dtoin_v));
    vecCore__MaskedAssignFunc(dtoout_v, from_inside, dist_v);
    dtoout = Min(dtoout, ReduceMin(dtoout_v));
    return true;
  }

  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  T SafetySq(Vector3D<Real_v> const &point, int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;
    //    Vector3D<Real_v> pointv(point);
    Real_v safetyv = DistPlanes(point);
    T distancesq   = InfinityLength<T>();
    // Find the projection of the point on each plane
    Vector3D<Real_v> intersectionv = point - safetyv * fNormals;
    Bool_v withinBound;
    InsideCluster(intersectionv, withinBound);
    if (ToIn)
      withinBound &= safetyv > Real_v(-kTolerance);
    else
      withinBound &= safetyv < Real_v(kTolerance);
    safetyv *= safetyv;

    isurf = -1;
    if (vecCore::MaskFull(withinBound)) {
      // loop over lanes to get minimum positive value.
      for (size_t i = 0; i < kVecSize; ++i) {
        auto saflane = vecCore::Get(safetyv, i);
        if (saflane < distancesq) {
          distancesq = saflane;
          isurf      = fIfacets[i];
        }
      }
      return distancesq;
    }

    Vector3D<Real_v> safetyv_outbound = InfinityLength<Real_v>();
    for (size_t ivert = 0; ivert < NVERT; ++ivert) {
      safetyv_outbound[ivert] =
          DistanceToLineSegmentSquared2(fVertices[ivert], fVertices[(ivert + 1) % NVERT], point, !withinBound);
    }
    Real_v safety_outv = safetyv_outbound.Min();
    vecCore::MaskedAssign(safetyv, !withinBound, safety_outv);

    // loop over lanes to get minimum positive value.
    for (size_t i = 0; i < kVecSize; ++i) {
      auto saflane = vecCore::Get(safetyv, i);
      if (saflane < distancesq) {
        distancesq = saflane;
        isurf      = fIfacets[i];
      }
    }
    return distancesq;
  }

  /*
    VECCORE_ATT_HOST_DEVICE
    void DistanceToInScalar(Vector3D<T> const &point, Vector3D<T> const &direction, T const &stepMax, T &distance,
                            int &isurf)
    {
      distance = InfinityLength<T>();
      isurf    = -1;
      T distfacet;
      for (size_t i = 0; i < kVecSize; ++i) {
        distfacet = fFacets[i]->DistanceToIn(point, direction, stepMax);
        if (distfacet < distance) {
          distance = distfacet;
          isurf    = fIfacets[i];
        }
      }
    }

    VECCORE_ATT_HOST_DEVICE
    void DistanceToOutScalar(Vector3D<T> const &point, Vector3D<T> const &direction, T const &stepMax, T &distance,
                             int &isurf)
    {
      distance = InfinityLength<T>();
      isurf    = -1;
      T distfacet;
      for (size_t i = 0; i < kVecSize; ++i) {
        distfacet = fFacets[i]->DistanceToOut(point, direction, stepMax);
        if (distfacet < distance) {
          distance = distfacet;
          isurf    = fIfacets[i];
        }
      }
    }

    template <bool ToIn>
    VECCORE_ATT_HOST_DEVICE
    T SafetySqScalar(Vector3D<T> const &point, int &isurf)
    {
      T distance = InfinityLength<T>();
      T distfacet;
      for (size_t i = 0; i < kVecSize; ++i) {
        distfacet = fFacets[i]->template SafetySq<ToIn>(point, isurf);
        if (distfacet < distance) {
          distance = distfacet;
          isurf    = fIfacets[i];
        }
      }
      return distance;
    }
  */
};

std::ostream &operator<<(std::ostream &os, TessellatedCluster<3, typename vecgeom::VectorBackend::Real_v> const &tcl);
std::ostream &operator<<(std::ostream &os, TessellatedCluster<4, typename vecgeom::VectorBackend::Real_v> const &tcl);

} // namespace VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
