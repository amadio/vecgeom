/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_
#define VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_

#define TEST_TCPERF = 1

#include <VecCore/VecCore>

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Vector.h"
#include "volumes/kernel/GenericKernels.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

constexpr size_t kVecSize = vecCore::VectorSize<vecgeom::VectorBackend::Real_v>();

//______________________________________________________________________________
// Basic structure of indices to 3 vertices making a triangle.
// The vertices making the triangle have to be given in anti-clockwise
// order looking from the outsider of the solid where it belongs.
// This helper structure is used temporarily by the tesselated solid in the
// creation and clustering phase.
//______________________________________________________________________________
template <typename T = double>
struct TriangleFacet {
  Vector3D<T> fVertices[3]; ///< vertices of the triangle
  Vector3D<int> fIndices;   ///< indices for 3 distinct vertices
  Vector<int> fNeighbors;   ///< indices to triangle neighbors
  T fSurfaceArea = 0;       ///< surface area
  Vector3D<T> fNormal;      ///< normal vector pointing outside
  T fDistance;              ///< distance between the origin and the triangle plane
#ifdef TEST_TCPERF
  Vector3D<T> fSideVectors[3]; ///< side vectors perpendicular to edges
#endif

  VECCORE_ATT_HOST_DEVICE
  TriangleFacet() { fNeighbors.reserve(3); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool SetVertices(Vector3D<T> const &vtx0, Vector3D<T> const &vtx1, Vector3D<T> const &vtx2, int ind0, int ind1,
                   int ind2)
  {
    fVertices[0] = vtx0;
    fVertices[1] = vtx1;
    fVertices[2] = vtx2;
    fIndices.Set(ind0, ind1, ind2);
    // Check validity
    Vector3D<T> e1 = vtx1 - vtx0;
    Vector3D<T> e2 = vtx2 - vtx0;
    double eMag1   = e1.Mag();
    double eMag2   = e2.Mag();
    double eMag3   = (e2 - e1).Mag();

    if (eMag1 <= kTolerance || eMag2 <= kTolerance || eMag3 <= kTolerance) {
      // TO DO: add more verbosity
      std::cout << "Length of sides of facet are too small." << std::endl;
      return false;
    }
    // Compute normal
    fNormal = e1.Cross(e2).Unit();
#ifdef TEST_TCPERF
    // Compute side vectors (testing)
    for (int i        = 0; i < 3; i++)
      fSideVectors[i] = fNormal.Cross(fVertices[(i + 1) % 3] - fVertices[i]).Normalized();
#endif
    // Distace to facet
    fDistance    = -fNormal.Dot(vtx0);
    fSurfaceArea = 0.5 * (e1.Cross(e2)).Mag();
    if (fSurfaceArea < kTolerance * kTolerance) {
      // TO DO: add more verbosity
      std::cout << "Flat triangle." << std::endl;
      return false;
    }
    // Any more fields to fill to be added here
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSurrounded() const { return fNeighbors.size() > 2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsNeighbor(TriangleFacet const &other)
  {
    // Check if a segment is common
    int ncommon = 0;
    int i1      = 0;
    int i2      = 0;
    for (int ind1 = 0; ind1 < 3; ++ind1) {
      for (int ind2 = 0; ind2 < 3; ++ind2) {
        if (fIndices[ind1] == other.fIndices[ind2]) {
          ncommon++;
          i1 = ind1;
          i2 = ind2;
        }
      }
    }
    // In case we detect a single common vertex, it is still possible that the
    // facets are neighbors
    // if (ncommon == 1) DetectCollision(other)
    return (ncommon == 2);
  }

#ifdef TEST_TCPERF
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Precision> const &point) const
  {
    // Check id point within the triangle plane is inside the triangle.
    bool inside = true;
    for (size_t i = 0; i < 3; ++i) {
      Precision saf = (point - fVertices[i]).Dot(fSideVectors[i]);
      inside &= saf > -kTolerance;
    }
    return inside;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T DistPlanes(Vector3D<Precision> const &point) const
  {
    // Returns distance from point to plane. This is positive if the point is on
    // the outside halfspace, negative otherwise.
    return (point.Dot(fNormal) + fDistance);
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                 Precision const & /*stepMax*/) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlanes(point);
    bool valid = ndd < 0. && saf > -kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<Precision> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Inside(point_prop)) return InfinityLength<T>();
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                          Precision const & /*stepMax*/) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlanes(point);
    bool valid = ndd > 0. && saf < kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<Precision> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Inside(point_prop)) return InfinityLength<T>();
    return distance;
  }

  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  Precision SafetySq(Vector3D<Precision> const &point, int &isurf) const
  {
    T safety = DistPlanes(point);
    // Find the projection of the point on each plane
    Vector3D<Precision> intersection = point - safety * fNormal;
    bool withinBound                 = Inside(intersection);
    if (ToIn)
      withinBound &= safety > -kTolerance;
    else
      withinBound &= safety < kTolerance;
    safety *= safety;
    if (withinBound) return safety;

    Vector3D<T> safety_outbound = InfinityLength<T>();
    for (int ivert = 0; ivert < 3; ++ivert) {
      safety_outbound[ivert] =
          DistanceToLineSegmentSquared<kScalar>(fVertices[ivert], fVertices[(ivert + 1) % 3], point);
    }
    return (safety_outbound.Min());
  }
#endif
};

//______________________________________________________________________________
// Structure used for vectorizing queries on groups of triangles. Stores Real_v
// types matched with the compiled SIMD vectors of double.
template <typename Real_v>
struct TessellatedCluster {
  using T = typename vecCore::ScalarType<Real_v>::Type;

  Vector3D<Real_v> fNormals;        ///< Normals to facet components
  Real_v fDistances;                ///< Distances from origin to facets
  Vector3D<Real_v> fSideVectors[3]; ///< Side vectors of the triangular facets
  Vector3D<Real_v> fVertices[3];    ///< Vertices stored in SIMD format
#ifdef TEST_TCPERF
  TriangleFacet<T> fFacets[kVecSize];
#endif
  Vector3D<T> fMinExtent; ///< Minimum extent
  Vector3D<T> fMaxExtent; ///< Maximum extent

  VECCORE_ATT_HOST_DEVICE
  TessellatedCluster() {}

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void GetVertex(size_t ifacet, size_t ivert, Vector3D<T> &vertex) const
  {
    vertex[0] = fVertices[ivert].x()[ifacet];
    vertex[1] = fVertices[ivert].y()[ifacet];
    vertex[2] = fVertices[ivert].z()[ifacet];
  }
  /** @brief Fill the components 'i' of the cluster with facet data
    * @param index Triangle index, equivalent to SIMD lane index
    * @param facet Triangle facet data
    */
  VECCORE_ATT_HOST_DEVICE
  void AddFacet(size_t index, TriangleFacet<T> const &facet)
  {
    // Fill the facet normal by accessing individual SIMD lanes
    assert(index < kVecSize);
    fNormals.x()[index] = facet.fNormal.x();
    fNormals.y()[index] = facet.fNormal.y();
    fNormals.z()[index] = facet.fNormal.z();
    // Fill the distance to the plane
    fDistances[index] = facet.fDistance;
    // Compute side vectors and fill them using the store operation per SIMD lane
    for (size_t ivert = 0; ivert < 3; ++ivert) {
      Vector3D<T> c0                            = facet.fVertices[ivert];
      if (c0.x() < fMinExtent[0]) fMinExtent[0] = c0.x();
      if (c0.y() < fMinExtent[1]) fMinExtent[1] = c0.y();
      if (c0.z() < fMinExtent[2]) fMinExtent[2] = c0.z();
      if (c0.x() > fMaxExtent[0]) fMaxExtent[0] = c0.x();
      if (c0.y() > fMaxExtent[1]) fMaxExtent[1] = c0.y();
      if (c0.z() > fMaxExtent[2]) fMaxExtent[2] = c0.z();
      Vector3D<T> c1                            = facet.fVertices[(ivert + 1) % 3];
      Vector3D<T> sideVector                    = facet.fNormal.Cross(c1 - c0).Normalized();
      fSideVectors[ivert].x()[index]            = sideVector.x();
      fSideVectors[ivert].y()[index]            = sideVector.y();
      fSideVectors[ivert].z()[index]            = sideVector.z();
      fVertices[ivert].x()[index]               = c0.x();
      fVertices[ivert].y()[index]               = c0.y();
      fVertices[ivert].z()[index]               = c0.z();
    }
#ifdef TEST_TCPERF
    fFacets[index] = facet;
#endif
  }

  // === Navigation functionality === //
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void InsideCluster(Vector3D<Real_v> const &point, typename vecCore::Mask<Real_v> &inside) const
  {
    // Check if the points are inside some of the triangles. The points are assumed
    // to be already propagated on the triangle planes.
    using Bool_v = vecCore::Mask<Real_v>;

    inside = Bool_v(true);
    for (size_t i = 0; i < 3; ++i) {
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
  void DistanceToIn(Vector3D<T> const &point, Vector3D<T> const &direction, T const & /*stepMax*/, T &distance,
                    int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    distance    = InfinityLength<T>();
    Real_v dist = InfinityLength<Real_v>();
    isurf       = -1;
    Vector3D<Real_v> pointv(point);
    Vector3D<Real_v> dirv(direction);
    Real_v ndd   = NonZero(dirv.Dot(fNormals));
    Real_v saf   = DistPlanes(pointv);
    Bool_v valid = ndd < Real_v(0.) && saf > Real_v(-kTolerance);
    //    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid)) return;

    vecCore__MaskedAssignFunc(dist, valid, -saf / ndd);
    // Since we can make no assumptions on convexity, we need to actually check
    // which surface is actually crossed. First propagate the point with the
    // distance to each plane.
    pointv += dist * dirv;
    // Check if propagated points hit the triangles
    Bool_v hit;
    InsideCluster(pointv, hit);
    valid &= hit;
    // Now we need to return the minimum distance for the hit facets
    if (vecCore::MaskEmpty(valid)) return;

    for (size_t i = 0; i < kVecSize; ++i) {
      if (valid[i] && dist[i] < distance) {
        distance = dist[i];
        isurf    = i;
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  void DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction, T const & /*stepMax*/, T &distance,
                     int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    distance    = InfinityLength<T>();
    Real_v dist = InfinityLength<Real_v>();
    isurf       = -1;
    Vector3D<Real_v> pointv(point);
    Vector3D<Real_v> dirv(direction);
    Real_v ndd   = NonZero(dirv.Dot(fNormals));
    Real_v saf   = DistPlanes(pointv);
    Bool_v valid = ndd > Real_v(0.) && saf < Real_v(kTolerance);
    if (vecCore::EarlyReturnAllowed() && vecCore::MaskEmpty(valid)) return;

    vecCore__MaskedAssignFunc(dist, valid, -saf / ndd);
    // Since we can make no assumptions on convexity, we need to actually check
    // which surface is actually crossed. First propagate the point with the
    // distance to each plane.
    pointv += dist * dirv;
    // Check if propagated points hit the triangles
    Bool_v hit;
    InsideCluster(pointv, hit);
    valid &= hit;
    // Now we need to return the minimum distance for the hit facets
    if (vecCore::MaskEmpty(valid)) return;
    for (size_t i = 0; i < kVecSize; ++i) {
      if (valid[i] && dist[i] < distance) {
        distance = dist[i];
        isurf    = i;
      }
    }
  }

  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  T SafetySq(Vector3D<T> const &point, int &isurf) const
  {
    using Bool_v = vecCore::Mask<Real_v>;
    Vector3D<Real_v> pointv(point);
    Real_v safetyv = DistPlanes(pointv);
    T distancesq   = InfinityLength<T>();
    // Find the projection of the point on each plane
    Vector3D<Real_v> intersectionv = pointv - safetyv * fNormals;
    Bool_v withinBound;
    InsideCluster(intersectionv, withinBound);
    if (ToIn)
      withinBound &= safetyv > Real_v(-kTolerance);
    else
      withinBound &= safetyv < Real_v(kTolerance);
    safetyv *= safetyv;

    if (vecCore::MaskFull(withinBound)) {
      // loop over lanes to get minimum positive value.
      for (size_t i = 0; i < kVecSize; ++i) {
        if (safetyv[i] < distancesq) {
          distancesq = safetyv[i];
          isurf      = i;
        }
      }
      return distancesq;
    }

    Vector3D<Real_v> safetyv_outbound = InfinityLength<Real_v>();
    for (int ivert = 0; ivert < 3; ++ivert) {
      safetyv_outbound[ivert] =
          DistanceToLineSegmentSquared2(fVertices[ivert], fVertices[(ivert + 1) % 3], pointv, !withinBound);
    }
    Real_v safety_outv = safetyv_outbound.Min();
    vecCore::MaskedAssign(safetyv, !withinBound, safety_outv);

    // loop over lanes to get minimum positive value.
    for (size_t i = 0; i < kVecSize; ++i) {
      if (safetyv[i] < distancesq) {
        distancesq = safetyv[i];
        isurf      = i;
      }
    }
    return distancesq;
  }

#ifdef TEST_TCPERF
  VECCORE_ATT_HOST_DEVICE
  //  __attribute__((optimize("no-tree-vectorize")))
  void DistanceToInScalar(Vector3D<T> const &point, Vector3D<T> const &direction, T const &stepMax, T &distance,
                          int &isurf)
  {
    distance = InfinityLength<T>();
    isurf    = -1;
    T distfacet;
    for (size_t i = 0; i < kVecSize; ++i) {
      distfacet = fFacets[i].DistanceToIn(point, direction, stepMax);
      if (distfacet < distance) {
        distance = distfacet;
        isurf    = i;
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
      distfacet = fFacets[i].DistanceToOut(point, direction, stepMax);
      if (distfacet < distance) {
        distance = distfacet;
        isurf    = i;
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
      distfacet = fFacets[i].SafetySq<ToIn>(point, isurf);
      if (distfacet < distance) {
        distance = distfacet;
        isurf    = i;
      }
    }
    return distance;
  }
#endif
};

std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet);
std::ostream &operator<<(std::ostream &os, TessellatedCluster<typename vecgeom::VectorBackend::Real_v> const &tcl);

} // end VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
