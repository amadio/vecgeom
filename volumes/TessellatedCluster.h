/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_
#define VECGEOM_VOLUMES_TESSELLATEDCLUSTER_H_

#include <VecCore/VecCore>

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Vector.h"

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
                    int &isurf)
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
    for (int i = 0; i < kVecSize; ++i) {
      if (valid[i] && dist[i] < distance) {
        distance = dist[i];
        isurf    = i;
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  void DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction, T const & /*stepMax*/, T &distance,
                     int &isurf)
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
    for (int i = 0; i < kVecSize; ++i) {
      if (valid[i] && dist[i] < distance) {
        distance = dist[i];
        isurf    = i;
      }
    }
  }
};

std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet);
std::ostream &operator<<(std::ostream &os, TessellatedCluster<typename vecgeom::VectorBackend::Real_v> const &tcl);

} // end VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
