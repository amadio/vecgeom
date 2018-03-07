/// \file TessellatedSection.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDSECTION_H_
#define VECGEOM_VOLUMES_TESSELLATEDSECTION_H_

#include "volumes/TessellatedCluster.h"

#include "management/HybridManager2.h"
#include "navigation/HybridNavigator2.h"
#include "management/ABBoxManager.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// Navigation helper for adjacent quadrilateral facets forming a closed
// convex section in between 2 Z planes. A base class defines the scalar navigation
// interfaces.

template <typename T>
class TessellatedSection /* : public TessellatedSectionBase<T> */ {
  // Here we should be able to use vecgeom::Vector
  template <typename U>
  using vector_t  = std::vector<U>;
  using Real_v    = vecgeom::VectorBackend::Real_v;
  using Facet_t   = QuadrilateralFacet<T>;
  using Cluster_t = TessellatedCluster<4, Real_v>;

private:
  size_t fNfacets = 0;    ///< Number of triangle facets on the section
  T fZ            = 0;    ///<< Z position of the section
  T fDz           = 0;    ///< Half-length in Z
  T fCubicVolume  = 0;    ///< Cubic volume
  T fSurfaceArea  = 0;    ///< Surface area
  Vector3D<T> fMinExtent; ///< Minimum extent
  Vector3D<T> fMaxExtent; ///< Maximum extent
  bool fSameZ = false;    ///< All facets are at same Z
  T fUpNorm   = 0;        ///< Up normal in case of sameZ (+1 or -1)

  vector_t<Vector3D<T>> fVertices; ///< Vector of unique vertices
  vector_t<Facet_t *> fFacets;     ///< Vector of quadrilateral convex facets
  vector_t<Cluster_t *> fClusters; ///< Vector of facet clusters

protected:
  VECCORE_ATT_HOST_DEVICE
  void AddFacet(Facet_t *facet)
  {
    // Method adding a facet to the structure. The vertices are added to the
    // list of all vertices (including duplications) and the extent is re-adjusted.
    if (fSameZ) {
      Vector3D<T> const &normal = facet->GetNormal();
      assert(normal.Perp() < kTolerance);
      if (fUpNorm == 0.) fUpNorm = vecCore::math::CopySign(1., normal.z());
      assert(fUpNorm * normal.z() > 0);
    }
    fFacets.push_back(facet);
    // Adjust extent
    using vecCore::math::Min;
    using vecCore::math::Max;
    T xmin = Min(Min(facet->fVertices[0].x(), facet->fVertices[1].x()),
                 Min(facet->fVertices[2].x(), facet->fVertices[3].x()));
    T ymin = Min(Min(facet->fVertices[0].y(), facet->fVertices[1].y()),
                 Min(facet->fVertices[2].y(), facet->fVertices[3].y()));
    fMinExtent[0] = Min(fMinExtent[0], xmin);
    fMinExtent[1] = Min(fMinExtent[1], ymin);
    fMinExtent[2] = fZ - fDz;
    T xmax        = Max(Max(facet->fVertices[0].x(), facet->fVertices[1].x()),
                 Max(facet->fVertices[2].x(), facet->fVertices[3].x()));
    T ymax = Max(Min(facet->fVertices[0].y(), facet->fVertices[1].y()),
                 Max(facet->fVertices[2].y(), facet->fVertices[3].y()));
    fMaxExtent[0] = Max(fMaxExtent[0], xmax);
    fMaxExtent[1] = Max(fMaxExtent[1], ymax);
    fMaxExtent[2] = fZ + fDz;
    // Check if we can create a Tessellated cluster
    size_t nfacets = fFacets.size();
    assert(nfacets <= fNfacets && "Cannot add extra facets to section");
    if (nfacets % kVecSize == 0 || nfacets == fNfacets) {
      size_t istart      = nfacets - (nfacets - 1) % kVecSize - 1;
      size_t i           = 0;
      Cluster_t *cluster = new Cluster_t();
      for (; istart < nfacets; ++istart) {
        cluster->AddFacet(i++, fFacets[istart], istart);
      }
      // The last cluster may not be yet full: fill with last facet
      for (; i < kVecSize; ++i)
        cluster->AddFacet(i, facet, nfacets - 1);
      fClusters.push_back(cluster);
    }
    if (nfacets == fNfacets) {
      assert(CalculateConvexity() == true);
    }
  }

  /** &brief Calculate convexity of the section with respect to itself */
  VECCORE_ATT_HOST_DEVICE
  bool CalculateConvexity()
  {
    size_t nconvex = 0;
    for (size_t i = 0; i < fNfacets; ++i) {
      Facet_t *facet = fFacets[i];
      bool convex    = true;
      for (size_t j = 0; j < fNfacets; ++j) {
        if (j == i) continue;
        for (size_t ivert = 0; ivert < 4; ++ivert) {
          convex &= facet->DistPlane(fFacets[j]->fVertices[ivert]) < kTolerance;
        }
        if (!convex) continue;
      }
      facet->fConvex = convex;
      if (convex) nconvex++;
    }
    for (auto cluster : fClusters)
      cluster->CalculateConvexity();
    if (nconvex == fNfacets) return true;
    return false;
  }

public:
  VECCORE_ATT_HOST_DEVICE
  TessellatedSection(int nfacets, T zmin, T zmax) : fNfacets(nfacets), fZ(0.5 * (zmin + zmax)), fDz(0.5 * (zmax - zmin))
  {
    assert(zmax >= zmin && "zmin is greater than zmax");
    if (fDz < kTolerance) {
      fSameZ = true;
      fDz    = 0.;
    }
    fMinExtent.Set(InfinityLength<T>());
    fMaxExtent.Set(-InfinityLength<T>());
  }

  VECCORE_ATT_HOST_DEVICE
  bool AddQuadrilateralFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2,
                             Vector3D<T> const &vt3, bool absolute = true)
  {
    // Quadrilateral facet, normal pointing outside
    Facet_t *facet = new Facet_t;
    if (absolute) {
      if (!facet->SetVertices(vt0, vt1, vt2, vt3)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
    } else {
      if (!facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2, vt0 + vt1 + vt2 + vt3)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
    }
    return true;
  }

  /** @brief Fast check using the extent if the point is outside */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsOutside(Vector3D<T> const &point)
  {
    return ((point - fMinExtent).Min() < -kTolerance || (point - fMaxExtent).Max() > kTolerance);
  }

  VECCORE_ATT_HOST_DEVICE
  Inside_t Inside(Vector3D<Real_v> const &point) const
  {
    // All lanes of point contain the same scalar point
    using Bool_v = vecCore::Mask<Real_v>;

    // Assuming the fast check on extent was already done using the scalar point
    size_t nclusters = fClusters.size();
    // Convex polygone on top/bottom
    Real_v distPlanes;
    Bool_v inside(true), outside(false);
    for (size_t i = 0; i < nclusters; ++i) {
      distPlanes = fClusters[i]->DistPlanes(point);
      outside |= distPlanes > Real_v(kTolerance);
      //        if (!vecCore::MaskEmpty(outside)) return kOutside;
      inside &= distPlanes < -kTolerance;
    }
    if (!vecCore::MaskEmpty(outside)) return kOutside;
    if (vecCore::MaskFull(inside)) return kInside;
    return kSurface;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNfacets() const { return fFacets.size(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNclusters() const { return fClusters.size(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Cluster_t const &GetCluster(size_t i) const { return *fClusters[i]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Facet_t const &GetFacet(size_t i) const { return *fFacets[i]; }

  /* @brief Check if point is inside the section. Note that the Z range is not checked */
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Real_v> const &point) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    // Check if point is in the bounding box
    // if ((point - fMinExtent).Min() < 0. || (point - fMaxExtent).Max() > 0.) return kOutside;

    size_t nclusters = fClusters.size();
    // Convex polygone on top/bottom
    Bool_v outside(false);

    for (size_t i = 0; i < nclusters; ++i) {
      Real_v distPlanes = fClusters[i]->DistPlanes(point);
      outside |= distPlanes > Real_v(0);
      if (!vecCore::MaskEmpty(outside)) return false;
    }
    return true;
  }

  template <bool skipZ = true>
  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<T> const &point, Vector3D<T> const &direction, T invdirz, T stepmax) const
  {
    // Compute distance to segment from outside point.
    if (fSameZ) {
      // All facets are on the plane at fZ
      // Distance to plane
      T pz = point.z() - fZ;
      // If wrong direction or opposite side, no hit
      if (fUpNorm * direction.z() > 0 || pz * fUpNorm < -kTolerance) return InfinityLength<T>();
      T distance = -pz * invdirz;
      // Still need to check that the propagated point is in the section
      Vector3D<T> propagated(point + distance * direction);
      const int nclusters = fClusters.size();
      for (int i = 0; i < nclusters; ++i) {
        if (fClusters[i]->Contains(propagated)) return distance;
      }
      return InfinityLength<T>();
    }

    T pz = point.z() - fZ;
    if (!skipZ) {
      if ((vecCore::math::Abs(pz) - fDz) > -kTolerance && pz * direction.z() >= 0) return InfinityLength<T>();
    }
    const T ddz   = vecCore::math::CopySign(fDz, invdirz);
    const T distz = -(pz + ddz) * invdirz;
    T distance    = distz;
    T limit       = vecCore::math::Min(-(pz - ddz) * invdirz, stepmax);

    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      bool canhit =
          (fClusters[i]->DistanceToInConvex(point, direction, distance, limit)) && (distance < limit - kTolerance);
      if (!canhit) return InfinityLength<T>();
    }
    if (skipZ) {
      if (distance > distz) return distance;
      return InfinityLength<T>();
    }
    return distance;
  }

  template <bool skipZ = true>
  VECCORE_ATT_HOST_DEVICE
  T DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction) const
  {
    // Compute distance to segment from point inside, returning also the crossed
    // facet.
    T pz         = point.z() - fZ;
    const T safz = vecCore::math::Abs(pz) - fDz;
    if (safz > kTolerance) return -kTolerance;
    const T vz = direction.z();
    T distance = (vecCore::math::CopySign(fDz, vz) - pz) / NonZero(vz);
    T dist;
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      if (!fClusters[i]->DistanceToOutConvex(point, direction, dist)) return dist;
      if (dist < distance) distance = dist;
    }
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToOutRange(Vector3D<T> const &point, Vector3D<T> const &direction, T invdirz) const
  {
    // Compute distance to segment from point inside, returning also the crossed
    // facet.
    if (fSameZ) {
      // All facets are on the plane at z = fZ
      // Distance to plane
      T pz = point.z() - fZ;
      // If wrong direction or opposite side, no hit
      if (fUpNorm * direction.z() < 0 || pz * fUpNorm > kTolerance) return InfinityLength<T>();
      T distance = -pz * invdirz;
      // Still need to check that the propagated point is in the section
      Vector3D<T> propagated(point + distance * direction);
      const int nclusters = fClusters.size();
      for (int i = 0; i < nclusters; ++i) {
        if (fClusters[i]->Contains(propagated)) return distance;
      }
      return InfinityLength<T>();
    }
    T pz                = point.z() - fZ;
    T dmax              = (vecCore::math::CopySign(fDz, invdirz) - pz) * invdirz;
    T dmin              = (-vecCore::math::CopySign(fDz, invdirz) - pz) * invdirz;
    T dtoin             = dmin;                // will be reduced Max for all clusters
    T dtoout            = InfinityLength<T>(); // will be reduced Min for all clusters
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      bool hit = fClusters[i]->DistanceToInOut(point, direction, dtoin, dtoout);
      if (!hit) return InfinityLength<T>();
    }

    if (dtoout > dtoin - kTolerance && dtoout < dmax) return dtoout;
    return InfinityLength<T>();
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, int &isurf) const
  {
    // Compute distance to segment from point inside, returning also the crossed
    // facet.
    isurf               = -1;
    T distance          = InfinityLength<T>();
    T stepmax           = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      int isurfcrt = -1;
      T distcrt;
      fClusters[i]->DistanceToOut(point, direction, stepmax, distcrt, isurfcrt);
      if (distcrt < distance) {
        distance = distcrt;
        isurf    = isurfcrt;
      }
    }
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToIn(Vector3D<T> const &point) const
  {
    // Compute approximate safety for the convex case
    T safety            = vecCore::math::Max(fZ - fDz - point.z(), point.z() - fZ - fDz);
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      const Real_v safcl       = fClusters[i]->DistPlanes(point);
      const T saf              = vecCore::ReduceMax(safcl);
      if (saf > safety) safety = saf;
    }
    return safety;
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToInSq(Vector3D<Real_v> const &point, int &isurf) const
  {
    // Compute safety squared to segment from point outside, returning also the crossed
    // facet.
    isurf               = -1;
    T safetysq          = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      int isurfcrt     = -1;
      const T safsqcrt = fClusters[i]->template SafetySq<true>(point, isurfcrt);
      if (safsqcrt < safetysq) {
        safetysq = safsqcrt;
        isurf    = isurfcrt;
      }
    }
    return safetysq;
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToOut(Vector3D<T> const &point) const
  {
    // Compute approximate safety for the convex case
    T safety            = vecCore::math::Max(fZ - fDz - point.z(), point.z() - fZ - fDz);
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      const Real_v safcl       = fClusters[i]->DistPlanes(point);
      const T saf              = vecCore::ReduceMax(safcl);
      if (saf > safety) safety = saf;
    }
    return -safety;
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToOutSq(Vector3D<Real_v> const &point, int &isurf) const
  {
    // Compute safety squared to segment from point inside, returning also the crossed
    // facet.
    isurf               = -1;
    T safetysq          = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i = 0; i < nclusters; ++i) {
      int isurfcrt     = -1;
      const T safsqcrt = fClusters[i]->template SafetySq<false>(point, isurfcrt);
      if (safsqcrt < safetysq) {
        safetysq = safsqcrt;
        isurf    = isurfcrt;
      }
    }
    return safetysq;
  }

  VECCORE_ATT_HOST_DEVICE
  void Normal(Vector3D<T> const & /*point*/, Vector3D<T> & /*normal*/, bool & /*valid*/) const
  {
    // Compute normal to segment surface in given point near surface.
  }
};

std::ostream &operator<<(std::ostream &os, TessellatedSection<double> const &ts);

} // namespace VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
