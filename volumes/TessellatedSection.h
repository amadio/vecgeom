/// \file TessellatedSection.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_
#define VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_

#include "TessellatedCluster.h"

#include "management/HybridManager2.h"
#include "navigation/HybridNavigator2.h"
#include "management/ABBoxManager.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// Navigation helper for adjacent quadrilateral facets forming a closed
// section in between 2 Z planes. A base class defines the scalar navigation
// interfaces.

template <typename T>
class TessellatedSectionBase {
public:
  VECCORE_ATT_HOST_DEVICE
  Inside_t Inside(Vector3D<T> const &point) const = 0;

  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<T> const &point) const = 0;

  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<T> const &point, Vector3D<T> const &direction,
                 int &isurf) = 0;

  VECCORE_ATT_HOST_DEVICE
  T DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction,
                 int &isurf) = 0;

  VECCORE_ATT_HOST_DEVICE
  double SafetyToInSq(Vector3D<T> const &point) = 0;

  VECCORE_ATT_HOST_DEVICE
  double SafetyToOutSq(Vector3D<T> const &point) = 0;

  VECCORE_ATT_HOST_DEVICE
  void Normal(Vector3D<double> const &point, Vector3D<double> &normal, bool &valid) = 0;
};

// Derived templated tessellated section on convexity, making a right prism or
// making a flat surface at same z.

template <typename T, bool Convex=false, bool Right=false, bool SameZ=false>
class TessellatedSection : public TessellatedSectionBase<T> {
  // Here we should be able to use vecgeom::Vector
  template <typename U>
  using vector_t = std::vector<U>;
  using Real_v  = vecgeom::VectorBackend::Real_v;
  using Facet_t = TriangleFacet<T>;

private:
  int fNfacets;              ///< Number of triangle facets on the section
  T fZmin = 0;               ///< Minimum Z
  T fZmax = 0;               ///< Maximum Z
  T fCubicVolume      = 0;   ///< Cubic volume
  T fSurfaceArea      = 0;   ///< Surface area
  Vector3D<T> fMinExtent;    ///< Minimum extent
  Vector3D<T> fMaxExtent;    ///< Maximum extent

  vector_t<Vector3D<T>> fVertices;                  ///< Vector of unique vertices
  vector_t<Facet_t *> fFacets;                      ///< Vector of triangular facets
  vector_t<TessellatedCluster<Real_v> *> fClusters; ///< Vector of facet clusters

protected:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddFacet(Facet_t *facet)
  {
    // Method adding a facet to the structure. The vertices are added to the
    // list of all vertices (including duplications) and the extent is re-adjusted.
    if (Right)
      assert(vecCore::math::Abs(facet.fNormal[2]) < kTolerance && "Section is supposed to be right");
    fFacets.push_back(facet);
    // Adjust extent
    T xmin        = vecCore::math::Min(facet->fVertices[0].x(), facet->fVertices[1].x(), facet->fVertices[2].x());
    T ymin        = vecCore::math::Min(facet->fVertices[0].y(), facet->fVertices[1].y(), facet->fVertices[2].y());
    T zmin        = vecCore::math::Min(facet->fVertices[0].z(), facet->fVertices[1].z(), facet->fVertices[2].z());
    fMinExtent[0] = vecCore::math::Min(fMinExtent[0], xmin);
    fMinExtent[1] = vecCore::math::Min(fMinExtent[1], ymin);
    fMinExtent[2] = vecCore::math::Min(fMinExtent[2], zmin);
    T xmax        = vecCore::math::Max(facet->fVertices[0].x(), facet->fVertices[1].x(), facet->fVertices[2].x());
    T ymax        = vecCore::math::Max(facet->fVertices[0].y(), facet->fVertices[1].y(), facet->fVertices[2].y());
    T zmax        = vecCore::math::Max(facet->fVertices[0].z(), facet->fVertices[1].z(), facet->fVertices[2].z());
    fMaxExtent[0] = vecCore::math::Max(fMaxExtent[0], xmax);
    fMaxExtent[1] = vecCore::math::Max(fMaxExtent[1], ymax);
    fMaxExtent[2] = vecCore::math::Max(fMaxExtent[2], zmax);
    // Check if we can create a Tessellated cluster
    int nfacets = fFacets.size();
    assert(nfacets <= fNfacets && "Cannot add extra facets to section");
    if (nfacets % kVecSize == 0 || nfacets == fNfacets) {
      int istart = nfacets - (nfacets-1) % kVecSize - 1;
      int i = 0;
      TessellatedCluster<T> *cluster = new TessellatedCluster<Real_v>();      
      for (; istart<nfacets; ++istart) {
        cluster->AddFacet(i++, fFacets[istart], istart);
      }
      // The last cluster may not be yet full: fill with last facet
      for (; i < kVecSize; ++i)
        tcl->AddFacet(i, facet, nfacets-1);
    }
  }

public:
  VECCORE_ATT_HOST_DEVICE
  TessellatedSection(int nfacets, T zmin, T zmax) : fNfacets(2*nfacets), fZmin(zmin), fZmax(zmax)
  {
    if (SameZ)
      assert(VecCore::math::Abs(zmax - zmin) < kTolerance && "zmin and zmax are not equal for sameZ section");
    else
      assert(zmax > zmin && "zmin is greater than zmax");
  }

  VECCORE_ATT_HOST_DEVICE
  bool AddQuadrilateralFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2,
                             Vector3D<T> const &vt3, bool absolute = true)
  {
    // We should check the quadrilateral convexity to correctly define the
    // triangle facets
    // CheckConvexity()vt0, vt1, vt2, vt3, absolute);
    Facet_t *facet = new Facet_t;
    if (absolute) {
      if (!facet->SetVertices(vt0, vt1, vt2)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
      facet = new Facet_t;
      if (!facet->SetVertices(vt0, vt2, vt3)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
    } else {
      if (!facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
      facet = new Facet_t;
      if (!facet->SetVertices(vt0, vt0 + vt1 + vt2, vt0 + vt1 + vt2, vt3)) {
        delete facet;
        return false;
      }
      AddFacet(facet);
    }
    return true;
  }


  VECCORE_ATT_HOST_DEVICE
  Inside_t Inside(Vector3D<T> const &point) const
  {
    using Bool_v = vecCore::Mask<Real_v>;

    // Check if point is in the bounding box
    if (point.x() - fMinExtent.x() < -kTolerance ||
        point.y() - fMinExtent.y() < -kTolerance ||
        point.z() - fMinExtent.z() < -kTolerance) return kOutside;
    if (point.x() - fMaxExtent.x() > kTolerance ||
        point.y() - fMaxExtent.y() > kTolerance ||
        point.z() - fMaxExtent.z() > kTolerance) return kOutside;
    
    if (SameZ) {
      // Section lies on same Z plane. The point can only be outside or on the
      // surface.
      if (vecCore::math::Abs(point.z() - fZmin) > kTolerance)
        return kOutside;
      Real_v pointv(point);
      Bool_v inside;
      for (int i=0; i<nclusters; ++i) {
        fClusters[i]->InsideCluster(pointv, inside);
        if (vecCore::MaskFull(inside))
          return kSurface;
      }
      return kOutside;
    }
    
    if (Convex) {
      // Convex polygone on top/bottom
      if (point.z() < fZmin - kTolerance || point.z() > fZmax + kTolerance)
        return kOutside;
      Real_v distplanes;
      Real_v pointv(point);
      Bool_v outside(Bool_v(false));
      Bool_v inside, outside;
      for (int i=0; i<nclusters; ++i) {
        distPlanes = fClusters[i]->DistPlanes(pointv);
        outside |= distPlanes > kTolerance;
        if (vecCore::MaskAny(outside))
          return kOutside;
        inside &= distPlanes < -kTolerance;
      }
      if (vecCore::MaskFull(inside))
        return kInside;
      return kSurface;          
    }
    
    // The general case.
    
    T distanceToIn = InfinityLength<T>();
    T disanceToOut = InfinityLength<T>();
    int isurfToIn = -1;
    int isurfToOut = -1;

    Real_v clusterToIn, clusterToOut;
    int icrtToIn, icrtToOut;
    for (int i=0; i<nclusters; ++i) {
      tessellated.fClusters[i]->DistanceToCluster(point, fTestDir, clusterToIn, clusterToOut, icrtToIn,
                                                             icrtToOut);

      // Update distanceToIn/Out
      if (icrtToIn >= 0 && clusterToIn < distanceToIn) {
        distanceToIn = clusterToIn;
        isurfToIn    = icrtToIn;
      }

      if (icrtToOut >= 0 && clusterToOut < distanceToOut) {
        distanceToOut = clusterToOut;
        isurfToOut    = icrtToOut;
      }
    }
    if (isurfToOut < 0) return kOutside;
    if (isurfToIn >= 0 && distanceToIn < distanceToOut &&
        distanceToIn * fTestDir.Dot(fFacets[isurfToIn]->fNormal) < -kTolerance)
      return kOutside;

    if (distanceToOut < 0 || distanceToOut * fTestDir.Dot(fFacets[isurfToOut]->fNormal) < kTolerance)
      return kSurface;

    if (isurfToIn < 0 || distanceToOut < distanceToIn)
      return kInside;

    if (distanceToIn < 0 || distanceToIn * fTestDir.Dot(tfFacets[isurfToIn]->fNormal) > -kTolerance)
      return kSurface;
  }

  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<T> const &point) const;

  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<T> const &point, Vector3D<T> const &direction,
                      int &isurf)
  {
    // Compute distance to segment from outside point, returning also the crossed 
    // facet.
    isurf = -1;
    T distance = InfinityLength<T>();
    T stepmax = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i=0; i<nclusters; ++i) {
      int isurfcrt = -1;
      T distcrt;
      fClusters[i]->DistanceToIn(point, dir, stepmax, distcrt, isurfcrt);
      if (distcrt < distance) {
        distance = distcrt;
        isurf = isurfcrt;
      }
    }
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction,
                      int &isurf)
  {
    // Compute distance to segment from point inside, returning also the crossed 
    // facet.
    isurf = -1;
    T distance = InfinityLength<T>();
    T stepmax = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i=0; i<nclusters; ++i) {
      int isurfcrt = -1;
      T distcrt;
      fClusters[i]->DistanceToOut(point, dir, stepmax, distcrt, isurfcrt);
      if (distcrt < distance) {
        distance = distcrt;
        isurf = isurfcrt;
      }
    }
    return distance;  
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToInSq(Vector3D<T> const &point)
  {
    // Compute safety squared to segment from point outside, returning also the crossed 
    // facet.
    isurf = -1;
    T safetysq = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i=0; i<nclusters; ++i) {
      int isurfcrt = -1;
      T safsqcrt = fClusters[i]->SafetySq<true>(point, isurfcrt);
      if (safsqcrt < safetysq) {
        safetysq = safetysqcrt;
        isurf = isurfcrt;
      }
    }
    return safetysq;
  }

  VECCORE_ATT_HOST_DEVICE
  T SafetyToOutSq(Vector3D<T> const &point)
  {
    // Compute safety squared to segment from point inside, returning also the crossed 
    // facet.
    isurf = -1;
    T safetysq = InfinityLength<T>();
    const int nclusters = fClusters.size();
    for (int i=0; i<nclusters; ++i) {
      int isurfcrt = -1;
      T safsqcrt = fClusters[i]->SafetySq<false>(point, isurfcrt);
      if (safsqcrt < safetysq) {
        safetysq = safetysqcrt;
        isurf = isurfcrt;
      }
    }
    return safetysq;
  }

  VECCORE_ATT_HOST_DEVICE
  void Normal(Vector3D<T> const &point, Vector3D<T> &normal, bool &valid)
  {
    // Compute normal to segment surface in given point near surface.
  }

};
} // end VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif


