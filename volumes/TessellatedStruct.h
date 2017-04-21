/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_
#define VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_

#include "TesselatedCluster.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// Structure used for vectorizing queries on groups of triangles

template <typename T = double>
class TessellatedStruct {

  using Real_v = vecgeom::VectorBackend::Real_v;

private:
  bool fSolidClosed;      ///< Closure of the solid
  int fNsurrounded = 0;   ///< Number of surrounded facets
  T fCubicVolume   = 0;   ///< cubic volume
  Vector3D<T> fMinExtent; ///< Minimum extent
  Vector3D<T> fMaxExtent; ///< Maximum extent

  // Here we have a pointer to the aligned bbox structure
  // ABBoxanager *fABBoxManager;

  Vector<Vector3D<T>> fVertices;                ///< Vector of unique vertices
  Vector<TriangleFacet> fFacets;                ///< Vector of triangular facets
  Vector<TessellatedCluster<Real_v>> fClusters; ///< Vector of facet clusters

protected:
  VECCORE_ATT_HOST_DEVICE
  int AddVertex(Vector3D<T> const &vtx)
  {
    // This method needs to check if the vertex is duplicated. The
    // index of the vertex is returned.

    // The trivial loop makes AddVertex a N squared problem...
    // *** TO DO: uniform grid store/search, ...
    constexpr Precision tolerancesq = kTolerance * kTolerance;
    int ivert                       = 0;
    for (auto vertex : fVertices) {
      if ((vtx - vertex).Mag2() < tolerancesq) return ivert;
      ivert++;
    }
    fVertices.push_back(vtx);
    return ivert;
  }

  VECCORE_ATT_HOST_DEVICE
  void FindNeighbors(TriangleFacet &facet, int ifacet)
  {
    // Loop non-closed facets.
    int nfacets = fFacets.size();
    for (int icrt = 0; icrt < nfacets; ++icrt) {
      if (fFacets[icrt].IsSurrounded()) continue;
      if (facet.IsNeighbor(fFacets[icrt])) {
        facet.fNeighbors.push_back(icrt);
        fFacets[icrt].fNeighbors.push_back(ifacet);
        if (fFacets[icrt].IsSurrounded()) fNsurrounded++;
        if (facet.IsSurrounded()) {
          fNsurrounded++ return;
        }
      }
    }
  }

public:
  VECCORE_ATT_HOST_DEVICE
  TesselatedStruct() { fClusterSize = vecCore::VectorSize<vecgeom::VectorBackend::Real_v>(); }

  /* @brief Methods for adding a new facet
   * @detailed The method akes 4 parameters to define the three fVertices:
   *      1) UFacetvertexType = "ABSOLUTE": in this case Pt0, vt1 and vt2 are
   *         the 3 fVertices in anti-clockwise order looking from the outsider.
   *      2) UFacetvertexType = "RELATIVE": in this case the first vertex is Pt0,
   *         the second vertex is Pt0+vt1 and the third vertex is Pt0+vt2, all
   *         in anti-clockwise order when looking from the outsider.
   */
  VECCORE_ATT_HOST_DEVICE
  bool AddTriangularFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2, bool absolute = true)
  {
    TriangleFacet facet;
    int ind0, ind1, ind2;
    ind0 = AddVertex(vt0);
    if (absolute) {
      ind1 = AddVertex(vt1);
      ind2 = AddVertex(vt2);
    } else {
      ind1 = AddVertex(vt0 + vt1);
      ind2 = AddVertex(vt0 + vt1 + vt2);
    }
    if (!facet.AddVertices(fVertices[ind0], fVertices[ind1], fVertices[ind2], ind0, ind1, ind2)) return false;
    FindNeighbors(facet);
    fFacets.push_back(facet);
    if (fNsurrounded == fFacets.size()) fClosed = true;
  }

  VECCORE_ATT_HOST_DEVICE
  bool AddQuadrilateralFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2,
                             Vector3D<T> const &vt3, bool absolute = true)
  {
    // We should check the quadrilateral convexity to correctly define the
    // triangle facets
    // CheckConvexity()vt0, vt1, vt2, vt3, absolute);
    TriangleFacet facet1, facet2;
    int ind0, ind1, ind2, ind22;
    ind0 = AddVertex(vt0);
    if (absolute) {
      ind1  = AddVertex(vt1);
      ind2  = AddVertex(vt2);
      ind22 = AddVertex(vt3);
    } else {
      ind1  = AddVertex(vt0 + vt1);
      ind2  = AddVertex(vt0 + vt1 + vt2);
      ind22 = AddVertex(vt0 + vt1 + vt2 + vt3);
    }
    bool added1 = facet1.AddVertices(fVertices[ind0], fVertices[ind1], fVertices[ind2], ind0, ind1, ind2);
    bool added2 = facet2.AddVertices(fVertices[ind0], fVertices[ind2], fVertices[ind22], ind0, ind2, ind22);

    // The two facets are neighbors
    int ifacet1 = fFacets.size();
    if (added1) facet2.fNeighbors.push_back(ifacet1);
    if (added2) facet1.fNeighbors.push_back(ifacet1 + 1);
    if (added1) {
      FindNeighbors(facet1);
      fFacets.push_back(facet1);
    }
    if (added2) {
      FindNeighbors(facet2);
      fFacets.push_back(facet2);
    }
    if (fNsurrounded == fFacets.size()) fClosed = true;
    return (added1 || added2);
  }
};
}
} // end

#endif
