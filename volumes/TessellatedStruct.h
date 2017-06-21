/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_
#define VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_

#include "TessellatedCluster.h"
#include "base/BitSet.h"
#include "base/RNG.h"
#include <vector>

#include "base/Stopwatch.h"
#include "management/HybridManager2.h"
#include "navigation/HybridNavigator2.h"
#include "management/ABBoxManager.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// Structure used for vectorizing queries on groups of triangles

template <typename T = double>
class TessellatedStruct {

  using BitSet  = veccore::BitSet;
  using Real_v  = vecgeom::VectorBackend::Real_v;
  using Facet_t = TriangleFacet<T>;

  // Here we should be able to use vecgeom::Vector
  template <typename U>
  using vector_t = std::vector<U>;

  using BVHStructure = HybridManager2::HybridBoxAccelerationStructure;

  //__________________________________________________________________________
  struct GridCell {
    vector_t<int> fArray; // Array of facet indices
    bool fUsed = false;   // Used flag

    VECCORE_ATT_HOST_DEVICE
    GridCell() { /* fArray.reserve(4); */}
  };

  //__________________________________________________________________________
  struct GridHelper {
    int fNgrid       = 0;           ///< Grid size
    int fNcells      = 0;           ///< Number of cells in the grid
    int fNcached     = 0;           ///< number of cached cells
    GridCell **fGrid = nullptr;     ///< Grid for clustering facets
    Vector3D<T> fMinExtent;         ///< Minimum extent
    Vector3D<T> fMaxExtent;         ///< Maximum extent
    Vector3D<T> fInvExtSize;        ///< Inverse extent size
    vector_t<Vector3D<T>> fAllVert; ///< Full list of vertices

    GridHelper() {}

    ~GridHelper()
    {
      if (fGrid) {
        for (int i = 0; i < fNcells; ++i)
          delete fGrid[i];
        delete[] fGrid;
      }
    }

    void CreateCells(int ngrid)
    {
      if (fNgrid) return;
      fNgrid  = ngrid;
      fNcells = ngrid * ngrid * ngrid;
      fGrid   = new GridCell *[fNcells];
      for (int i = 0; i < fNcells; ++i)
        fGrid[i] = new GridCell();
    }

    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    void ClearCells()
    {
      for (int icell = 0; icell < fNcells; ++icell)
        fGrid[icell]->fArray.clear();
    }

    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    GridCell *GetCell(int ind[3]) { return fGrid[fNgrid * fNgrid * ind[0] + fNgrid * ind[1] + ind[2]]; }

    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    GridCell *GetCell(Vector3D<T> const &point, int ind[3])
    {
      Vector3D<T> ratios = (point - fMinExtent) * fInvExtSize;
      assert(ratios[0] <= 1. && ratios[1] <= 1 && ratios[2] <= 1);
      for (int i = 0; i < 3; ++i) {
        ind[i] = ratios[i] * fNgrid;
        ind[i] = vecCore::math::Max(ind[i], 0);
        ind[i] = vecCore::math::Min(ind[i], fNgrid - 1);
      }
      return (GetCell(ind));
    }
  };

public:
  bool fSolidClosed   = false;        ///< Closure of the solid
  T fCubicVolume      = 0;            ///< Cubic volume
  T fSurfaceArea      = 0;            ///< Surface area
  GridHelper *fHelper = nullptr;      ///< Grid helper
  Vector3D<T> fMinExtent;             ///< Minimum extent
  Vector3D<T> fMaxExtent;             ///< Maximum extent
  Vector3D<T> fInvExtSize;            ///< Inverse extent size
  Vector3D<T> fTestDir;               ///< Test direction for Inside function
  BVHStructure *fNavHelper = nullptr; ///< Navigation helper using bounding boxes

  // Here we have a pointer to the aligned bbox structure
  // ABBoxanager *fABBoxManager;

  vector_t<int> fCluster;                           ///< Cluster of facets storing just the indices
  vector_t<int> fCandidates;                        ///< Candidates for the current cluster
  vector_t<Vector3D<T>> fVertices;                  ///< Vector of unique vertices
  vector_t<Facet_t *> fFacets;                      ///< Vector of triangular facets
  vector_t<TessellatedCluster<Real_v> *> fClusters; ///< Vector of facet clusters
  BitSet *fSelected          = nullptr;             ///< Facets already in clusters
  int fNcldist[kVecSize + 1] = {0};                 ///< Distribution of number of cluster size

private:
  void CreateABBoxes()
  {
    using Boxes_t           = ABBoxManager::ABBoxContainer_t;
    using BoxCorner_t       = ABBoxManager::ABBox_s;
    int nclusters           = fClusters.size();
    BoxCorner_t *boxcorners = new BoxCorner_t[2 * nclusters];
    for (int i = 0; i < nclusters; ++i) {
      boxcorners[2 * i]     = fClusters[i]->fMinExtent;
      boxcorners[2 * i + 1] = fClusters[i]->fMaxExtent;
    }
    Boxes_t boxes = &boxcorners[0];
    fNavHelper    = HybridManager2::Instance().BuildStructure(boxes, nclusters);
  }

public:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TessellatedStruct()
  {
    fMinExtent = InfinityLength<T>();
    fMaxExtent = -InfinityLength<T>();
    fHelper    = new GridHelper();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ~TessellatedStruct()
  {
    delete fHelper;
    if (fSelected) BitSet::ReleaseInstance(fSelected);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddFacet(Facet_t *facet)
  {
    // Method adding a facet to the structure. The vertices are added to the
    // list of all vertices (including duplications) and the extent is re-adjusted.
    fFacets.push_back(facet);
    int ind = fHelper->fAllVert.size();
    // Add the three vertices
    for (int i = 0; i < 3; ++i) {
      fHelper->fAllVert.push_back(facet->fVertices[i]);
      facet->fIndices[i] = ind + i;
    }
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
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int AddVertex(Vector3D<T> const &vertex)
  {
    // Add a non-duplicated vertex to the solid. Duplications are only checked
    // in the grid cell containing the vertex. An index to the unique vertex is
    // added to the cell, while the vertex positionis added to the list fVertices
    // Get the cell in which to add the vertex
    int ind[3];
    GridCell *cell = fHelper->GetCell(vertex, ind);
    // Loop existing vertices in the cell and check for a duplicate
    constexpr Precision tolerancesq = kTolerance * kTolerance;
    for (int ivert : cell->fArray) {
      // existing vertex?
      if ((fVertices[ivert] - vertex).Mag2() < tolerancesq) return ivert;
    }
    // Push new vertex into the tessellated structure
    int ivertnew = fVertices.size();
    fVertices.push_back(vertex);
    // Update the cell with the new vertex index
    cell->fArray.push_back(ivertnew);
    return ivertnew;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Extent(Vector3D<T> &amin, Vector3D<T> &amax)
  {
    amin = fMinExtent;
    amax = fMaxExtent;
  }

  void Close()
  {
    // The solid becomes now closed. A cell grid is computed base on the extent
    // to fasten up the search for duplicates and neighbors.
    int ind[3];
    fInvExtSize = fMaxExtent - fMinExtent;
    if (fInvExtSize[0] * fInvExtSize[1] * fInvExtSize[2] < kTolerance) {
      std::cout << "Tessellated structure is flat - not allowed\n";
      return;
    }
    fInvExtSize          = 1. / fInvExtSize;
    fHelper->fMinExtent  = fMinExtent;
    fHelper->fMaxExtent  = fMaxExtent;
    fHelper->fInvExtSize = fInvExtSize;
    // Make a grid with ~kVecSize facets per cell
    int ngrid = 1 + size_t(vecCore::math::Pow<T>(T(fFacets.size()), 1. / 3.));
    Stopwatch timer;
    timer.Start();
    fHelper->CreateCells(ngrid);
    auto time = timer.Stop();
    std::cout << "CreateCells: " << time << " sec\n";

    timer.Start();
    // Loop over facets and their vertices, fill list of vertices free of
    // duplications.
    for (auto facet : fFacets) {
      for (int ivert = 0; ivert < 3; ++ivert) {
        facet->fIndices[ivert] = AddVertex(facet->fVertices[ivert]);
      }
    }
    time = timer.Stop();
    std::cout << "Remove duplicates: " << time << " sec\n";

    timer.Start();
    // Clear vertices and store facet indices in the grid helper
    fHelper->ClearCells();
    unsigned ifacet = 0;
    for (auto facet : fFacets) {
      fHelper->GetCell(facet->fCenter, ind)->fArray.push_back(ifacet);
      //      for (int ivert = 0; ivert < 3; ++ivert) {
      //        fHelper->GetCell(facet->fVertices[ivert], ind)->fArray.push_back(ifacet);
      //      }
      ifacet++;
    }
    time = timer.Stop();
    std::cout << "Store facets into grid: " << time << " sec\n";

    // Make clusters
    timer.Start();
    //    std::cout << "=== Using dummy clusters\n";
    //    CreateDummyClusters();

    const int nfacets = fFacets.size();
    fSelected         = BitSet::MakeInstance(nfacets);
    fSelected->ResetAllBits();
    fCandidates.clear();
    ifacet = 0;
    fCandidates.push_back(ifacet);
    fSelected->SetBitNumber(ifacet);
    TessellatedCluster<Real_v> *cluster;
    while (fCandidates.size()) {
      // Use existing candidates in fCandidates to create the cluster
      cluster               = CreateCluster();
      if (!cluster) cluster = MakePartialCluster();
      fClusters.push_back(cluster);
      // Fill cluster from the same cell or from a neighbor cell
      if (!fCandidates.size()) {
        ifacet = fSelected->FirstNullBit();
        if (ifacet == fFacets.size()) break;
        fCandidates.push_back(ifacet);
        fSelected->SetBitNumber(ifacet);
      }
    }

    time = timer.Stop();
    std::cout << "Clusterizer: " << time << " sec\n";

    // Create navigation helper to be used in TessellatedImplementation
    timer.Start();
    CreateABBoxes(); // to navigate, see: TestHybridBVH.cpp/HybridNavigator2.h/HybridSafetyEstimator.h
    time = timer.Stop();
    std::cout << "Create AABoxes: " << time << " sec\n";
    // Generate random direction non-parallel to any of the surfaces
    constexpr T tolerance(1.e-8);
    while (1) {
      RandomDirection(fTestDir);
      // Loop over triangles and check that dot product is not close to 0
      for (auto facet : fFacets) {
        if (vecCore::math::Abs(facet->fNormal.Dot(fTestDir)) < tolerance) break;
      }
      break;
    }
    fSolidClosed = true;
  }

  void RandomDirection(Vector3D<double> &direction)
  {

    double phi    = RNG::Instance().uniform(0., 2. * kPi);
    double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
    direction.x() = std::sin(theta) * std::cos(phi);
    direction.y() = std::sin(theta) * std::sin(phi);
    direction.z() = std::cos(theta);
  }

  void CreateDummyClusters()
  {
    // Loop over facets and group them in clusters in the order of definition
    TessellatedCluster<Real_v> *tcl = nullptr;
    int i                           = 0;
    int j                           = 0;
    for (auto facet : fFacets) {
      i = i % kVecSize;
      if (i == 0) {
        if (tcl) fClusters.push_back(tcl);
        tcl = new TessellatedCluster<Real_v>();
      }
      tcl->AddFacet(i++, facet, j++);
    }
    // The last cluster may not be yet full
    for (; i < kVecSize; ++i)
      tcl->AddFacet(i, tcl->fFacets[0], tcl->fIfacets[0]);
  }
  TessellatedCluster<Real_v> *CreateCluster()
  {
    // Create cluster starting from fCandidates list
    unsigned nfacets = 0;
    assert(fCandidates.size() > 0); // call the method with at least one candidate in the list
    constexpr int rankmax = 3;      // ??? how to determine an appropriate value ???
    int rank              = 0;
    fCluster.clear();
    int ifacet = fCandidates[0];
    fCluster.push_back(ifacet);
    while (fCandidates.size() < kVecSize && rank < rankmax) {
      GatherNeighborCandidates(fCandidates[0], rank++);
    }
    if (fCandidates.size() < kVecSize) return nullptr;
    fCandidates.erase(fCandidates.begin());
    // Add facets with maximum neighborhood weight to existing cluster
    int iref = 0;
    while (nfacets < kVecSize) {
      nfacets = AddCandidatesToCluster(4 >> iref++); // 4 common vertices, 2, 1 or none
    }

    // The cluster is now complete, create a tessellated cluster object
    TessellatedCluster<Real_v> *tcl = new TessellatedCluster<Real_v>();
    int i                           = 0;
    for (auto ifct : fCluster) {
      Facet_t *facet = fFacets[ifct];
      tcl->AddFacet(i++, facet, ifct);
    }
    fNcldist[fCluster.size()]++;
    return tcl;
  }

  TessellatedCluster<Real_v> *MakePartialCluster()
  {
    // Create partial cluster starting from fCandidates list
    for (auto ifacet : fCandidates) {
      fCluster.push_back(ifacet);
    }
    fNcldist[fCluster.size()]++;
    fCandidates.clear();
    while (fCluster.size() < kVecSize)
      fCluster.push_back(fCluster[0]);

    // The cluster is now complete, create a tessellated cluster object
    TessellatedCluster<Real_v> *tcl = new TessellatedCluster<Real_v>();
    int i                           = 0;
    for (auto ifacet : fCluster) {
      Facet_t *facet = fFacets[ifacet];
      tcl->AddFacet(i++, facet, ifacet);
    }
    return tcl;
  }

  int AddCandidatesToCluster(int weightmin)
  {
    // Add all candidate having the required weight to the cluster, until cluster
    // is complete
    for (auto it = fCandidates.begin(); it != fCandidates.end() && fCluster.size() < kVecSize;) {
      int weight = NeighborToCluster(*it);
      if (weight >= weightmin) {
        fCluster.push_back(*it);
        it = fCandidates.erase(it);
      } else {
        ++it;
      }
    }
    return fCluster.size();
  }

  int NeighborToCluster(int ifacet)
  {
    // Get neighborhood 'weight' of a candidate with respect to the existing cluster
    Facet_t *facet = fFacets[ifacet];
    int weight     = 0;
    for (auto icand : fCluster) {
      Facet_t *other = fFacets[icand];
      weight += facet->IsNeighbor(*other);
    }
    return weight;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddCandidatesFromCell(int ind[3])
  {
    GridCell *cell = fHelper->GetCell(ind);
    if (cell->fUsed) return;
    for (auto neighbor : cell->fArray) {
      if (!fSelected->TestBitNumber(neighbor)) {
        fSelected->SetBitNumber(neighbor);
        fCandidates.push_back(neighbor);
      }
    }
    cell->fUsed = true;
  }

  int GatherNeighborCandidates(int ifacet, int rank)
  {
    // Gather candidates from cells neighboring the cell containing the center
    // of the facet
    int ind0[3], ind[3];
    const Facet_t *facet = fFacets[ifacet];
    fHelper->GetCell(facet->fCenter, ind0);
    if (rank == 0) {
      AddCandidatesFromCell(ind0);
      return (fCandidates.size());
    }

    // Establish cell index limits for the requested rank
    int limits[6];
    bool limited[6] = {false};
    for (int i = 0; i < 3; ++i) {
      limits[2 * i] = ind0[i] - rank;
      if (limits[2 * i] < 0) {
        limits[2 * i]  = 0;
        limited[2 * i] = true;
      }
      limits[2 * i + 1] = ind0[i] + rank;
      if (limits[2 * i + 1] > fHelper->fNgrid - 1) {
        limits[2 * i + 1]  = fHelper->fNgrid - 1;
        limited[2 * i + 1] = true;
      }
    }
    // Gather all cells for the given rank
    for (int iax1 = 0; iax1 < 3; ++iax1) {
      int iax2 = (iax1 + 1) % 3;
      int iax3 = (iax1 + 2) % 3;
      for (int iside = 0; iside < 2; ++iside) {
        if (limited[2 * iax1 + iside]) continue;
        ind[iax1] = limits[2 * iax1 + iside];
        for (ind[iax2] = limits[2 * iax2]; ind[iax2] <= limits[2 * iax2 + 1]; ind[iax2]++) {
          for (ind[iax3] = limits[2 * iax3]; ind[iax3] <= limits[2 * iax3 + 1]; ind[iax3]++) {
            AddCandidatesFromCell(ind);
          }
        }
      }
    }
    return (fCandidates.size());
  }

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
    Facet_t *facet = new Facet_t;
    bool added     = false;
    if (absolute)
      added = facet->SetVertices(vt0, vt1, vt2);
    else
      added = facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2);
    if (!added) {
      delete facet;
      return false;
    }
    AddFacet(facet);
    return true;
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

  // Navigation functions
  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  void DistanceToSolid(Vector3D<T> const &point, Vector3D<T> const &direction, T const &stepMax, T &distance,
                       int &isurf) const
  {
    // int ntries = 0;
    // Check if the bounding box is hit

    Vector3D<Precision> invdir(1. / direction.x(), 1. / direction.y(), 1. / direction.z());
    Vector3D<int> sign;
    sign[0]  = invdir.x() < 0;
    sign[1]  = invdir.y() < 0;
    sign[2]  = invdir.z() < 0;
    distance = BoxImplementation::IntersectCachedKernel2<T, T>(&fMinExtent, point, invdir, sign.x(), sign.y(), sign.z(),
                                                               -kTolerance, InfinityLength<T>());
    if (distance >= InfinityLength<T>()) return;

    // Define the user hook calling DistanceToIn for the cluster with the same
    // index as the bounding box
    distance      = InfinityLength<T>();
    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the distance to the current box is bigger than the
      // requested limit or than the current distance
      if (hitbox.second > vecCore::math::Min(stepMax, distance)) return true;
      // Compute distance to the cluster
      T distcrt;
      int isurfcrt;
      if (ToIn)
        fClusters[hitbox.first]->DistanceToIn(point, direction, stepMax, distcrt, isurfcrt);
      else
        fClusters[hitbox.first]->DistanceToOut(point, direction, stepMax, distcrt, isurfcrt);
      if (distcrt < distance) {
        distance = distcrt;
        isurf    = fClusters[hitbox.first]->fIfacets[isurfcrt];
      }
      // ntries++;
      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    // intersect ray with the BVH structure and use hook
    boxNav->BVHSortedIntersectionsLooper(*fNavHelper, point, direction, userhook);
    //    printf("ntries = %d\n", ntries);
  }

  bool Contains(Vector3D<T> const &point) const
  {
    int isurf;
    T stepMax = InfinityLength<T>();
    T distOut, distIn;
    DistanceToSolid<false>(point, fTestDir, stepMax, distOut, isurf);
    // If distance to out is infinite the point is outside
    if (distOut >= stepMax) return false;

    DistanceToSolid<true>(point, fTestDir, stepMax, distIn, isurf);
    // If distance to out is finite and less than distance to in, the point is inside
    if (distOut < distIn) return true;
    return false;
  }

  EnumInside Inside(Vector3D<T> const &point) const
  {
    int isurf;
    T stepMax = InfinityLength<T>();
    T distOut, distIn;
    DistanceToSolid<false>(point, fTestDir, stepMax, distOut, isurf);
    // If distance to out is infinite the point is outside
    if (distOut >= stepMax) return kOutside;
    if (distOut < 0 || distOut * fTestDir.Dot(fFacets[isurf]->fNormal) < kTolerance) return kSurface;

    DistanceToSolid<true>(point, fTestDir, stepMax, distIn, isurf);
    // If distance to out is finite and less than distance to in, the point is inside
    if (distOut < distIn) return kInside;
    if (distIn < 0 || distIn * fTestDir.Dot(fFacets[isurf]->fNormal) > -kTolerance) return kSurface;
    return kOutside;
  }

}; // end class
}
} // end namespace

#endif
