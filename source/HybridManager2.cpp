/*
 * HybridManager.cpp
 *
 *  Created on: 03.08.2015
 *      Author: yang.zhang@cern.ch
 */

#include "management/HybridManager2.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/GeoManager.h"
#include "management/ABBoxManager.h"
#include "base/SOA3D.h"
#include "volumes/utilities/VolumeUtilities.h"
#include <map>
#include <vector>
#include <sstream>
#include <queue>
#include <set>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void HybridManager2::InitStructure(LogicalVolume const *lvol)
{
  if (fVolumeToABBoxes.size() != GeoManager::Instance().GetRegisteredVolumesCount()) {
    fVolumeToABBoxes.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
  }
  if (fVolumeToABBoxes_v.size() != GeoManager::Instance().GetRegisteredVolumesCount()) {
    fVolumeToABBoxes_v.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
  }
  if (fNodeToDaughters.size() != GeoManager::Instance().GetRegisteredVolumesCount()) {
    fNodeToDaughters.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
  }
  if (fNodeToDaughters_v.size() != GeoManager::Instance().GetRegisteredVolumesCount()) {
    fNodeToDaughters_v.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
  }
  if (fVolumeToABBoxes[lvol->id()] != nullptr || fVolumeToABBoxes_v[lvol->id()] != nullptr) {
    RemoveStructure(lvol);
  }

  // BuildStructure(lvol);
  BuildStructure_v(lvol);
}

void HybridManager2::BuildStructure(LogicalVolume const *vol)
{
  int numberOfFirstLevelNodes = vol->GetDaughters().size() / vecCore::VectorSize<Float_v>() +
                                (vol->GetDaughters().size() % vecCore::VectorSize<Float_v>() == 0 ? 0 : 1);
  int numberOfNodes = numberOfFirstLevelNodes + vol->GetDaughters().size();
  std::vector<std::vector<int>> clusters(numberOfFirstLevelNodes);
  SOA3D<Precision> centers(numberOfFirstLevelNodes);
  SOA3D<Precision> allvolumecenters(vol->GetDaughters().size());
  InitClustersWithKMeans(vol, clusters, centers, allvolumecenters);

  EqualizeClusters(clusters, centers, allvolumecenters, vecCore::VectorSize<Float_v>());

  fNodeToDaughters[vol->id()] = new std::vector<int>[ numberOfFirstLevelNodes ];

  int nDaughters;
  ABBoxManager::ABBoxContainer_t daughterboxes = ABBoxManager::Instance().GetABBoxes(vol, nDaughters);

  ABBoxContainer_t boxes = new ABBox_s[numberOfNodes * 2];

  // init internal nodes for unvectorized

  int vectorindex = 0;
  for (int i = 0; i < numberOfFirstLevelNodes; ++i) {
    Vector3D<Precision> lowerCornerFirstLevelNode(kInfLength), upperCornerFirstLevelNode(-kInfLength);
    vectorindex += 2;
    for (size_t d = 0; d < clusters[i].size(); ++d) {
      int daughterIndex               = clusters[i][d];
      Vector3D<Precision> lowerCorner = daughterboxes[2 * daughterIndex];
      Vector3D<Precision> upperCorner = daughterboxes[2 * daughterIndex + 1];
      for (int axis = 0; axis < 3; axis++) {
        lowerCornerFirstLevelNode[axis] = std::min(lowerCornerFirstLevelNode[axis], lowerCorner[axis]);
        upperCornerFirstLevelNode[axis] = std::max(upperCornerFirstLevelNode[axis], upperCorner[axis]);
      }
      boxes[vectorindex++] = lowerCorner;
      boxes[vectorindex++] = upperCorner;

      fNodeToDaughters[vol->id()][i].push_back(daughterIndex);
    }
    boxes[2 * i * (vecCore::VectorSize<Float_v>() + 1)]     = lowerCornerFirstLevelNode;
    boxes[2 * i * (vecCore::VectorSize<Float_v>() + 1) + 1] = upperCornerFirstLevelNode;
  }
  fVolumeToABBoxes[vol->id()] = boxes;
}

/**
 * build bvh bruteforce AND vectorized
 */
void HybridManager2::BuildStructure_v(LogicalVolume const *vol)
{
  if (vol->GetDaughters().size() == 0) return;
  constexpr auto kVS             = vecCore::VectorSize<HybridManager2::Float_v>();
  size_t numberOfFirstLevelNodes = vol->GetDaughters().size() / kVS + (vol->GetDaughters().size() % kVS == 0 ? 0 : 1);
  size_t numberOfNodes           = numberOfFirstLevelNodes + vol->GetDaughters().size();
  size_t vectorsize =
      numberOfFirstLevelNodes / kVS + (numberOfFirstLevelNodes % kVS == 0 ? 0 : 1) + numberOfFirstLevelNodes;

  std::vector<std::vector<int>> clusters(numberOfFirstLevelNodes);
  SOA3D<Precision> centers(numberOfFirstLevelNodes);
  SOA3D<Precision> allvolumecenters(vol->GetDaughters().size());

  InitClustersWithKMeans(vol, clusters, centers, allvolumecenters);

  // EqualizeClusters(clusters, centers, allvolumecenters, HybridManager2::vecCore::VectorSize<Float_v>());

  fNodeToDaughters[vol->id()]   = new std::vector<int>[ numberOfFirstLevelNodes ];
  fNodeToDaughters_v[vol->id()] = new std::vector<int>[ numberOfFirstLevelNodes ];

  int nDaughters;
  // we are using the existing aligned bounding box list
  HybridManager2::ABBoxContainer_t daughterboxes = ABBoxManager::Instance().GetABBoxes(vol, nDaughters);

  ABBoxContainer_t boxes   = new ABBox_s[numberOfNodes * 2];
  ABBoxContainer_v boxes_v = new ABBox_v[vectorsize * 2];

  // init internal nodes for unvectorized
  int vectorindex   = 0;
  int vectorindex_v = 0;
  for (size_t i = 0; i < numberOfFirstLevelNodes; ++i) {
    Vector3D<Precision> lowerCornerFirstLevelNode(kInfLength), upperCornerFirstLevelNode(-kInfLength);
    vectorindex += 2;
    for (size_t d = 0; d < clusters[i].size(); ++d) {
      int daughterIndex               = clusters[i][d];
      Vector3D<Precision> lowerCorner = daughterboxes[2 * daughterIndex];
      Vector3D<Precision> upperCorner = daughterboxes[2 * daughterIndex + 1];
      for (unsigned int axis = 0; axis < 3; axis++) {
        lowerCornerFirstLevelNode[axis] = std::min(lowerCornerFirstLevelNode[axis], lowerCorner[axis]);
        upperCornerFirstLevelNode[axis] = std::max(upperCornerFirstLevelNode[axis], upperCorner[axis]);
      }
      boxes[vectorindex++] = lowerCorner;
      boxes[vectorindex++] = upperCorner;

      fNodeToDaughters[vol->id()][i].push_back(daughterIndex);
      fNodeToDaughters_v[vol->id()][i].push_back(daughterIndex);
    }
    boxes[2 * i * (kVS + 1)]     = lowerCornerFirstLevelNode;
    boxes[2 * i * (kVS + 1) + 1] = upperCornerFirstLevelNode;
  }
  fVolumeToABBoxes[vol->id()] = boxes;

  // init boxes_v to -inf
  for (size_t i = 0; i < vectorsize * 2; i++) {
    boxes_v[i] = -InfinityLength<typename HybridManager2::Float_v>();
  }

  // init internal nodes for vectorized
  for (size_t i = 0; i < numberOfFirstLevelNodes; ++i) {
    if (i % kVS == 0) {
      vectorindex_v += 2;
    }
    Vector3D<Precision> lowerCornerFirstLevelNode(kInfLength), upperCornerFirstLevelNode(-kInfLength);
    for (size_t d = 0; d < clusters[i].size(); ++d) {
      int daughterIndex               = clusters[i][d];
      Vector3D<Precision> lowerCorner = daughterboxes[2 * daughterIndex];
      Vector3D<Precision> upperCorner = daughterboxes[2 * daughterIndex + 1];

      using vecCore::AssignLane;

      AssignLane(boxes_v[vectorindex_v].x(), d, lowerCorner.x());
      AssignLane(boxes_v[vectorindex_v].y(), d, lowerCorner.y());
      AssignLane(boxes_v[vectorindex_v].z(), d, lowerCorner.z());
      AssignLane(boxes_v[vectorindex_v + 1].x(), d, upperCorner.x());
      AssignLane(boxes_v[vectorindex_v + 1].y(), d, upperCorner.y());
      AssignLane(boxes_v[vectorindex_v + 1].z(), d, upperCorner.z());
      for (int axis = 0; axis < 3; axis++) {
        lowerCornerFirstLevelNode[axis] = std::min(lowerCornerFirstLevelNode[axis], lowerCorner[axis]);
        upperCornerFirstLevelNode[axis] = std::max(upperCornerFirstLevelNode[axis], upperCorner[axis]);
      }
    }
    vectorindex_v += 2;
    // insert internal node ABBOX in boxes_v
    int indexForInternalNode  = i / kVS;
    indexForInternalNode      = 2 * (kVS + 1) * indexForInternalNode;
    int offsetForInternalNode = i % kVS;
    using vecCore::AssignLane;
    AssignLane(boxes_v[indexForInternalNode].x(), offsetForInternalNode, lowerCornerFirstLevelNode.x());
    AssignLane(boxes_v[indexForInternalNode].y(), offsetForInternalNode, lowerCornerFirstLevelNode.y());
    AssignLane(boxes_v[indexForInternalNode].z(), offsetForInternalNode, lowerCornerFirstLevelNode.z());
    AssignLane(boxes_v[indexForInternalNode + 1].x(), offsetForInternalNode, upperCornerFirstLevelNode.x());
    AssignLane(boxes_v[indexForInternalNode + 1].y(), offsetForInternalNode, upperCornerFirstLevelNode.y());
    AssignLane(boxes_v[indexForInternalNode + 1].z(), offsetForInternalNode, upperCornerFirstLevelNode.z());
  }

  fVolumeToABBoxes_v[vol->id()] = boxes_v;
}

void HybridManager2::RemoveStructure(LogicalVolume const *lvol)
{
  if (fVolumeToABBoxes[lvol->id()]) delete[] fVolumeToABBoxes[lvol->id()];
  if (fVolumeToABBoxes_v[lvol->id()]) delete[] fVolumeToABBoxes_v[lvol->id()];
}

/**
 * assign daughter volumes to its closest cluster using the cluster centers stored in centers.
 * clusters need to be empty before this function is called
 */
void HybridManager2::AssignVolumesToClusters(std::vector<std::vector<int>> &clusters, SOA3D<Precision> const &centers,
                                             SOA3D<Precision> const &allvolumecenters)
{

  assert(centers.size() == clusters.size());
  int numberOfDaughers = allvolumecenters.size();
  int numberOfClusters = clusters.size();

  Precision minDistance;
  int closestCluster;

  for (int d = 0; d < numberOfDaughers; d++) {
    minDistance    = kInfLength;
    closestCluster = -1;

    Precision dist;
    for (int c = 0; c < numberOfClusters; ++c) {
      dist = (allvolumecenters[d] - centers[c]).Length2();
      if (dist < minDistance) {
        minDistance    = dist;
        closestCluster = c;
      }
    }

    clusters.at(closestCluster).push_back(d);
  }
}

void HybridManager2::RecalculateCentres(SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                                        std::vector<std::vector<int>> const &clusters)
{
  assert(centers.size() == clusters.size());
  auto numberOfClusters = centers.size();
  for (size_t c = 0; c < numberOfClusters; ++c) {
    Vector3D<Precision> newCenter(0);
    for (size_t clustersize = 0; clustersize < clusters[c].size(); ++clustersize) {
      int daughterIndex = clusters[c][clustersize];
      newCenter += allvolumecenters[daughterIndex];
    }
    newCenter /= clusters[c].size();
    centers.set(c, newCenter);
  }
}

template <typename Container_t>
void HybridManager2::InitClustersWithKMeans(LogicalVolume const *lvol, Container_t &clusters, SOA3D<Precision> &centers,
                                            SOA3D<Precision> &allvolumecenters, int const numberOfIterations)
{
  int numberOfClusters  = clusters.size();
  int numberOfDaughters = lvol->GetDaughters().size();

  int size;
  ABBoxManager::ABBoxContainer_t boxes = ABBoxManager::Instance().GetABBoxes(lvol, size);
  Vector3D<Precision> meanCenter(0);
  std::set<int> daughterSet;
  for (int i = 0; i < numberOfDaughters; ++i) {
    Vector3D<Precision> center = 0.5 * (boxes[2 * i] + boxes[2 * i + 1]);
    allvolumecenters.set(i, center);
    daughterSet.insert(i);
    meanCenter += allvolumecenters[i];
  }
  meanCenter /= numberOfDaughters;

  size_t clustersize = (numberOfDaughters + numberOfClusters - 1) / numberOfClusters;
  for (int clusterindex = 0; clusterindex < numberOfClusters && !daughterSet.empty(); ++clusterindex) {
    Vector3D<Precision> clusterMean(0);

    while (clusters[clusterindex].size() < clustersize) {
      int addDaughter =
          clusters[clusterindex].size() == 0
              ? *std::max_element(daughterSet.begin(), daughterSet.end(),
                                  [&](int a, int b) {
                                    return (allvolumecenters[a] - meanCenter).Length() <
                                           (allvolumecenters[b] - meanCenter).Length();
                                  })
              : *std::min_element(daughterSet.begin(), daughterSet.end(), [&](int a, int b) {
                  return (allvolumecenters[a] - clusterMean).Length() < (allvolumecenters[b] - clusterMean).Length();
                });

      daughterSet.erase(addDaughter);
      clusters[clusterindex].push_back(addDaughter);

      if (daughterSet.empty()) break;
      meanCenter  = (meanCenter * (daughterSet.size() + 1) - allvolumecenters[addDaughter]) / daughterSet.size();
      clusterMean = (clusterMean * (clusters[clusterindex].size() - 1) + allvolumecenters[addDaughter]) /
                    clusters[clusterindex].size();
    }
  }
}

/*
template<typename Container_t>
void HybridManager2::InitClustersWithKMeans(LogicalVolume const * lvol, Container_t & clusters, SOA3D<Precision> &
centers, SOA3D<Precision> & allvolumecenters, int const numberOfIterations) {
    int numberOfClusters  = clusters.size();
    int numberOfDaughters = lvol->GetDaughters().size();

    int size;
    ABBoxManager::ABBoxContainer_t boxes = ABBoxManager::Instance().GetABBoxes(lvol, size);
    for (int i = 0; i < numberOfDaughters; ++i) {
        Vector3D<Precision> center = 0.5 * (boxes[2 * i] + boxes[2 * i + 1]);
        allvolumecenters.set(i, center);
    }

        // inits cluster centers
    std::set<int> startpoints;
    int index = 0;
    while (startpoints.size() < numberOfClusters) {
        int randomDaughterIndex = std::rand() % numberOfDaughters;
        if (startpoints.find(randomDaughterIndex) == startpoints.end()) {
            startpoints.insert(randomDaughterIndex);
            centers.set(index, allvolumecenters[randomDaughterIndex]);
            index++;
        }

    }

    //volumeUtilities::FillUncontainedPoints(*lvol, centers);
    for (int i = 0 ; i < numberOfIterations; ++i) {
        for (int c = 0 ; c < numberOfClusters; ++c) {
            clusters[c].clear();
        }
        AssignVolumesToClusters(clusters, centers, allvolumecenters);
        RecalculateCentres(centers, allvolumecenters, clusters);
    }
}*/

template <typename Container_t>
void HybridManager2::EqualizeClusters(Container_t &clusters, SOA3D<Precision> &centers,
                                      SOA3D<Precision> const &allvolumecenters, size_t const maxNodeSize)
{
  // clusters need to be sorted
  size_t numberOfClusters = clusters.size();
  sort(clusters, IsBiggerCluster);
  for (size_t c = 0; c < numberOfClusters; ++c) {
    size_t clustersize = clusters[c].size();
    if (clustersize > maxNodeSize) {
      RecalculateCentres(centers, allvolumecenters, clusters);
      distanceQueue clusterelemToCenterMap; // pairs of index of elem in cluster and distance to cluster center
      for (size_t clusterElem = 0; clusterElem < clustersize; ++clusterElem) {
        Precision distance2 = (centers[c] - allvolumecenters[clusters[c][clusterElem]]).Length2();
        clusterelemToCenterMap.push(std::make_pair(clusters[c][clusterElem], distance2));
      }

      while (clusters[c].size() > maxNodeSize) {
        // int daughterIndex = clusters[c][it];
        int daughterIndex = clusterelemToCenterMap.top().first;
        clusterelemToCenterMap.pop();
        distanceQueue clusterToCenterMap2;
        for (size_t nextclusterIndex = c + 1; nextclusterIndex < numberOfClusters; nextclusterIndex++) {
          if (clusters[nextclusterIndex].size() < maxNodeSize) {
            Precision distanceToOtherCenters = (centers[nextclusterIndex] - allvolumecenters[daughterIndex]).Length2();
            clusterToCenterMap2.push(std::make_pair(nextclusterIndex, -distanceToOtherCenters));
          }
        }
        // find nearest cluster to daughterIndex

        clusters[c].erase(std::find(clusters[c].begin(), clusters[c].end(), daughterIndex));
        clusters[clusterToCenterMap2.top().first].push_back(daughterIndex);
        // RecalculateCentres(centers, allvolumecenters, clusters);
      }

      std::sort(clusters.begin() + c + 1, clusters.end(), IsBiggerCluster);
    }
  }
}

VPlacedVolume const *HybridManager2::PrintHybrid(LogicalVolume const *lvol) const
{
  return 0;
}
}
}
