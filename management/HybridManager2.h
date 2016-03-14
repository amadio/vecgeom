/*
 * HybridManager2.h
 *
 *  Created on: 27.08.2015 by yang.zhang@cern.ch
 *  integrated into main development line by sandro.wenzel@cern.ch 24.11.2015
 *
 */

#ifndef VECGEOM_HYBRIDMANAGER_H
#define VECGEOM_HYBRIDMANAGER_H

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "base/AlignmentAllocator.h"

#ifdef VECGEOM_VC
#include "backend/vc/Backend.h"
#endif

#include <queue>
#include <map>
#include <vector>
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a singleton class which manages a particular helper structure for voxelized navigation
// the helper structure is a hybrid between a flat list of aligned bounding boxed and a pure boundary volume hierarchy (BVH)
// and is hence called "HybridManager": TODO: come up with a more appropriate name
class HybridManager2 {

public:
#ifdef VECGEOM_VC // just temporary typedef ---> will be removed with new backend structure
  typedef Vc::float_v Real_v;
  typedef Vc::float_m Bool_v;
  constexpr static unsigned int Real_vSize = Real_v::Size;
#else
  typedef float Real_v;
  typedef bool Bool_v;
  constexpr static unsigned int Real_vSize = 1;
#endif

  typedef float Real_t;
  typedef Vector3D<Real_v> ABBox_v;

  // scalar or vector vectors
  typedef Vector3D<Precision> ABBox_t;
  // use old style arrays here as std::vector has some problems
  // with Vector3D<kVc::Double_t>
  typedef ABBox_t *ABBoxContainer_t;
  typedef ABBox_v *ABBoxContainer_v;

  // first index is # daughter index, second is step
  typedef std::pair<int, double> BoxIdDistancePair_t;
  using HitContainer_t = std::vector<BoxIdDistancePair_t>;

  // we have to make this thread safe
  HitContainer_t fAllocatedHitContainer;

  struct HitBoxComparatorFunctor {
    bool operator()(BoxIdDistancePair_t const &a, BoxIdDistancePair_t const &b) { return a.second < b.second; }
  };

private:
  // this might be duplicated with the ABBoxManager??
  std::vector<ABBoxContainer_t> fVolumeToABBoxes;   // at lvol->id() stores continuous bounding box array
  std::vector<ABBoxContainer_v> fVolumeToABBoxes_v; // at lvol->id() stores continuous bounding box array

  std::vector<std::vector<int> *> fNodeToDaughters; // access by internal node index, returns vector of daughterindices
  std::vector<std::vector<int> *> fNodeToDaughters_v; //access by internal node index, returns vector of daughterindices

public:
  // initialized the helper structure for a given logical volume
  void InitStructure(LogicalVolume const *lvol);

  // initialized the helper structure for the complete geometry
  void InitVoxelStructureForCompleteGeometry() {
    std::vector<LogicalVolume const *> logicalvolumes;
    GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);
    // size containers
    fVolumeToABBoxes.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
    fVolumeToABBoxes_v.resize(GeoManager::Instance().GetRegisteredVolumesCount(), nullptr);
    for (auto lvol : logicalvolumes) {
          InitStructure(lvol);
        }
  }

  static HybridManager2 &Instance() {
    static HybridManager2 manager;
    return manager;
  }

  // removed/deletes the helper structure for a given logical volume
  void RemoveStructure(LogicalVolume const *lvol);

  template <typename C, typename Compare> static void sort(C &v, Compare cmp) { std::sort(v.begin(), v.end(), cmp); }

  // verbose output of helper structure
  VPlacedVolume const *PrintHybrid(LogicalVolume const *) const;

  // returns half the number of nodes in vector, e.g. half the size of fVolumeToABBoxes[lvol->id()]
  ABBoxContainer_t GetABBoxes(LogicalVolume const *lvol, int &numberOfNodes) const {
    int numberOfFirstLevelNodes =
        lvol->GetDaughters().size() / Real_vSize + (lvol->GetDaughters().size() % Real_vSize == 0 ? 0 : 1);
    numberOfNodes = numberOfFirstLevelNodes + lvol->GetDaughters().size();
    assert(fVolumeToABBoxes[lvol->id()]!=nullptr);
    return fVolumeToABBoxes[lvol->id()];
  }

  // returns half the number of vector registers to store all the nodes
  ABBoxContainer_v GetABBoxes_v(LogicalVolume const *lvol, int &size, int &numberOfNodes) const {
    int numberOfFirstLevelNodes =
        lvol->GetDaughters().size() / Real_vSize + (lvol->GetDaughters().size() % Real_vSize == 0 ? 0 : 1);
    numberOfNodes = numberOfFirstLevelNodes + lvol->GetDaughters().size();
    size = numberOfFirstLevelNodes / Real_vSize + (numberOfFirstLevelNodes % Real_vSize == 0 ? 0 : 1) +
           numberOfFirstLevelNodes;
    assert(fVolumeToABBoxes[lvol->id()]!=nullptr);
    return fVolumeToABBoxes_v[lvol->id()];
  }

  std::vector<int> *GetNodeToDaughters(LogicalVolume const *lvol) { return fNodeToDaughters[lvol->id()]; }

private:
  // private methods use

  template <typename Container_t>
  void EqualizeClusters(Container_t &clusters, SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                        size_t const maxNodeSize);
  template <typename Container_t>
  void InitClustersWithKMeans(LogicalVolume const *, Container_t &, SOA3D<Precision> &, SOA3D<Precision> &,
                              int const numberOfInterations = 50);

  void RecalculateCentres(SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                          std::vector<std::vector<int>> const &clusters);
  struct OrderByDistance {
    bool operator()(std::pair<int, Precision> const &a, std::pair<int, Precision> const &b) {
      return a.second < b.second;
    }
  };
  typedef std::priority_queue<std::pair<int, Precision>, std::vector<std::pair<int, Precision>>, OrderByDistance>
      distanceQueue;
  void AssignVolumesToClusters(std::vector<std::vector<int>> &clusters, SOA3D<Precision> const &centers,
                               SOA3D<Precision> const &allvolumecenters);
  void BuildStructure(LogicalVolume const *vol);
  void BuildStructure_v(LogicalVolume const *vol);
  static bool IsBiggerCluster(std::vector<int> const &first, std::vector<int> const &second) {
    return first.size() > second.size();
  }

}; // end class

}} // end namespace

#endif
