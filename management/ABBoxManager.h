/*
 * ABBoxManager.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#pragma once

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"

#ifdef VECGEOM_VC
#include "backend/vc/Backend.h"
#endif

#include <map>
#include <vector>
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// Singleton class for ABBox manager
// keeps a (centralized) map of volume pointers to vectors of aligned bounding boxes
// the alternative would be to include such a thing into logical volumes
class ABBoxManager {
public:
  // scalar or vector vectors
  typedef Vector3D<Precision> ABBox_t;
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

  // use old style arrays here as std::vector has some problems
  // with Vector3D<kVc::Double_t>
  typedef ABBox_t *ABBoxContainer_t;
  typedef ABBox_v *ABBoxContainer_v;

  typedef std::pair<int, double> BoxIdDistancePair_t;
  //typedef std::vector<BoxIdDistancePair_t> HitContainer_t;

  // build an abstraction of sort to sort vectors and lists portably
  template <typename C, typename Compare> static void sort(C &v, Compare cmp) { std::sort(v.begin(), v.end(), cmp); }

  struct HitBoxComparatorFunctor {
    bool operator()(BoxIdDistancePair_t const &left, BoxIdDistancePair_t const &right) {
      return left.second < right.second;
    }
  };

  using FP_t = HitBoxComparatorFunctor;

private:
  std::vector<ABBoxContainer_t> fVolToABBoxesMap;
  std::vector<ABBoxContainer_v> fVolToABBoxesMap_v;

  // we have to make this thread safe
  //HitContainer_t fAllocatedHitContainer;


public:

  // computes the aligned bounding box for a certain placed volume
  static void ComputeABBox(VPlacedVolume const *pvol, ABBox_t *lower, ABBox_t *upper);
  static void ComputeSplittedABBox(VPlacedVolume const *pvol, std::vector<ABBox_t> &lower, std::vector<ABBox_t> &upper, int numOfSlices);

  static ABBoxManager &Instance() {
    static ABBoxManager instance;
    return instance;
  }

  // initialize ABBoxes for a certain logical volume
  // very first version that just creates as many boxes as there are daughters
  // in reality we might have a lot more boxes than daughters (but not less)
  void InitABBoxes(LogicalVolume const *lvol);

  // doing the same for many logical volumes
  template <typename Container> void InitABBoxes(Container const &lvolumes) {
    for (auto lvol : lvolumes) {
      InitABBoxes(lvol);
    }
  }

  void InitABBoxesForCompleteGeometry() {
    std::vector<LogicalVolume const *> logicalvolumes;
    GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);
    // size containers
    fVolToABBoxesMap.resize( GeoManager::Instance().GetRegisteredVolumesCount(), nullptr );
    fVolToABBoxesMap_v.resize( GeoManager::Instance().GetRegisteredVolumesCount(), nullptr );
    InitABBoxes(logicalvolumes);
  }

  // remove the boxes from the list
  void RemoveABBoxes(LogicalVolume const *lvol);

  // returns the Container for a given logical volume or nullptr if
  // it does not exist
  ABBoxContainer_t GetABBoxes(LogicalVolume const *lvol, int &size) {
    size = lvol->GetDaughtersp()->size();
    return fVolToABBoxesMap[lvol->id()];
  }

  // returns the Container for a given logical volume or nullptr if
  // it does not exist
  ABBoxContainer_v GetABBoxes_v(LogicalVolume const *lvol, int &size) {
    int ndaughters = lvol->GetDaughtersp()->size();
    int extra = (ndaughters % Real_vSize > 0) ? 1 : 0;
    size = ndaughters / Real_vSize + extra;
    return fVolToABBoxesMap_v[lvol->id()];
  }

  //HitContainer_t &GetAllocatedHitContainer() { return fAllocatedHitContainer; }
};

// output for hitboxes
template <typename stream> stream &operator<<(stream &s, std::vector<ABBoxManager::BoxIdDistancePair_t> const &list) {
  for (auto i : list) {
    s << "(" << i.first << "," << i.second << ")"
      << " ";
  }
  return s;
}

}} // end namespace

