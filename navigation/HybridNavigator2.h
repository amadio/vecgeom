/*
 * HybridNavigator2.h
 *
 *  Created on: 27.08.2015
 *      Author: yang.zhang@cern.ch and sandro.wenzel@cern.ch
 */

#ifndef VECGEOM_HYBRIDNAVIGATOR
#define VECGEOM_HYBRIDNAVIGATOR

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "management/HybridManager2.h"
#include "navigation/VNavigator.h"
#include "navigation/HybridSafetyEstimator.h"
#include "navigation/SimpleABBoxNavigator.h"

#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A navigator using a shallow tree of aligned bounding boxes (hybrid approach) to quickly exclude
// potential hit targets.
// This navigator goes into the direction of "voxel" navigators used in Geant4
// and ROOT. Checking single-rays against a set of aligned bounding boxes can be done
// in a vectorized fashion.
template <bool MotherIsConvex = false>
class HybridNavigator : public VNavigatorHelper<HybridNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  HybridManager2 &fAccelerationManager;
  HybridNavigator()
      : VNavigatorHelper<HybridNavigator<MotherIsConvex>, MotherIsConvex>(SimpleABBoxSafetyEstimator::Instance()),
        fAccelerationManager(HybridManager2::Instance())
  {
  }

  static VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int const daughterIndex)
  {
    return lvol->GetDaughters()[daughterIndex];
  }

  // a simple sort class (based on insertionsort)
  template <typename T> //, typename Cmp>
  static void insertionsort(T *arr, unsigned int N)
  {
    for (unsigned short i = 1; i < N; ++i) {
      T value    = arr[i];
      short hole = i;

      for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
        arr[hole] = arr[hole - 1];

      arr[hole] = value;
    }
  }

  /**
   * Returns hitlist of daughter candidates (pairs of [daughter index, step to bounding box]) crossed by ray.
   */
  size_t GetHitCandidates_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                            Vector3D<Precision> const &point, Vector3D<Precision> const &dir,
                            HybridManager2::BoxIdDistancePair_t *hitlist) const
  {
    size_t count = 0;
    Vector3D<Precision> invdir(1. / dir.x(), 1. / dir.y(), 1. / dir.z());
    Vector3D<int> sign;
    sign[0] = invdir.x() < 0;
    sign[1] = invdir.y() < 0;
    sign[2] = invdir.z() < 0;
    int numberOfNodes, size;
    auto boxes_v                = fAccelerationManager.GetABBoxes_v(accstructure, size, numberOfNodes);
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      HybridManager2::Float_v distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
          &boxes_v[index], point, invdir, sign.x(), sign.y(), sign.z(), 0, InfinityLength<float>());
      auto hit = distance < InfinityLength<float>();
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0 /*hit.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
                &boxes_v[index + 2 * (i + 1)], point, invdir, sign.x(), sign.y(), sign.z(), 0, InfinityLength<float>());
            auto hit1 = distance < InfinityLength<float>();
            if (!vecCore::MaskEmpty(hit1)) {
              for (size_t j = 0 /*hit1.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(hit1, j)) {
                  assert(count < VECGEOM_MAXDAUGHTERS);
                  hitlist[count] = HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j],
                                                                       vecCore::LaneAt(distance, j));
                  count++;
                }
              }
            }
          }
        }
      }
    }
    return count;
  }

  size_t GetSafetyCandidates_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                               Vector3D<Precision> const &point, HybridManager2::BoxIdDistancePair_t *hitlist,
                               Precision upper_squared_limit) const
  {
    using Float_v = vecgeom::VectorBackend::Float_v;
    size_t count  = 0;
    int numberOfNodes, size;
    auto boxes_v                = fAccelerationManager.GetABBoxes_v(accstructure, size, numberOfNodes);
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;

    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());

    // Loop internal nodes (internal node = kVS clusters)
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      // index = start index for internal node
      // nodeindex = cluster index
      Float_v distmaxsqr; // Maximum distance that may still touch the box
      Float_v safetytonodesqr =
          ABBoxImplementation::ABBoxSafetyRangeSqr(boxes_v[index], boxes_v[index + 1], pointfloat, distmaxsqr);
      // Find minimum of distmaxsqr
      for (size_t i = 0; i < kVS; ++i) {
        Precision distmaxsqr_s                                      = vecCore::LaneAt(distmaxsqr, i);
        if (distmaxsqr_s < upper_squared_limit) upper_squared_limit = distmaxsqr_s;
      }
      auto hit = safetytonodesqr < ABBoxManager::Real_t(upper_squared_limit);
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            Float_v safetytoboxsqr = ABBoxImplementation::ABBoxSafetyRangeSqr(
                boxes_v[index + 2 * (i + 1)], boxes_v[index + 2 * (i + 1) + 1], pointfloat, distmaxsqr);

            auto hit1 = safetytoboxsqr < ABBoxManager::Real_t(upper_squared_limit);
            if (!vecCore::MaskEmpty(hit1)) {
              // loop bounding boxes in the cluster
              for (size_t j = 0; j < kVS; ++j) {
                if (vecCore::MaskLaneAt(hit1, j)) {
                  assert(count < VECGEOM_MAXDAUGHTERS);
                  hitlist[count] = HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j],
                                                                       vecCore::LaneAt(safetytoboxsqr, j));
                  Precision distmaxsqr_s = vecCore::LaneAt(distmaxsqr, j);
                  // Reduce the upper limit
                  if (distmaxsqr_s < upper_squared_limit) upper_squared_limit = distmaxsqr_s;
                  count++;
                }
              }
            }
          }
        }
      }
    }
    return count;
  }

public:
  // we provide hit detection on the local level and reuse the generic implementations from
  // VNavigatorHelper<SimpleABBoxNavigator>

  // a generic looper function that
  // given an acceleration structure (an aligned bounding box hierarchy),
  // a hit-query will be performed, the intersected boxes sorted, looped over
  // and a user hook called for processing
  // the user hook needs to indicate with a boolean return value whether to continue looping (false)
  // or whether we are done (true) and can exit

  // FIXME: might be generic enough to work for all possible kinds of BVH structures
  // FIXME: offer various sorting directions, etc.
  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE
  void BVHSortedIntersectionsLooper(AccStructure const &accstructure, Vector3D<Precision> const &localpoint,
                                    Vector3D<Precision> const &localdir, Func &&userhook) const
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    auto ncandidates = GetHitCandidates_v(accstructure, localpoint, localdir, hitlist);
    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      // here we got the hit candidates
      // now we execute user specific code to process this "hitbox"
      auto done = userhook(hitbox);
      if (done) break;
    }
  }

  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE
  void BVHSortedSafetyLooper(AccStructure const &accstructure, Vector3D<Precision> const &localpoint, Func &&userhook,
                             Precision upper_squared_limit) const
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    // Get candidates using HybridSafetyEstimator
    //    HybridSafetyEstimator *hse = static_cast<HybridSafetyEstimator*>(HybridSafetyEstimator::Instance());
    //    auto ncandidates = hse->GetSafetyCandidates_v(accstructure, localpoint, hitlist, upper_squared_limit);

    // Get candidates in vectorized mode
    auto ncandidates = GetSafetyCandidates_v(accstructure, localpoint, hitlist, upper_squared_limit);

    // sort candidates according to their bounding volume safety distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      // here we got the hit candidates
      // now we execute user specific code to process this "hitbox"
      auto done = userhook(hitbox);
      if (done) break;
    }
  }

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    if (lvol->GetDaughtersp()->size() == 0) return false;
    auto &accstructure = *fAccelerationManager.GetAccStructure(lvol);

    BVHSortedIntersectionsLooper(accstructure, localpoint, localdir, [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // only consider those hitboxes which are within potential reach of this step
      if (!(step < hitbox.second)) {
        VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);
        Precision ddistance            = candidate->DistanceToIn(localpoint, localdir, step);
        const auto valid               = !IsInf(ddistance) && ddistance < step &&
                           !((ddistance <= 0.) && in_state && in_state->GetLastExited() == candidate);
        hitcandidate = valid ? candidate : hitcandidate;
        step         = valid ? ddistance : step;
        return false; // not yet done; need to continue in looper
      }
      return true; // mark done in this case
    });
    return false;
  }

  static VNavigator *Instance()
  {
    static HybridNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "HybridNavigator";
  typedef SimpleABBoxSafetyEstimator SafetyEstimator_t;
};
}
} // End global namespace

#endif
