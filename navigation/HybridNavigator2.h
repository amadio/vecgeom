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

  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int const daughterIndex) const
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
  size_t GetHitCandidates_v(LogicalVolume const *lvol, Vector3D<Precision> const &point, Vector3D<Precision> const &dir,
                            HybridManager2::BoxIdDistancePair_t *hitlist) const
  {
    size_t count = 0;
    Vector3D<Precision> invdir(1. / dir.x(), 1. / dir.y(), 1. / dir.z());
    Vector3D<int> sign;
    sign[0] = invdir.x() < 0;
    sign[1] = invdir.y() < 0;
    sign[2] = invdir.z() < 0;
    int numberOfNodes, size;
    auto boxes_v                      = fAccelerationManager.GetABBoxes_v(lvol, size, numberOfNodes);
    constexpr auto kVS                = vecCore::VectorSize<HybridManager2::Float_v>();
    std::vector<int> *nodeToDaughters = fAccelerationManager.GetNodeToDaughters(lvol);
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      HybridManager2::Float_v distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
          &boxes_v[index], point, invdir, sign.x(), sign.y(), sign.z(), 0, static_cast<float>(vecgeom::kInfinity));
      auto hit = distance < static_cast<float>(vecgeom::kInfinity);
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0 /*hit.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
                &boxes_v[index + 2 * (i + 1)], point, invdir, sign.x(), sign.y(), sign.z(), 0,
                static_cast<float>(vecgeom::kInfinity));
            auto hit = distance < static_cast<float>(vecgeom::kInfinity);
            if (!vecCore::MaskEmpty(hit)) {
              for (size_t j = 0 /*hit.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(hit, j)) {
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

public:
  // we provide hit detection on the local level and reuse the generic implementations from
  // VNavigatorHelper<SimpleABBoxNavigator>

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    // static __thread HybridManager2::BoxIdDistancePair_t hitlist[VECGEOM_MAXDAUGHTERS] = {};
    HybridManager2::BoxIdDistancePair_t hitlist[VECGEOM_MAXDAUGHTERS];
    if (lvol->GetDaughtersp()->size() == 0) return false;

    auto ncandidates = GetHitCandidates_v(lvol, localpoint, localdir, hitlist);

    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox                    = hitlist[index];
      VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);

      // only consider those hitboxes which are within potential reach of this step
      if (!(step < hitbox.second)) {
        //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
        //        if (hitbox.second < 0) {
        //          bool checkindaughter = candidate->Contains(localpoint);
        //          if (checkindaughter == true) {
        //            // need to relocate
        //            step = 0;
        //            hitcandidate = candidate;
        //            // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
        //            break;
        //          }
        //        }
        Precision ddistance = candidate->DistanceToIn(localpoint, localdir, step);
#ifdef VERBOSE
        std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
#endif
        const auto valid = !IsInf(ddistance) && ddistance < step &&
                           !((ddistance <= 0.) && in_state && in_state->GetLastExited() == candidate);
        hitcandidate = valid ? candidate : hitcandidate;
        step         = valid ? ddistance : step;
        //        hitcandidate = (ddistance < step) ? candidate : hitcandidate;
        //        step = (ddistance < step) ? ddistance : step;
      } else {
        break;
      }
    }
    return false; // no assembly seen
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
