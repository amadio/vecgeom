/*
 * SimpleABBoxSafetyEstimator.h
 *
 *  Created on: Aug 28, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_
#define NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_

#include "VecGeom/navigation/VSafetyEstimator.h"
#include "VecGeom/management/ABBoxManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a safety estimator using a (vectorized) search through bounding boxes to exclude certain daughter volumes
//! to talk to
class SimpleABBoxSafetyEstimator : public VSafetyEstimatorHelper<SimpleABBoxSafetyEstimator> {

private:
  // we keep a reference to the ABBoxManager ( avoids calling Instance() on this guy all the time )
  ABBoxManager &fABBoxManager;

  SimpleABBoxSafetyEstimator()
      : VSafetyEstimatorHelper<SimpleABBoxSafetyEstimator>(), fABBoxManager(ABBoxManager::Instance())
  {
  }

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const
  {
    assert(id >= 0 && "access with negative index");
    assert(size_t(id) < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator[](id);
  }

public:
  // helper function calculating some candidate volumes
  VECCORE_ATT_HOST_DEVICE
  static size_t GetSafetyCandidates_v(Vector3D<Precision> const &point, ABBoxManager::ABBoxContainer_v const &corners,
                                      size_t size, ABBoxManager::BoxIdDistancePair_t *boxsafetypairs,
                                      Precision upper_squared_limit)
  {
    size_t count = 0;
    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());
    size_t vecsize = size;
    for (size_t box = 0; box < vecsize; ++box) {
      ABBoxManager::Float_v safetytoboxsqr =
          ABBoxImplementation::ABBoxSafetySqr(corners[2 * box], corners[2 * box + 1], pointfloat);

      auto hit           = safetytoboxsqr < ABBoxManager::Real_t(upper_squared_limit);
      constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            boxsafetypairs[count] =
                ABBoxManager::BoxIdDistancePair_t(box * kVS + i, vecCore::LaneAt(safetytoboxsqr, i));
            count++;
          }
        }
      }
    }
    return count;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision TreatSafetyToIn(Vector3D<Precision> const &localpoint, LogicalVolume const *lvol, Precision outsafety) const
  {
    // a stack based workspace array
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = ABBoxManager::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    double safety    = outsafety; // we use the outsafety estimate as starting point
    double safetysqr = safety * safety;

    // safety to bounding boxes
    if (safety > 0. && lvol->GetDaughtersp()->size() > 0) {
      int size;

      ABBoxManager::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
      // calculate squared bounding box safeties in vectorized way
      auto ncandidates = GetSafetyCandidates_v(localpoint, bboxes, size, boxsafetylist, safetysqr);
      // not sorting the candidate list ( which one could do )
      for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
        auto boxsafetypair = boxsafetylist[candidate];
        if (boxsafetypair.second < safetysqr) {
          VPlacedVolume const *cand = LookupDaughter(lvol, boxsafetypair.first);
          if (boxsafetypair.first > lvol->GetDaughtersp()->size()) break;
          auto candidatesafety = cand->SafetyToIn(localpoint);
#ifdef VERBOSE
          if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
            std::cerr << "real safety smaller than boxsafety \n";
#endif
          if (candidatesafety < safety) {
            safety    = candidatesafety;
            safetysqr = safety * safety;
          }
        }
      }
    }
    return safety;
  }

public:
  static constexpr const char *gClassNameString = "SimpleABBoxSafetyEstimator";

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override
  {

    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);
    return TreatSafetyToIn(localpoint, pvol->GetLogicalVolume(), safety);
  }

  // estimate just the safety to daughters for a local point with respect to a logical volume
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                          LogicalVolume const *lvol) const override
  {
    return TreatSafetyToIn(localpoint, lvol, kInfLength);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Real_v ComputeSafetyForLocalPoint(Vector3D<Real_v> const &localpoint, VPlacedVolume const *pvol,
                                            Bool_v m) const override
  {
    using vecCore::AssignLane;
    using vecCore::LaneAt;
    Real_v safety(0.);
    if (!vecCore::MaskEmpty(m)) {
      // SIMD safety to mother
      safety                    = pvol->SafetyToOut(localpoint);
      LogicalVolume const *lvol = pvol->GetLogicalVolume();
      // now loop over the voxelized treatment of safety to in
      for (unsigned int i = 0; i < vecCore::VectorSize<Real_v>(); ++i) {
        if (vecCore::MaskLaneAt(m, i)) {
          vecCore::AssignLane(safety, i,
                              TreatSafetyToIn(Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i),
                                                                  LaneAt(localpoint.z(), i)),
                                              lvol, vecCore::LaneAt(safety, i)));
        } else {
          vecCore::AssignLane(safety, i, 0.);
        }
      }
    }
    return safety;
  }

  // vector interface
  VECGEOM_FORCE_INLINE
  virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const &localpoints, VPlacedVolume const *pvol,
                                           Precision *safeties) const override
  {
    // a stack based workspace array
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = ABBoxManager::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    // safety to mother -- using vector interface
    pvol->SafetyToOut(localpoints, safeties);

    // safety to bounding boxes
    LogicalVolume const *lvol = pvol->GetLogicalVolume();
    if (!(lvol->GetDaughtersp()->size() > 0)) return;

    // get bounding boxes (they are the same for all tracks)
    int numberofboxes;
    auto bboxes = fABBoxManager.GetABBoxes_v(lvol, numberofboxes);

    // now loop over particles
    for (int i = 0, ntracks = localpoints.size(); i < ntracks; ++i) {
      double safety = safeties[i];
      if (safeties[i] > 0.) {
        double safetysqr = safeties[i] * safeties[i];
        auto lpoint      = localpoints[i];
        // vectorized search through bounding boxes -- quickly excluding many candidates
        auto ncandidates = GetSafetyCandidates_v(lpoint, bboxes, numberofboxes, boxsafetylist, safetysqr);
        // loop over remaining candidates
        for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
          auto boxsafetypair = boxsafetylist[candidate];
          if (boxsafetypair.second < safetysqr) {
            VPlacedVolume const *cand = LookupDaughter(lvol, boxsafetypair.first);
            if (boxsafetypair.first > lvol->GetDaughtersp()->size()) break;
            auto candidatesafety = cand->SafetyToIn(lpoint);
#ifdef VERBOSE
            if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
              std::cerr << "real safety smaller than boxsafety \n";
#endif
            if (candidatesafety < safety) {
              safety    = candidatesafety;
              safetysqr = safety * safety;
            }
          }
        }
      }
      // write back result
      safeties[i] = safety;
    }
  }

  static VSafetyEstimator *Instance()
  {
    static SimpleABBoxSafetyEstimator instance;
    return &instance;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_ */
