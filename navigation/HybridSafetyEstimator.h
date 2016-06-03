/*
 *  HybridSafetyEstimator.h
 *
 *  Created on: 22.11.2015
 *      Author: sandro.wenzel@cern.ch
 *
 *  (based on prototype implementation by Yang Zhang (Sep 2015)
 */

#ifndef NAVIGATION_HYBRIDSAFETYESTIMATOR_H_
#define NAVIGATION_HYBRIDSAFETYESTIMATOR_H_

#include "navigation/VSafetyEstimator.h"
#include "management/HybridManager2.h"

#include <exception>
#include <stdexcept>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a safety estimator using a (vectorized) search through bounding boxes to exclude certain daughter volumes
//! to talk to
class HybridSafetyEstimator : public VSafetyEstimatorHelper<HybridSafetyEstimator> {

private:
  // we keep a reference to the ABBoxManager ( avoids calling Instance() on this guy all the time )
  HybridManager2 &fAccelerationStructureManager;

  HybridSafetyEstimator() : VSafetyEstimatorHelper<HybridSafetyEstimator>(), fAccelerationStructureManager(HybridManager2::Instance()) {}

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const {
    assert(id >= 0 && "access with negative index");
    assert(size_t(id) < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator[](id);
  }

  // helper structure to find the candidate set for safety calculations
  size_t GetSafetyCandidates_v(LogicalVolume const *lvol, Vector3D<Precision> const &point,
                               HybridManager2::BoxIdDistancePair_t *boxsafetypairs, Precision upper_squared_limit) const {
    size_t count = 0;
#ifdef VECGEOM_VC
    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());
    int halfvectorsize, numberOfNodes;
    auto boxes_v = fAccelerationStructureManager.GetABBoxes_v(lvol, halfvectorsize, numberOfNodes);
    std::vector<int> *nodeToDaughters = fAccelerationStructureManager.GetNodeToDaughters(lvol);
    size_t simdsize = kVcFloat::precision_v::Size;

    for (int index = 0, nodeindex = 0; index < halfvectorsize * 2; index += 2 * (simdsize + 1), nodeindex += simdsize) {
      HybridManager2::Real_v safetytoboxsqr = ABBoxImplementation::ABBoxSafetySqr<kVcFloat, HybridManager2::Real_t>(
          boxes_v[index], boxes_v[index + 1], pointfloat);
      HybridManager2::Bool_v closer = safetytoboxsqr < HybridManager2::Real_v(upper_squared_limit);
      if (Any(closer)) {
        for (size_t i = closer.firstOne(); i < simdsize; ++i) {
          if (closer[i]) {
            safetytoboxsqr = ABBoxImplementation::ABBoxSafetySqr<kVcFloat, HybridManager2::Real_t>(
                boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3], pointfloat);
            HybridManager2::Bool_v closer = safetytoboxsqr < HybridManager2::Real_v(upper_squared_limit);
            if (Any(closer)) {
              for (size_t j = closer.firstOne(); j < simdsize; ++j) { // leaf node
                if (closer[j]) {
                  boxsafetypairs[count]=HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j], safetytoboxsqr[j]);
                  count++;
                }
              }
            }
          }
        }
      }
    }
    return count;
#else
	throw std::runtime_error("unimplemented function called: HybridSafetyEstimator::GetSafetyCandidates_v()");
#endif
  }

public:
  static constexpr const char *gClassNameString = "HybridSafetyEstimator";

#ifdef VECGEOM_BACKEND_PRECISION_NOT_SCALAR
  VECGEOM_INLINE
   virtual VECGEOM_BACKEND_PRECISION_TYPE ComputeSafetyForLocalPoint(Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &localpoint,
                                                VPlacedVolume const *pvol, VECGEOM_BACKEND_PRECISION_TYPE::Mask m) const override {
     VECGEOM_BACKEND_PRECISION_TYPE safety(0.);
     if (Any(m)) {
       // SIMD safety to mother
       auto safety = pvol->SafetyToOut(localpoint);

       // now loop over the voxelized treatment of safety to in
       for (unsigned int i = 0; i < VECGEOM_BACKEND_PRECISION_TYPE::Size; ++i) {
         if (m[i]) {
           safety[i] = TreatSafetyToIn(Vector3D<Precision>(localpoint.x()[i], localpoint.y()[i], localpoint.z()[i]),
                                       pvol, safety[i]);
         } else {
           safety[i] = 0;
         }
       }
     }
     return safety;
  }
#endif

  VECGEOM_INLINE
  Precision TreatSafetyToIn(Vector3D<Precision> const &localpoint, VPlacedVolume const *pvol,
                            Precision outsafety) const {
    // a stack based workspace array
    static __thread HybridManager2::BoxIdDistancePair_t boxsafetylist[VECGEOM_MAXDAUGHTERS] = {};

    double safety = outsafety; // we use the outsafety estimate as starting point
    double safetysqr = safety * safety;

    // safety to bounding boxes
    LogicalVolume const *lvol = pvol->GetLogicalVolume();
    if (safety > 0. && lvol->GetDaughtersp()->size() > 0) {
      // calculate squared bounding box safeties in vectorized way
      auto ncandidates = GetSafetyCandidates_v(lvol, localpoint, boxsafetylist, safetysqr);
      // not sorting the candidate list ( which one could do )
      for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
        auto boxsafetypair = boxsafetylist[candidate];
        if (boxsafetypair.second < safetysqr) {
          VPlacedVolume const *candidate = LookupDaughter(lvol, boxsafetypair.first);
          if (size_t(boxsafetypair.first) > lvol->GetDaughtersp()->size())
            break;
          auto candidatesafety = candidate->SafetyToIn(localpoint);
#ifdef VERBOSE
          if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
            std::cerr << "real safety smaller than boxsafety \n";
#endif
          if (candidatesafety < safety) {
            safety = candidatesafety;
            safetysqr = safety * safety;
          }
        }
      }
    }
    return safety;
  }

  // this is (almost) the same code as in SimpleABBoxSafetyEstimator --> avoid this
  VECGEOM_INLINE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override {
    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);
    return TreatSafetyToIn(localpoint,pvol,safety);
  }

  // vector interface
  VECGEOM_INLINE
    virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const & /*localpoints*/, VPlacedVolume const * /*pvol*/,
                                             Precision * /*safeties*/) const override {
//    // a stack based workspace array
//    static __thread ABBoxManager::BoxIdDistancePair_t boxsafetylist[VECGEOM_MAXDAUGHTERS] = {};
//
//    // safety to mother -- using vector interface
//    pvol->SafetyToOut(localpoints, safeties);
//
//    // safety to bounding boxes
//    LogicalVolume const *lvol = pvol->GetLogicalVolume();
//    if (!(lvol->GetDaughtersp()->size() > 0))
//      return;
//
//    // get bounding boxes (they are the same for all tracks)
//    int numberofboxes;
//    auto bboxes = fABBoxManager.GetABBoxes_v(lvol, numberofboxes);
//
//    // now loop over particles
//    for (int i = 0, ntracks = localpoints.size(); i < ntracks; ++i) {
//      double safety = safeties[i];
//      if (safeties[i] > 0.) {
//        double safetysqr = safeties[i] * safeties[i];
//        auto lpoint = localpoints[i];
//        // vectorized search through bounding boxes -- quickly excluding many candidates
//        auto ncandidates = GetSafetyCandidates_v(lpoint, bboxes, numberofboxes, boxsafetylist, safetysqr);
//        // loop over remaining candidates
//        for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
//          auto boxsafetypair = boxsafetylist[candidate];
//          if (boxsafetypair.second < safetysqr) {
//            VPlacedVolume const *candidate = LookupDaughter(lvol, boxsafetypair.first);
//            if (boxsafetypair.first > lvol->GetDaughtersp()->size())
//              break;
//            auto candidatesafety = candidate->SafetyToIn(lpoint);
//#ifdef VERBOSE
//            if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
//              std::cerr << "real safety smaller than boxsafety \n";
//#endif
//            if (candidatesafety < safety) {
//              safety = candidatesafety;
//              safetysqr = safety * safety;
//            }
//          }
//        }
//      }
//      // write back result
//      safeties[i] = safety;
//    }
  }

  static VSafetyEstimator *Instance() {
    static HybridSafetyEstimator instance;
    return &instance;
  }

}; // end class

}} // end namespace


#endif /* NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_ */
