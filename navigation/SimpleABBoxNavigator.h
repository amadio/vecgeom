/*
 * SimpleABBoxNavigator.h
 *
 *  Created on: Nov 23, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXNAVIGATOR_H_
#define NAVIGATION_SIMPLEABBOXNAVIGATOR_H_

#include "VNavigator.h"
#include "SimpleABBoxSafetyEstimator.h"
#include "management/ABBoxManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A basic implementation of a navigator which is SIMD accelerated by using a flat aligned bounding box list
template <bool MotherIsConvex=false>
class SimpleABBoxNavigator : public VNavigatorHelper<SimpleABBoxNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  ABBoxManager &fABBoxManager;
  SimpleABBoxNavigator() : VNavigatorHelper<SimpleABBoxNavigator<MotherIsConvex>, MotherIsConvex>(SimpleABBoxSafetyEstimator::Instance()), fABBoxManager(ABBoxManager::Instance()) {}

  // convert index to physical daugher
  VPlacedVolume const * LookupDaughter( LogicalVolume const *lvol, int id ) const {
    assert( id >= 0 && "access with negative index");
    assert( size_t(id) < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator []( id );
  }

  // a simple sort class (based on insertionsort)
  //template <typename T, typename Cmp>
  static void insertionsort(ABBoxManager::BoxIdDistancePair_t *arr, unsigned int N) {
    for (unsigned short i = 1; i < N; ++i) {
      ABBoxManager::BoxIdDistancePair_t value = arr[i];
      short hole = i;

      for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
        arr[hole] = arr[hole - 1];

      arr[hole] = value;
    }
  }

  // vector version
size_t GetHitCandidates_v(LogicalVolume const * /*lvol*/, Vector3D<Precision> const &point,
                          Vector3D<Precision> const &dir, ABBoxManager::ABBoxContainer_v const &corners,
                          size_t size, ABBoxManager::BoxIdDistancePair_t *hitlist) const {
    size_t vecsize = size;
    size_t hitcount = 0;
    Vector3D<float> invdirfloat(1.f / (float)dir.x(), 1.f / (float)dir.y(), 1.f / (float)dir.z());
    Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());
    int sign[3];
    sign[0] = invdirfloat.x() < 0;
    sign[1] = invdirfloat.y() < 0;
    sign[2] = invdirfloat.z() < 0;
    using Float_v = ABBoxManager::Float_v;
    using Bool_v  = vecCore::Mask_v<Float_v>;
    for (size_t box = 0; box < vecsize; ++box) {
      Float_v distance = BoxImplementation::IntersectCachedKernel2<Float_v, float>(
          &corners[2 * box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0, static_cast<float>(vecgeom::kInfinity));
      Bool_v hit = distance < static_cast<float>(vecgeom::kInfinity);
      if (!vecCore::MaskEmpty(hit)) {
        constexpr auto kVS = vecCore::VectorSize<Float_v>();

        // VecCore does not have firstOne() function; iterating from zero
        // consider putting a firstOne into vecCore or in VecGeom
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)){
            hitlist[hitcount]=(ABBoxManager::BoxIdDistancePair_t(box * kVS + i, vecCore::LaneAt(distance, i)));
            hitcount++;
          }
        }
      }
    }
    return hitcount;
  }

public:
  // we provide hit detection on the local level and reuse the generic implementations from
  // VNavigatorHelper<SimpleABBoxNavigator>

  VECGEOM_INLINE
  virtual Precision ComputeStepAndHittingBoundaryForLocalPoint(Vector3D<Precision> const &localpoint,
                                                               Vector3D<Precision> const &localdir,
                                                               Precision step_limit, NavigationState const &in_state,
                                                               NavigationState &out_state) const override {
    static __thread ABBoxManager::BoxIdDistancePair_t hitlist[VECGEOM_MAXDAUGHTERS]={};

    auto currentvolume = in_state.Top();
    auto lvol = currentvolume->GetLogicalVolume();
#if defined(VERBOSE) && !defined(VECGEOM_NVCC)
    __thread static size_t counter=0;
    if (counter % 1 == 0) {
      std::cerr << GetClassName() << " navigating in " << currentvolume->GetLabel() << " stepnumber " << counter << "step_limit" << step_limit
                << "localpoint " << localpoint << "localdir" << globaldir;
    }
    counter++;
#endif

    Precision step;
    assert(currentvolume != nullptr && "currentvolume is null in navigation");
    step = currentvolume->DistanceToOut(localpoint, localdir, step_limit);
    decltype(currentvolume) nexthitvolume=nullptr; // nullptr means mother

    // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. SET TO INFINITY SO THAT WE CAN BETTER HANDLE IT
    // LATER ON
    if (step < 0.) {
      step = kInfinity;
    }

    int size;
    ABBoxManager::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
    auto ncandidates = GetHitCandidates_v(lvol, localpoint, localdir, bboxes, size, hitlist);

    // sort candidates according to their bounding volume hit distance
    insertionsort( hitlist, ncandidates );

    for(size_t index=0;index < ncandidates;++index)
    {
        auto hitbox = hitlist[index];
        VPlacedVolume const * candidate = LookupDaughter( lvol, hitbox.first );

        // only consider those hitboxes which are within potential reach of this step
        if( ! (step < hitbox.second )) {
        //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
         if( hitbox.second < 0 ){
            bool checkindaughter = candidate->Contains( localpoint );
            if( checkindaughter == true ){
                // need to relocate
                step = 0;
                nexthitvolume = candidate;
                // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
                break;
            }
        }
        Precision ddistance = candidate->DistanceToIn( localpoint, localdir, step );
#ifdef VERBOSE
        std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
#endif
        nexthitvolume = (ddistance < step) ? candidate : nexthitvolume;
        step      = (ddistance < step) ? ddistance  : step;
        }
        else {
          break;
        }
    }
    bool done;
    return VNavigator::PrepareOutState(in_state, out_state, step, step_limit, nexthitvolume, done);
  }

  VECGEOM_INLINE
    virtual bool
    CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const & localpoint, Vector3D<Precision> const & localdir,
                               NavigationState const & /*in_state*/, NavigationState & /*out_state*/, Precision & step, VPlacedVolume const *&hitcandidate) const override {


      static __thread ABBoxManager::BoxIdDistancePair_t hitlist[VECGEOM_MAXDAUGHTERS]={};
      if(lvol->GetDaughtersp()->size() == 0) return false;

      int size;
      ABBoxManager::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
      auto ncandidates = GetHitCandidates_v(lvol, localpoint, localdir, bboxes, size, hitlist);

      // sort candidates according to their bounding volume hit distance
      insertionsort( hitlist, ncandidates );

      for(size_t index=0;index < ncandidates;++index)
      {
          auto hitbox = hitlist[index];
          VPlacedVolume const * candidate = LookupDaughter( lvol, hitbox.first );

          // only consider those hitboxes which are within potential reach of this step
          if( ! (step < hitbox.second )) {
          //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
//           if( hitbox.second < 0 ){
//            //   std::cerr << "funny2\n";
//              bool checkindaughter = candidate->Contains( localpoint );
//              if( checkindaughter == true ){
//                  std::cerr << "funny\n";
//
//                  // need to relocate
//                  step = 0;
//                  hitcandidate = candidate;
//                  // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
//                  break;
//              }
//          }
          Precision ddistance = candidate->DistanceToIn( localpoint, localdir, step );
  #ifdef VERBOSE
          std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
  #endif
          auto valid = !IsInf(ddistance) && ddistance < step;
          hitcandidate = valid ? candidate : hitcandidate;
          step      = valid ? ddistance  : step;
          }
          else {
            break;
          }
      }
      return false;
    }



  static VNavigator *Instance() {
    static SimpleABBoxNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "SimpleABBoxNavigator";
  typedef SimpleABBoxSafetyEstimator SafetyEstimator_t;
}; // end of class


}} // end namespace

#endif /* NAVIGATION_SIMPLEABBOXNAVIGATOR_H_ */
