/*
 * NewSimpleNavigator.h
 *
 *  Created on: 17.09.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_NEWSIMPLENAVIGATOR_H_
#define NAVIGATION_NEWSIMPLENAVIGATOR_H_

#include "VNavigator.h"
#include "SimpleSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A very basic implementation of a navigator ( brute force which scales linearly with the number of daughters )
template <bool MotherIsConvex=false>
class NewSimpleNavigator : public VNavigatorHelper<class NewSimpleNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  NewSimpleNavigator() {}
  virtual ~NewSimpleNavigator() {}

public:
  VECGEOM_INLINE
  virtual Precision ComputeStepAndHittingBoundaryForLocalPoint(Vector3D<Precision> const &localpoint,
                                                               Vector3D<Precision> const &localdir,
                                                               Precision step_limit, NavigationState const &in_state,
                                                               NavigationState &out_state) const override {

    auto currentvolume = in_state.Top();
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
    // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. SET TO INFINITY SO THAT WE CAN BETTER HANDLE IT
    // LATER ON
    if (step < 0.) {
      step = kInfinity;
    }

    // iterate over all daughters
    auto *daughters = currentvolume->GetLogicalVolume()->GetDaughtersp();
    decltype(currentvolume) nexthitvolume=nullptr;
    auto ndaughters = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter = daughters->operator[](d);
//    previous distance becomes step estimate, distance to daughter returned in workspace
// SW: this makes the navigation more robust and it appears that I have to
// put this at the moment since not all shapes respond yet with a negative distance if
// the point is actually inside the daughter
#ifdef CHECKCONTAINS
#pragma message "CHECKCONTAINS"
      bool contains = daughter->Contains(localpoint);
      if (!contains) {
#endif
        Precision ddistance = daughter->DistanceToIn(localpoint, localdir, step);

        // if distance is negative; we are inside that daughter and should relocate
        // unless distance is minus infinity
        bool valid = (ddistance < step && !IsInf(ddistance));
        nexthitvolume = valid ? daughter : nexthitvolume;
        step = valid ? ddistance : step;
#ifdef CHECKCONTAINS
      } else {
        std::cerr << " INDA ";
        step = -1.;
        nexthitvolume = daughter;
        break;
      }
#endif
    }
    return VNavigator::PrepareOutState(in_state, out_state, step, step_limit, nexthitvolume);
  }

  VECGEOM_INLINE
  virtual bool
  CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const & localpoint, Vector3D<Precision> const & localdir,
                             NavigationState const & in_state, NavigationState & out_state, Precision & step, VPlacedVolume const *&hitcandidate) const override {

    // iterate over all daughters
    auto *daughters = lvol->GetDaughtersp();
    auto ndaughters = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter = daughters->operator[](d);
//    previous distance becomes step estimate, distance to daughter returned in workspace
// SW: this makes the navigation more robust and it appears that I have to
// put this at the moment since not all shapes respond yet with a negative distance if
// the point is actually inside the daughter
#ifdef CHECKCONTAINS
#pragma message "CHECKCONTAINS"
      bool contains = daughter->Contains(localpoint);
      if (!contains) {
#endif
        Precision ddistance = daughter->DistanceToIn(localpoint, localdir, step);

        // if distance is negative; we are inside that daughter and should relocate
        // unless distance is minus infinity
        bool valid = (ddistance < step && !IsInf(ddistance));
        hitcandidate = valid ? daughter : hitcandidate;
        step = valid ? ddistance : step;
#ifdef CHECKCONTAINS
      } else {
        std::cerr << " INDA ";
        step = -1.;
        hitcandidate = daughter;
        break;
      }
#endif
    }
  }

//  template <typename Backend>
//  VECGEOM_INLINE
//  virtual Bool_v
//  CheckDaughterIntersectionsT(LogicalVolume const *lvol, Vector3D<typename Backend::Real_v> const &localpoint,
//                              Vector3D<typename Backend::Real_v> const &localdir, NavStatePool const &in_state,
//                              NavStatePool &out_state, unsigned int from_index, unsigned int to_index,
//                              typename Backend::Real_v &step, VPlacedVolume const **&hitcandidates) const override {
//
//    // iterate over all daughters
//    using Real_v = typename Backend::Real_v;
//    using Bool_v = Real_v::Mask;
//    const auto *daughters = lvol->GetDaughtersp();
//    const auto ndaughters = daughters->size();
//    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
//      auto daughter = daughters->operator[](d);
//      // SW: this makes the navigation more robust and it appears that I have to
//      // put this at the moment since not all shapes respond yet with a negative distance if
//      // the point is actually inside the daughter
//      //  #ifdef CHECKCONTAINS
//      // #pragma message "CHECKCONTAINS"
//      //        Bool_v contains = daughter->Contains(localpoint);
//      //        if (!contains) {
//      //  #endif
//      Real_v ddistance = daughter->DistanceToIn(localpoint, localdir, step);
//
//      // if distance is negative; we are inside that daughter and should relocate
//      // unless distance is minus infinity
//      Bool_v valid = (ddistance < step && !IsInf(ddistance));
//      // serial treatment of hit candidate until I find a better way
//      for (unsigned int index = valid.firstOne(); index < Real_v::Size; ++index) {
//        hitcandidates[i] = valid[i] ? daughter : hitcandidates[i];
//      }
//      step(valid) = ddistance; // or maskedAssign
//                               //  #ifdef CHECKCONTAINS
//                               //        } else {
//                               //          std::cerr << " INDA ";
//                               //          step = -1.;
//                               //          hitcandidate = daughter;
//                               //          break;
//                               //        }
//                               //  #endif
//    }
//    return false;
//  }

  static VNavigator *Instance() {
    static NewSimpleNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "NewSimpleNavigator";
  typedef SimpleSafetyEstimator SafetyEstimator_t;
}; // end of class

}
} // end namespace



#endif /* NAVIGATION_NEWSIMPLENAVIGATOR_H_ */
