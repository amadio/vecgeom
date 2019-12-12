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
template <bool MotherIsConvex = false>
class NewSimpleNavigator : public VNavigatorHelper<class NewSimpleNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  VECCORE_ATT_DEVICE
  NewSimpleNavigator()
      : VNavigatorHelper<class NewSimpleNavigator<MotherIsConvex>, MotherIsConvex>(SimpleSafetyEstimator::Instance()) {
  } VECCORE_ATT_DEVICE virtual ~NewSimpleNavigator() {}

public:
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
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
      bool contains = daughter->Contains(localpoint);
      if (!contains) {
#endif
        Precision ddistance = daughter->DistanceToIn(localpoint, localdir, step);

        // if distance is negative; we are inside that daughter and should relocate
        // unless distance is minus infinity
        const bool valid = (ddistance < step && !IsInf(ddistance)) &&
                           !((ddistance <= 0.) && in_state && in_state->GetLastExited() == daughter);
        hitcandidate = valid ? daughter : hitcandidate;
        step         = valid ? ddistance : step;
#ifdef CHECKCONTAINS
      } else {
        std::cerr << " INDA ";
        step         = -1.;
        hitcandidate = daughter;
        break;
      }
#endif
    }
    return false;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                                          Vector3D<Precision> const & /*localdir*/, VPlacedVolume const * /*blocked*/,
                                          Precision & /*step*/, VPlacedVolume const *& /*hitcandidate*/) const override
  {
    return false;
  }

  // Vector specialization for NewSimpleNavigator
  // TODO: unify with scalar use case
  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void DaughterIntersectionsLooper(VNavigator const * /*nav*/, LogicalVolume const *lvol,
                                          Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                          NavigationState const* const* /*in_states*/, NavigationState ** /*out_states*/,
                                          unsigned int from_index, Precision *out_steps,
                                          VPlacedVolume const *hitcandidates[ChunkSize])
  {
    // dispatch to vector implementation
    // iterate over all daughters
    // using Real_v = typename Backend::Real_v;
    // using Bool_v = Real_v::Mask;
    T step(vecCore::FromPtr<T>(out_steps + from_index));
    const auto *daughters = lvol->GetDaughtersp();
    auto ndaughters       = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter = daughters->operator[](d);
      // SW: this makes the navigation more robust and it appears that I have to
      // put this at the moment since not all shapes respond yet with a negative distance if
      // the point is actually inside the daughter
      //  #ifdef CHECKCONTAINS
      //        Bool_v contains = daughter->Contains(localpoint);
      //        if (!contains) {
      //  #endif
      const T ddistance = daughter->DistanceToIn(localpoint, localdir, step);

      // if distance is negative; we are inside that daughter and should relocate
      // unless distance is minus infinity
      const auto valid = (ddistance < step && !IsInf(ddistance));

      // serial treatment of hit candidate until I find a better way
      // we might do this using a cast to a double vector with subsequent masked assignment
      if (!vecCore::MaskEmpty(valid)) {
        for (unsigned int i = 0 /*valid.firstOne()*/; i < ChunkSize; ++i) {
          hitcandidates[i] = vecCore::MaskLaneAt(valid, i) ? daughter : hitcandidates[i];
        }

        vecCore::MaskedAssign(step, valid, ddistance);
        //  #ifdef CHECKCONTAINS
        //        } else {
        //          std::cerr << " INDA ";
        //          step = -1.;
        //          hitcandidate = daughter;
        //          break;
        //        }
        //  #endif
      }
    }
    vecCore::Store(step, out_steps + from_index);
  }

  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void DaughterIntersectionsLooper(VNavigator const *nav, LogicalVolume const *lvol,
                                          Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                          NavigationState const *const *in_states, unsigned int from_index,
                                          Precision *out_steps, VPlacedVolume const *hitcandidates[ChunkSize])
  {
    NewSimpleNavigator<MotherIsConvex>::template DaughterIntersectionsLooper<T, ChunkSize>(
        nav, lvol, localpoint, localdir, in_states, nullptr, from_index, out_steps, hitcandidates);
  }

#ifndef VECCORE_CUDA
  static VNavigator *Instance()
  {
    static NewSimpleNavigator instance;
    return &instance;
  }
#else
  VECCORE_ATT_DEVICE
  static VNavigator *Instance();
#endif

  static constexpr const char *gClassNameString = "NewSimpleNavigator";
  typedef SimpleSafetyEstimator SafetyEstimator_t;
}; // end of class
}
} // end namespace

#endif /* NAVIGATION_NEWSIMPLENAVIGATOR_H_ */
