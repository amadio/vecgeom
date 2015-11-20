/*
 * VSimpleSafetyEstimator.h
 *
 *  Created on: 28.08.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLESAFETYESTIMATOR_H_
#define NAVIGATION_SIMPLESAFETYESTIMATOR_H_

#include "navigation/VSafetyEstimator.h"
#ifdef VECGEOM_VC
#include <Vc/Vc>
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a simple safety estimator based on a brute force (O(N)) approach
class SimpleSafetyEstimator : public VSafetyEstimatorHelper<SimpleSafetyEstimator> {

public:
  static constexpr const char *gClassNameString = "SimpleSafetyEstimator";

  VECGEOM_INLINE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override {
    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);

    // safety to daughters
    auto daughters = pvol->GetLogicalVolume()->GetDaughtersp();
    auto numberdaughters = daughters->size();
    for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
      VPlacedVolume const *daughter = daughters->operator[](d);
      double tmp = daughter->SafetyToIn(localpoint);
      safety = Min(safety, tmp);
    }
    return safety;
  }

  VECGEOM_INLINE
  virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const &localpoints, VPlacedVolume const *pvol,
                                           Precision *safeties) const override {
    pvol->SafetyToOut(localpoints, safeties);

    // safety to daughters; brute force but each function (possibly) vectorized
    Vector<Daughter> const *daughters = pvol->GetLogicalVolume()->GetDaughtersp();
    auto numberdaughters = daughters->size();
    for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
      VPlacedVolume const *daughter = daughters->operator[](d);
      daughter->SafetyToInMinimize(localpoints, safeties);
    }
    // here the safeties array automatically contains the right values
  }

  // bring in interface from higher up
  using VSafetyEstimatorHelper<SimpleSafetyEstimator>::ComputeVectorSafety;

  // implementation (using VC that does not need a workspace) by doing the whole algorithm in little vector chunks
  virtual void ComputeVectorSafety(SOA3D<Precision> const &globalpoints, NavStatePool &states,
                                   Precision *safeties) const override {
// this is a temporary implementation until we have the new generalized backends
#ifdef VECGEOM_VC
    VPlacedVolume const *pvol = states[0]->Top();
    auto npoints = globalpoints.size();
    for (decltype(npoints) i = 0; i < npoints; i += Vc::double_v::Size) {
      // do the transformation to local
      Vector3D<Vc::double_v> local;
      for (size_t j = 0; j < Vc::double_v::Size; ++j) {
        Transformation3D m;
        states[i + j]->TopMatrix(m);
        auto v = m.Transform(globalpoints[i + j]);
        local.x()[j] = v.x();
        local.y()[j] = v.y();
        local.z()[j] = v.z();
      }

      Vc::double_v safety = pvol->SafetyToOut(local);
      auto daughters = pvol->GetLogicalVolume()->GetDaughtersp();
      auto numberdaughters = daughters->size();
      for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
        VPlacedVolume const *daughter = daughters->operator[](d);
        safety = Min(safety, daughter->SafetyToIn(local));
      }
      safety.store(safeties + i);
    }
#endif
  }

  static VSafetyEstimator *Instance() {
    static SimpleSafetyEstimator instance;
    return &instance;
  }

}; // end class


}} // end namespaces


#endif /* NAVIGATION_SIMPLESAFETYESTIMATOR_H_ */
