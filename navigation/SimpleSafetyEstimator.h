/*
 * VSimpleSafetyEstimator.h
 *
 *  Created on: 28.08.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLESAFETYESTIMATOR_H_
#define NAVIGATION_SIMPLESAFETYESTIMATOR_H_

#include "navigation/VSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a simple safety estimator based on a brute force (O(N)) approach
class SimpleSafetyEstimator : public VSafetyEstimatorHelper<SimpleSafetyEstimator> {

public:
  static constexpr const char *gClassNameString = "SimpleSafetyEstimator";

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
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
      safety = vecCore::math::Min(safety, tmp);
    }
    return safety;
  }


  VECGEOM_INLINE
  virtual Real_v ComputeSafetyForLocalPoint(Vector3D<Real_v> const &localpoint,
                                                VPlacedVolume const *pvol, Bool_v m) const override {
     // safety to mother
    Real_v safety(0.);
    if (! vecCore::MaskEmpty(m)) {
       safety = pvol->SafetyToOut(localpoint);

       // safety to daughters
       auto daughters = pvol->GetLogicalVolume()->GetDaughtersp();
       auto numberdaughters = daughters->size();
       for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
         VPlacedVolume const *daughter = daughters->operator[](d);
         auto tmp = daughter->SafetyToIn(localpoint);
         safety = vecCore::math::Min(safety, tmp);
       }
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
    VPlacedVolume const *pvol = states[0]->Top();
    auto npoints = globalpoints.size();

    using Real_v = vecgeom::VectorBackend::Real_v;
    constexpr auto kVS = vecCore::VectorSize<Real_v>();
    for (decltype(npoints) i = 0; i < npoints; i += kVS) {
      // do the transformation to local
      Vector3D<Real_v> local;
      for (size_t j = 0; j < kVS; ++j) {
        Transformation3D m;
        states[i + j]->TopMatrix(m);
        auto v = m.Transform(globalpoints[i + j]);

        using vecCore::AssignLane;
        AssignLane(local.x(), j, v.x());
        AssignLane(local.y(), j, v.y());
        AssignLane(local.z(), j, v.z());
      }

      auto safety = pvol->SafetyToOut(local);
      auto daughters = pvol->GetLogicalVolume()->GetDaughtersp();
      auto numberdaughters = daughters->size();
      for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
        VPlacedVolume const *daughter = daughters->operator[](d);
        safety = vecCore::math::Min(safety, daughter->SafetyToIn(local));
      }
      vecCore::Store(safety, safeties + i);
    }
  }

#ifndef VECGEOM_NVCC
  static VSafetyEstimator *Instance() {
    static SimpleSafetyEstimator instance;
    return &instance;
  }
#else
  VECGEOM_CUDA_HEADER_DEVICE
  static VSafetyEstimator *Instance();
#endif

}; // end class


}} // end namespaces


#endif /* NAVIGATION_SIMPLESAFETYESTIMATOR_H_ */
