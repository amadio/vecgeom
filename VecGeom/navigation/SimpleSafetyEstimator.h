/*
 * VSimpleSafetyEstimator.h
 *
 *  Created on: 28.08.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLESAFETYESTIMATOR_H_
#define NAVIGATION_SIMPLESAFETYESTIMATOR_H_

#include "VecGeom/navigation/VSafetyEstimator.h"
//#include "VecGeom/base/Array.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a simple safety estimator based on a brute force (O(N)) approach
class SimpleSafetyEstimator : public VSafetyEstimatorHelper<SimpleSafetyEstimator> {

public:
  static constexpr const char *gClassNameString = "SimpleSafetyEstimator";

  // estimate just the safety to daughters for a local point with respect to a logical volume
  // TODO: use this function in other interfaces to avoid code duplication
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                          LogicalVolume const *lvol) const override
  {
    // safety to daughters
    double safety(kInfLength);
    auto daughters       = lvol->GetDaughtersp();
    auto numberdaughters = daughters->size();
    for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
      VPlacedVolume const *daughter = daughters->operator[](d);
      double tmp                    = daughter->SafetyToIn(localpoint);
      safety                        = Min(safety, tmp);
    }
    return safety;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override
  {
    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);

    // safety to daughters
    auto daughters       = pvol->GetLogicalVolume()->GetDaughtersp();
    auto numberdaughters = daughters->size();
    for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
      VPlacedVolume const *daughter = daughters->operator[](d);
      double tmp                    = daughter->SafetyToIn(localpoint);
      safety                        = Min(safety, tmp);
    }
    return safety;
  }

#ifndef VECCORE_CUDA
  static VSafetyEstimator *Instance()
  {
    static SimpleSafetyEstimator instance;
    return &instance;
  }
#else
  VECCORE_ATT_DEVICE
  static VSafetyEstimator *Instance();
#endif

}; // end class
}
} // end namespaces

#endif /* NAVIGATION_SIMPLESAFETYESTIMATOR_H_ */
