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

/// Keep in dest the minimum between dest,temp for each corresponding element
static void VectMin(unsigned int nelem, Precision *__restrict__ dest, const Precision *__restrict__ temp) {
  using Real_v = vecgeom::VectorBackend::Real_v;
  // loop over all elements
  unsigned int i = 0;
  constexpr unsigned int nlanes = vecCore::VectorSize<Real_v>();
  const auto ilast = nelem - (nlanes-1);
  for ( ; i < ilast; i += nlanes) {
    Real_v &dest1 = * reinterpret_cast<Real_v*>(dest + i);
    const Real_v &temp1 = * reinterpret_cast<const Real_v*>(temp + i);
    vecCore::MaskedAssign(dest1, vecCore::Mask_v<Real_v>(dest1 > temp1), temp1);
  }

  // fall back to scalar interface for tail treatment
  for (; i < nelem; ++i) {
    if ( dest[i] > temp[i] ) dest[i] = temp[i];
  }
}

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

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Real_v ComputeSafetyForLocalPoint(Vector3D<Real_v> const &localpoint, VPlacedVolume const *pvol,
                                            Bool_v m) const override
  {
    // safety to mother
    Real_v safety(0.);
    if (!vecCore::MaskEmpty(m)) {
      safety = pvol->SafetyToOut(localpoint);

      // safety to daughters
      auto daughters       = pvol->GetLogicalVolume()->GetDaughtersp();
      auto numberdaughters = daughters->size();
      for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
        VPlacedVolume const *daughter = daughters->operator[](d);
        auto tmp                      = daughter->SafetyToIn(localpoint);
        safety                        = Min(safety, tmp);
      }
    }
    return safety;
  }

  // GL: This function uses a deprecated function, SafetyToInMinimize(). Is it considered deprecated too?
  // Trying alternative SafetyToIn() instead, but there is no minimization for now!
  VECGEOM_FORCE_INLINE
  virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const &localpoints, VPlacedVolume const *pvol,
                                           Precision *safeties) const override
  {
    auto npoints = localpoints.size();
    // a stack based workspace array
    // The following construct reserves stackspace WITHOUT initializing it
    char stackspace[npoints * sizeof(Precision)];
    Precision *tmpSafeties = reinterpret_cast<Precision *>(&stackspace);

    // // Array-based - transparently ensures proper alignment
    // // NavigBenchmark performance is actually worse than above though
    // Array<Precision> stackspace(npoints);
    // Precision *tmpSafeties = reinterpret_cast<Precision *>(stackspace.begin());

    pvol->SafetyToOut(localpoints, safeties);

    // safety to daughters; brute force but each function (possibly) vectorized
    Vector<Daughter> const *daughters = pvol->GetLogicalVolume()->GetDaughtersp();
    auto numberdaughters              = daughters->size();
    for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
      VPlacedVolume const *daughter = daughters->operator[](d);
      //daughter->SafetyToInMinimize(localpoints, safeties);
      daughter->SafetyToIn(localpoints, tmpSafeties);

      //.. keep track of closest daughter
      VectMin(npoints, safeties, tmpSafeties);
    }
    // here the safeties array automatically contains the right values
  }

  // bring in interface from higher up
  using VSafetyEstimatorHelper<SimpleSafetyEstimator>::ComputeVectorSafety;

  // implementation (using VC that does not need a workspace) by doing the whole algorithm in little vector chunks
  virtual void ComputeVectorSafety(SOA3D<Precision> const &globalpoints, NavStatePool &states,
                                   Precision *safeties) const override
  {
    VPlacedVolume const *pvol = states[0]->Top();
    auto npoints              = globalpoints.size();

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

      auto safety          = pvol->SafetyToOut(local);
      auto daughters       = pvol->GetLogicalVolume()->GetDaughtersp();
      auto numberdaughters = daughters->size();
      for (decltype(numberdaughters) d = 0; d < numberdaughters; ++d) {
        VPlacedVolume const *daughter = daughters->operator[](d);
        safety                        = Min(safety, daughter->SafetyToIn(local));
      }
      vecCore::Store(safety, safeties + i);
    }
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
