/*
 * VSafetyEstimator.h
 *
 *  Created on: 28.08.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_VSAFETYESTIMATOR_H_
#define NAVIGATION_VSAFETYESTIMATOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/NavStatePool.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/NavStateFwd.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// some forward declarations
template <typename T>
class Vector3D;
// class NavigationState;
class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

//! base class defining basic interface for safety estimators;
//! safety estimators calculate the safety of a track in a logical volume containing other objects
//! sub classes implement optimized algorithms for logical volumes
//
// safety estimators can be called standalone or be used from the navigators
class VSafetyEstimator {

public:
  //! computes the safety of a point given in global coordinates for a geometry location specified
  //! by the navigationstate
  //! this function is a convenience interface to ComputeSafetyForLocalPoint and will usually be implemented in terms of
  //! the latter
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafety(Vector3D<Precision> const & /*globalpoint*/,
                                  NavigationState const & /*state*/) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const & /*localpoint*/,
                                               VPlacedVolume const * /*pvol*/) const = 0;

  // estimate just the safety to daughters for a local point with respect to a logical volume
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const & /*localpoint*/,
                                                          LogicalVolume const * /*lvol*/) const = 0;

  // in addition useful to offer an explicit SIMD interface
  // which could be used from other clients (such as VNavigator when it treats basket data)
  // the mask is supposed to indicate which lane needs a safety result since often the track is
  // on a boundary where the safety is zero anyway

  using Real_v = vecgeom::VectorBackend::Real_v;
  using Bool_v = vecCore::Mask_v<Real_v>;

  VECCORE_ATT_HOST_DEVICE
  virtual Real_v ComputeSafetyForLocalPoint(Vector3D<Real_v> const & /*localpoint*/, VPlacedVolume const * /*pvol*/,
                                            Bool_v /*m*/) const = 0;

  // interfaces to treat vectors/collections of points (uses the approach with intermediate storage and passing down the
  // loops to shapes)
  virtual void ComputeVectorSafety(SOA3D<Precision> const & /*globalpoints*/, NavStatePool &states,
                                   SOA3D<Precision> & /*workspace*/, Precision * /*safeties*/) const = 0;

  // interfaces to treat vectors/collections of points (uses the approach without intermediate storage; requires access
  // to new SIMD interface)
  virtual void ComputeVectorSafety(SOA3D<Precision> const & /*globalpoints*/, NavStatePool & /*states*/,
                                   Precision * /*safeties*/) const = 0;

private:
  virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const & /*localpoints*/, VPlacedVolume const * /*pvol*/,
                                           Precision * /*safeties*/) const = 0;

public:
  VECCORE_ATT_HOST_DEVICE
  virtual ~VSafetyEstimator() {}

  // get name of implementing class
  virtual const char *GetName() const = 0;
}; // end class VSafetyEstimator

//! template class providing a standard implementation for
//! some interfaces in VSafetyEstimator (using the CRT pattern)
template <typename Impl>
class VSafetyEstimatorHelper : public VSafetyEstimator {

public:
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafety(Vector3D<Precision> const &globalpoint, NavigationState const &state) const override
  {
    // calculate local point from global point
    Transformation3D m;
    state.TopMatrix(m);
    Vector3D<Precision> localpoint = m.Transform(globalpoint);
    // std::cerr << "##### " << localpoint << "\n";
    // "suck in" algorithm from Impl
    return ((Impl *)this)->Impl::ComputeSafetyForLocalPoint(localpoint, state.Top());
  }

  // interfaces to treat vectors/collections of points (uses the approach without intermediate storage; requires access
  // to new SIMD interface)
  virtual void ComputeVectorSafety(SOA3D<Precision> const & /*globalpoints*/, NavStatePool & /*states*/,
                                   Precision * /*safeties*/) const override
  {
    assert(0 && "not implemented yet, requires access to new SIM interface");
  }

  virtual void ComputeVectorSafety(SOA3D<Precision> const &globalpoints, NavStatePool &states,
                                   SOA3D<Precision> &localpointworkspace, Precision *safeties) const override
  {
    // calculate local point from global point
    auto np = globalpoints.size();
    for (auto i = decltype(np){0}; i < np; ++i) {
      Transformation3D m;
      states[i]->TopMatrix(m);
      localpointworkspace.set(i, m.Transform(globalpoints[i]));
    }
    // "suck in" algorithm from Impl
    ((Impl *)this)->Impl::ComputeSafetyForLocalPoints(localpointworkspace, states[0]->Top(), safeties);
  }

  static const char *GetClassName() { return Impl::gClassNameString; }

  virtual const char *GetName() const override { return GetClassName(); }
}; // end class VSafetyEstimatorHelper

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_VSAFETYESTIMATOR_H_ */
