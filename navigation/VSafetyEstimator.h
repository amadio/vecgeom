/*
 * VSafetyEstimator.h
 *
 *  Created on: 28.08.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_VSAFETYESTIMATOR_H_
#define NAVIGATION_VSAFETYESTIMATOR_H_

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/Transformation3D.h"
#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"
#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// some forward declarations
template <typename T> class Vector3D;
class NavigationState;
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
  virtual Precision ComputeSafety(Vector3D<Precision> const & /*globalpoint*/,
                                  NavigationState const & /*state*/) const = 0;

  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const & /*localpoint*/,
                                               VPlacedVolume const * /*pvol*/) const = 0;

  // TODO: We might want to add an explicit vector interface
  //
  // virtual SIMDTYPE ComputeSafetyForLocalPoint(Vector3D<SIMDTYPE> const &, ) = 0;

  // interfaces to treat vectors/collections of points (uses the approach with intermediate storage and passing down the
  // loops to shapes)
  virtual void ComputeVectorSafety(SOA3D<Precision> const & /*globalpoints*/, NavStatePool &states,
                                   SOA3D<Precision> & /*workspace*/, Precision * /*safeties*/) const = 0;

  // interfaces to treat vectors/collections of points (uses the approach without intermediate storage; requires access
  // to new SIMD interface)
  virtual void ComputeVectorSafety(SOA3D<Precision> const & /*globalpoints*/, NavStatePool &states,
                                   Precision * /*safeties*/) const {};

  virtual void ComputeSafetyForLocalPoints(SOA3D<Precision> const & /*localpoints*/, VPlacedVolume const * /*pvol*/,
                                           Precision * /*safeties*/) const = 0;

  virtual ~VSafetyEstimator() {}

  // get name of implementing class
  virtual const char *GetName() const = 0;
}; // end class VSafetyEstimator

//! template class providing a standard implementation for
//! some interfaces in VSafetyEstimator (using the CRT pattern)
template <typename Impl>
class VSafetyEstimatorHelper : public VSafetyEstimator {

public:
  virtual Precision ComputeSafety(Vector3D<Precision> const &globalpoint, NavigationState const &state) const override {
    // calculate local point from global point
    Transformation3D m;
    state.TopMatrix(m);
    Vector3D<Precision> localpoint = m.Transform(globalpoint);
    // std::cerr << "##### " << localpoint << "\n";
    // "suck in" algorithm from Impl
    return ((Impl *)this)->Impl::ComputeSafetyForLocalPoint(localpoint, state.Top());
  }

  virtual void ComputeVectorSafety(SOA3D<Precision> const &globalpoints, NavStatePool &states,
                                   SOA3D<Precision> &localpointworkspace, Precision *safeties) const override {
    // calculate local point from global point
    auto np = globalpoints.size();
    for (auto i=decltype(np){0}; i < np; ++i) {
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
}} // end namespaces

#endif /* NAVIGATION_VSAFETYESTIMATOR_H_ */
