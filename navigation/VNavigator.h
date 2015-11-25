/*
 * VNavigator.h
 *
 *  Created on: 17.09.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_VNAVIGATOR_H_
#define NAVIGATION_VNAVIGATOR_H_

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/Transformation3D.h"
#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"
#include "navigation/GlobalLocator.h"
#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "navigation/VSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// some forward declarations
template <typename T> class Vector3D;
class NavigationState;
class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

//! base class defining basic interface for navigation ( hit-detection )
//! sub classes implement optimized algorithms for logical volumes
class VNavigator {

public:
  VNavigator() : fSafetyEstimator(nullptr) {}

  //! computes the step (distance) to the next object in the geometry hierarchy obtained
  //! by propagating with step along the ray
  //! the next object could be after a boundary and does not necessarily coincide with the object
  //! hit by the ray

  //! this methods transforms the global coordinates into local ones usually calls more specialized methods
  //! like the hit detection on local coordinates
  virtual Precision ComputeStepAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                  Vector3D<Precision> const & /*globaldir*/,
                                                  Precision /*(physics) step limit */,
                                                  NavigationState const & /*in_state*/,
                                                  NavigationState & /*out_state*/) const = 0;

  // a similar interface also returning the local coordinates as a result
  // might be reused by other calculations such as Safety
  virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                 Vector3D<Precision> const & /*globaldir*/,
                                                 Precision /*(physics) step limit */,
                                                 NavigationState const & /*in_state*/,
                                                 NavigationState & /*out_state*/, Precision & /*safety_out*/) const = 0;

  virtual Precision ComputeStepAndHittingBoundaryForLocalPoint(Vector3D<Precision> const & /*localpoint*/,
                                                               Vector3D<Precision> const & /*localdir*/,
                                                               Precision /*(physics) step limit */,
                                                               NavigationState const & /*in_state*/,
                                                               NavigationState & /*out_state*/) const = 0;

  // think about a method that operates only on logical volume level and that treats only the daughters
  // (this would further modularize the navigators and reduce code); this is true since the DistanceToOut treatment is almost the same
  // virtual Precision CollideDaughters(Vector3D<Precision> const & /*localpoint*/,
  //                                    Vector3D<Precision> const & /*localdir*/,
  //                                    LogicalVolume const *lvol,
  //                                    Precision /*(physics) step limit*/) const = 0;

  // the bool return type indicates if out_state was already modified; this may happen in assemblies; in this case we don't need to copy the in_state to the out state later on
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const & /*localpoint*/, Vector3D<Precision> const & /*localdir*/,
                                          NavigationState const & /*in_state*/, NavigationState & /*out_state*/, Precision & step, VPlacedVolume const *& /*hitcandidate*/) const {return false;}

  // for vector navigation
  //! this methods transforms the global coordinates into local ones usually calls more specialized methods
  //! like the hit detection on local coordinates
  virtual void ComputeStepsAndPropagatedStates(SOA3D<Precision> const & /*globalpoints*/,
                                               SOA3D<Precision> const & /*globaldirs*/,
                                               Precision const */*(physics) step limits */,
                                               NavStatePool const & /*in_states*/,
                                               NavStatePool & /*out_states*/, Precision */*out_steps*/) const {};

protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  virtual void Relocate( Vector3D<Precision> /*localpoint*/, NavigationState const & /*in_state*/,
                         NavigationState & /*out_state*/) const = 0;

  // a common function to be used by all navigators to ensure consistency in transporting points
  // after a boundary
  VECGEOM_INLINE
  Vector3D<Precision> MovePointAfterBoundary( Vector3D<Precision> const &localpoint, Vector3D<Precision> const &dir, Precision step ) const {
      const Precision extra=1E-6; //TODO: to be revisited (potentially going for a more relative approach)
      return localpoint + (step+extra)*dir;
  }



public:
  virtual ~VNavigator() {};

  // get name of implementing class
  virtual const char *GetName() const = 0;

  typedef VSafetyEstimator SafetyEstimator_t;

protected:
  VSafetyEstimator *fSafetyEstimator; // a pointer to the safetyEstimator which can be used by the Navigator

  // some common code to prepare the
  VECGEOM_INLINE
  static Precision PrepareOutState(NavigationState const &in_state, NavigationState &out_state, Precision geom_step,
                                   Precision step_limit, VPlacedVolume const *hitcandidate){
    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);

    // if the following is the case we are in the wrong volume;
    // assuming that DistanceToIn return negative number when point is inside
    // do nothing (step=0) and retry one level higher

    // TODO: put diagnostic code here ( like in original SimpleNavigator )
    if (geom_step == kInfinity && step_limit > 0.) {
      geom_step = vecgeom::kTolerance;
      out_state.SetBoundaryState(true);
      out_state.Pop();
      return geom_step;
    }

    // is geometry further away than physics step?
    // this is a physics step
    if (geom_step > step_limit) {
      // don't need to do anything
      geom_step = step_limit;
      out_state.SetBoundaryState(false);
      return geom_step;
    }

    // otherwise it is a geometry step
    out_state.SetBoundaryState(true);
    if (hitcandidate)
      out_state.Push(hitcandidate);

    if (geom_step < 0.) {
      // std::cerr << "WARNING: STEP NEGATIVE; NEXTVOLUME " << nexthitvolume << std::endl;
      // InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
      geom_step = 0.;
    }
    return geom_step;
  }

  VECGEOM_INLINE
  static Precision TreatDistanceToMother( VPlacedVolume const *pvol, Vector3D<Precision> const &localpoint , Vector3D<Precision> const &localdir, Precision step_limit ){
    Precision step;
    assert(pvol != nullptr && "currentvolume is null in navigation");
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);
    // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. SET TO INFINITY SO THAT WE CAN BETTER HANDLE IT
    // LATER ON
    if (step < 0.) {
      step = kInfinity;
    }
    return step;
  }

};

//! template class providing a standard implementation for
//! some interfaces in VSafetyEstimator (using the CRT pattern)
template <typename Impl, bool MotherIsConvex=false>
class VNavigatorHelper : public VNavigator {

public:
    // a generic implementation
    // may be completely specialized in child-navigators
#define OLD
#ifdef OLD
    virtual
    Precision
    ComputeStepAndPropagatedState(Vector3D<Precision> const & globalpoint,
                                  Vector3D<Precision> const & globaldir,
                                  Precision step_limit,
                                  NavigationState const & in_state,
                                  NavigationState & out_state) const override {
      // calculate local point from global point
      Transformation3D m;
      in_state.TopMatrix(m);
      auto localpoint = m.Transform(globalpoint);
      auto localdir = m.TransformDirection(globaldir);
      // "suck in" algorithm from Impl and treat hit detection in local coordinates
      Precision step = ((Impl*)this)->Impl::ComputeStepAndHittingBoundaryForLocalPoint(
              localpoint, localdir,
              step_limit, in_state, out_state);

      // step was physics limited
      if( ! out_state.IsOnBoundary() )
        return step;

      // otherwise if necessary do a relocation

      // try relocation to refine out_state to correct location after the boundary
      ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
      return step;
    }
#else
    // a future, more modular and potentially faster implementation may look like this
    virtual
    Precision
    ComputeStepAndPropagatedState(Vector3D<Precision> const & globalpoint,
                                  Vector3D<Precision> const & globaldir,
                                  Precision step_limit,
                                  NavigationState const & in_state,
                                  NavigationState & out_state) const override {
      static size_t counter=0;
      // calculate local point from global point
      Transformation3D m;
      in_state.TopMatrix(m);
      auto localpoint = m.Transform(globalpoint);
      auto localdir = m.TransformDirection(globaldir);
      Precision step = step_limit;
      VPlacedVolume const *hitcandidate=nullptr;
      auto pvol= in_state.Top();
      auto lvol= pvol->GetLogicalVolume();

      // think about calling template specializations instead of branching
      if( MotherIsConvex ){
        // if mother is convex we may not need to do treatment of mother
        // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
        ((Impl *)this)
            ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, in_state, out_state, step, hitcandidate);
        if(hitcandidate == nullptr)
            step = TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      } else {
        // need to calc DistanceToOut first
        step = TreatDistanceToMother( pvol, localpoint, localdir, step_limit );
          // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
        ((Impl *)this)->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, in_state, out_state, step, hitcandidate);
        if(hitcandidate==nullptr) counter++;
      }

      // fix state
      step = PrepareOutState(in_state, out_state, step, step_limit, hitcandidate);

      // step was physics limited
      if (!out_state.IsOnBoundary())
        return step;

      // otherwise if necessary do a relocation
      // try relocation to refine out_state to correct location after the boundary
      ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
      return step;
    }
#endif


    // a similar interface also returning the safety
    virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const & globalpoint,
                                                             Vector3D<Precision> const & globaldir,
                                                             Precision step_limit,
                                                             NavigationState const & in_state,
                                                             NavigationState & out_state, Precision & safety_out) const override {
        // calculate local point from global point
        Transformation3D m;
        in_state.TopMatrix(m);
        auto localpoint = m.Transform(globalpoint);
        auto localdir = m.TransformDirection(globaldir);

        // get safety first ( the only benefit here is when we reuse the local points
        using SafetyE_t = typename Impl::SafetyEstimator_t;
        if (!in_state.IsOnBoundary()) {
          // call the appropriate safety Estimator
          safety_out = ((SafetyE_t*) fSafetyEstimator)->SafetyE_t::ComputeSafetyForLocalPoint(localpoint, in_state.Top());
        }

        // "suck in" algorithm from Impl and treat hit detection in local coordinates
        Precision step = ((Impl*)this)->Impl::ComputeStepAndHittingBoundaryForLocalPoint(
                localpoint, localdir,
                step_limit, in_state, out_state);

        // step was physics limited
        if( ! out_state.IsOnBoundary() )
          return step;

        // otherwise if necessary do a relocation

        // try relocation to refine out_state to correct location after the boundary
        ((Impl*)this)->Impl::Relocate( MovePointAfterBoundary( localpoint, localdir, step ), in_state, out_state);
        return step;
    }



protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  VECGEOM_INLINE
  virtual void Relocate(Vector3D<Precision> pointafterboundary, NavigationState const & in_state,
                        NavigationState & out_state) const override {
    // this means that we are leaving the mother
    // alternatively we could use nextvolumeindex like before
    if( out_state.Top() == in_state.Top() ){
        GlobalLocator::RelocatePointFromPath( pointafterboundary, out_state );
    }
    else {
        // continue directly further down ( next volume should have been stored in out_state already )
      VPlacedVolume const *nextvol = out_state.Top();
      out_state.Pop();
      GlobalLocator::LocateGlobalPoint(nextvol, nextvol->GetTransformation()->Transform(pointafterboundary), out_state,
                                       false);

      assert(in_state.Top() != out_state.Top() && " error relocating when entering ");
      return;
    }
  }

public:
  static const char *GetClassName() { return Impl::gClassNameString; }

  virtual const char *GetName() const override { return GetClassName(); }
}; // end class VSafetyEstimatorHelper





}
} // end namespace

#endif /* NAVIGATION_VNAVIGATOR_H_ */
