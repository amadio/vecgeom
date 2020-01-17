/*
 * VNavigator.h
 *
 *  Created on: 17.09.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_VNAVIGATOR_H_
#define NAVIGATION_VNAVIGATOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/VSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// some forward declarations
template <typename T>
class Vector3D;
class NavigationState;
class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

//! base class defining basic interface for navigation ( hit-detection )
//! sub classes implement optimized algorithms for logical volumes
class VNavigator {

public:
  VNavigator() : fSafetyEstimator(nullptr) {}

  VECCORE_ATT_HOST_DEVICE
  VSafetyEstimator const *GetSafetyEstimator() const { return fSafetyEstimator; }

  //! computes the step (distance) to the next object in the geometry hierarchy obtained
  //! by propagating with step along the ray
  //! the next object could be after a boundary and does not necessarily coincide with the object
  //! hit by the ray

  //! this methods transforms the global coordinates into local ones usually calls more specialized methods
  //! like the hit detection on local coordinates
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                  Vector3D<Precision> const & /*globaldir*/,
                                                  Precision /*(physics) step limit */,
                                                  NavigationState const & /*in_state*/,
                                                  NavigationState & /*out_state*/) const = 0;

  //! computes the step (distance) to the next object in the geometry hierarchy obtained
  //! by propagating with step along the ray
  //! updates out_state to contain information about the next hitting boundary:
  //!   - if a daugher is hit: out_state.Top() will be daughter
  //!   - if ray leaves volume: out_state.Top() will point to current volume
  //!   - if step limit > step: out_state == in_state
  //!
  //! This function is essentialy equal to ComputeStepAndPropagatedState without
  //! the relocation part
  virtual Precision ComputeStep(Vector3D<Precision> const & /*globalpoint*/, Vector3D<Precision> const & /*globaldir*/,
                                Precision /*(physics) step limit */, NavigationState const & /*in_state*/,
                                NavigationState & /*out_state*/) const = 0;

  //! as above ... also returns the safety ... does not give_back an out_state
  //! but the in_state might be modified to contain the next daughter when
  //! user specifies indicateDaughterHit = true
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafety(Vector3D<Precision> const & /*globalpoint*/,
                                         Vector3D<Precision> const & /*globaldir*/, Precision /*(physics) step limit */,
                                         NavigationState & /*in_state*/, bool /*calcsafety*/, Precision & /*safety*/,
                                         bool indicateDaughterHit = false) const = 0;

  // an alias interface ( using TGeo name )
  VECCORE_ATT_HOST_DEVICE
  void FindNextBoundaryAndStep(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                               NavigationState const &in_state, NavigationState &out_state, Precision step_limit,
                               Precision &step) const
  {
    step = ComputeStepAndPropagatedState(globalpoint, globaldir, step_limit, in_state, out_state);
  }

  // an alias interface ( using TGeo name )
  void FindNextBoundaryAndStepAndSafety(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                        NavigationState const &in_state, NavigationState &out_state,
                                        Precision step_limit, Precision &step, bool calcsafety, Precision &safety) const
  {
    step = ComputeStepAndSafetyAndPropagatedState(globalpoint, globaldir, step_limit, in_state, out_state, calcsafety,
                                                  safety);
  }

  // a similar interface, in addition also returning the safety as a result
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                           Vector3D<Precision> const & /*globaldir*/,
                                                           Precision /*(physics) step limit */,
                                                           NavigationState const & /*in_state*/,
                                                           NavigationState & /*out_state*/, bool /*calcsafefty*/,
                                                           Precision & /*safety_out*/) const = 0;

  // the bool return type indicates if out_state was already modified; this may happen in assemblies;
  // in this case we don't need to copy the in_state to the out state later on
  // NavigationState a pointer since we might want to pass nullptr
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                                          Vector3D<Precision> const & /*localdir*/,
                                          NavigationState const * /*in_state*/, NavigationState * /*out_state*/,
                                          Precision & /*step*/, VPlacedVolume const *& /*hitcandidate*/) const = 0;

  /// check if a ray given by localpoint, localdir intersects with any daughter. Possibility
  /// to pass a volume which is blocked/should be ignored in the query. Updates the step as well as the hitcandidate
  /// volume. (This version is useful for G4; assemblies not supported)
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                                          Vector3D<Precision> const & /*localdir*/, VPlacedVolume const * /*blocked*/,
                                          Precision & /*step*/, VPlacedVolume const *& /*hitcandidate*/) const
  {
    return false;
  }

  // interfaces for vector/basket navigation
  virtual void ComputeStepsAndPropagatedStates(SOA3D<Precision> const & /*globalpoints*/,
                                               SOA3D<Precision> const & /*globaldirs*/,
                                               Precision const * /*(physics) step limits */,
                                               NavigationState const* const* /*in_states*/,
                                               NavigationState ** /*out_states*/, Precision * /*out_steps*/) const = 0;

  // for vector navigation
  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafetiesAndPropagatedStates(SOA3D<Precision> const & points,
                                                          SOA3D<Precision> const & dirs,
                                                          Precision const *psteps,
                                                          NavStatePool const &instates,
                                                          NavStatePool &outstates,
                                                          Precision *outsteps,
                                                          bool const *calcsafety,
                                                          Precision *safeties) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafetiesAndPropagatedStates(SOA3D<Precision> const & /*globalpoints*/,
                                                          SOA3D<Precision> const & /*globaldirs*/,
                                                          Precision const * /*(physics) step limits */,
                                                          NavigationState const* const* /*in_states*/,
                                                          NavigationState ** /*out_states*/,
                                                          Precision * /*out_steps*/,
                                                          bool const * /*calcsafety*/,
                                                          Precision * /*out_safeties*/) const = 0;

  // for vector navigation (not doing relocation) -- interface for GeantV
  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafeties(SOA3D<Precision> const & /*globalpoints*/,
                                       SOA3D<Precision> const & /*globaldirs*/,
                                       Precision const * /*(physics) step limits */,
                                       NavigationState const* const* /*in_states*/, Precision * /*out_steps*/,
                                       bool const * /*calcsafety*/, Precision * /*out_safeties*/) const = 0;

protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  VECCORE_ATT_HOST_DEVICE
  virtual void Relocate(Vector3D<Precision> const & /*localpoint*/, NavigationState const &__restrict__ /*in_state*/,
                        NavigationState &__restrict__ /*out_state*/) const = 0;

  // a common function to be used by all navigators to ensure consistency in transporting points
  // after a boundary
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> MovePointAfterBoundary(Vector3D<Precision> const &localpoint,
                                                    Vector3D<Precision> const &dir, Precision step)
  {
    const Precision extra = 1E-6; // TODO: to be revisited (potentially going for a more relative approach)
    return localpoint + (step + extra) * dir;
  }

public:
  VECCORE_ATT_DEVICE
  virtual ~VNavigator(){};

  // get name of implementing class
  virtual const char *GetName() const = 0;

  typedef VSafetyEstimator SafetyEstimator_t;

protected:
  VECCORE_ATT_HOST_DEVICE
  VNavigator(VSafetyEstimator *s) : fSafetyEstimator(s) {}
  VSafetyEstimator *fSafetyEstimator; // a pointer to the safetyEstimator which can be used by the Navigator

  // some common code to prepare the outstate
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Precision PrepareOutState(NavigationState const &__restrict__ in_state,
                                   NavigationState &__restrict__ out_state, Precision geom_step, Precision step_limit,
                                   VPlacedVolume const *hitcandidate, bool &doneafterthisstep)
  {
    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    doneafterthisstep = false;

    // if the following is the case we are in the wrong volume;
    // assuming that DistanceToIn returns negative number when point is inside
    // do nothing (step=0) and retry one level higher

    // TODO: put diagnostic code here ( like in original SimpleNavigator )
    if (geom_step == kInfLength && step_limit > 0.) {
      geom_step = vecgeom::kTolerance;
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      doneafterthisstep = true;
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
    if (hitcandidate) out_state.Push(hitcandidate);

    if (geom_step < 0.) {
      // std::cerr << "WARNING: STEP NEGATIVE; NEXTVOLUME " << nexthitvolume << std::endl;
      // InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
      geom_step = 0.;
    }
    return geom_step;
  }

  // kernel to be used with both scalar and vector types
  template <typename T>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static T TreatDistanceToMother(VPlacedVolume const *pvol, Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                 T step_limit)
  {
    T step;
    assert(pvol != nullptr && "currentvolume is null in navigation");
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);
    vecCore::MaskedAssign(step, step < T(0.), T(0.));
    return step;
  }

  // default static function doing the global to local transformation
  // may be redefined in concrete implementations ( for instance in cases where we know the form of the global matrix
  // a-priori )
  // input
  // TODO: think about how we can have scalar + SIMD version
  // note: the last argument is a trick to pass information across function calls ( only exploited in specialized
  // navigators )
  template <typename T>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DoGlobalToLocalTransformation(NavigationState const &in_state, Vector3D<T> const &globalpoint,
                                            Vector3D<T> const &globaldir, Vector3D<T> &localpoint,
                                            Vector3D<T> &localdir)
  {
    // calculate local point/dir from global point/dir
    Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
  }

  // version used for SIMD processing
  // should be specialized in Impl for faster treatment
  template <typename T, unsigned int ChunkSize>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DoGlobalToLocalTransformations(NavigationState const *const *in_states, SOA3D<Precision> const &globalpoints,
                                             SOA3D<Precision> const &globaldirs, unsigned int from_index,
                                             Vector3D<T> &localpoint, Vector3D<T> &localdir)
  {
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      Transformation3D m;
      //  assert(in_states[trackid]->Top()->GetLogicalVolume() == lvol &&
      //         "not all states in same logical volume"); // the logical volume of all the states should be the same
      in_states[trackid]->TopMatrix(m);              // could benefit from interal vec
      auto tmp = m.Transform(globalpoints[trackid]); // could benefit from internal vec

      using vecCore::AssignLane;
      AssignLane(localpoint.x(), i, tmp.x());
      AssignLane(localpoint.y(), i, tmp.y());
      AssignLane(localpoint.z(), i, tmp.z());
      tmp = m.TransformDirection(globaldirs[trackid]); // could benefit from internal vec
      AssignLane(localdir.x(), i, tmp.x());
      AssignLane(localdir.y(), i, tmp.y());
      AssignLane(localdir.z(), i, tmp.z());
    }
  }
};

//! template class providing a standard implementation for
//! some interfaces in VNavigator (using the CRT pattern)
template <typename Impl, bool MotherIsConvex = false>
class VNavigatorHelper : public VNavigator {
protected:
  using VNavigator::VNavigator;

public:
  // the default implementation for hit detection with daughters for a chunk of data
  // is to loop over the implementation for the scalar case
  // this static function may be overridden by the specialized implementations (such as done in NewSimpleNavigator)
  // the from_index, to_index indicate which states from the NavigationState ** are actually treated
  // in the worst case, we might have to implement this stuff over there
  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void DaughterIntersectionsLooper(VNavigator const *nav, LogicalVolume const *lvol,
                                          Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                          NavigationState const *const *in_states, NavigationState **out_states,
                                          unsigned int from_index, Precision *out_steps,
                                          VPlacedVolume const *hitcandidates[ChunkSize])
  {
    // dispatch to ordinary implementation ( which itself might be vectorized )
    using vecCore::LaneAt;
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      ((Impl *)nav)
          ->Impl::CheckDaughterIntersections(
              lvol,
              Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i), LaneAt(localpoint.z(), i)),
              Vector3D<Precision>(LaneAt(localdir.x(), i), LaneAt(localdir.y(), i), LaneAt(localdir.z(), i)),
              in_states[trackid], out_states[trackid], out_steps[trackid], hitcandidates[i]);
    }
  }

  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void DaughterIntersectionsLooper(VNavigator const *nav, LogicalVolume const *lvol,
                                          Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                          NavStatePool const &in_states, NavStatePool &out_states,
                                          unsigned int from_index, Precision *out_steps,
                                          VPlacedVolume const *hitcandidates[ChunkSize])
  {
    // dispatch to ordinary implementation ( which itself might be vectorized )
    using vecCore::LaneAt;
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      ((Impl *)nav)
          ->Impl::CheckDaughterIntersections(
              lvol,
              Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i), LaneAt(localpoint.z(), i)),
              Vector3D<Precision>(LaneAt(localdir.x(), i), LaneAt(localdir.y(), i), LaneAt(localdir.z(), i)),
              in_states[trackid], out_states[trackid], out_steps[trackid], hitcandidates[i]);
    }
  }

  // the default implementation for hit detection with daughters for a chunk of data
  // is to loop over the implementation for the scalar case
  // similar to previous version; does not care about out_state
  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void DaughterIntersectionsLooper(VNavigator const *nav, LogicalVolume const *lvol,
                                          Vector3D<T> const &localpoint, Vector3D<T> const &localdir,
                                          NavigationState const *const *in_states, unsigned int from_index,
                                          Precision *out_steps, VPlacedVolume const *hitcandidates[ChunkSize])
  {
    // dispatch to ordinary implementation ( which itself might be vectorized )
    using vecCore::LaneAt;
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      ((Impl *)nav)
          ->Impl::CheckDaughterIntersections(
              lvol,
              Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i), LaneAt(localpoint.z(), i)),
              Vector3D<Precision>(LaneAt(localdir.x(), i), LaneAt(localdir.y(), i), LaneAt(localdir.z(), i)),
              in_states[trackid], nullptr, out_steps[trackid], hitcandidates[i]);
    }
  }

  // the default implementation for safety calculation for a chunk of data
  template <typename T, unsigned int ChunkSize> // we may go to Backend as template parameter in future
  VECCORE_ATT_HOST_DEVICE static void SafetyLooper(VNavigator const *nav, VPlacedVolume const *pvol,
                                                   Vector3D<T> const &localpoint, unsigned int from_index,
                                                   bool const *calcsafeties, Precision *out_safeties)
  {
    // dispatch to ordinary implementation ( which itself might be vectorized )
    // TODO: check calcsafeties if we need to do this at all
    using SafetyE_t = typename Impl::SafetyEstimator_t;
    using Bool_v    = vecCore::Mask_v<T>;
    Bool_v m;
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      vecCore::AssignMaskLane(m, i, calcsafeties[from_index + i]);
    }

    T safety(0.);
    if (!vecCore::MaskEmpty(m)) {
      safety = ((SafetyE_t *)nav->GetSafetyEstimator())->ComputeSafetyForLocalPoint(localpoint, pvol, m);
    }
    vecCore::Store(safety, &out_safeties[from_index]);
  }

public:
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndPropagatedState(Vector3D<Precision> const &globalpoint,
                                                  Vector3D<Precision> const &globaldir, Precision step_limit,
                                                  NavigationState const &in_state,
                                                  NavigationState &out_state) const override
  {
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    Precision step                    = step_limit;
    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();

    if (MotherIsConvex) {
      // if mother is convex we may not need to do treatment of mother
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
      if (hitcandidate == nullptr) step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    } else {
      // need to calc DistanceToOut first
      step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
    }

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) {
      if (out_state.Top() != nullptr) {
        assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
      return step;
    }
    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    // otherwise if necessary do a relocation
    // try relocation to refine out_state to correct location after the boundary
    ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
    if (out_state.Top() != nullptr) {
      while (out_state.Top()->IsAssembly()) {
        out_state.Pop();
      }
      assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
    return step;
  }

  virtual Precision ComputeStep(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                Precision step_limit, NavigationState const &in_state,
                                NavigationState &out_state) const override
  {
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    Precision step                    = step_limit;
    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();

    if (MotherIsConvex) {
      // if mother is convex we may not need to do treatment of mother
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
      if (hitcandidate == nullptr) step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    } else {
      // need to calc DistanceToOut first
      step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
    }

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) {
      if (out_state.Top() != nullptr) {
        assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
      return step;
    }
    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    return step;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafety(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                         Precision step_limit, NavigationState &in_state, bool calcsafety,
                                         Precision &safety, bool indicateDaughterHit = false) const override
  {
// FIXME: combine this kernel and the one for ComputeStep() into one generic function
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    NavigationState *out_state = nullptr;

    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    // get safety first ( the benefit here is that we reuse the local points )
    using SafetyE_t = typename Impl::SafetyEstimator_t;
    if (calcsafety) {
      // call the appropriate safety Estimator
      safety = ((SafetyE_t *)fSafetyEstimator)->SafetyE_t::ComputeSafetyForLocalPoint(localpoint, in_state.Top());
    }

    Precision step                    = step_limit;
    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();

    // is the next object certainly further away than the safety
    bool safetydone = calcsafety && safety >= step;

    if (!safetydone) {
      if (MotherIsConvex) {
        // if mother is convex we may not need to do treatment of mother
        // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
        ((Impl *)this)
            ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, out_state, step, hitcandidate);
        if (hitcandidate == nullptr) step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      } else {
        // need to calc DistanceToOut first
        step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
        // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
        ((Impl *)this)
            ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, out_state, step, hitcandidate);
      }
    }
    if (indicateDaughterHit && hitcandidate) in_state.Push(hitcandidate);
    return Min(step, step_limit);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafeties(SOA3D<Precision> const &globalpoints, SOA3D<Precision> const &globaldirs,
                                       Precision const *step_limits, NavigationState const *const *in_states,
                                       Precision *out_steps, bool const *calcsafeties,
                                       Precision *out_safeties) const override
  {
    // process SIMD part and TAIL part
    using Real_v = vecgeom::VectorBackend::Real_v;
    (void)(Real_v *)(nullptr); // Avoid spurrious warning: unused type alias 'Real_v' [-Wunused-local-typedef]
    const auto size = globalpoints.size();
    auto pvol       = in_states[0]->Top();
    auto lvol       = pvol->GetLogicalVolume();
    // loop over all tracks in chunks
    auto i             = decltype(size){0};
    constexpr auto kVS = vecCore::VectorSize<Real_v>();
    for (; i < (size - (kVS - 1)); i += kVS) {
      NavigateAChunkNoReloc<Real_v, kVS>(this, pvol, lvol, globalpoints, globaldirs, step_limits, in_states, out_steps,
                                         calcsafeties, out_safeties, i);
    }

    // fall back to scalar interface for tail treatment
    for (; i < size; ++i) {
      out_steps[i] = ((Impl *)this)
                         ->Impl::ComputeStepAndSafety(globalpoints[i], globaldirs[i], step_limits[i],
                                                      *const_cast<NavigationState *>(in_states[i]), calcsafeties[i],
                                                      out_safeties[i]);
    }
    i = decltype(size){0};
    for (; i < size; ++i)
      out_steps[i] = vecCore::math::Min(out_steps[i], step_limits[i]);
  }

  // this kernel is a generic implementation to navigate with chunks of data
  // can be used also for the scalar implementation
  template <typename T, unsigned int ChunkSize>
  static void NavigateAChunk(VNavigator const *__restrict__ nav, VPlacedVolume const *__restrict__ pvol,
                             LogicalVolume const *__restrict__ lvol, SOA3D<Precision> const &__restrict__ globalpoints,
                             SOA3D<Precision> const &__restrict__ globaldirs, Precision const *__restrict__ step_limits,
                             NavigationState const* const* __restrict__ in_states, NavigationState** __restrict__ out_states,
                             Precision *__restrict__ out_steps, unsigned int from_index)
  {

    VPlacedVolume const *hitcandidates[ChunkSize] = {}; // initialize all to nullptr

    Vector3D<T> localpoint, localdir;
    Impl::template DoGlobalToLocalTransformations<T, ChunkSize>(in_states, globalpoints, globaldirs, from_index,
                                                                localpoint, localdir);

    T slimit(vecCore::FromPtr<T>(step_limits + from_index));
    if (MotherIsConvex) {
      vecCore::Store(slimit, out_steps + from_index);
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, out_states,
                                                               from_index, out_steps, hitcandidates);
      // parse the hitcandidates pointer as double to apply a mask
      T step(vecCore::FromPtr<T>(out_steps + from_index));
      auto nothitdaughter = step == T(kInfLength);
      if (!vecCore::MaskEmpty(nothitdaughter)) {
        auto dout = Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit);
        vecCore::MaskedAssign(step, nothitdaughter, dout);
        vecCore::Store(step, out_steps + from_index);
      }

    } else {
      // need to calc DistanceToOut first
      T step = Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit);
      vecCore::Store(step, out_steps + from_index);

      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, out_states,
                                                               from_index, out_steps, hitcandidates);
    }

    // fix state ( seems to be serial so we iterate over indices )
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      bool done;
      out_steps[trackid] = Impl::PrepareOutState(*in_states[trackid], *out_states[trackid], out_steps[trackid],
                                                 vecCore::LaneAt(slimit, i), hitcandidates[i], done);
      if (done) continue;
      // step was physics limited
      if (!out_states[trackid]->IsOnBoundary()) continue;
      // otherwise if necessary do a relocation
      // try relocation to refine out_state to correct location after the boundary
      using vecCore::LaneAt;
      ((Impl *)nav)
          ->Impl::Relocate(
              MovePointAfterBoundary(
                  Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i), LaneAt(localpoint.z(), i)),
                  Vector3D<Precision>(LaneAt(localdir.x(), i), LaneAt(localdir.y(), i), LaneAt(localdir.z(), i)),
                  out_steps[trackid]),
              *in_states[trackid], *out_states[trackid]);
    }
  }

  // this kernel is a generic implementation to navigate with chunks of data
  // can be used also for the scalar imple
  // same as above but including safety treatment
  // TODO: remerge with above to avoid code duplication
  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void NavigateAChunk(VNavigator const *__restrict__ nav, VPlacedVolume const *__restrict__ pvol,
                             LogicalVolume const *__restrict__ lvol, SOA3D<Precision> const &__restrict__ globalpoints,
                             SOA3D<Precision> const &__restrict__ globaldirs, Precision const *__restrict__ step_limits,
                             NavigationState const *const *__restrict__ in_states, NavigationState **__restrict__ out_states,
                             Precision *__restrict__ out_steps, bool const *__restrict__ calcsafeties,
                             Precision *__restrict__ out_safeties, unsigned int from_index)
  {

    VPlacedVolume const *hitcandidates[ChunkSize] = {}; // initialize all to nullptr

    Vector3D<T> localpoint, localdir;
    Impl::template DoGlobalToLocalTransformations<T, ChunkSize>(in_states, globalpoints, globaldirs, from_index,
                                                                localpoint, localdir);

    // safety part
    Impl::template SafetyLooper<T, ChunkSize>(nav, pvol, localpoint, from_index, calcsafeties, out_safeties);

    T slimit(vecCore::FromPtr<T>(step_limits + from_index)); // will only work with new ScalarWrapper
    if (MotherIsConvex) {
      vecCore::Store(slimit, out_steps + from_index);
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, out_states,
                                                               from_index, out_steps, hitcandidates);
      // parse the hitcandidates pointer as double to apply a mask
      T step(vecCore::FromPtr<T>(out_steps + from_index));
      auto nothitdaughter = step == T(kInfLength);
      if (!vecCore::MaskEmpty(nothitdaughter)) {
        vecCore__MaskedAssignFunc(step, nothitdaughter,
                                  Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit));
        vecCore::Store(step, out_steps + from_index);
      }
    } else {
      // need to calc DistanceToOut first
      T step = Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit);
      vecCore::Store(step, out_steps + from_index);

      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, out_states,
                                                               from_index, out_steps, hitcandidates);
    }

    using vecCore::LaneAt;

    // fix state ( seems to be serial so we iterate over indices )
    for (unsigned int i = 0; i < ChunkSize; ++i) {
      unsigned int trackid = from_index + i;
      bool done;
      out_steps[trackid] = Impl::PrepareOutState(*in_states[trackid], *out_states[trackid], out_steps[trackid],
                                                 LaneAt(slimit, i), hitcandidates[i], done);
      if (done) continue;
      // step was physics limited
      if (!out_states[trackid]->IsOnBoundary()) continue;
      // otherwise if necessary do a relocation
      // try relocation to refine out_state to correct location after the boundary
      ((Impl *)nav)
          ->Impl::Relocate(
              MovePointAfterBoundary(
                  Vector3D<Precision>(LaneAt(localpoint.x(), i), LaneAt(localpoint.y(), i), LaneAt(localpoint.z(), i)),
                  Vector3D<Precision>(LaneAt(localdir.x(), i), LaneAt(localdir.y(), i), LaneAt(localdir.z(), i)),
                  out_steps[trackid]),
              *in_states[trackid], *out_states[trackid]);
    }
  }

  // this kernel is a generic implementation to navigate with chunks of data
  // can be used also for the scalar imple
  // same as above but including safety treatment
  // TODO: remerge with above to avoid code duplication
  template <typename T, unsigned int ChunkSize>
  VECCORE_ATT_HOST_DEVICE
  static void NavigateAChunkNoReloc(VNavigator const *__restrict__ nav, VPlacedVolume const *__restrict__ pvol,
                                    LogicalVolume const *__restrict__ lvol,
                                    SOA3D<Precision> const &__restrict__ globalpoints,
                                    SOA3D<Precision> const &__restrict__ globaldirs,
                                    Precision const *__restrict__ step_limits,
                                    NavigationState const *const *__restrict__ in_states, Precision *__restrict__ out_steps,
                                    bool const *__restrict__ calcsafeties, Precision *__restrict__ out_safeties,
                                    unsigned int from_index)
  {
    VPlacedVolume const *hitcandidates[ChunkSize] = {}; // initialize all to nullptr

    Vector3D<T> localpoint, localdir;
    Impl::template DoGlobalToLocalTransformations<T, ChunkSize>(in_states, globalpoints, globaldirs, from_index,
                                                                localpoint, localdir);

    // safety part
    Impl::template SafetyLooper<T, ChunkSize>(nav, pvol, localpoint, from_index, calcsafeties, out_safeties);

    T slimit(vecCore::FromPtr<T>(step_limits + from_index));
    if (MotherIsConvex) {
      vecCore::Store(slimit, out_steps + from_index);
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, from_index,
                                                               out_steps, hitcandidates);
      // parse the hitcandidates pointer as double to apply a mask
      T step(vecCore::FromPtr<T>(out_steps + from_index));
      auto nothitdaughter = step == T(kInfLength);
      if (!vecCore::MaskEmpty(nothitdaughter)) {
        vecCore__MaskedAssignFunc(step, nothitdaughter,
                                  Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit));
        vecCore::Store(step, out_steps + from_index);
      }
    } else {
      // need to calc DistanceToOut first
      T step = Impl::template TreatDistanceToMother<T>(pvol, localpoint, localdir, slimit);
      vecCore::Store(step, out_steps + from_index);

      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      Impl::template DaughterIntersectionsLooper<T, ChunkSize>(nav, lvol, localpoint, localdir, in_states, from_index,
                                                               out_steps, hitcandidates);
    }
  }

  // generic implementation for the vector interface
  // this implementation tries to process everything in vector CHUNKS
  // at the very least this enables at least the DistanceToOut call to be vectorized
  virtual void ComputeStepsAndPropagatedStates(SOA3D<Precision> const &__restrict__ globalpoints,
                                               SOA3D<Precision> const &__restrict__ globaldirs,
                                               Precision const *__restrict__ step_limit,
                                               NavigationState const *const *__restrict__ in_states,
                                               NavigationState **__restrict__ out_states,
                                               Precision *__restrict__ out_steps) const override
  {

    // process SIMD part and TAIL part
    // something like
    using Real_v = vecgeom::VectorBackend::Real_v;
    (void)(Real_v *)(nullptr); // Avoid spurrious warning: unused type alias 'Real_v' [-Wunused-local-typedef]
    const auto size = globalpoints.size();
    auto pvol       = in_states[0]->Top();
    auto lvol       = pvol->GetLogicalVolume();
    // loop over all tracks in chunks
    auto i             = decltype(size){0};
    constexpr auto kVS = vecCore::VectorSize<Real_v>();
    for (; i < (size - (kVS - 1)); i += kVS) {
      NavigateAChunk<Real_v, kVS>(this, pvol, lvol, globalpoints, globaldirs, step_limit, in_states, out_states,
                                  out_steps, i);
    }

    // fall back to scalar interface for tail treatment
    for (; i < size; ++i) {
      out_steps[i] = ((Impl *)this)
                         ->Impl::ComputeStepAndPropagatedState(globalpoints[i], globaldirs[i], step_limit[i],
                                                               *in_states[i], *out_states[i]);
    }
  }

  // generic implementation for the vector interface -- using NavStatePool instead of NavigationState**
  // this implementation tries to process everything in vector CHUNKS
  // at the very least this enables at least the DistanceToOut call to be vectorized
  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafetiesAndPropagatedStates(
      SOA3D<Precision> const &__restrict__ globalpoints, SOA3D<Precision> const &__restrict__ globaldirs,
      Precision const *__restrict__ step_limit, NavStatePool const &in_states, NavStatePool &out_states,
      Precision *__restrict__ out_steps,
      bool const *__restrict__ calcsafeties, Precision *__restrict__ out_safeties) const override
  {
    // setup navstate** arrays -- must be cleaned up before returning!
    const NavigationState **inarray = nullptr;
    in_states.ToPlainPointerArray(inarray);
    NavigationState **outarray = nullptr;
    out_states.ToPlainPointerArray(outarray);

    // process SIMD part and TAIL part
    // something like
    const auto size = globalpoints.size();
    auto pvol       = in_states[0]->Top();
    auto lvol       = pvol->GetLogicalVolume();
    // loop over all tracks in chunks
    auto i             = decltype(size){0};
    constexpr auto kVS = vecCore::VectorSize<Real_v>();
    for (; i < (size - (kVS - 1)); i += kVS) {
      NavigateAChunk<Real_v, kVS>(this, pvol, lvol, globalpoints, globaldirs, step_limit, inarray, outarray,
                                  out_steps, calcsafeties, out_safeties, i);
    }
    // save results back -- could benefit from internal vectorization
    for (i = 0; i < (size - (kVS - 1)); ++i) {
      *out_states[i] = *outarray[i];
    }

    // fall back to scalar interface for tail treatment
    for (; i < size; ++i) {
      out_steps[i] = ((Impl *)this)
         ->Impl::ComputeStepAndSafetyAndPropagatedState(globalpoints[i], globaldirs[i], step_limit[i],
                                                        *in_states[i], *out_states[i], calcsafeties[i],
                                                        out_safeties[i]);
    }

    // cleanup
    delete[] inarray;
    delete[] outarray;
  }

  // generic implementation for the vector interface
  // this implementation tries to process everything in vector CHUNKS
  // at the very least this enables at least the DistanceToOut call to be vectorized
  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafetiesAndPropagatedStates(
      SOA3D<Precision> const &__restrict__ globalpoints, SOA3D<Precision> const &__restrict__ globaldirs,
      Precision const *__restrict__ step_limit, NavigationState const *const *__restrict__ in_states,
      NavigationState **__restrict__ out_states, Precision *__restrict__ out_steps,
      bool const *__restrict__ calcsafeties, Precision *__restrict__ out_safeties) const override
  {

    // process SIMD part and TAIL part
    // something like
    using Real_v = vecgeom::VectorBackend::Real_v;
    (void)(Real_v *)(nullptr); // Avoid spurrious warning: unused type alias 'Real_v' [-Wunused-local-typedef]
    const auto size = globalpoints.size();
    auto pvol       = in_states[0]->Top();
    auto lvol       = pvol->GetLogicalVolume();
    // loop over all tracks in chunks
    auto i             = decltype(size){0};
    constexpr auto kVS = vecCore::VectorSize<Real_v>();
    for (; i < (size - (kVS - 1)); i += kVS) {
      NavigateAChunk<Real_v, kVS>(this, pvol, lvol, globalpoints, globaldirs, step_limit, in_states, out_states,
                                  out_steps, calcsafeties, out_safeties, i);
    }
    // fall back to scalar interface for tail treatment
    for (; i < size; ++i) {
      out_steps[i] = ((Impl *)this)
                         ->Impl::ComputeStepAndSafetyAndPropagatedState(globalpoints[i], globaldirs[i], step_limit[i],
                                                                        *in_states[i], *out_states[i], calcsafeties[i],
                                                                        out_safeties[i]);
    }
  }

#ifdef TRIVIAL
  // another generic implementation for the vector interface
  // this implementation just loops over the scalar interface
  virtual void ComputeStepsAndPropagatedStates(SOA3D<Precision> const &__restrict__ globalpoints,
                                               SOA3D<Precision> const &__restrict__ globaldirs,
                                               Precision const *__restrict__ step_limit,
                                               NavigationState const *const *__restrict__ in_states,
                                               NavigationState **__restrict__ out_states,
                                               Precision *__restrict__ out_steps) const override
  {
    for (unsigned int i = 0; i < globalpoints.size(); ++i) {
      out_steps[i] = ((Impl *)this)
                         ->Impl::ComputeStepAndPropagatedState(globalpoints[i], globaldirs[i], step_limit[i],
                                                               *in_states[i], *out_states[i]);
    }
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void ComputeStepsAndSafetiesAndPropagatedStates(
      SOA3D<Precision> const &__restrict__ globalpoints, SOA3D<Precision> const &__restrict__ globaldirs,
      Precision const *__restrict__ step_limit, NavigationState const *const *__restrict__ in_states,
      NavigationState **__restrict__ out_states, Precision *__restrict__ out_steps,
      bool const *__restrict__ calcsafeties, Precision *__restrict__ out_safeties) const override
  {

    for (unsigned int i = 0; i < globalpoints.size(); ++i) {
      out_steps[i] = ((Impl *)this)
                         ->Impl::ComputeStepAndSafetyAndPropagatedState(globalpoints[i], globaldirs[i], step_limit[i],
                                                                        *in_states[i], *out_states[i], calcsafeties[i],
                                                                        out_safeties[i]);
    }
  }
#endif

  // a similar interface also returning the safety
  // TODO: reduce this evident code duplication with ComputeStepAndPropagatedState
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const &globalpoint,
                                                           Vector3D<Precision> const &globaldir, Precision step_limit,
                                                           NavigationState const &__restrict__ in_state,
                                                           NavigationState &__restrict__ out_state, bool calcsafety,
                                                           Precision &safety_out) const override
  {
    // calculate local point/dir from global point/dir
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    // get safety first ( the benefit here is that we reuse the local points )
    using SafetyE_t = typename Impl::SafetyEstimator_t;
    safety_out      = 0.;
    if (calcsafety) {
      // call the appropriate safety Estimator
      safety_out = ((SafetyE_t *)fSafetyEstimator)->SafetyE_t::ComputeSafetyForLocalPoint(localpoint, in_state.Top());
    }

    Precision step                    = step_limit;
    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();

    if (MotherIsConvex) {
      // if mother is convex we may not need to do treatment of mother
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
      if (hitcandidate == nullptr) step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    } else {
      // need to calc DistanceToOut first
      step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
    }

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) return step;

    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    // otherwise if necessary do a relocation
    // try relocation to refine out_state to correct location after the boundary
    ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
    return step;
  }

protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual void Relocate(Vector3D<Precision> const &pointafterboundary, NavigationState const &__restrict__ in_state,
                        NavigationState &__restrict__ out_state) const override
  {
    // this means that we are leaving the mother
    // alternatively we could use nextvolumeindex like before
    if (out_state.Top() == in_state.Top()) {
      GlobalLocator::RelocatePointFromPathForceDifferent(pointafterboundary, out_state);
#ifdef CHECK_RELOCATION_ERRORS
      assert(in_state.Distance(out_state) != 0 && " error relocating when leaving ");
#endif
    } else {
      // continue directly further down ( next volume should have been stored in out_state already )
      VPlacedVolume const *nextvol = out_state.Top();
      out_state.Pop();
      GlobalLocator::LocateGlobalPoint(nextvol, nextvol->GetTransformation()->Transform(pointafterboundary), out_state,
                                       false);
#ifdef CHECK_RELOCATION_ERRORS
      assert(in_state.Distance(out_state) != 0 && " error relocating when entering ");
#endif
      return;
    }
  }

public:
  static const char *GetClassName() { return Impl::gClassNameString; }

  virtual const char *GetName() const override { return GetClassName(); }
}; // end class VNavigatorHelper
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_VNAVIGATOR_H_ */
