/// \file BVHNavigator.h
/// \author Guilherme Amadio

#ifndef VECGEOM_NAVIGATION_BVHNAVIGATOR_H_
#define VECGEOM_NAVIGATION_BVHNAVIGATOR_H_

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/navigation/BVHSafetyEstimator.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Navigator class using the bounding volume hierarchy of each logical volume for acceleration.
 */

template <bool MotherIsConvex = false>
class BVHNavigator : public VNavigatorHelper<BVHNavigator<MotherIsConvex>, MotherIsConvex> {
private:
  /** Constructor. Private since this is a singleton class accessed only via the @c Instance() static method. */
  BVHNavigator()
      : VNavigatorHelper<BVHNavigator<MotherIsConvex>, MotherIsConvex>(BVHSafetyEstimator::Instance())
  {
  }

public:
  using SafetyEstimator_t = BVHSafetyEstimator;

  static constexpr const char *gClassNameString = "BVHNavigator";

  /** Returns the instance of this singleton class. */
  static VNavigator *Instance()
  {
    static BVHNavigator instance;
    return &instance;
  }

  /**
   * Checks for intersections against child volumes of logical volume @p lvol, using the BVH
   * associated with it.
   * @param[in] lvol Logical volume being checked.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] localdir Direction in the local coordinates of the logical volume.
   * @param[in] in_state Incoming navigation state.
   * @param[in] out_state Outgoing navigation state (not used by this method).
   * @param[in] step Maximum step size. Volumes beyond this distance are ignored.
   * @param[out] hitcandidate
   * @returns Whether @p out_state has been modified or not. Always false for this method.
   */
  bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /* out_state */, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const final
  {
    VPlacedVolume const *last = in_state ? in_state->GetLastExited() : nullptr;
    BVHManager::GetBVH(lvol)->CheckDaughterIntersections(localpoint, localdir, step, last, hitcandidate);
    return false; /* return value indicates whether out_state has been modified */
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
