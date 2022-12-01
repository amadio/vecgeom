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
  VECCORE_ATT_DEVICE
  BVHNavigator() : VNavigatorHelper<BVHNavigator<MotherIsConvex>, MotherIsConvex>(BVHSafetyEstimator::Instance()) {}

public:
  using SafetyEstimator_t = BVHSafetyEstimator;
  using Base              = VNavigatorHelper<BVHNavigator<MotherIsConvex>, MotherIsConvex>;
  using Base::CheckDaughterIntersections;

  static constexpr const char *gClassNameString = "BVHNavigator";

#ifndef VECCORE_CUDA
  /** Returns the instance of this singleton class. */
  static VNavigator *Instance()
  {
    static BVHNavigator instance;
    return &instance;
  }
#else
  // If used on device, this needs to be implemented in a .cu file rather than in this header
  // This hack is used also by NewSimpleNavigator, implemented in LogicalVolume.cpp
  // This is now implemented in BVHManager.cu
  VECCORE_ATT_DEVICE
  static VNavigator *Instance();
#endif

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
  VECCORE_ATT_HOST_DEVICE
  bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                  Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                  NavigationState * /* out_state */, Precision &step,
                                  VPlacedVolume const *&hitcandidate) const final
  {
    if (auto bvh = BVHManager::GetBVH(lvol)) {
      VPlacedVolume const *last = in_state ? in_state->GetLastExited() : nullptr;
      bvh->CheckDaughterIntersections(localpoint, localdir, step, last, hitcandidate);
    }
    return false; /* return value indicates whether out_state has been modified */
  }

  /// @brief Relocates the point on boundary after crossing.
  /// @param[in] pointafterboundary Propagated point on boundary, in the reference frame of in_state.Top().
  /// @param[in] in_state Mother volume being exited.
  /// @param[out] out_state State being exited, or daughter being entered.
  VECCORE_ATT_HOST_DEVICE
  void Relocate(Vector3D<Precision> const &pointafterboundary, NavigationState const &__restrict__ in_state,
                NavigationState &__restrict__ out_state) const final
  {
    // this means that we are leaving the mother
    // alternatively we could use nextvolumeindex like before
    if (out_state.Top() == in_state.Top()) {
      RelocatePointFromPathForceDifferent(pointafterboundary, out_state);
    } else {
      // continue directly further down ( next volume should have been stored in out_state already )
      VPlacedVolume const *nextvol = out_state.Top();
      out_state.Pop();
      LocateGlobalPoint(nextvol, nextvol->GetTransformation()->Transform(pointafterboundary), out_state, false);
      return;
    }
  }

  /// @brief Locate a point starting from a volume
  /// @param[in] vol Current volume to start the search from
  /// @param[in] point Point in current volume frame
  /// @param[in] path Navigation state pointing to the mother of vol
  /// @param[in] top Should the top volume be checked
  /// @param[in] exclude Volume excluded from search
  /// @return Deepest placed volume containing the point
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *LocateGlobalPoint(VPlacedVolume const *vol, Vector3D<Precision> const &point,
                                         NavigationState &path, bool top, VPlacedVolume const *exclude = nullptr) const
  {
    if (top) {
      assert(vol != nullptr);
      if (!vol->UnplacedContains(point)) return nullptr;
    }

    path.Push(vol);

    Vector3D<Precision> currentpoint(point);
    Vector3D<Precision> daughterlocalpoint;

    for (auto v = vol; v->GetDaughters().size() > 0;) {
      auto bvh = vecgeom::BVHManager::GetBVH(v->GetLogicalVolume()->id());

      if (!bvh->LevelLocate(exclude, currentpoint, v, daughterlocalpoint)) break;

      currentpoint = daughterlocalpoint;
      path.Push(v);
      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      exclude = nullptr;
    }

    return path.Top();
  }

  /// @brief Special version of locate point function that excludes searching a given volume
  /// (useful when we know that a particle must have traversed a boundary).
  /// @param vol Current volume to start the search from
  /// @param exclvol Volume to be excluded from search
  /// @param point Point in current volume frame
  /// @param path Navigation state pointing to the mother of vol
  /// @param top Should the top volume be checked
  /// @return Deepest placed volume containing the point
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *LocateGlobalPointExclVolume(VPlacedVolume const *vol, VPlacedVolume const *exclvol,
                                                   Vector3D<Precision> const &point, NavigationState &path,
                                                   bool top) const
  {
    VPlacedVolume const *candvolume = vol;
    Vector3D<Precision> currentpoint(point);
    if (top) {
      assert(vol != nullptr);
      candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
    }
    if (candvolume) {
      path.Push(candvolume);
      LogicalVolume const *lvol         = candvolume->GetLogicalVolume();
      Vector<Daughter> const *daughters = lvol->GetDaughtersp();

      bool godeeper = true;
      while (daughters->size() > 0 && godeeper) {
        // returns nextvolume; and transformedpoint; modified path
        Vector3D<Precision> transformedpoint;
        godeeper = BVHManager::GetBVH(lvol)->LevelLocate(exclvol, currentpoint, candvolume, transformedpoint);
        if (godeeper) {
          lvol         = candvolume->GetLogicalVolume();
          daughters    = lvol->GetDaughtersp();
          currentpoint = transformedpoint;
          path.Push(candvolume);
        }
      }
    }
    return candvolume;
  }

  /// @brief Relocation function called when exiting the current volume.
  /// @param[in] localpoint Point in current volume path coordinates
  /// @param path Path to volume being exited
  /// @return Location of point after exiting
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *RelocatePointFromPathForceDifferent(Vector3D<Precision> const &localpoint,
                                                           NavigationState &path) const
  {
    // idea: do the following:
    // ----- is localpoint still in current mother ? : then go down
    // if not: have to go up until we reach a volume that contains the
    // localpoint and then go down again (neglecting the volumes currently stored in the path)
    VPlacedVolume const *currentmother = path.Top();
    VPlacedVolume const *entryvol      = currentmother;

    if (currentmother != nullptr) {
      Vector3D<Precision> tmp = localpoint;
      while (currentmother) {
        if (currentmother == entryvol || currentmother->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly() ||
            !currentmother->UnplacedContains(tmp)) {
          path.Pop();
          Vector3D<Precision> pointhigherup = currentmother->GetTransformation()->InverseTransform(tmp);
          tmp                               = pointhigherup;
          currentmother                     = path.Top();
        } else {
          break;
        }
      }

      if (currentmother) {
        path.Pop();
        return LocateGlobalPointExclVolume(currentmother, entryvol, tmp, path, false);
      }
    }
    return currentmother;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
