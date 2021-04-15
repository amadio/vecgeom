// \file BVHLevelLocator.h
// \author Guilherme Amadio

#ifndef VECGEOM_NAVIGATION_BVHLEVELLOCATOR_H_
#define VECGEOM_NAVIGATION_BVHLEVELLOCATOR_H_

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/navigation/VLevelLocator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

/**
 * @brief Level locator class using the bounding volume hierarchy of each logical volume for acceleration.
 */

class BVHLevelLocator : public VLevelLocator {
private:
  /** Constructor. Private since this is a singleton class accessed only via the @c Instance() static method. */
  BVHLevelLocator() = default;

public:
  /** Returns the instance of this singleton class. */
  static VLevelLocator const *GetInstance()
  {
    static BVHLevelLocator instance;
    return &instance;
  }

  /** Returns the name of this class. */
  std::string GetName() const final { return "BVHLevelLocator"; }

  /**
   * Find child volume of @p lvol inside which the given point @p localpoint is located.
   * @param[in] lvol Logical volume being checked.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] daughterpvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const final
  {
    return BVHManager::GetBVH(lvol).LevelLocate(localpoint, pvol, daughterlocalpoint);
  }

  /**
   * Find child volume of @p lvol inside which the given point @p localpoint is located.
   * @param[in] lvol Logical volume being checked.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] outstate Navigation state. Gets updated if point is relocated to another volume.
   * @param[out] daughterlocalpoint Point in the local coordinates of newly located volume.
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &state,
                   Vector3D<Precision> &daughterlocalpoint) const final
  {
    return BVHManager::GetBVH(lvol).LevelLocate(localpoint, state, daughterlocalpoint);
  }

  /**
   * Find child volume of @p lvol inside which the given point @p localpoint is located.
   * @param[in] lvol Logical volume being checked.
   * @param[in] exclvol Placed volume that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] pvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                          Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                          Vector3D<Precision> &daughterlocalpoint) const final
  {
    return BVHManager::GetBVH(lvol).LevelLocate(exclvol, localpoint, pvol, daughterlocalpoint);
  }

  /**
   * Find child volume of @p lvol inside which the given point @p localpoint is located.
   * @param[in] lvol Logical volume being checked.
   * @param[in] exclvol Placed volume that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] localdir Direction in the local coordinates of the logical volume.
   * @param[out] pvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                          Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdirection,
                          VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint) const final
  {
    return BVHManager::GetBVH(lvol).LevelLocate(exclvol, localpoint, localdirection, pvol, daughterlocalpoint);
  }
}; // end class declaration

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
