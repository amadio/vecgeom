/// \file BVHManager.h
/// \author Guilherme Amadio

#ifndef VECGEOM_MANAGEMENT_BVHMANAGER_H_
#define VECGEOM_MANAGEMENT_BVHMANAGER_H_

#include "VecGeom/base/BVH.h"
#include "VecGeom/volumes/LogicalVolume.h"

#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief The @c BVHManager class is a singleton class to manage the association between
 * logical volumes and their bounding volume hierarchies, using the logical volumes' ids.
 */

class BVHManager {
private:
  /** Constructor. Private since this is a singleton class accessed only via the @c Instance() static method. */
  BVHManager() = default;

public:
  /** Returns a reference to this singleton's instance. */
  static BVHManager &Instance()
  {
    static BVHManager instance;
    return instance;
  }

  /**
   * Initializes the bounding volume hierarchies for all logical volumes in the geometry.
   * Since it uses the ABBoxManager to fetch the pre-computed bounding boxes for each logical volume,
   * it must be called after the bounding boxes have already been computed. The depth is not specified,
   * to allow the BVH class to choose the depth dynamically based on the number of children of each
   * logical volume. This function is called automatically when the geometry is closed.
   *
   * The BVHManager assumes all volumes have an associated BVH, but only BVHs for volumes whose
   * navigator is set to the BVHNavigator are actually accessed at runtime.
   */
  static void Init();

  static BVH const &GetBVH(LogicalVolume const *v) { return *(Instance().fBVHs[v->id()]); }

private:
  std::vector<BVH *> fBVHs; ///< Vector of @c BVH instances for each logical volume.
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
