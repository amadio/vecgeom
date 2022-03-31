/// \file BVHManager.h
/// \author Guilherme Amadio

#ifndef VECGEOM_MANAGEMENT_BVHMANAGER_H_
#define VECGEOM_MANAGEMENT_BVHMANAGER_H_

#include "VecGeom/base/BVH.h"
#include "VecGeom/volumes/LogicalVolume.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
inline std::vector<BVH *> hBVH;
} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECGEOM_ENABLE_CUDA
namespace cuda {
inline VECCORE_ATT_HOST_DEVICE BVH *dBVH;
}
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief The @c BVHManager class is a singleton class to manage the association between
 * logical volumes and their bounding volume hierarchies, using the logical volumes' ids.
 */

class BVHManager {
public:
  BVHManager() = delete;

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

#ifdef VECGEOM_CUDA_INTERFACE
  /** Initializes bounding volume hierarchies on the GPU. */
  static void DeviceInit();
#endif

  VECCORE_ATT_HOST_DEVICE
  static BVH const *GetBVH(int id)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return &cuda::dBVH[id];
#else
    return hBVH[id];
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  static BVH const *GetBVH(LogicalVolume const *v) { return GetBVH(v->id()); }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
