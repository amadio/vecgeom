/// \file BVH.h
/// \author Guilherme Amadio

#ifndef VECGEOM_BASE_BVH_H_
#define VECGEOM_BASE_BVH_H_

#include "VecGeom/base/AABB.h"
#include "VecGeom/navigation/NavStateFwd.h"

#include <vector>

namespace vecgeom {
VECGEOM_DEVICE_FORWARD_DECLARE(class BVH;);
VECGEOM_DEVICE_DECLARE_CONV(class, BVH);
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

/**
 * @brief Bounding Volume Hierarchy class to represent an axis-aligned bounding volume hierarchy.
 * @details BVH instances can be associated with logical volumes to accelerate queries to their child volumes.
 */

class BVH {
public:
  /** Maximum depth. */
  static constexpr int BVH_MAX_DEPTH = 32;

  /**
   * Constructor.
   * @param volume Pointer to logical volume for which the BVH will be created.
   * @param depth Depth of the BVH binary tree. Defaults to zero, in which case
   * the actual depth will be chosen dynamically based on the number of child volumes.
   * When a fixed depth is chosen, it cannot be larger than @p BVH_MAX_DEPTH.
   */
  BVH(LogicalVolume const &volume, int depth = 0);

  /** Destructor. */
  ~BVH();

#ifdef VECGEOM_ENABLE_CUDA
  /**
   * Constructor for GPU. Takes as input pre-constructed BVH buffers.
   * @param volume  Reference to logical volume on the device
   * @param depth Depth of the BVH binary tree stored in the device buffers.
   * @param dPrimId Device buffer with child volume ids
   * @param dAABBs  Device buffer with AABBs of child volumes
   * @param dOffset Device buffer with offsets in @c dPrimId for first child of each BVH node
   * @param dNChild Device buffer with number of children for each BVH node
   * @param dNodes AABBs of BVH nodes
   */
  VECCORE_ATT_DEVICE
  BVH(LogicalVolume const *volume, int depth, int *dPrimId, AABB *dAABBs, int *dOffset, int *NChild, AABB *dNodes);
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  /** Copy and construct an instance of this BVH on the device, at the device address @p addr. */
  DevicePtr<cuda::BVH> CopyToGpu(void *addr) const;
#endif

  /** Print a summary of BVH contents */
  VECCORE_ATT_HOST_DEVICE
  void Print(bool verbose = false) const;

  /**
   * Check ray defined by <tt>localpoint + t * localdir</tt> for intersections with children
   * of the logical volume associated with the BVH, and within a maximum distance of @p step
   * along the ray, while ignoring the @p last volume.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] localdir Direction in the local coordinates of the logical volume.
   * @param[in,out] step Maximum step distance for which intersections should be considered.
   * @param[in] last Last volume. This volume is ignored when reporting intersections.
   * @param[out] hitcandidate Pointer to volume for which closest intersection was found.
   */
  VECCORE_ATT_HOST_DEVICE
  void CheckDaughterIntersections(Vector3D<Precision> localpoint, Vector3D<Precision> localdir, Precision &step,
                                  VPlacedVolume const *last, VPlacedVolume const *&hitcandidate) const;

  /**
   * Check ray defined by <tt>localpoint + t * localdir</tt> for intersections with bounding
   * boxes of children of the logical volume associated with the BVH, and within a maximum
   * distance of @p step along the ray. Returns the distance to the first crossed box.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] localdir Direction in the local coordinates of the logical volume.
   * @param[in,out] step Maximum step distance for which intersections should be considered.
   * @param[in] last Last volume. This volume is ignored when reporting intersections.
   */
  VECCORE_ATT_HOST_DEVICE
  void ApproachNextDaughter(Vector3D<Precision> localpoint, Vector3D<Precision> localdir, Precision &step,
                            VPlacedVolume const *last) const;

  /**
   * Compute safety against children of the logical volume associated with the BVH.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] safety Maximum safety. Volumes further than this are not checked.
   * @returns Minimum between safety to the closest child of logical volume and input @p safety.
   */
  VECCORE_ATT_HOST_DEVICE
  Precision ComputeSafety(Vector3D<Precision> localpoint, Precision safety) const;

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] daughterpvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const;

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] outstate Navigation state. Gets updated if point is relocated to another volume.
   * @param[out] daughterlocalpoint Point in the local coordinates of newly located volume.
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(Vector3D<Precision> const &localpoint, NavigationState &state,
                   Vector3D<Precision> &daughterlocalpoint) const;

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] exclvol Placed volume that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] pvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(VPlacedVolume const *exclvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const;

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] exclvol Placed volume that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[in] localdir Direction in the local coordinates of the logical volume.
   * @param[out] pvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(VPlacedVolume const *exclvol, Vector3D<Precision> const &localpoint,
                   Vector3D<Precision> const &localdirection, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const;

private:
  /**
   * Compute internal nodes of the BVH recursively.
   * @param[in] id Node id of node to be computed.
   * @param[in] first Iterator pointing to the position of this node's first volume in @c fPrimId.
   * @param[in] last Iterator pointing to the position of this node's last volume in @c fPrimId.
   * @param[in] nodes Number of nodes for this BVH.
   *
   * @remark This function computes the bounding box of a node, then chooses a split plane and reorders
   * the elements of @c fPrimId within the first,last range such that volumes for its left child all come
   * before volumes for its right child, then launches itself to compute bounding boxes of each child.
   * Recursion stops when all children lie on one side of the splitting plane, or when the current node
   * contains only a single child volume.
   */
  void ComputeNodes(unsigned int id, int *first, int *last, unsigned int nodes);

  LogicalVolume const &fLV; ///< Logical volume this BVH was constructed for
  int *fPrimId;             ///< Child volume ids for each BVH node
  int *fOffset;             ///< Offset in @c fPrimId for first child of each BVH node
  int *fNChild;             ///< Number of children for each BVH node
  AABB *fNodes;             ///< AABBs of BVH nodes
  AABB *fAABBs;             ///< AABBs of children of logical volume @c fLV
  int fDepth;               ///< Depth of the BVH
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
