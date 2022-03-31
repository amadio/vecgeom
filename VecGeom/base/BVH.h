/// \file BVH.h
/// \author Guilherme Amadio

#ifndef VECGEOM_BASE_BVH_H_
#define VECGEOM_BASE_BVH_H_

#include "VecGeom/base/AABB.h"
#include "VecGeom/navigation/NavStateIndex.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

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
  /*
   * BVH::ComputeDaughterIntersections() computes the intersection of a ray against all children of
   * the logical volume. A stack is kept of the node ids that need to be checked. It needs to be at
   * most as deep as the binary tree itself because we always first pop the current node, and then
   * add at most the two children. For example, for depth two, we pop the root node, then at most we
   * add both of its leaves onto the stack to be checked. We initialize ptr with &stack[1] such that
   * when we pop the first time as we enter the loop, the position we read from is the first position
   * of the stack, which contains the id 0 for the root node. When we pop the stack such that ptr
   * points before &stack[0], it means we've checked all we needed and the loop can be terminated.
   * In order to determine if a node of the tree is internal or not, we check if the node id of its
   * left child is past the end of the array (in which case we know we are at the maximum depth), or
   * if the sum of children in both leaves is the same as in the current node, as for leaf nodes, the
   * sum of children in the left+right child nodes will be less than for the current node.
   */
  VECCORE_ATT_HOST_DEVICE
  void CheckDaughterIntersections(Vector3D<Precision> localpoint, Vector3D<Precision> localdir, Precision &step,
                                  VPlacedVolume const *last, VPlacedVolume const *&hitcandidate) const
  {
    unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

    /* Calculate and reuse inverse direction to save on divisions */
    Vector3D<Precision> invdir(1.0 / NonZero(localdir[0]), 1.0 / NonZero(localdir[1]), 1.0 / NonZero(localdir[2]));

    do {
      unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

      if (fNChild[id] >= 0) {
        /* For leaf nodes, loop over children */
        for (int i = 0; i < fNChild[id]; ++i) {
          int prim = fPrimId[fOffset[id] + i];
          /* Check AABB first, then the volume itself if needed */
          if (fAABBs[prim].IntersectInvDir(localpoint, invdir, step)) {
            auto vol  = fLV.GetDaughters()[prim];
            auto dist = vol->DistanceToIn(localpoint, localdir, step);
            /* If distance to current child is smaller than current step, update step and hitcandidate */
            if (dist < step && !(dist <= 0.0 && vol == last)) step = dist, hitcandidate = vol;
          }
        }
      } else {
        unsigned int childL = 2 * id + 1;
        unsigned int childR = 2 * id + 2;

        /* For internal nodes, check AABBs to know if we need to traverse left and right children */
        Precision tminL = kInfLength, tmaxL = -kInfLength, tminR = kInfLength, tmaxR = -kInfLength;

        fNodes[childL].ComputeIntersectionInvDir(localpoint, invdir, tminL, tmaxL);
        fNodes[childR].ComputeIntersectionInvDir(localpoint, invdir, tminR, tmaxR);

        bool traverseL = tminL <= tmaxL && tmaxL >= 0.0 && tminL < step;
        bool traverseR = tminR <= tmaxR && tmaxR >= 0.0 && tminR < step;

        /*
         * If both left and right nodes need to be checked, check closest one first.
         * This ensures step gets short as fast as possible so we can skip more nodes without checking.
         */
        if (tminR < tminL) {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        } else {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        }
      }
    } while (ptr > stack);
  }

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
  /*
   * BVH::ComputeSafety is very similar to the method above regarding traversal of the tree, but it
   * computes only the safety instead of the intersection using a ray, so the logic is a bit simpler.
   */
  VECCORE_ATT_HOST_DEVICE
  Precision ComputeSafety(Vector3D<Precision> localpoint, Precision safety) const
  {
    unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

    do {
      unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Safety(localpoint) < safety) {
            Precision dist = fLV.GetDaughters()[prim]->SafetyToIn(localpoint);
            if (dist < safety) safety = dist;
          }
        }
      } else {
        unsigned int childL = 2 * id + 1;
        unsigned int childR = 2 * id + 2;

        Precision safetyL = fNodes[childL].Safety(localpoint);
        Precision safetyR = fNodes[childR].Safety(localpoint);

        bool traverseL = safetyL < safety;
        bool traverseR = safetyR < safety;

        if (safetyR < safetyL) {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        } else {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        }
      }
    } while (ptr > stack);

    return safety;
  }

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] daughterpvol Placed volume in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of @p daughterpvol
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const
  {
    VPlacedVolume const *exclvol = nullptr;
    return LevelLocate(exclvol, localpoint, pvol, daughterlocalpoint);
  }

  /**
   * Find child volume inside which the given point @p localpoint is located.
   * @param[in] localpoint Point in the local coordinates of the logical volume.
   * @param[out] outstate Navigation state. Gets updated if point is relocated to another volume.
   * @param[out] daughterlocalpoint Point in the local coordinates of newly located volume.
   * @returns Whether @p localpoint falls within a child volume of @p lvol.
   */
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocate(Vector3D<Precision> const &localpoint, NavigationState &state,
                   Vector3D<Precision> &daughterlocalpoint) const
  {
    VPlacedVolume const *exclvol = nullptr;
    VPlacedVolume const *pvol    = nullptr;
    bool Result                  = LevelLocate(exclvol, localpoint, pvol, daughterlocalpoint);
    if (Result) {
      state.Push(pvol);
    }
    return Result;
  }

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
                   Vector3D<Precision> &daughterlocalpoint) const
  {
    unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

    do {
      unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Contains(localpoint)) {
            auto vol = fLV.GetDaughters()[prim];
            if (vol != exclvol && vol->Contains(localpoint, daughterlocalpoint)) {
              pvol = vol;
              return true;
            }
          }
        }
      } else {
        unsigned int childL = 2 * id + 1;
        if (fNodes[childL].Contains(localpoint)) *ptr++ = childL;

        unsigned int childR = 2 * id + 2;
        if (fNodes[childR].Contains(localpoint)) *ptr++ = childR;
      }
    } while (ptr > stack);

    return false;
  }

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
                   Vector3D<Precision> &daughterlocalpoint) const
  {
    unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

    do {
      unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Contains(localpoint)) {
            auto v = fLV.GetDaughters()[prim];

            if (v == exclvol) continue;

            const auto T = v->GetTransformation();
            const auto u = v->GetUnplacedVolume();
            const auto p = T->Transform(localpoint);

            auto Entering = [&]() {
              Vector3D<Precision> normal, dir = T->TransformDirection(localdirection);
              u->Normal(p, normal);
              return Vector3D<Precision>::Dot(normal, dir) < 0.0;
            };

            auto inside = u->Inside(p);

            if (inside == kInside || (inside == kSurface && Entering())) {
              pvol = v, daughterlocalpoint = p;
              return true;
            }
          }
        }
      } else {
        unsigned int childL = 2 * id + 1;
        if (fNodes[childL].Contains(localpoint)) *ptr++ = childL;

        unsigned int childR = 2 * id + 2;
        if (fNodes[childR].Contains(localpoint)) *ptr++ = childR;
      }
    } while (ptr > stack);

    return false;
  }

private:
  enum class ConstructionAlgorithm : unsigned int;
  /**
   * Compute internal nodes of the BVH recursively.
   * @param[in] id Node id of node to be computed.
   * @param[in] first Iterator pointing to the position of this node's first volume in @c fPrimId.
   * @param[in] last Iterator pointing to the position of this node's last volume in @c fPrimId.
   * @param[in] nodes Number of nodes for this BVH.
   * @param[in] constructionAlgorithm Index of the splitting function to use.
   *
   * @remark This function computes the bounding box of a node, then chooses a split plane and reorders
   * the elements of @c fPrimId within the first,last range such that volumes for its left child all come
   * before volumes for its right child, then launches itself to compute bounding boxes of each child.
   * Recursion stops when all children lie on one side of the splitting plane, or when the current node
   * contains only a single child volume.
   */
  void ComputeNodes(unsigned int id, int *first, int *last, unsigned int nodes, ConstructionAlgorithm);

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
