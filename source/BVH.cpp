/// \file BVH.cpp
/// \author Guilherme Amadio

#include "VecGeom/base/BVH.h"

#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

constexpr int BVH::BVH_MAX_DEPTH;

static int ClosestAxis(Vector3D<Precision> v)
{
  v = v.Abs();
  return v[0] > v[2] ? (v[0] > v[1] ? 0 : 1) : (v[1] > v[2] ? 1 : 2);
}

/*
 * The BVH class is based on a complete binary tree stored contiguously in an array.
 *
 * For a given node id, 2*id + 1 gives the left child id, and 2*id + 2 gives the right child id.
 * For example, node 0 has its children at positions 1 and 2 in the vector, and for node 2, the
 * child nodes are at positions 5 and 6. The tree has 1 + 2 + 4 + ... + d nodes in total, where
 * d is the depth of the tree, or 1, 3, 7, ..., 2^d - 1 nodes in total. Visually, the ids for each
 * node look like shown below for a tree of depth 3:
 *
 *                                              0
 *                                             / \
 *                                            1   2
 *                                           / \ / \
 *                                          3  4 5  6
 *
 * with 2^3 - 1 = 7 nodes in total. For each node id, fNChild[id] gives the number of children of
 * the logical volume that belong to that node, and fOffset[id] gives the offset in the id map
 * fPrimId where the ids of the child volumes are stored, such that accessing fPrimId[fOffset[id]]
 * gives the first child id, then fPrimId[fOffset[id] + 1] gives the second child, up to fNChild[id]
 * children. The bounding boxes stored in fAABBs are in the original order, so they are accessed by
 * the original child number (i.e. the id stored in fPrimId, not by a node id of the tree itself).
 */

BVH::BVH(LogicalVolume const &volume, int depth) : fLV(volume)
{
  int n;

  /* ptr is a pointer to ndaughters times (min, max) corner vectors of each AABB */
  Vector3D<Precision> *ptr = ABBoxManager::Instance().GetABBoxes(&volume, n);

  assert(n > 0);

  fAABBs.resize(n);
  for (int i = 0; i < n; ++i)
    fAABBs[i] = AABB(ptr[2 * i], ptr[2 * i + 1]);

  /* Initialize map of primitive ids (i.e. child volume ids) as {0, 1, 2, ...}. */
  fPrimId.resize(n);
  std::iota(fPrimId.begin(), fPrimId.end(), 0);

  /*
   * If depth = 0, choose depth dynamically based on the number of child volumes, up to the fixed
   * maximum depth. We use n/2 here because that creates a tree with roughly one node for every two
   * volumes, or roughly at most 4 children per leaf node. For example, for 1000 volumes, the
   * default depth would be log2(500) = 8.96 -> 8, with 2^8 - 1 = 511 nodes, and 256 leaf nodes.
   */
  depth = std::min(depth ? depth : (int)std::log2(n / 2), BVH_MAX_DEPTH);

  int nodes = (2 << depth) - 1;

  fNodes.resize(nodes);
  fNChild.resize(nodes);
  fOffset.resize(nodes);

  /* Recursively initialize BVH nodes starting at the root node */
  ComputeNodes(0, fPrimId.begin(), fPrimId.end());

  /* Mark internal nodes with a negative number of children to simplify traversal */
  for (int id = 0; id < nodes / 2; ++id)
    if (fNChild[id] > 8 && (fNChild[id] == fNChild[2 * id + 1] + fNChild[2 * id + 2])) fNChild[id] = -1;
}

/*
 * BVH::ComputeNodes() initializes nodes of the BVH. It first computes the number of children that
 * belong to the current node based on the iterator range that is passed as input, as well as the
 * offset where the children of this node start. Then, it computes the overall bounding box of the
 * current node as the union of all bounding boxes of its child volumes. Then, if recursion should
 * continue, a splitting plane is chosen based on the longest dimension of the bounding box for the
 * current node, and the children are sorted such that all children on each side of the splitting
 * plane are stored contiguously. Then the function is called recursively with the iterator
 * sub-ranges for volumes on each side of the splitting plane to construct its left and right child
 * nodes. Recursion stops if a child node is deeper than the maximum depth, if the iterator range
 * is empty (i.e. no volumes on this node, maybe because all child volumes' centroids are on the
 * same side of the splitting plane), or if the node contains only a single volume.
 */

void BVH::ComputeNodes(unsigned int id, std::vector<int>::iterator first, std::vector<int>::iterator last)
{
  const Vector3D<Precision> basis[] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  if (id >= fNodes.size() || first == last) return;

  fNChild[id] = std::distance(first, last);
  fOffset[id] = std::distance(fPrimId.begin(), first);

  fNodes[id] = fAABBs[*first];
  for (auto it = std::next(first); it != last; ++it)
    fNodes[id] = AABB::Union(fNodes[id], fAABBs[*it]);

  if (std::next(first) == last) return;

  Vector3D<Precision> p = fNodes[id].Center();
  Vector3D<Precision> v = basis[ClosestAxis(fNodes[id].Size())];

  auto mid =
      std::partition(first, last, [&](size_t i) { return Vector3D<Precision>::Dot(fAABBs[i].Center() - p, v) < 0.0; });

  ComputeNodes(2 * id + 1, first, mid);
  ComputeNodes(2 * id + 2, mid, last);
}

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

void BVH::CheckDaughterIntersections(Vector3D<Precision> localpoint, Vector3D<Precision> localdir, Precision &step,
                                     VPlacedVolume const *last, VPlacedVolume const *&hitcandidate) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

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

      bool intersectL = tminL <= tmaxL && tmaxL >= 0.0 && tminL < step;
      bool intersectR = tminR <= tmaxR && tmaxR >= 0.0 && tminR < step;

      /*
       * If both left and right nodes need to be checked, check closest one first.
       * This ensures step gets short as fast as possible so we can skip more nodes without checking.
       */
      if (intersectL && intersectR && tminL > tminR) std::swap(childL, childR);

      /* Push node id of intersecting nodes onto the stack */
      if (intersectL) *ptr++ = childL;
      if (intersectR) *ptr++ = childR;
    }
  } while (ptr > stack);
}

/*
 * BVH::ComputeSafety is very similar to the method above regarding traversal of the tree, but it
 * computes only the safety instead of the intersection using a ray, so the logic is a bit simpler.
 */

Precision BVH::ComputeSafety(Vector3D<Precision> localpoint, Precision safety) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

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

      if (traverseL && traverseR && safetyR < safetyL) std::swap(childL, childR);

      if (traverseL) *ptr++ = childL;
      if (traverseR) *ptr++ = childR;
    }
  } while (ptr > stack);

  return safety;
}

bool BVH::LevelLocate(Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                      Vector3D<Precision> &daughterlocalpoint) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

  do {
    unsigned int id = *--ptr;

    if (fNChild[id] >= 0) {
      for (int i = 0; i < fNChild[id]; ++i) {
        int prim = fPrimId[fOffset[id] + i];
        if (fAABBs[prim].Contains(localpoint)) {
          auto vol = fLV.GetDaughters()[prim];
          if (vol->Contains(localpoint, daughterlocalpoint)) {
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

bool BVH::LevelLocate(Vector3D<Precision> const &localpoint, NavigationState &state,
                      Vector3D<Precision> &daughterlocalpoint) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

  do {
    unsigned int id = *--ptr;

    if (fNChild[id] >= 0) {
      for (int i = 0; i < fNChild[id]; ++i) {
        int prim = fPrimId[fOffset[id] + i];
        if (fAABBs[prim].Contains(localpoint)) {
          auto vol = fLV.GetDaughters()[prim];
          if (vol->Contains(localpoint, daughterlocalpoint)) {
            state.Push(vol);
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

bool BVH::LevelLocate(VPlacedVolume const *exclvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                      Vector3D<Precision> &daughterlocalpoint) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

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

bool BVH::LevelLocate(VPlacedVolume const *exclvol, Vector3D<Precision> const &localpoint,
                      Vector3D<Precision> const &localdirection, VPlacedVolume const *&pvol,
                      Vector3D<Precision> &daughterlocalpoint) const
{
  int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

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

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
