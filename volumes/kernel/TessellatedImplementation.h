//===-- kernel/TessellatedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TessellatedImplementation.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/TessellatedStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TessellatedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TessellatedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated;
template <typename T>
class TessellatedStruct;
class UnplacedTessellated;

struct TessellatedImplementation {

  using PlacedShape_t    = PlacedTessellated;
  using UnplacedStruct_t = TessellatedStruct<double>;
  using UnplacedVolume_t = UnplacedTessellated;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //  printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedTessellated<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    inside = Bool_v(false);
    int isurf;
    Real_v stepMax = InfinityLength<Real_v>();
    Real_v distOut, distIn;
    DistanceToSolid<Real_v, false>(tessellated, point, tessellated.fTestDir, stepMax, distOut, isurf);
    // If distance to out is infinite the point is outside
    if (distOut >= stepMax) return;

    DistanceToSolid<Real_v, true>(tessellated, point, tessellated.fTestDir, stepMax, distIn, isurf);
    // If distance to out is finite and less than distance to in, the point is inside
    if (distOut < distIn) inside = Bool_v(true);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    inside = Inside_v(kOutside);
    int isurf;
    Real_v stepMax = InfinityLength<Real_v>();
    Real_v distOut, distIn;
    DistanceToSolid<Real_v, false>(tessellated, point, tessellated.fTestDir, stepMax, distOut, isurf);
    // If distance to out is infinite the point is outside
    if (distOut >= stepMax) return;
    if (distOut < 0 || distOut * tessellated.fTestDir.Dot(tessellated.fFacets[isurf]->fNormal) < kTolerance) {
      inside = Inside_v(kSurface);
      return;
    }

    DistanceToSolid<Real_v, true>(tessellated, point, tessellated.fTestDir, stepMax, distIn, isurf);
    // If distance to out is finite and less than distance to in, the point is inside
    if (distOut < distIn) {
      inside = Inside_v(kInside);
      return;
    }
    if (distIn < 0 || distIn * tessellated.fTestDir.Dot(tessellated.fFacets[isurf]->fNormal) > -kTolerance)
      inside = Inside_v(kSurface);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf;
    DistanceToSolid<Real_v, true>(tessellated, point, direction, stepMax, distance, isurf);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf;
    DistanceToSolid<Real_v, false>(tessellated, point, direction, stepMax, distance, isurf);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    int isurf;
    Real_v safetysq = SafetySq<Real_v, true>(tessellated, point, isurf);
    safety          = vecCore::math::Sqrt(safetysq);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    int isurf;
    Real_v safetysq = SafetySq<Real_v, false>(tessellated, point, isurf);
    safety          = vecCore::math::Sqrt(safetysq);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    valid = true;
    int isurf;
    // We may need to check the value of safety to declare the validity of the normal
    SafetySq<Real_v, false>(tessellated, point, isurf);
    return tessellated.fFacets[isurf]->fNormal;
  }

  template <typename Real_v, bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToSolid(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                              Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance, int &isurf)
  {
    // Common method providing DistanceToIn/Out functionality
    // Check if the bounding box is hit
    Vector3D<Real_v> invdir(1. / direction.x(), 1. / direction.y(), 1. / direction.z());
    Vector3D<int> sign;
    sign[0]  = invdir.x() < 0;
    sign[1]  = invdir.y() < 0;
    sign[2]  = invdir.z() < 0;
    distance = BoxImplementation::IntersectCachedKernel2<Real_v, Real_v>(
        &tessellated.fMinExtent, point, invdir, sign.x(), sign.y(), sign.z(), -kTolerance, InfinityLength<Real_v>());
    if (distance >= InfinityLength<Real_v>()) return;

    // Define the user hook calling DistanceToIn for the cluster with the same
    // index as the bounding box
    distance      = InfinityLength<Real_v>();
    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the distance to the current box is bigger than the
      // requested limit or than the current distance
      if (hitbox.second > vecCore::math::Min(stepMax, distance)) return true;
      // Compute distance to the cluster
      Real_v distcrt;
      int isurfcrt;
      if (ToIn)
        tessellated.fClusters[hitbox.first]->DistanceToIn(point, direction, stepMax, distcrt, isurfcrt);
      else
        tessellated.fClusters[hitbox.first]->DistanceToOut(point, direction, stepMax, distcrt, isurfcrt);
      if (distcrt < distance) {
        distance = distcrt;
        isurf    = tessellated.fClusters[hitbox.first]->fIfacets[isurfcrt];
      }
      // ntries++;
      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    // intersect ray with the BVH structure and use hook
    boxNav->BVHSortedIntersectionsLooper(*tessellated.fNavHelper, point, direction, userhook);
  }

  template <typename Real_v, bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  static Real_v SafetySq(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, int &isurf)
  {
    Real_v safetysq = InfinityLength<Real_v>();

    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the safety to the current cluster is bigger than the
      // current safety
      if (hitbox.second > safetysq) return true;
      // Compute distance to the cluster
      int isurfcrt;
      Real_v safetycrt = tessellated.fClusters[hitbox.first]->template SafetySq<ToIn>(point, isurfcrt);
      if (safetycrt < safetysq) {
        safetysq = safetycrt;
        isurf    = isurfcrt;
      }
      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    // Use the BVH structure and connect hook
    boxNav->BVHSortedSafetyLooper(*tessellated.fNavHelper, point, userhook, safetysq);
    return safetysq;
  }

}; // end TessellatedImplementation
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
