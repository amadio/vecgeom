//===-- kernel/TessellatedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TessellatedImplementation.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TessellatedStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TessellatedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TessellatedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated;
template <size_t NVERT, typename T>
class TessellatedStruct;
class UnplacedTessellated;

struct TessellatedImplementation {

  using PlacedShape_t    = PlacedTessellated;
  using UnplacedStruct_t = TessellatedStruct<3, Precision>;
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
    int isurfOut, isurfIn;
    Real_v distOut, distIn;
    DistanceToSolid<Real_v, false>(tessellated, point, tessellated.fTestDir, InfinityLength<Real_v>(), distOut,
                                   isurfOut, distIn, isurfIn);
    if (isurfOut >= 0) inside = Bool_v(true);
    /*
        DistanceToSolid<Real_v, true>(tessellated, point, tessellated.fTestDir, stepMax, distIn, isurf);
        // If distance to out is finite and less than distance to in, the point is inside
        if (distOut < distIn) inside = Bool_v(true);
    */
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Inside_v &inside)
  {
    inside = Inside_v(kOutside);
    int isurfOut, isurfIn;
    Real_v distOut, distIn;
    DistanceToSolid<Real_v, false>(tessellated, point, tessellated.fTestDir, InfinityLength<Real_v>(), distOut,
                                   isurfOut, distIn, isurfIn);
    // If no surface is hit then the point is outside
    if (isurfOut < 0) return;
    if (distOut < 0 || distOut * tessellated.fTestDir.Dot(tessellated.fFacets[isurfOut]->fNormal) < kTolerance) {
      inside = Inside_v(kSurface);
      return;
    }

    // DistanceToSolid<Real_v, true>(tessellated, point, tessellated.fTestDir, stepMax, distIn, isurf);
    // If distance to out is finite and less than distance to in, the point is inside
    if (isurfIn < 0 || distOut < distIn) {
      inside = Inside_v(kInside);
      return;
    }
    if (distIn < 0 || distIn * tessellated.fTestDir.Dot(tessellated.fFacets[isurfIn]->fNormal) > -kTolerance)
      inside = Inside_v(kSurface);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf, isurfOut;
    Real_v distOut;
    DistanceToSolid<Real_v, true>(tessellated, point, direction, stepMax, distance, isurf, distOut, isurfOut);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
  {
    int isurf, isurfIn;
    Real_v distIn;
    DistanceToSolid<Real_v, false>(tessellated, point, direction, stepMax, distance, isurf, distIn, isurfIn);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v inside;
    TessellatedImplementation::Contains<Real_v, Bool_v>(tessellated, point, inside);
    if (inside) {
      safety = -1.;
      return;
    }
    int isurf;
    Real_v safetysq = SafetySq<Real_v, true>(tessellated, point, isurf);
    safety          = vecCore::math::Sqrt(safetysq);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v inside;
    TessellatedImplementation::Contains<Real_v, Bool_v>(tessellated, point, inside);
    if (!inside) {
      safety = -1.;
      return;
    }
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
                              Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance, int &isurf,
                              Real_v &distother, int &isurfother)
  {
// Common method providing DistanceToIn/Out functionality
// Real_v here is scalar, we need to pass vector point/direction
#ifndef VECGEOM_ENABLE_CUDA
    using Float_v = vecgeom::VectorBackend::Real_v;
#else
    using Float_v             = vecgeom::ScalarBackend::Real_v;
#endif
    isurf      = -1;
    isurfother = -1;
    if (ToIn) {
      // Check if the bounding box is hit
      const Vector3D<Real_v> invdir(Real_v(1.0) / NonZero(direction.x()), Real_v(1.0) / NonZero(direction.y()),
                                    Real_v(1.0) / NonZero(direction.z()));
      Vector3D<int> sign;
      sign[0]  = invdir.x() < 0;
      sign[1]  = invdir.y() < 0;
      sign[2]  = invdir.z() < 0;
      distance = BoxImplementation::IntersectCachedKernel2<Real_v, Real_v>(
          &tessellated.fMinExtent, point, invdir, sign.x(), sign.y(), sign.z(), -kTolerance, InfinityLength<Real_v>());
      if (distance >= stepMax) return;
    }

    // Define the user hook calling DistanceToIn for the cluster with the same
    // index as the bounding box
    Vector3D<Float_v> pointv(point);
    Vector3D<Float_v> dirv(direction);
    distance             = InfinityLength<Real_v>();
    distother            = InfinityLength<Real_v>();
    Real_v distanceToIn  = InfinityLength<Real_v>();
    Real_v distanceToOut = InfinityLength<Real_v>();
    int isurfToIn        = -1;
    int isurfToOut       = -1;
    auto userhook        = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the distance to the current box is bigger than the
      // requested limit or than the current distance
      if (hitbox.second > vecCore::math::Min(stepMax, distance)) return true;
      // Compute distance to the cluster (in both ToIn or ToOut assumptions)
      Real_v clusterToIn, clusterToOut;
      int icrtToIn, icrtToOut;
      tessellated.fClusters[hitbox.first]->DistanceToCluster(pointv, dirv, clusterToIn, clusterToOut, icrtToIn,
                                                             icrtToOut);

      // Update distanceToIn/Out
      if (icrtToIn >= 0 && clusterToIn < distanceToIn) {
        distanceToIn = clusterToIn;
        isurfToIn    = icrtToIn;
        if (ToIn) {
          isurf    = isurfToIn;
          distance = distanceToIn;
        } else {
          isurfother = isurfToIn;
          distother  = distanceToIn;
        }
      }

      if (icrtToOut >= 0 && clusterToOut < distanceToOut) {
        distanceToOut = clusterToOut;
        isurfToOut    = icrtToOut;
        if (!ToIn) {
          isurf    = isurfToOut;
          distance = distanceToOut;
        } else {
          isurfother = isurfToOut;
          distother  = distanceToOut;
        }
      }
      return false;
    };

#ifdef USEEMBREE
    EmbreeNavigator<> *boxNav = (EmbreeNavigator<> *)EmbreeNavigator<>::Instance();
    // intersect ray with the BVH structure and use hook
    boxNav->BVHSortedIntersectionsLooper(*tessellated.fNavHelper2, point, direction, 1E20, userhook);
#else
    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHSortedIntersectionsLooper(*tessellated.fNavHelper2, point, direction, stepMax, userhook);
#endif

    // Treat special cases
    if (ToIn) {
      if (isurfToIn < 0) {
        if (isurfToOut >= 0 && distanceToOut * direction.Dot(tessellated.fFacets[isurfToOut]->fNormal) > kTolerance)
          distance = -1.; // point inside or on boundary
        // else not hitting, distance already inf
      } else {
        if (isurfToOut >= 0 && distanceToOut > kTolerance && distanceToOut < distanceToIn)
          distance = -1.; // point inside exiting first then re-entering
        // else valid entry point, distance already set
      }
    } else {
      if (isurfToOut < 0)
        distance = -1.; // point outside
      else {
        if (isurfToIn >= 0 && distanceToIn < distanceToOut &&
            distanceToIn * direction.Dot(tessellated.fFacets[isurfToIn]->fNormal) < -kTolerance) {
          distance = -1.; // point outside (first entering then exiting)
          isurf    = -1;
        }
      }
    }
  }

  template <typename Real_v, bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  static Real_v SafetySq(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, int &isurf)
  {
#ifndef VECGEOM_ENABLE_CUDA
    using Float_v = vecgeom::VectorBackend::Real_v;
#else
    using Float_v = vecgeom::ScalarBackend::Real_v;
#endif
    Real_v safetysq = InfinityLength<Real_v>();
    isurf           = -1;
    Vector3D<Float_v> pointv(point);

    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the safety to the current cluster is bigger than the
      // current safety
      if (hitbox.second > safetysq) return true;
      // Compute distance to the cluster
      int isurfcrt;
      Real_v safetycrt = tessellated.fClusters[hitbox.first]->template SafetySq<ToIn>(pointv, isurfcrt);
      if (safetycrt < safetysq) {
        safetysq = safetycrt;
        isurf    = isurfcrt;
      }
      return false;
    };

    HybridSafetyEstimator *safEstimator = (HybridSafetyEstimator *)HybridSafetyEstimator::Instance();
    // Use the BVH structure and connect hook
    safEstimator->BVHSortedSafetyLooper(*tessellated.fNavHelper, point, userhook, safetysq);
    return safetysq;
  }

}; // end TessellatedImplementation
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
