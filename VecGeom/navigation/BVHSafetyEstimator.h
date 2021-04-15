/// \file BVHSafetyEstimator.h
/// \author Guilherme Amadio

#ifndef VECGEOM_NAVIGATION_BVHSAFETYESTIMATOR_H_
#define VECGEOM_NAVIGATION_BVHSAFETYESTIMATOR_H_

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/navigation/VSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Safety estimator class using the bounding volume hierarchy of each
 * logical volume for acceleration.
 */

class BVHSafetyEstimator : public VSafetyEstimatorHelper<BVHSafetyEstimator> {
private:
  /** Constructor. Private since this is a singleton class accessed only via the @c Instance() static method. */
  BVHSafetyEstimator() : VSafetyEstimatorHelper<BVHSafetyEstimator>() {}

public:
  static constexpr const char *gClassNameString = "BVHSafetyEstimator";

  /** Return instance of this singleton class. */
  static VSafetyEstimator *Instance()
  {
    static BVHSafetyEstimator instance;
    return &instance;
  }

  /**
   * Compute safety of a point given in the local coordinates of the placed volume @p pvol.
   * @param[in] localpoint Point in the local coordinates of the placed volume.
   * @param[in] pvol Placed volume.
   */
  Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint, VPlacedVolume const *pvol) const final
  {
    Precision safety = pvol->SafetyToOut(localpoint);

    if (safety > 0.0 && pvol->GetDaughters().size() > 0)
      safety = BVHManager::GetBVH(pvol->GetLogicalVolume()).ComputeSafety(localpoint, safety);

    return safety;
  }

  /**
   * Compute safety of a point given in the local coordinates of the logical volume @p lvol against
   * all its child volumes. Uses the bounding volume hierarchy associated with the logical volume
   * for acceleration.
   * @param[in] localpoint Point in the local coordinates of the placed volume.
   * @param[in] lvol Logical volume.
   */
  Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                  LogicalVolume const *lvol) const final
  {
    return BVHManager::GetBVH(lvol).ComputeSafety(localpoint, kInfLength);
  }

  /**
   * Compute safety of a point given in the local coordinates of the placed volume @p pvol.
   * @param[in] localpoint Points in SIMD layout in the local coordinates of the placed volume.
   * @param[in] pvol Placed volume.
   * @param[in] m Mask of active SIMD lanes.
   */
  Real_v ComputeSafetyForLocalPoint(Vector3D<Real_v> const &localpoint, VPlacedVolume const *pvol, Bool_v m) const final
  {
    using vecCore::Get;
    using vecCore::Set;
    using vecCore::VectorSize;

    Real_v safeties(kInfLength);
    for (size_t i = 0; i < VectorSize<Real_v>(); ++i) {
      if (Get(m, i) == true) {
        Vector3D<Precision> point(Get(localpoint[0], i), Get(localpoint[1], i), Get(localpoint[2], i));
        Set(safeties, i, ComputeSafetyForLocalPoint(point, pvol));
      }
    }
    return safeties;
  }

  /**
   * Vector interface to compute safety for a set of points given in the local coordinates of the placed volume @p pvol.
   * @param[in] localpoints Points in the local coordinates of the placed volume.
   * @param[in] pvol Placed volume.
   * @param[out] safeties Output safeties.
   */
  void ComputeSafetyForLocalPoints(SOA3D<Precision> const &localpoints, VPlacedVolume const *pvol,
                                   Precision *safeties) const final
  {
    for (size_t i = 0; i < localpoints.size(); ++i)
      safeties[i] = ComputeSafetyForLocalPoint({localpoints.x(i), localpoints.y(i), localpoints.z(i)}, pvol);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
