#ifndef VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_
#define VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_

#include "base/Global.h"
#include "management/HybridManager2.h"
#include "navigation/HybridNavigator2.h"
#include "management/ABBoxManager.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 @brief Struct containing placed volume components representing a multiple union solid
 @author mihaela.gheata@cern.ch
*/

struct MultiUnionStruct {
  template <typename U>
  using vector_t     = vecgeom::Vector<U>;
  using BVHStructure = HybridManager2::HybridBoxAccelerationStructure;

  vector_t<VPlacedVolume *> fVolumes; ///< Component placed volumes
  BVHStructure *fNavHelper = nullptr; ///< Navigation helper using bounding boxes

  Vector3D<double> fMinExtent;
  Vector3D<double> fMaxExtent;
  mutable double fCapacity    = -1; ///< Capacity of the multiple union
  mutable double fSurfaceArea = -1; ///< Surface area of the multiple union

  VECCORE_ATT_HOST_DEVICE
  MultiUnionStruct()
  {
    fMinExtent.Set(kInfLength);
    fMaxExtent.Set(-kInfLength);
  }

  VECCORE_ATT_HOST_DEVICE
  void AddNode(VPlacedVolume *volume)
  {
    using vecCore::math::Min;
    using vecCore::math::Max;
    Vector3D<double> amin, amax;
    ABBoxManager::ComputeABBox(volume, &amin, &amax);
    fMinExtent.Set(Min(fMinExtent.x(), amin.x()), Min(fMinExtent.y(), amin.y()), Min(fMinExtent.z(), amin.z()));
    fMaxExtent.Set(Max(fMaxExtent.x(), amax.x()), Max(fMaxExtent.y(), amax.y()), Max(fMaxExtent.z(), amax.z()));
    fVolumes.push_back(volume);
  }

  VECCORE_ATT_HOST_DEVICE
  void Close()
  {
    // This method prepares the navigation structure
    using Boxes_t           = ABBoxManager::ABBoxContainer_t;
    using BoxCorner_t       = ABBoxManager::ABBox_s;
    size_t nboxes           = fVolumes.size();
    BoxCorner_t *boxcorners = new BoxCorner_t[2 * nboxes];
    Vector3D<double> amin, amax;
    for (size_t i = 0; i < nboxes; ++i)
      ABBoxManager::ComputeABBox(fVolumes[i], &boxcorners[2 * i], &boxcorners[2 * i + 1]);
    Boxes_t boxes = &boxcorners[0];
    fNavHelper    = HybridManager2::Instance().BuildStructure(boxes, nboxes);
  }

}; // End struct

} // End impl namespace
} // End global namespace

#endif /* VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_ */
