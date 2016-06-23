/*
 * HybridLevelLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: Yang Zhang and Sandro Wenzel (sandro.wenzel@cern.ch)
 */

#ifndef NAVIGATION_HYBRIDLEVELLOCATOR_H_
#define NAVIGATION_HYBRIDLEVELLOCATOR_H_

#include "navigation/VLevelLocator.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/HybridManager2.h"
#include "volumes/kernel/BoxImplementation.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a LevelLocator using a flat list of bounding boxes
class HybridLevelLocator : public VLevelLocator {

private:
  HybridManager2 &fAccelerationStructure;

  HybridLevelLocator() : fAccelerationStructure(HybridManager2::Instance()) {}

public:
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &transformedPoint) const override
  {
    int halfvectorsize, numberOfNodes;
    auto boxes_v                      = fAccelerationStructure.GetABBoxes_v(lvol, halfvectorsize, numberOfNodes);
    std::vector<int> *nodeToDaughters = fAccelerationStructure.GetNodeToDaughters(lvol);
    constexpr auto kVS                = vecCore::VectorSize<HybridManager2::Float_v>();

    for (int index = 0, nodeindex = 0; index < halfvectorsize * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      using Bool_v = vecCore::Mask_v<HybridManager2::Float_v>;
      Bool_v inChildNodes;
      ABBoxImplementation::ABBoxContainsKernel(boxes_v[index], boxes_v[index + 1], localpoint, inChildNodes);
      if (!vecCore::MaskEmpty(inChildNodes)) {
        for (size_t i = 0 /*inChildNodes.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(inChildNodes, i)) {
            Bool_v inDaughterBox;
            ABBoxImplementation::ABBoxContainsKernel(boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3], localpoint,
                                                     inDaughterBox);
            if (!vecCore::MaskEmpty(inDaughterBox)) {
              for (size_t j = 0 /*inDaughterBox.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(inDaughterBox, j) &&
                    lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]]->Contains(localpoint, transformedPoint)) {
                  pvol = lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]];
                  return true;
                }
              }
            }
          }
        }
      }
    }
    return false;
  }

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &transformedPoint) const override
  {
    int halfvectorsize, numberOfNodes;
    auto boxes_v                      = fAccelerationStructure.GetABBoxes_v(lvol, halfvectorsize, numberOfNodes);
    std::vector<int> *nodeToDaughters = fAccelerationStructure.GetNodeToDaughters(lvol);
    constexpr auto kVS                = vecCore::VectorSize<HybridManager2::Float_v>();

    for (int index = 0, nodeindex = 0; index < halfvectorsize * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      using Bool_v = vecCore::Mask_v<HybridManager2::Float_v>;
      Bool_v inChildNodes;

      ABBoxImplementation::ABBoxContainsKernel(boxes_v[index], boxes_v[index + 1], localpoint, inChildNodes);
      if (!vecCore::MaskEmpty(inChildNodes)) {
        for (size_t i = 0 /*inChildNodes.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(inChildNodes, i)) {
            Bool_v inDaughterBox;
            ABBoxImplementation::ABBoxContainsKernel(boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3], localpoint,
                                                     inDaughterBox);
            if (!vecCore::MaskEmpty(inDaughterBox)) {
              for (size_t j = 0 /*inDaughterBox.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(inDaughterBox, j)) {
                  auto daughter = lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]];
                  if (daughter == exclvol) continue;
                  if (daughter->Contains(localpoint, transformedPoint)) {
                    pvol = daughter;
                    return true;
                  }
                }
              }
            }
          }
        }
      }
    }
    return false;
  }

  static std::string GetClassName() { return "HybridLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static VLevelLocator const *GetInstance()
  {
    static HybridLevelLocator instance;
    return &instance;
  }

}; // end class declaration
}
} // end namespace

#endif /* NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_ */
