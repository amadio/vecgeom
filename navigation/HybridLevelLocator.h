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
#include "navigation/SimpleLevelLocator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a LevelLocator using a flat list of bounding boxes
template <bool IsAssemblyAware = false>
class THybridLevelLocator : public VLevelLocator {

private:
  HybridManager2 &fAccelerationStructure;

  THybridLevelLocator() : fAccelerationStructure(HybridManager2::Instance()) {}

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                                        Vector3D<Precision> const &localpoint, NavigationState *state,
                                                        VPlacedVolume const *&pvol,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    auto accstructure = fAccelerationStructure.GetAccStructure(lvol);
    int halfvectorsize, numberOfNodes;
    auto boxes_v                = fAccelerationStructure.GetABBoxes_v(*accstructure, halfvectorsize, numberOfNodes);
    auto const *nodeToDaughters = accstructure->fNodeToDaughters;
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();

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

                  // final candidate check
                  VPlacedVolume const *candidate = lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]];
                  if (ExclV)
                    if (candidate == exclvol) continue;

                  if (CheckCandidateVol<IsAssemblyAware, ModifyState>(candidate, localpoint, state, pvol,
                                                                      daughterlocalpoint)) {
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

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernelWithDirection(LogicalVolume const *lvol,
                                                                     VPlacedVolume const *exclvol,
                                                                     Vector3D<Precision> const &localpoint,
                                                                     Vector3D<Precision> const &localdir,
                                                                     NavigationState *state, VPlacedVolume const *&pvol,
                                                                     Vector3D<Precision> &daughterlocalpoint) const
  {
    auto accstructure = fAccelerationStructure.GetAccStructure(lvol);
    int halfvectorsize, numberOfNodes;
    auto boxes_v                = fAccelerationStructure.GetABBoxes_v(*accstructure, halfvectorsize, numberOfNodes);
    auto const *nodeToDaughters = accstructure->fNodeToDaughters;
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();

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

                  // final candidate check
                  VPlacedVolume const *candidate = lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]];
                  if (ExclV) {
                    if (candidate == exclvol) {
                      continue;
                    }
                  }

                  if (CheckCandidateVolWithDirection<IsAssemblyAware, ModifyState>(candidate, localpoint, localdir,
                                                                                   state, pvol, daughterlocalpoint)) {
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

public:
  static std::string GetClassName() { return "HybridLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<false, false>(lvol, nullptr, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &state,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    VPlacedVolume const *pvol;
    return LevelLocateKernel<false, true>(lvol, nullptr, localpoint, &state, pvol, daughterlocalpoint);
  }

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<true, false>(lvol, exclvol, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdir,
                                  VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernelWithDirection<true, false>(lvol, exclvol, localpoint, localdir, nullptr, pvol,
                                                       daughterlocalpoint);
  }

  static VLevelLocator const *GetInstance()
  {
    static THybridLevelLocator instance;
    return &instance;
  }

}; // end class declaration

template <>
inline std::string THybridLevelLocator<true>::GetClassName()
{
  return "AssemblyAwareHybridLevelLocator";
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_HYBRIDLEVELLOCATOR_H_ */
