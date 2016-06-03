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

#include <exception>
#include <stdexcept>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


// a LevelLocator using a flat list of bounding boxes
class HybridLevelLocator : public VLevelLocator {

private:
  HybridManager2 &fAccelerationStructure;

  HybridLevelLocator() : fAccelerationStructure(HybridManager2::Instance()) {}

public:

  virtual bool
  LevelLocate(LogicalVolume const * lvol,
              Vector3D<Precision> const & localpoint, VPlacedVolume const *&pvol,
              Vector3D<Precision> & transformedPoint) const override {
#ifdef VECGEOM_VC
	  int halfvectorsize, numberOfNodes;
      auto boxes_v = fAccelerationStructure.GetABBoxes_v(lvol, halfvectorsize, numberOfNodes);
      std::vector<int> * nodeToDaughters = fAccelerationStructure.GetNodeToDaughters(lvol);
      auto simdsize = kVcFloat::precision_v::Size;

      for (int index = 0, nodeindex = 0; index < halfvectorsize * 2;
           index += 2 * (simdsize + 1), nodeindex += simdsize) {
        HybridManager2::Bool_v inChildNodes;
        ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(boxes_v[index], boxes_v[index + 1], localpoint,
                                                           inChildNodes);
        if (Any(inChildNodes)) {
          for (size_t i = inChildNodes.firstOne(); i < simdsize; ++i) {
            if (inChildNodes[i]) {
              HybridManager2::Bool_v inDaughterBox;
              ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(
                  boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3], localpoint, inDaughterBox);
              if (Any(inDaughterBox)) {
                for (size_t j = inDaughterBox.firstOne(); j < simdsize; ++j) { // leaf node
                  if (inDaughterBox[j] &&
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
#else
    throw std::runtime_error("unimplemented function called: HybridLevelLocator::LevelLocate()");
#endif
  }

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &transformedPoint) const override {
#ifdef VECGEOM_VC
	  int halfvectorsize, numberOfNodes;
    auto boxes_v = fAccelerationStructure.GetABBoxes_v(lvol, halfvectorsize, numberOfNodes);
    std::vector<int> *nodeToDaughters = fAccelerationStructure.GetNodeToDaughters(lvol);
    auto simdsize = kVcFloat::precision_v::Size;

    for (int index = 0, nodeindex = 0; index < halfvectorsize * 2; index += 2 * (simdsize + 1), nodeindex += simdsize) {
      HybridManager2::Bool_v inChildNodes;
      ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(boxes_v[index], boxes_v[index + 1], localpoint, inChildNodes);
      if (Any(inChildNodes)) {
        for (size_t i = inChildNodes.firstOne(); i < simdsize; ++i) {
          if (inChildNodes[i]) {
            HybridManager2::Bool_v inDaughterBox;
            ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3],
                                                               localpoint, inDaughterBox);
            if (Any(inDaughterBox)) {
              for (size_t j = inDaughterBox.firstOne(); j < simdsize; ++j) { // leaf node
                if (inDaughterBox[j]) {
                  auto daughter = lvol->GetDaughters()[nodeToDaughters[nodeindex + i][j]];
                  if (daughter == exclvol)
                    continue;
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
#else
    throw std::runtime_error("unimplemented function called: HybridLevelLocator::LevelLocateExclVol()");
#endif
  }

  static std::string GetClassName() { return "HybridLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static
  VLevelLocator const *GetInstance(){
    static HybridLevelLocator instance;
    return &instance;
  }


}; // end class declaration


}} // end namespace


#endif /* NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_ */
