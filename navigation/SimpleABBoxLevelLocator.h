/*
 * SimpleABBoxLevelLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_
#define NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_

#include "navigation/VLevelLocator.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/ABBoxManager.h"
#include "volumes/kernel/BoxImplementation.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


// a LevelLocator using a flat list of bounding boxes
class SimpleABBoxLevelLocator : public VLevelLocator {

private:
  ABBoxManager & fAccelerationStructure;
    SimpleABBoxLevelLocator() : fAccelerationStructure(ABBoxManager::Instance()) {}

public:
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override {

    int size;
    ABBoxManager::ABBoxContainer_v alignedbboxes =
        fAccelerationStructure.GetABBoxes_v(lvol, size);

    auto daughters = lvol->GetDaughtersp();
    // here the loop is over groups of bounding boxes
    // it is basically linear but vectorizable search
    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
      typename kVcFloat::bool_v inBox;
      ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(alignedbboxes[2 * boxgroupid],
                                                         alignedbboxes[2 * boxgroupid + 1], localpoint, inBox);
      if (Any(inBox)) {
        // TODO: could start directly at first 1 in inBox
        for (size_t ii = 0; ii < kVcFloat::precision_v::Size; ++ii) {
          auto daughterid = boxgroupid * kVcFloat::precision_v::Size + ii;
          if (daughterid < daughters->size() && inBox[ii]) {
            VPlacedVolume const *daughter = (*daughters)[daughterid];
            if (daughter->Contains(localpoint, daughterlocalpoint)) {
              pvol = daughter;
              // careful here: we also want to break on external loop
              return true;
            }
          }
        }
      }
    }
    return false;
  } // end function

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &daughterlocalpoint) const override {

    int size;
    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    auto daughters = lvol->GetDaughtersp();
    // here the loop is over groups of bounding boxes
    // it is basically linear but vectorizable search
    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
      typename kVcFloat::bool_v inBox;
      ABBoxImplementation::ABBoxContainsKernel<kVcFloat>(alignedbboxes[2 * boxgroupid],
                                                         alignedbboxes[2 * boxgroupid + 1], localpoint, inBox);
      if (Any(inBox)) {
        // TODO: could start directly at first 1 in inBox
        for (size_t ii = 0; ii < kVcFloat::precision_v::Size; ++ii) {
          auto daughterid = boxgroupid * kVcFloat::precision_v::Size + ii;
          if (daughterid < daughters->size() && inBox[ii]) {
            VPlacedVolume const *daughter = (*daughters)[daughterid];
            if (daughter == exclvol)
              continue;
            if (daughter->Contains(localpoint, daughterlocalpoint)) {
              pvol = daughter;
              // careful here: we also want to break on external loop
              return true;
            }
          }
        }
      }
    }
    return false;
  } // end function

  static std::string GetClassName() { return "SimpleABBoxLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static
  VLevelLocator const *GetInstance(){
    static SimpleABBoxLevelLocator instance;
    return &instance;
  }


}; // end class declaration


}} // end namespace


#endif /* NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_ */
