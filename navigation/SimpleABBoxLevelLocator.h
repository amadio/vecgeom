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

#include "navigation/SimpleLevelLocator.h"

#include <exception>
#include <stdexcept>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a LevelLocator using a flat list of bounding boxes
template <bool IsAssemblyAware = false>
class TSimpleABBoxLevelLocator : public VLevelLocator {

private:
  ABBoxManager &fAccelerationStructure;
  TSimpleABBoxLevelLocator() : fAccelerationStructure(ABBoxManager::Instance()) {}

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                                        Vector3D<Precision> const &localpoint, NavigationState *state,
                                                        VPlacedVolume const *&pvol,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    int size;
    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    auto daughters = lvol->GetDaughtersp();
    // here the loop is over groups of bounding boxes
    // it is basically linear but vectorizable search
    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
      using Bool_v = vecCore::Mask_v<ABBoxManager::Float_v>;
      Bool_v inBox;
      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
                                               localpoint, inBox);
      if (!vecCore::MaskEmpty(inBox)) {
        constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
        // TODO: could start directly at first 1 in inBox
        for (size_t ii = 0; ii < kVS; ++ii) {
          auto daughterid = boxgroupid * kVS + ii;
          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {

            VPlacedVolume const *daughter = (*daughters)[daughterid];
            if (ExclV) {
              if (daughter == exclvol) continue;
            }
            // call final treatment of candidate
            if (CheckCandidateVol<IsAssemblyAware, ModifyState>(daughter, localpoint, state, pvol, daughterlocalpoint))
              return true;
          }
        }
      }
    }
    return false;
  }

public:
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<false, false>(lvol, nullptr, localpoint, nullptr, pvol, daughterlocalpoint);
    //    int size;
    //    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    //    auto daughters = lvol->GetDaughtersp();
    //    // here the loop is over groups of bounding boxes
    //    // it is basically linear but vectorizable search
    //    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
    //      using Bool_v = vecCore::Mask_v<ABBoxManager::Float_v>;
    //      Bool_v inBox;
    //      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
    //                                               localpoint, inBox);
    //      if (!vecCore::MaskEmpty(inBox)) {
    //        constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
    //        // TODO: could start directly at first 1 in inBox
    //        for (size_t ii = 0; ii < kVS; ++ii) {
    //          auto daughterid = boxgroupid * kVS + ii;
    //          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {
    //            VPlacedVolume const *daughter = (*daughters)[daughterid];
    //            if (daughter->Contains(localpoint, daughterlocalpoint)) {
    //              pvol = daughter;
    //              // careful here: we also want to break on external loop
    //              return true;
    //            }
    //          }
    //        }
    //      }
    //    }
    //    return false;
  } // end function

  // version that directly modifies the navigation state
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &outstate,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    VPlacedVolume const *pvol;
    return LevelLocateKernel<false, true>(lvol, nullptr, localpoint, &outstate, pvol, daughterlocalpoint);
    //    int size;
    //    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    //    auto daughters = lvol->GetDaughtersp();
    //    // here the loop is over groups of bounding boxes
    //    // it is basically linear but vectorizable search
    //    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
    //      using Bool_v = vecCore::Mask_v<ABBoxManager::Float_v>;
    //      Bool_v inBox;
    //      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
    //                                               localpoint, inBox);
    //      if (!vecCore::MaskEmpty(inBox)) {
    //        constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
    //        // TODO: could start directly at first 1 in inBox
    //        for (size_t ii = 0; ii < kVS; ++ii) {
    //          auto daughterid = boxgroupid * kVS + ii;
    //          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {
    //            VPlacedVolume const *daughter = (*daughters)[daughterid];
    //            if (daughter->Contains(localpoint, daughterlocalpoint)) {
    //              outstate.Push(daughter);
    //              // careful here: we also want to break on external loop
    //              return true;
    //            }
    //          }
    //        }
    //      }
    //    }
    //    return false;
  }

  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<true, false>(lvol, exclvol, localpoint, nullptr, pvol, daughterlocalpoint);
    //    int size;
    //    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    //    auto daughters = lvol->GetDaughtersp();
    //    // here the loop is over groups of bounding boxes
    //    // it is basically linear but vectorizable search
    //    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
    //      using Bool_v = vecCore::Mask_v<ABBoxManager::Float_v>;
    //      Bool_v inBox;
    //      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
    //                                               localpoint, inBox);
    //      if (!vecCore::MaskEmpty(inBox)) {
    //        constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
    //        // TODO: could start directly at first 1 in inBox
    //        for (size_t ii = 0; ii < kVS; ++ii) {
    //          auto daughterid = boxgroupid * kVS + ii;
    //          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {
    //            VPlacedVolume const *daughter = (*daughters)[daughterid];
    //            if (daughter == exclvol) continue;
    //            if (daughter->Contains(localpoint, daughterlocalpoint)) {
    //              pvol = daughter;
    //              // careful here: we also want to break on external loop
    //              return true;
    //            }
    //          }
    //        }
    //      }
    //    }
    //    return false;
  } // end function

  static std::string GetClassName() { return "SimpleABBoxLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static VLevelLocator const *GetInstance()
  {
    static TSimpleABBoxLevelLocator instance;
    return &instance;
  }

}; // end class declaration

using SimpleABBoxLevelLocator              = TSimpleABBoxLevelLocator<>;
using SimpleAssemblyAwareABBoxLevelLocator = TSimpleABBoxLevelLocator<true>;

//// template specializations
// template <>
// inline
// bool TSimpleABBoxLevelLocator<true>::LevelLocate(LogicalVolume const * lvol, Vector3D<Precision> const & localpoint,
//                                                 NavigationState & outstate, Vector3D<Precision> & daughterlocalpoint)
//                                                 const
//{
//    int size;
//    ABBoxManager::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

//    auto daughters = lvol->GetDaughtersp();
//    // here the loop is over groups of bounding boxes
//    // it is basically linear but vectorizable search
//    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
//      using Bool_v = vecCore::Mask_v<ABBoxManager::Float_v>;
//      Bool_v inBox;
//      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
//                                               localpoint, inBox);
//      if (!vecCore::MaskEmpty(inBox)) {
//        constexpr auto kVS = vecCore::VectorSize<ABBoxManager::Float_v>();
//        // TODO: could start directly at first 1 in inBox
//        for (size_t ii = 0; ii < kVS; ++ii) {
//          auto daughterid = boxgroupid * kVS + ii;
//          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {
//            VPlacedVolume const *daughter = (*daughters)[daughterid];
//            if (daughter->Contains(localpoint, daughterlocalpoint)) {
//              outstate.Push(daughter);
//              // careful here: we also want to break on external loop
//              return true;
//            }
//          }
//        }
//      }
//    }
//    return false;
//}

template <>
inline std::string TSimpleABBoxLevelLocator<true>::GetClassName()
{
  return "SimpleAssemblyAwareABBoxLevelLocator";
}
}
} // end namespace

#endif /* NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_ */
