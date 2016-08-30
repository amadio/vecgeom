/*
 * SimpleLevelLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLELEVELLOCATOR_H_
#define NAVIGATION_SIMPLELEVELLOCATOR_H_

#include "navigation/VLevelLocator.h"
#include "volumes/LogicalVolume.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedAssembly.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// shared kernel for many locators
// treats the actual final check (depending on which interface to serve)
template <bool IsAssemblyAware, bool ModifyState>
__attribute__((always_inline)) inline static bool CheckCandidateVol(VPlacedVolume const *nextvolume,
                                                                    Vector3D<Precision> const &localpoint,
                                                                    NavigationState *state, VPlacedVolume const *&pvol,
                                                                    Vector3D<Precision> &daughterlocalpoint)
{
  if (IsAssemblyAware && ModifyState) {
    if (nextvolume->GetUnplacedVolume()->IsAssembly()) {
      // in this case we call a special version of Contains
      // offered by the assembly
      assert(ModifyState == true);
      if (((PlacedAssembly *)nextvolume)->Contains(localpoint, daughterlocalpoint, *state)) {
        return true;
      }
    } else {
      if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
        state->Push(nextvolume);
        return true;
      }
    }
  } else {
    if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
      if (ModifyState) {
        state->Push(nextvolume);
      } else {
        pvol = nextvolume;
      }
      return true;
    }
  }
  return false;
}

// a simple version of a LevelLocator offering a generic brute force algorithm
template <bool IsAssemblyAware = false>
class TSimpleLevelLocator : public VLevelLocator {

private:
  TSimpleLevelLocator() {}

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                                        Vector3D<Precision> const &localpoint, NavigationState *state,
                                                        VPlacedVolume const *&pvol,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    auto daughters = lvol->GetDaughtersp();
    for (size_t i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *nextvolume = (*daughters)[i];
      if (ExclV) {
        if (exclvol == nextvolume) continue;
      }
      if (CheckCandidateVol<IsAssemblyAware, ModifyState>(nextvolume, localpoint, state, pvol, daughterlocalpoint))
        return true;

      //      if (IsAssemblyAware) {
      //        if (nextvolume->GetUnplacedVolume()->IsAssembly()) {
      //          // in this case we call a special version of Contains
      //          // offered by the assembly
      //          assert(ModifyState == true);
      //          if (((PlacedAssembly *)nextvolume)->Contains(localpoint, daughterlocalpoint, *state)) {
      //            return true;
      //          }
      //        } else {
      //          if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
      //            state->Push(nextvolume);
      //            return true;
      //          }
      //        }
      //      } else {
      //        if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
      //          if (ModifyState) {
      //            state->Push(nextvolume);
      //          } else {
      //            pvol = nextvolume;
      //          }
      //          return true;
      //        }
      //      }
    }
    return false;
  }

public:
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<false, false>(lvol, nullptr, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  VECGEOM_CUDA_HEADER_BOTH
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

  static std::string GetClassName() { return "SimpleLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static VLevelLocator const *GetInstance()
  {
    static TSimpleLevelLocator instance;
    return &instance;
  }

}; // end class declaration

using SimpleLevelLocator = TSimpleLevelLocator<>;

template <>
inline std::string TSimpleLevelLocator<true>::GetClassName()
{
  return "SimpleAssemblyLevelLocator";
}
using SimpleAssemblyLevelLocator = TSimpleLevelLocator<true>;
}
} // end namespace

#endif /* NAVIGATION_SIMPLELEVELLOCATOR_H_ */
