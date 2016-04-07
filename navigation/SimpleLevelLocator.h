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

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a simple version of a LevelLocator offering a generic brute force algorithm
class SimpleLevelLocator : public VLevelLocator {

private:
  SimpleLevelLocator() {}

public:
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override {
    auto daughters = lvol->GetDaughtersp();
    for (int i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *nextvolume = (*daughters)[i];
      if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
        pvol = nextvolume;
        return true;
      }
    }
    return false;
  }

  // might call for a shared kernel to avoid code duplication
  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override {
    auto daughters = lvol->GetDaughtersp();
    for (int i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *nextvolume = (*daughters)[i];
      if(exclvol==nextvolume) continue;
      if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
        pvol = nextvolume;
        return true;
      }
    }
    return false;
  }
  static std::string GetClassName() { return "SimpleLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static
  VLevelLocator const *GetInstance() {
    static SimpleLevelLocator instance;
    return &instance;
  }


}; // end class declaration
}} // end namespace


#endif /* NAVIGATION_SIMPLELEVELLOCATOR_H_ */
