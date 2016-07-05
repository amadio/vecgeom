/*
 * GlobalLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_GLOBALLOCATOR_H_
#define NAVIGATION_GLOBALLOCATOR_H_

#include "base/Vector3D.h"
#include "volumes/LogicalVolume.h"
#include "navigation/VLevelLocator.h"
#include "base/Assert.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

//! basic algorithm locate a global point

//! basic generic algorithm that locate a global point in a geometry hierarchy
//! dispatches to specific LevelLocators
//! its a namespace rather than a class since it offers only a static function
namespace GlobalLocator {

// this function is a generic variant which can pick from each volume
// the best (or default) LevelLocator
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
VPlacedVolume const *LocateGlobalPoint(VPlacedVolume const *vol, Vector3D<Precision> const &point,
                                       NavigationState &path, bool top)
{
  VPlacedVolume const *candvolume = vol;
  Vector3D<Precision> currentpoint(point);
  if (top) {
    assert(vol != nullptr);
    candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
  }
  if (candvolume) {
    path.Push(candvolume);
    LogicalVolume const *lvol         = candvolume->GetLogicalVolume();
    Vector<Daughter> const *daughters = lvol->GetDaughtersp();

    bool godeeper = true;
    while (daughters->size() > 0 && godeeper) {
      godeeper = false;
      // returns nextvolume; and transformedpoint; modified path
      VLevelLocator const *locator = lvol->GetLevelLocator();
      if (locator != nullptr) { // if specialized/optimized technique attached to logical volume
        Vector3D<Precision> transformedpoint;
        godeeper = locator->LevelLocate(lvol, currentpoint, candvolume, transformedpoint);
        if (godeeper) {
          lvol         = candvolume->GetLogicalVolume();
          daughters    = lvol->GetDaughtersp();
          currentpoint = transformedpoint;
          path.Push(candvolume);
        }
      } else { // otherwise do a default implementation
        for (size_t i = 0; i < daughters->size(); ++i) {
          VPlacedVolume const *nextvolume = (*daughters)[i];
          Vector3D<Precision> transformedpoint;
          if (nextvolume->Contains(currentpoint, transformedpoint)) {
            path.Push(nextvolume);
            currentpoint = transformedpoint;
            candvolume   = nextvolume;
            daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
            godeeper     = true;
            break;
          }
        }
      }
    }
  }
  return candvolume;
}

// special version of locate point function that excludes searching a given volume
// (useful when we know that a particle must have traversed a boundary)
VECGEOM_FORCE_INLINE
VPlacedVolume const *LocateGlobalPointExclVolume(VPlacedVolume const *vol, VPlacedVolume const *excludedvolume,
                                                 Vector3D<Precision> const &point, NavigationState &path, bool top)
{
  VPlacedVolume const *candvolume = vol;
  Vector3D<Precision> currentpoint(point);
  if (top) {
    assert(vol != nullptr);
    candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
  }
  if (candvolume) {
    path.Push(candvolume);
    LogicalVolume const *lvol         = candvolume->GetLogicalVolume();
    Vector<Daughter> const *daughters = lvol->GetDaughtersp();

    bool godeeper = true;
    while (daughters->size() > 0 && godeeper) {
      godeeper = false;
      // returns nextvolume; and transformedpoint; modified path
      VLevelLocator const *locator = lvol->GetLevelLocator();
      if (locator != nullptr) { // if specialized/optimized technique attached to logical volume
        Vector3D<Precision> transformedpoint;
        godeeper = locator->LevelLocateExclVol(lvol, excludedvolume, currentpoint, candvolume, transformedpoint);
        if (godeeper) {
          lvol         = candvolume->GetLogicalVolume();
          daughters    = lvol->GetDaughtersp();
          currentpoint = transformedpoint;
          path.Push(candvolume);
        }
      } else { // otherwise do a default implementation
        for (size_t i = 0; i < daughters->size(); ++i) {
          VPlacedVolume const *nextvolume = (*daughters)[i];
          if (nextvolume != excludedvolume) {
            Vector3D<Precision> transformedpoint;
            if (nextvolume->Contains(currentpoint, transformedpoint)) {
              path.Push(nextvolume);
              currentpoint = transformedpoint;
              candvolume   = nextvolume;
              daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
              godeeper     = true;
              break;
            }
          } // end if excludedvolume
        }
      }
    }
  }
  return candvolume;
}

VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
static VPlacedVolume const *RelocatePointFromPath(Vector3D<Precision> const &localpoint, NavigationState &path)
{
  // idea: do the following:
  // ----- is localpoint still in current mother ? : then go down
  // if not: have to go up until we reach a volume that contains the
  // localpoint and then go down again (neglecting the volumes currently stored in the path)
  VPlacedVolume const *currentmother = path.Top();
  if (currentmother != nullptr) {
    Vector3D<Precision> tmp = localpoint;
    // go up iteratively
    while (currentmother && !currentmother->UnplacedContains(tmp)) {
      path.Pop();
      Vector3D<Precision> pointhigherup = currentmother->GetTransformation()->InverseTransform(tmp);
      tmp                               = pointhigherup;
      currentmother                     = path.Top();
    }

    if (currentmother) {
      path.Pop();
      // return LocateGlobalPointExclVolume(currentmother, currentmother, tmp, path, false);
      return LocateGlobalPoint(currentmother, tmp, path, false);
    }
  }
  return currentmother;
}

} // end GlobalLocator namespace
}
} // end namespaces

#endif /* NAVIGATION_GLOBALLOCATOR_H_ */
