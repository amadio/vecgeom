// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \author created by Sandro Wenzel

#ifndef VECGEOM_MANAGEMENT_FLATVOXELMANAGER_H_
#define VECGEOM_MANAGEMENT_FLATVOXELMANAGER_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/FlatVoxelHashMap.h"
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class VPlacedVolume;

// A singleton class which manages structures for fast voxelized safety lookup
class FlatVoxelManager {

public:
  // the actual class encapsulating the Embree structures
  struct VoxelStructure {
    FlatVoxelHashMap<float, true> *fVoxels = nullptr; // voxels keeping track of best known safety
    FlatVoxelHashMap<int, false> *fVoxelToCandidate =
        nullptr; // keep list of candidate objects to check for safety (-1 = mother, ...)
    FlatVoxelHashMap<int, false> *fVoxelToLocateCandidates =
        nullptr; // keep list of candidate objects to check for LevelLocate

    LogicalVolume const *fVol = nullptr; // keep track to which volume this belongs
  };

private:
  // keeps/registers an acceleration structure for logical volumes
  std::vector<VoxelStructure const *> fStructureHolder;

public:
  // initialized the helper structure for a given logical volume
  void InitStructure(LogicalVolume const *lvol);

  static FlatVoxelManager &Instance()
  {
    static FlatVoxelManager manager;
    return manager;
  }

  // removed/deletes the helper structure for a given logical volume
  void RemoveStructure(LogicalVolume const *lvol);

  VoxelStructure const *GetStructure(LogicalVolume const *lvol) const { return fStructureHolder[lvol->id()]; }

  static VoxelStructure *BuildStructure(LogicalVolume const *lvol);

  static FlatVoxelHashMap<int, false> *BuildLocateVoxels(LogicalVolume const *lvol);
  static FlatVoxelHashMap<int, false> *BuildSafetyVoxels(LogicalVolume const *lvol);
}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* MANAGEMENT_FLATVOXELMANAGER_H_ */
