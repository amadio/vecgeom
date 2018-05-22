/*
 * EmbreeManager.h
 *
 *  Created on: May 18, 2018
 *      Author: swenzel
 */

#ifndef VECGEOM_MANAGEMENT_EMBREEMANAGER_H_
#define VECGEOM_MANAGEMENT_EMBREEMANAGER_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include <vector>

#include <embree3/rtcore.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class VPlacedVolume;

// enumeration used to decide how to build the structure
enum class EmbreeBuildMode {
  kAABBox, // from axis aligned bounding boxes
  kBBox    // from arbitrary (rotated) bounding boxes
};

// A singleton class which manages Embree geometries/scenes/helper structure for voxelized navigation
// The helper structure is whatever Intel Embree decides to use
class EmbreeManager {

public:
  typedef Vector3D<Precision> ABBox_s;
  typedef ABBox_s *ABBoxContainer_t;

  // first index is # daughter index, second is step
  typedef std::pair<int, double> BoxIdDistancePair_t;
  using HitContainer_t = std::vector<BoxIdDistancePair_t>;

  // the actual class encapsulating the Embree structures
  struct EmbreeAccelerationStructure {
    RTCDevice fDevice;
    RTCScene fScene;
    Vector3D<float> *fNormals; // normals of triangles/quads made available to user algorithms
    int fNumberObjects;        // number of objects
  };

private:
  // keeps/registers an acceleration structure for logical volumes
  std::vector<EmbreeAccelerationStructure const *> fStructureHolder;

public:
  // initialized the helper structure for a given logical volume
  void InitStructure(LogicalVolume const *lvol);

  // initialized the helper structure for the complete geometry
  // (this might not be appropriate and will consume memory)
  void InitVoxelStructureForCompleteGeometry()
  {
    std::vector<LogicalVolume const *> logicalvolumes;
    GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);
    for (auto lvol : logicalvolumes) {
      InitStructure(lvol);
    }
  }

  static EmbreeManager &Instance()
  {
    static EmbreeManager manager;
    return manager;
  }

  // removed/deletes the helper structure for a given logical volume
  void RemoveStructure(LogicalVolume const *lvol);

  EmbreeAccelerationStructure const *GetAccStructure(LogicalVolume const *lvol) const
  {
    return fStructureHolder[lvol->id()];
  }

  // public method allowing to build Embree acceleration structure
  // given a vector of aligned bounding boxes (without reference to a logical volume)
  EmbreeAccelerationStructure *BuildStructureFromBoundingBoxes(ABBoxContainer_t alignedboxes,
                                                               size_t numberofboxes) const;

  // build structure for a given logical volume
  EmbreeAccelerationStructure *BuildStructureFromBoundingBoxes(LogicalVolume const *lvol) const;

  // we could add more such methods starting from other structures (non-aligned bounding boxes or any crude triangular
  // hulls)
  void SetBuildMode(EmbreeBuildMode mode) { fBuildMode = mode; }

private:
  // private methods use
  void BuildStructure(LogicalVolume const *lvol);

  // adds a particular bounding box to an Embree scene
  void AddBoxGeometryToScene(EmbreeAccelerationStructure &, Vector3D<Precision> const &lower,
                             Vector3D<Precision> const &upper,
                             Transformation3D const &transf = Transformation3D::kIdentity) const;

  // adds an arbitratry bounding box (not necessarily axis aligned) for pvol
  void AddArbitraryBBoxToScene(EmbreeAccelerationStructure &, VPlacedVolume const *pvol,
                               EmbreeBuildMode mode = EmbreeBuildMode::kBBox) const;

  EmbreeBuildMode fBuildMode = EmbreeBuildMode::kAABBox;

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* MANAGEMENT_EMBREEMANAGER_H_ */
