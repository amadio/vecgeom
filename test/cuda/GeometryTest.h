// Author: Stephan Hageboeck, CERN, 2021
// Shared C++/Cuda infrastructure for a geometry test.

#include "VecGeom/volumes/PlacedVolume.h"

#include <vector>
#include <cassert>

namespace vecgeom {
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_HOST_FORWARD_DECLARE(class VPlacedVolume;);
}

struct GeometryInfo {
  unsigned int depth = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().id()) id = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetChildId()) childId = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetCopyNo()) copyNo = 0;
  decltype(std::declval<vecgeom::LogicalVolume>().id()) logicalId = 0;
  vecgeom::Transformation3D trans;

  GeometryInfo() = default;

  template<typename Vol_t>
  VECCORE_ATT_HOST_DEVICE
  GeometryInfo(unsigned int theDepth, Vol_t & vol) :
  depth(theDepth),
  id(vol.id()),
  childId(vol.GetChildId()),
  copyNo(vol.GetCopyNo()),
  logicalId(vol.GetLogicalVolume()->id()),
  trans{*vol.GetTransformation()}
  { }

  bool operator==(const GeometryInfo& rhs) {
    return depth == rhs.depth && id == rhs.id && childId == rhs.childId
           && copyNo == rhs.copyNo && logicalId == rhs.logicalId
           && trans == rhs.trans;
  }
};

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume* volume);