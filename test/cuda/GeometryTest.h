// Author: Stephan Hageboeck, CERN, 2021
// Shared C++/Cuda infrastructure for a geometry test.

#include "VecGeom/volumes/PlacedVolume.h"

#include <vector>
#include <cassert>

namespace vecgeom {
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_HOST_FORWARD_DECLARE(class VPlacedVolume;);
} // namespace vecgeom

struct GeometryInfo {
  unsigned int depth                                                    = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().id()) id              = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetChildId()) childId = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetCopyNo()) copyNo   = 0;
  decltype(std::declval<vecgeom::LogicalVolume>().id()) logicalId       = 0;
  vecgeom::Transformation3D trans;
  vecgeom::Vector3D<vecgeom::Precision> amin;
  vecgeom::Vector3D<vecgeom::Precision> amax;

  GeometryInfo() = default;

  template <typename Vol_t>
  VECCORE_ATT_HOST_DEVICE
  GeometryInfo(unsigned int theDepth, Vol_t &vol)
      : depth(theDepth), id(vol.id()), childId(vol.GetChildId()), copyNo(vol.GetCopyNo()),
        logicalId(vol.GetLogicalVolume()->id()), trans{*vol.GetTransformation()}
  {
    vol.GetUnplacedVolume()->GetBBox(amin, amax);
    assert((amax - amin).Mag() > 0 && "Bounding box size must be positive");
  }

  bool operator==(const GeometryInfo &rhs)
  {
    return depth == rhs.depth && id == rhs.id && childId == rhs.childId && copyNo == rhs.copyNo &&
           logicalId == rhs.logicalId && trans == rhs.trans && amin == rhs.amin && amax == rhs.amax;
  }
};

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume);