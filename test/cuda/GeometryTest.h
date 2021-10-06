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
  // Don't use Vector3D which changes layout from device to host in vector mode
  // so cudaMemcpy gets wrong data
  vecgeom::Precision amin[3];
  vecgeom::Precision amax[3];

  GeometryInfo() = default;

  template <typename Vol_t>
  VECCORE_ATT_HOST_DEVICE
  GeometryInfo(unsigned int theDepth, Vol_t &vol)
      : depth(theDepth), id(vol.id()), childId(vol.GetChildId()), copyNo(vol.GetCopyNo()),
        logicalId(vol.GetLogicalVolume()->id()), trans{*vol.GetTransformation()}
  {
    vecgeom::Vector3D<vecgeom::Precision> aminv, amaxv;
    vol.GetUnplacedVolume()->GetBBox(aminv, amaxv);
    assert((amaxv - aminv).Mag() > 0 && "Bounding box size must be positive");
    for (auto i = 0; i < 3; ++i) {
      amin[i] = aminv[i];
      amax[i] = amaxv[i];
    }
  }

  bool operator==(const GeometryInfo &rhs)
  {
    bool same_bbox = amin[0] == rhs.amin[0] && amin[1] == rhs.amin[1] && amin[2] == rhs.amin[2] &&
                     amax[0] == rhs.amax[0] && amax[1] == rhs.amax[1] && amax[2] == rhs.amax[2];
    return depth == rhs.depth && id == rhs.id && childId == rhs.childId && copyNo == rhs.copyNo &&
           logicalId == rhs.logicalId && trans == rhs.trans && same_bbox;
  }
};

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume* volume, std::size_t numVols);
