#ifndef VECGEOM_VOLUMES_PLACEDTUBE_H_
#define VECGEOM_VOLUMES_PLACEDTUBE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TubeImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedTube.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTube;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTube);

inline namespace VECGEOM_IMPL_NAMESPACE {

// the base class of all placed tubes
// exists for stronger typing reasons and to be able
// to do runtime type inference on placed volumes

class PlacedTube : public VPlacedVolume {
  // some common functionality for all placed tubes
  // like constructors
public:
  using VPlacedVolume::VPlacedVolume;

#ifndef VECGEOM_NVCC
  PlacedTube(char const *const label, LogicalVolume const *const logical_volume,
             Transformation3D const *const transformation, ::vecgeom::PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedTube(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
             ::vecgeom::PlacedBox const *const boundingBox)
      : PlacedTube("", logical_volume, transformation, boundingBox)
  {
  }
#else
  __device__ PlacedTube(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id)
  {
  }
#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTube() {}

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const *ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif
};

// a placed tube knowing abouts its volume/structural specialization
template <typename UnplacedTube_t>
class SPlacedTube : public PlacedVolumeImplHelper<UnplacedTube_t, PlacedTube> {
  using Base = PlacedVolumeImplHelper<UnplacedTube_t, PlacedTube>;

public:
  typedef UnplacedTube UnplacedShape_t;
  using Base::Base;
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
