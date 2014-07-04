/// \file PlacedHype.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDHYPE_H_
#define VECGEOM_VOLUMES_PLACEDHYPE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedHype.h"

namespace VECGEOM_NAMESPACE {

class PlacedHype : public VPlacedVolume {

public:

  typedef UnplacedHype UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedHype(char const *const label,
                   LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedHype(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : PlacedHype("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedHype(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedHype() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedHype const* GetUnplacedVolume() const {
    return static_cast<UnplacedHype const *>(
        logical_volume()->unplaced_volume());
  }

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_