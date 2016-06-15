#ifndef VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
#define VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedBooleanVolume.h"

#ifdef VECGEOM_ROOT
  class TGeoShape;
#endif
#ifdef VECGEOM_GEANT4
  class G4VSolid;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedBooleanVolume; )
VECGEOM_DEVICE_DECLARE_CONV( class, PlacedBooleanVolume )

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBooleanVolume : public VPlacedVolume {

    typedef UnplacedBooleanVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC
  PlacedBooleanVolume(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedBooleanVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedBooleanVolume("", logicalVolume, transformation, boundingBox) {}
#else
  __device__
  PlacedBooleanVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedBooleanVolume() {}


  VECGEOM_CUDA_HEADER_BOTH
  UnplacedVol_t const* GetUnplacedVolume() const {
    return static_cast<UnplacedVol_t const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

//#ifndef VECGEOM_NVCC
  virtual Precision Capacity() override {
       // TODO: implement this
      return 0.;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType() ;}
#endif

  virtual Vector3D<Precision> GetPointOnSurface() const override;
//#endif

  VECGEOM_CUDA_HEADER_BOTH
  void PrintType() const override { };

  void PrintType(std::ostream &) const override { };

  // CUDA specific
  virtual int memory_size() const override { return sizeof(*this); }

  // Comparison specific

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const override {
   return this;
  }
#ifdef VECGEOM_ROOT
 virtual TGeoShape const* ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const* ConvertToUSolids() const override {
    // currently not supported in USOLIDS -- returning NULL
      return nullptr;
  }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override;
#endif
#endif // VECGEOM_BENCHMARK

}; // end class declaration

} // End impl namespace
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
