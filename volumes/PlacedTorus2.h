/// \file PlacedTorus2.h

#ifndef VECGEOM_VOLUMES_PLACEDTORUS2_H_
#define VECGEOM_VOLUMES_PLACEDTORUS2_H_

#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TorusImplementation2.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTorus2;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTorus2);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTorus2 : public VPlacedVolume {

public:
  typedef UnplacedTorus2 UnplacedShape_t;

#ifndef VECCORE_CUDA

  PlacedTorus2(char const *const label, LogicalVolume const *const logical_volume,
               Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedTorus2(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
               PlacedBox const *const boundingBox)
      : PlacedTorus2("", logical_volume, transformation, boundingBox)
  {
  }

#else

  __device__ PlacedTorus2(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                          PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id)
  {
  }

#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTorus2() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedTorus2 const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTorus2 const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmin() const { return GetUnplacedVolume()->rmin(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rmax() const { return GetUnplacedVolume()->rmax(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision rtor() const { return GetUnplacedVolume()->rtor(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision sphi() const { return GetUnplacedVolume()->sphi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dphi() const { return GetUnplacedVolume()->dphi(); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin() const { return GetUnplacedVolume()->rmin(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax() const { return GetUnplacedVolume()->rmax(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRtor() const { return GetUnplacedVolume()->rtor(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSPhi() const { return GetUnplacedVolume()->sphi(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDPhi() const { return GetUnplacedVolume()->dphi(); }

  void SetRmin(Precision arg) { const_cast<UnplacedTorus2 *>(GetUnplacedVolume())->SetRMin(arg); }
  void SetRmax(Precision arg) { const_cast<UnplacedTorus2 *>(GetUnplacedVolume())->SetRMax(arg); }
  void SetRtor(Precision arg) { const_cast<UnplacedTorus2 *>(GetUnplacedVolume())->SetRTor(arg); }
  void SetSPhi(Precision arg) { const_cast<UnplacedTorus2 *>(GetUnplacedVolume())->SetSPhi(arg); }
  void SetDPhi(Precision arg) { const_cast<UnplacedTorus2 *>(GetUnplacedVolume())->SetDPhi(arg); }

  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#ifndef VECCORE_CUDA
  virtual Vector3D<Precision> GetPointOnSurface() const override { return GetUnplacedVolume()->GetPointOnSurface(); }
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

#if defined(VECGEOM_USOLIDS)
  //  VECCORE_ATT_HOST_DEVICE
  std::ostream &StreamInfo(std::ostream &os) const override { return GetUnplacedVolume()->StreamInfo(os); }
#endif

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
#endif // VECCORE_CUDA
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTORUS2_H_
