/// \file PlacedTube.h
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDTUBE_H_
#define VECGEOM_VOLUMES_PLACEDTUBE_H_

#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TubeImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTube;)
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTube)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTube : public VPlacedVolume {

public:
  typedef UnplacedTube UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedTube(char const *const label, LogicalVolume const *const logical_volume,
             Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedTube(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
             PlacedBox const *const boundingBox)
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

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTube const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTube const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRMin() const { return GetUnplacedVolume()->rmin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInnerRadius() const { return GetUnplacedVolume()->rmin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRMax() const { return GetUnplacedVolume()->rmax(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetOuterRadius() const { return GetUnplacedVolume()->rmax(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->z(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetZHalfLength() const { return GetUnplacedVolume()->z(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSPhi() const { return GetUnplacedVolume()->sphi(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartPhiAngle() const { return GetUnplacedVolume()->sphi(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDPhi() const { return GetUnplacedVolume()->dphi(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDeltaPhiAngle() const { return GetUnplacedVolume()->dphi(); }

#ifndef VECGEOM_NVCC
  VECGEOM_INLINE
  void SetInnerRadius(Precision _rmin) { const_cast<UnplacedTube *>(GetUnplacedVolume())->SetRMin(_rmin); }

  VECGEOM_INLINE
  void SetOuterRadius(Precision _rmax) { const_cast<UnplacedTube *>(GetUnplacedVolume())->SetRMax(_rmax); }

  VECGEOM_INLINE
  void SetZHalfLength(Precision _z) { const_cast<UnplacedTube *>(GetUnplacedVolume())->SetDz(_z); }

  VECGEOM_INLINE
  void SetStartPhiAngle(Precision _sphi, bool /*compute=true*/)
  {
    const_cast<UnplacedTube *>(GetUnplacedVolume())->SetSPhi(_sphi);
  }

  VECGEOM_INLINE
  void SetDeltaPhiAngle(Precision _dphi) { const_cast<UnplacedTube *>(GetUnplacedVolume())->SetDPhi(_dphi); }

  VECGEOM_INLINE
  Precision SafetyFromInsideR(const Vector3D<Precision> &p, const Precision rho, bool precise = false) const
  {
    return GetUnplacedVolume()->SafetyFromInsideR(p, rho, precise);
  }

  VECGEOM_INLINE
  Precision SafetyFromOutsideR(const Vector3D<Precision> &p, const Precision rho, bool precise = false) const
  {
    return GetUnplacedVolume()->SafetyFromOutsideR(p, rho, precise);
  }

  virtual Vector3D<Precision> GetPointOnSurface() const override { return GetUnplacedVolume()->GetPointOnSurface(); }

  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual Precision SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }

  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#if defined(VECGEOM_USOLIDS)
  virtual std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType(); }
  //  VECGEOM_CUDA_HEADER_BOTH
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
#endif // VECGEOM_NVCC
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
