/*
 * PlacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */
#ifndef VECGEOM_VOLUMES_PLACEDCONE_H_
#define VECGEOM_VOLUMES_PLACEDCONE_H_

#include "base/Global.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#include <cmath>
#endif
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedCone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCone : public PlacedVolumeImplHelper<UnplacedCone, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedCone, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedCone(char const *const label, LogicalVolume const *const logicalVolume,
             Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
             vecgeom::PlacedBox const *const boundingBox)
      : PlacedCone("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  __device__ PlacedCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedCone() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedCone const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedCone const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

#if defined(VECGEOM_USOLIDS)
  //  VECCORE_ATT_HOST_DEVICE
  std::ostream &StreamInfo(std::ostream &os) const override { return GetUnplacedVolume()->StreamInfo(os); }
#endif

#ifndef VECCORE_CUDA
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
#endif // VECGEOM_BENCHMARK

  Precision GetRmin1() const { return GetUnplacedVolume()->GetRmin1(); }
  Precision GetRmax1() const { return GetUnplacedVolume()->GetRmax1(); }
  Precision GetRmin2() const { return GetUnplacedVolume()->GetRmin2(); }
  Precision GetRmax2() const { return GetUnplacedVolume()->GetRmax2(); }
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }
  Precision GetSPhi() const { return GetUnplacedVolume()->GetSPhi(); }
  Precision GetDPhi() const { return GetUnplacedVolume()->GetDPhi(); }
  Precision GetInnerSlope() const { return GetUnplacedVolume()->GetInnerSlope(); }
  Precision GetOuterSlope() const { return GetUnplacedVolume()->GetOuterSlope(); }
  Precision GetInnerOffset() const { return GetUnplacedVolume()->GetInnerOffset(); }
  Precision GetOuterOffset() const { return GetUnplacedVolume()->GetOuterOffset(); }

  // interface required by Geant4
  Precision GetInnerRadiusMinusZ() const { return GetUnplacedVolume()->GetRmin1(); }
  Precision GetOuterRadiusMinusZ() const { return GetUnplacedVolume()->GetRmax1(); }
  Precision GetInnerRadiusPlusZ() const { return GetUnplacedVolume()->GetRmin2(); }
  Precision GetOuterRadiusPlusZ() const { return GetUnplacedVolume()->GetRmax2(); }
  Precision GetZHalfLength() const { return GetUnplacedVolume()->GetDz(); }
  Precision GetStartPhiAngle() const { return GetUnplacedVolume()->GetSPhi(); }
  Precision GetDeltaPhiAngle() const { return GetUnplacedVolume()->GetDPhi(); }

  void SetInnerRadiusMinusZ(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetRmin1(xin); }
  void SetOuterRadiusMinusZ(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetRmax1(xin); }
  void SetInnerRadiusPlusZ(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetRmin2(xin); }
  void SetOuterRadiusPlusZ(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetRmax2(xin); }
  void SetZHalfLength(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetDz(xin); }
  void SetStartPhiAngle(Precision xin, bool) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetSPhi(xin); }
  void SetDeltaPhiAngle(Precision xin) { const_cast<UnplacedCone *>(GetUnplacedVolume())->SetDPhi(xin); }

#if !defined(VECCORE_CUDA)
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  virtual Vector3D<Precision> SamplePointOnSurface() const override
  {
    return GetUnplacedVolume()->SamplePointOnSurface();
  }

  virtual double SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }

#if defined(VECGEOM_USOLIDS)
  virtual std::string GetEntityType() const override { return "Cone"; }
#endif
#endif

}; // end class
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDCONE_H_
