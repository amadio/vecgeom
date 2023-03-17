/*
 * PlacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 * 20180320 Guilherme Lima  Adapted to new factory of unplaced volumes
 */
#ifndef VECGEOM_VOLUMES_PLACEDCONE_H_
#define VECGEOM_VOLUMES_PLACEDCONE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/ConeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCone : public VPlacedVolume {

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  PlacedCone(char const *const label, LogicalVolume const *const logicalVolume,
             Transformation3D const *const transformation)
      : VPlacedVolume(label, logicalVolume, transformation)
  {
  }

  PlacedCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedCone("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                                const int id, const int copy_no, const int child_id)
      : VPlacedVolume(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedCone() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

  VECCORE_ATT_HOST_DEVICE
  UnplacedCone const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedCone const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override { return GetUnplacedVolume()->ConvertToGeant4(GetName()); }
#endif
#endif // VECGEOM_CUDA

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
  Precision GetOuterConeApex() const { return GetUnplacedVolume()->GetOuterConeApex(); }

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
}; // end class

// a placed cone knowing abouts its volume/structural specialization
template <typename UnplacedCone_t>
class SPlacedCone : public PlacedVolumeImplHelper<UnplacedCone_t, PlacedCone> {
  using Base = PlacedVolumeImplHelper<UnplacedCone_t, PlacedCone>;

public:
  typedef UnplacedCone UnplacedShape_t;
  using Base::Base;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDCONE_H_
