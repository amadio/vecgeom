/// \file PlacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDSPHERE_H_
#define VECGEOM_VOLUMES_PLACEDSPHERE_H_

#include "base/Global.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#include <cmath>
#endif
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/SphereImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedSphere.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedSphere;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedSphere);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedSphere : public PlacedVolumeImplHelper<UnplacedSphere, VPlacedVolume> {

  using Base = PlacedVolumeImplHelper<UnplacedSphere, VPlacedVolume>;

public:
  using Base::Base;

#ifndef VECCORE_CUDA

  PlacedSphere(char const *const label, LogicalVolume const *const logical_volume,
               Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : Base(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedSphere(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
               PlacedBox const *const boundingBox)
      : PlacedSphere("", logical_volume, transformation, boundingBox)
  {
  }

#else

  VECCORE_ATT_DEVICE PlacedSphere(LogicalVolume const *const logical_volume,
                                  Transformation3D const *const transformation, PlacedBox const *const boundingBox,
                                  const int id)
      : Base(logical_volume, transformation, boundingBox, id)
  {
  }

#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedSphere() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  UnplacedSphere const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedSphere const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  evolution::Wedge const &GetWedge() const { return GetUnplacedVolume()->GetWedge(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ThetaCone const &GetThetaCone() const { return GetUnplacedVolume()->GetThetaCone(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInsideRadius() const { return GetUnplacedVolume()->GetInsideRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadius() const { return GetUnplacedVolume()->GetInnerRadius(); }

  void SetInnerRadius(Precision arg) { const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetInnerRadius(arg); }
  void SetOuterRadius(Precision arg) { const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetOuterRadius(arg); }
  void SetStartPhiAngle(Precision arg, bool compute = true)
  {
    const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetStartPhiAngle(arg, compute);
  }
  void SetDeltaPhiAngle(Precision arg) { const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetDeltaPhiAngle(arg); }
  void SetStartThetaAngle(Precision arg) { const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetStartThetaAngle(arg); }
  void SetDeltaThetaAngle(Precision arg) { const_cast<UnplacedSphere *>(GetUnplacedVolume())->SetDeltaThetaAngle(arg); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadius() const { return GetUnplacedVolume()->GetOuterRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartPhiAngle() const { return GetUnplacedVolume()->GetStartPhiAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaPhiAngle() const { return GetUnplacedVolume()->GetDeltaPhiAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartThetaAngle() const { return GetUnplacedVolume()->GetStartThetaAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaThetaAngle() const { return GetUnplacedVolume()->GetDeltaThetaAngle(); }

  // Functions to get Tolerance
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFRminTolerance() const { return GetUnplacedVolume()->GetFRminTolerance(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetMKTolerance() const { return GetUnplacedVolume()->GetMKTolerance(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetAngTolerance() const { return GetUnplacedVolume()->GetAngTolerance(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullSphere() const { return GetUnplacedVolume()->IsFullSphere(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullPhiSphere() const { return GetUnplacedVolume()->IsFullPhiSphere(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullThetaSphere() const { return GetUnplacedVolume()->IsFullThetaSphere(); }

  // Function to return all Trignometric values
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetHDPhi() const { return GetUnplacedVolume()->GetHDPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCPhi() const { return GetUnplacedVolume()->GetCPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEPhi() const { return GetUnplacedVolume()->GetEPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinCPhi() const { return GetUnplacedVolume()->GetSinCPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosCPhi() const { return GetUnplacedVolume()->GetCosCPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSPhi() const { return GetUnplacedVolume()->GetSinSPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSPhi() const { return GetUnplacedVolume()->GetCosSPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinEPhi() const { return GetUnplacedVolume()->GetSinEPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosEPhi() const { return GetUnplacedVolume()->GetCosEPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetETheta() const { return GetUnplacedVolume()->GetETheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSTheta() const { return GetUnplacedVolume()->GetSinSTheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSTheta() const { return GetUnplacedVolume()->GetCosSTheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinETheta() const { return GetUnplacedVolume()->GetSinETheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosETheta() const { return GetUnplacedVolume()->GetCosETheta(); }

  //****************************************************************
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta() const { return GetUnplacedVolume()->GetTanSTheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta() const { return GetUnplacedVolume()->GetTanETheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanSTheta() const { return GetUnplacedVolume()->GetFabsTanSTheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanETheta() const { return GetUnplacedVolume()->GetFabsTanETheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta2() const { return GetUnplacedVolume()->GetTanSTheta2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta2() const { return GetUnplacedVolume()->GetTanETheta2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiOT() const { return GetUnplacedVolume()->GetCosHDPhiOT(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiIT() const { return GetUnplacedVolume()->GetCosHDPhiIT(); }
  //****************************************************************

  // Old access functions
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return GetUnplacedVolume()->GetRmin(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return GetUnplacedVolume()->GetRmax(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return GetUnplacedVolume()->GetSPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return GetUnplacedVolume()->GetDPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSTheta() const { return GetUnplacedVolume()->GetSTheta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDTheta() const { return GetUnplacedVolume()->GetDTheta(); }

#ifndef VECCORE_CUDA
  VECGEOM_FORCE_INLINE
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  VECGEOM_FORCE_INLINE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    return GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override { return GetUnplacedVolume()->ConvertToGeant4(GetName()); }
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDSPHERE_H_
