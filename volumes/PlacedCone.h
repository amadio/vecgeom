/*
 * PlacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */
#ifndef VECGEOM_VOLUMES_PLACEDCONE_H_
#define VECGEOM_VOLUMES_PLACEDCONE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedCone.h"
#include "volumes/kernel/ConeImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCone : public VPlacedVolume {

public:
  typedef UnplacedCone UnplacedShape_t;

#ifndef VECCORE_CUDA

  PlacedCone(char const *const label, LogicalVolume const *const logical_volume,
             Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedCone(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
             PlacedBox const *const boundingBox)
      : PlacedCone("", logical_volume, transformation, boundingBox)
  {
  }

#else

  __device__ PlacedCone(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id)
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

/*
VECGEOM_FORCE_INLINE
double SafetyFromInsideR(const Vector3D<Precision> &p, const double rho, bool precise = false) const
{
  return GetUnplacedVolume()->SafetyFromInsideR(p, rho, precise);
}

VECGEOM_FORCE_INLINE
double SafetyFromOutsideR(const Vector3D<Precision> &p, const double rho, bool precise = false) const
{
  return GetUnplacedVolume()->SafetyFromOutsideR(p, rho, precise);
}
*/
#if !defined(VECCORE_CUDA)
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    /*
    bool valid;
    ConeImplementation<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::NormalKernel<kScalar>(
        *GetUnplacedVolume(), point, normal, valid);
    return valid;
    */
    return GetUnplacedVolume()->Normal(point, normal);
  }

  virtual Vector3D<Precision> SamplePointOnSurface() const override
  {
    return GetUnplacedVolume()->SamplePointOnSurface();
  }

  virtual double SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }

#if defined(VECGEOM_USOLIDS)
  virtual std::string GetEntityType() const override { return "Cone"; }
  virtual Vector3D<Precision> GetPointOnSurface() const override { return GetUnplacedVolume()->SamplePointOnSurface(); }
#endif
#endif

}; // end class
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDCONE_H_
