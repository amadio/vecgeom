/*
 * UnplacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDCONE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/Wedge.h"
#include <cmath>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * Class representing an unplaced cone; Encapsulated parameters of a cone and
 * functions that do not depend on how the cone is placed in a reference frame
 *
 * The unplaced cone is represented by the following parameters
 *
 * Member Data:
 *
 *  fDz half length in z direction;  ( the cone has height 2*fDz )
 *  fRmin1  inside radius at  -fDz ( in internal coordinate system )
 *  fRmin2  inside radius at  +fDz
 *  fRmax1  outside radius at -fDz
 *  fRmax2  outside radius at +fDz
 *  fSPhi starting angle of the segment in radians
 *  fDPhi delta angle of the segment in radians
 */
class UnplacedCone : public VUnplacedVolume, public AlignedBase {

private:
  Precision fRmin1;
  Precision fRmax1;
  Precision fRmin2;
  Precision fRmax2;
  Precision fDz;
  Precision fSPhi;
  Precision fDPhi;
  Wedge fPhiWedge; // the Phi bounding of the cone (not the cutout) -- will be able to get rid of the next angles
  // Precision innerSlantHeight,outerSlantHeight,partAreaInner,partAreaOuter,innerRadDiff,outerRadDiff,fDzInv;

  // vectors characterizing the normals of phi planes
  // makes task to detect phi sektors very efficient
  Vector3D<Precision> fNormalPhi1;
  Vector3D<Precision> fNormalPhi2;
  Precision fAlongPhi1x;
  Precision fAlongPhi1y;
  Precision fAlongPhi2x;
  Precision fAlongPhi2y;

  // Some precomputed values to avoid divisions etc
  Precision fInnerSlope;    // "gradient" of inner surface in z direction
  Precision fOuterSlope;    // "gradient" of outer surface in z direction
  Precision fInnerSlopeInv; // Inverse of innerSlope
  Precision fOuterSlopeInv; // Inverse of outerSlope
  Precision fInnerOffset;
  Precision fOuterOffset;
  Precision fOuterSlopeSquare;
  Precision fInnerSlopeSquare;
  Precision fOuterOffsetSquare;
  Precision fInnerOffsetSquare;

  // Values to be cached
  Precision fSqRmin1, fSqRmin2;
  Precision fSqRmax1, fSqRmax2;
  Precision fSqRmin1Tol, fSqRmin2Tol, fSqRmax1Tol, fSqRmax2Tol;
  Precision fTolIz, fTolOz;

  Precision fInnerConeApex;
  Precision fTanInnerApexAngle;
  Precision fTanInnerApexAngle2;

  Precision fOuterConeApex;
  Precision fTanOuterApexAngle;
  Precision fTanOuterApexAngle2;

  Precision fSecRMin;
  Precision fSecRMax;
  Precision fInvSecRMin;
  Precision fInvSecRMax;
  Precision fTanRMin;
  Precision fTanRMax;
  Precision fZNormInner;
  Precision fZNormOuter;
  Precision fRminAv;
  Precision fRmaxAv;
  bool fneedsRminTreatment;
  Precision fConeTolerance;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedCone(Precision rmin1, Precision rmax1, Precision rmin2, Precision rmax2, Precision dz, Precision phimin,
               Precision deltaphi);

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached();

  VECCORE_ATT_HOST_DEVICE
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y)
  {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetInvSecRMax() const { return fInvSecRMax; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetInvSecRMin() const { return fInvSecRMin; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTolIz() const { return fTolIz; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTolOz() const { return fTolOz; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetConeTolerane() const { return fConeTolerance; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin1() const { return fSqRmin1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin2() const { return fSqRmin2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax1() const { return fSqRmax1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax2() const { return fSqRmax2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin1Tol() const { return fSqRmin1Tol; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin2Tol() const { return fSqRmin2Tol; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax1Tol() const { return fSqRmax1Tol; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax2Tol() const { return fSqRmax2Tol; }
  VECCORE_ATT_HOST_DEVICE
  bool NeedsRminTreatment() const { return fneedsRminTreatment; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanRmax() const { return fTanRMax; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanRmin() const { return fTanRMin; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSecRmax() const { return fSecRMax; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSecRmin() const { return fSecRMin; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetZNormInner() const { return fZNormInner; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetZNormOuter() const { return fZNormOuter; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerConeApex() const { return fInnerConeApex; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTInner() const { return fTanInnerApexAngle; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTInner2() const { return fTanInnerApexAngle2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterConeApex() const { return fOuterConeApex; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTOuter() const { return fTanOuterApexAngle; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTOuter2() const { return fTanOuterApexAngle2; }

  // VECCORE_ATT_HOST_DEVICE
  // virtual bool IsConvex() const override;
  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();
  VECCORE_ATT_HOST_DEVICE
  Precision GetRminAv() const { return fRminAv; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmaxAv() const { return fRmaxAv; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin1() const { return fRmin1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax1() const { return fRmax1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin2() const { return fRmin2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax2() const { return fRmax2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDz() const { return fDz; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSPhi() const { return fSPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDPhi() const { return fDPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerSlope() const { return fInnerSlope; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterSlope() const { return fOuterSlope; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerSlopeInv() const { return fInnerSlopeInv; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterSlopeInv() const { return fOuterSlopeInv; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerOffset() const { return fInnerOffset; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterOffset() const { return fOuterOffset; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerSlopeSquare() const { return fInnerSlopeSquare; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterSlopeSquare() const { return fOuterSlopeSquare; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerOffsetSquare() const { return fInnerOffsetSquare; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterOffsetSquare() const { return fOuterOffsetSquare; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi1X() const { return fAlongPhi1x; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi1Y() const { return fAlongPhi1y; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi2X() const { return fAlongPhi2x; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi2Y() const { return fAlongPhi2y; }
  VECCORE_ATT_HOST_DEVICE
  Wedge const &GetWedge() const { return fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckSPhiAngle(Precision sPhi);

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckDPhiAngle(Precision dPhi);

  void SetRmin1(Precision const &arg)
  {
    fRmin1 = arg;
    CalculateCached();
  }
  void SetRmax1(Precision const &arg)
  {
    fRmax1 = arg;
    CalculateCached();
  }
  void SetRmin2(Precision const &arg)
  {
    fRmin2 = arg;
    CalculateCached();
  }
  void SetRmax2(Precision const &arg)
  {
    fRmax2 = arg;
    CalculateCached();
  }
  void SetDz(Precision const &arg)
  {
    fDz = arg;
    CalculateCached();
  }
  void SetSPhi(Precision const &arg)
  {
    fSPhi = arg;
    SetAndCheckSPhiAngle(fSPhi);
    DetectConvexity();
  }
  void SetDPhi(Precision const &arg)
  {
    fDPhi = arg;
    SetAndCheckDPhiAngle(fDPhi);
    DetectConvexity();
  }

  VECCORE_ATT_HOST_DEVICE
  bool IsFullPhi() const { return fDPhi == kTwoPi; }

  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;
  virtual void Print(std::ostream &os) const final;

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedCone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#ifndef VECCORE_CUDA
  Precision Capacity() const
  {
    return (fDz * fDPhi / 3.) *
           (fRmax1 * fRmax1 + fRmax2 * fRmax2 + fRmax1 * fRmax2 - fRmin1 * fRmin1 - fRmin2 * fRmin2 - fRmin1 * fRmin2);
  }

  Precision SurfaceArea() const
  {
    double mmin, mmax, dmin, dmax;
    mmin = (fRmin1 + fRmin2) * 0.5;
    mmax = (fRmax1 + fRmax2) * 0.5;
    dmin = (fRmin2 - fRmin1);
    dmax = (fRmax2 - fRmax1);

    return fDPhi * (mmin * std::sqrt(dmin * dmin + 4 * fDz * fDz) + mmax * std::sqrt(dmax * dmax + 4 * fDz * fDz) +
                    0.5 * (fRmax1 * fRmax1 - fRmin1 * fRmin1 + fRmax2 * fRmax2 - fRmin2 * fRmin2));
  }

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const;

  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;
  Vector3D<Precision> SamplePointOnSurface() const;

  // Helper funtion to detect edge points
  template <bool top>
  bool IsOnZPlane(Vector3D<Precision> const &point) const;
  template <bool start>
  bool IsOnPhiWedge(Vector3D<Precision> const &point) const;
  template <bool inner>
  bool IsOnConicalSurface(Vector3D<Precision> const &point) const;
  template <bool inner>
  Precision GetRadiusOfConeAtPoint(Precision const pointZ) const;

  bool IsOnEdge(Vector3D<Precision> &point) const;

  std::string GetEntityType() const { return "Cone"; }

#endif // !VECCORE_CUDA
};
}
} // End global namespace

#endif
