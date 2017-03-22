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
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedCone(Precision rmin1, Precision rmax1, Precision rmin2, Precision rmax2, Precision dz, Precision phimin,
               Precision deltaphi);

  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y)
  {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInvSecRMax() const { return fInvSecRMax; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInvSecRMin() const { return fInvSecRMin; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTolIz() const { return fTolIz; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTolOz() const { return fTolOz; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetConeTolerane() const { return fConeTolerance; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmin1() const { return fSqRmin1; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmin2() const { return fSqRmin2; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmax1() const { return fSqRmax1; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmax2() const { return fSqRmax2; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmin1Tol() const { return fSqRmin1Tol; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmin2Tol() const { return fSqRmin2Tol; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmax1Tol() const { return fSqRmax1Tol; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSqRmax2Tol() const { return fSqRmax2Tol; }
  VECGEOM_CUDA_HEADER_BOTH
  bool NeedsRminTreatment() const { return fneedsRminTreatment; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanRmax() const { return fTanRMax; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanRmin() const { return fTanRMin; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSecRmax() const { return fSecRMax; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSecRmin() const { return fSecRMin; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZNormInner() const { return fZNormInner; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZNormOuter() const { return fZNormOuter; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerConeApex() const { return fInnerConeApex; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTInner() const { return fTanInnerApexAngle; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTInner2() const { return fTanInnerApexAngle2; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterConeApex() const { return fOuterConeApex; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTOuter() const { return fTanOuterApexAngle; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTOuter2() const { return fTanOuterApexAngle2; }

  // VECGEOM_CUDA_HEADER_BOTH
  // virtual bool IsConvex() const override;
  VECGEOM_CUDA_HEADER_BOTH
  void DetectConvexity();
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRminAv() const { return fRminAv; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRmaxAv() const { return fRmaxAv; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRmin1() const { return fRmin1; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRmax1() const { return fRmax1; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRmin2() const { return fRmin2; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRmax2() const { return fRmax2; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDz() const { return fDz; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSPhi() const { return fSPhi; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDPhi() const { return fDPhi; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerSlope() const { return fInnerSlope; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterSlope() const { return fOuterSlope; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerSlopeInv() const { return fInnerSlopeInv; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterSlopeInv() const { return fOuterSlopeInv; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerOffset() const { return fInnerOffset; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterOffset() const { return fOuterOffset; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerSlopeSquare() const { return fInnerSlopeSquare; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterSlopeSquare() const { return fOuterSlopeSquare; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerOffsetSquare() const { return fInnerOffsetSquare; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterOffsetSquare() const { return fOuterOffsetSquare; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlongPhi1X() const { return fAlongPhi1x; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlongPhi1Y() const { return fAlongPhi1y; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlongPhi2X() const { return fAlongPhi2x; }
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlongPhi2Y() const { return fAlongPhi2y; }
  VECGEOM_CUDA_HEADER_BOTH
  Wedge const &GetWedge() const { return fPhiWedge; }

  void SetRmin1(Precision const &arg) { fRmin1 = arg; }
  void SetRmax1(Precision const &arg) { fRmax1 = arg; }
  void SetRmin2(Precision const &arg) { fRmin2 = arg; }
  void SetRmax2(Precision const &arg) { fRmax2 = arg; }
  void SetDz(Precision const &arg) { fDz = arg; }
  void SetSPhi(Precision const &arg) { fSPhi = arg; }
  void SetDPhi(Precision const &arg) { fDPhi = arg; }

  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullPhi() const { return fDPhi == kTwoPi; }

  virtual int MemorySize() const final { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;
  virtual void Print(std::ostream &os) const final;

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedCone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#ifndef VECGEOM_NVCC
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
  Vector3D<Precision> GetPointOnSurface() const;

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

#endif // !VECGEOM_NVCC
};
}
} // End global namespace

#endif
