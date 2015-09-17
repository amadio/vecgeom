/// @file UnplacedTube.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTUBE_H_
#define VECGEOM_VOLUMES_UNPLACEDTUBE_H_

#include "base/Global.h"
#include "base/RNG.h"
#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/Wedge.h"
#include <sstream>
#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedTube; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedTube )

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTube : public VUnplacedVolume, public AlignedBase {

private:
  // tube defining parameters
  Precision fRmin, fRmax, fZ, fSphi, fDphi;

  // cached values
  Precision fRmin2, fRmax2, fAlongPhi1x, fAlongPhi1y, fAlongPhi2x, fAlongPhi2y;
  Precision fTolIrmin2, fTolOrmin2, fTolIrmax2, fTolOrmax2, fTolIz, fTolOz;
  Precision fTolIrmin, fTolOrmin, fTolIrmax, fTolOrmax;
  Wedge fPhiWedge;

  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void calculateCached() {
    fTolIz = fZ - kTolerance;
    fTolOz = fZ + kTolerance;

    fRmin2 = fRmin * fRmin;
    fRmax2 = fRmax * fRmax;

    fTolOrmin = (fRmin - kTolerance);
    fTolIrmin = (fRmin + kTolerance);
    fTolOrmin2 = fTolOrmin * fTolOrmin;
    fTolIrmin2 = fTolIrmin * fTolIrmin;

    fTolOrmax = (fRmax + kTolerance);
    fTolIrmax = (fRmax - kTolerance);
    fTolOrmax2 = fTolOrmax * fTolOrmax;
    fTolIrmax2 = fTolIrmax * fTolIrmax;

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTube(Precision const& _rmin, Precision const& _rmax, Precision const& _z,
               Precision const& _sphi, Precision const& _dphi)
    : fRmin(_rmin<0.0?0.0:_rmin), fRmax(_rmax), fZ(_z), fSphi(_sphi), fDphi(_dphi),
fRmin2(0),
fRmax2(0),
fAlongPhi1x(0),
fAlongPhi1y(0),
fAlongPhi2x(0),
fAlongPhi2y(0),
fTolIrmin2(0),
fTolOrmin2(0),
fTolIrmax2(0),
fTolOrmax2(0),
fTolIz(0),
fTolOz(0),
fPhiWedge(_dphi,_sphi)
{
    calculateCached();
}

  VECGEOM_CUDA_HEADER_BOTH
     UnplacedTube(UnplacedTube const &other) : fRmin(other.fRmin), fRmax(other.fRmax), fZ(other.fZ), fSphi(other.fSphi), fDphi(other.fDphi),
fRmin2(other.fRmin2),
fRmax2(other.fRmax2),
fAlongPhi1x(other.fAlongPhi1x),
fAlongPhi1y(other.fAlongPhi1y),
fAlongPhi2x(other.fAlongPhi2x),
fAlongPhi2y(other.fAlongPhi2y),
fTolIrmin2(other.fTolIrmin2),
fTolOrmin2(other.fTolOrmin2),
fTolIrmax2(other.fTolIrmax2),
fTolOrmax2(other.fTolOrmax2),
fTolIz(other.fTolIz),
fTolOz(other.fTolOz),
fPhiWedge(other.fDphi,other.fSphi)
{  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return fRmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return fRmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return fZ; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision sphi() const { return fSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dphi() const { return fDphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetRMin(Precision const& _rmin) { fRmin = _rmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetRMax(Precision const& _rmax) { fRmax = _rmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetDz(Precision const& _z) { fZ = _z; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetSPhi(Precision const& _sphi) { CheckSPhiAngle(_sphi); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetDPhi(Precision const& _dphi) { CheckDPhiAngle(_dphi); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin2() const { return fRmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax2() const { return fRmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1x() const { return fAlongPhi1x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1y() const { return fAlongPhi1y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2x() const { return fAlongPhi2x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2y() const { return fAlongPhi2y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIz() const { return fTolIz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOz() const { return fTolOz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmin2() const { return fTolOrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmin2() const { return fTolIrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmax2() const { return fTolOrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmax2() const { return fTolIrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Wedge const & GetWedge() const { return fPhiWedge; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void CheckSPhiAngle(double sPhi) {
    // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0
    if (sPhi < 0) {
      fSphi = kTwoPi - std::fmod(std::fabs(sPhi), kTwoPi);
    }
    else {
      fSphi = std::fmod(sPhi, kTwoPi);
    }
    if (fSphi + fDphi > kTwoPi) {
      fSphi -= kTwoPi ;
    }
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void CheckDPhiAngle(double dPhi) {
    if (dPhi >= kTwoPi - 0.5*kAngTolerance) {
      fDphi = kTwoPi;
      fSphi = 0;
    }
    else {
      if (dPhi > 0) {
        fDphi = dPhi;
      }
      else {
        std::ostringstream message;
        message << "Invalid dphi.\n"
                << "Negative or zero delta-Phi (" << dPhi << ")\n";
        std::cout<<"UnplacedTube::CheckDPhiAngle(): Fatal error: "<< message.str().c_str() <<"\n";
        // UUtils::Exception("UTubs::CheckDPhiAngle()", "GeomSolids0002",
        //                   UFatalError, 1, message.str().c_str());
      }
    }
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void CheckPhiAngles(double sPhi, double dPhi) {
    CheckDPhiAngle(dPhi);
    if ((fDphi < kTwoPi) && (sPhi)) {
      CheckSPhiAngle(sPhi);
    }
    calculateCached();
  }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  double SafetyToPhi(Vector3D<Precision> const& p, double rho, bool& outside) const {

    double safePhi = 0.0;
    double cosCPhi, sinCPhi;
    GetAlongVectorToPhiSector(fSphi+0.5*fDphi, cosCPhi, sinCPhi);

    // Psi=angle from central phi to point
    double cosPsi = (p.x() * cosCPhi + p.y() * sinCPhi) / rho;
    outside = false;
    if (cosPsi < std::cos(fDphi * 0.5)) {
      // Point lies outside phi range
      outside=true;
      if ((p.y() * cosCPhi - p.x() * sinCPhi) <= 0) {
        safePhi = std::fabs(p.x() * fAlongPhi1y - p.y() * fAlongPhi1x);
      }
      else {
        safePhi = std::fabs(p.x() * fAlongPhi2y - p.y() * fAlongPhi2x);
      }
    }

    return safePhi;
  }

  // Safety From Inside R, used for UPolycone Section
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  double SafetyFromInsideR(Vector3D<Precision> const& p, double rho, bool /*precise = false*/) const {
    double safe = fRmax - rho;
    if (fRmin) {
      safe = Min<Precision>(safe, rho-fRmin);
    }

    // Check if phi divided, compute distance to closest phi plane
    if (fDphi<kTwoPi) {
      double safePhi1 = p.y() * fAlongPhi1x - p.x() * fAlongPhi1y;
      double safePhi2 = p.x() * fAlongPhi2y - p.y() * fAlongPhi2x;
      safe = Min<Precision>( safe, Min(safePhi1, safePhi2) );
    }

    return safe;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  double SafetyFromOutsideR(Vector3D<Precision> const& p, double rho, bool /*precise = false*/ ) const {
    // Safety for R, used in UPolycone for sections
    double safe = fRmax - rho;
    if (fRmin) {
      safe = Min<Precision>(safe, rho-fRmin);
    }

    // GL Note: I'm not sure point is checked whether inside or outside w.r.t. phi planes and DPhi
    // e.g. I don't see any distinction for DPhi<Pi or DPhi>Pi (maybe necessary?).  I plan to test it further.
    double safePhi;
    bool outside;
    if( fDphi<kTwoPi && (rho) ) {
      safePhi = SafetyToPhi(p,rho,outside);
      if ((outside) && (safePhi > safe)) {
        safe = safePhi;
      }
    }

    return safe; // not accurate safety
  }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    return fZ * (fRmax2 - fRmin2) * fDphi;
  }

//#if !defined(VECGEOM_NVCC)
  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const;

#ifndef VECGEOM_NVCC
  Vector3D<Precision> GetPointOnSurface() const;

  //VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const {
      return volume();
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea () const {
      return GetTopArea() + GetLateralPhiArea() + GetLateralROutArea() + GetLateralRInArea();
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetTopArea() const {            // Abhijit:: this is top and bottom circular area of tube
      return 2*0.5*(fRmax2 - fRmin2) * fDphi;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralPhiArea() const {     // Abhijit:: this is vertical Phi_start and phi_end opening
      // factor of 2 since fZ is half length
      return (fDphi < vecgeom::kTwoPi) ? 4.*fZ*(fRmax - fRmin) : 0.;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralRInArea() const {    // Abhijit:: this is Inner surface of tube along Z
      // factor of 2 since fZ is half length
      return 2.*fZ*fRmin*fDphi;
  }

  //VECGEOM_CUDA_HEADER_BOTH
  Precision GetLateralROutArea() const {  // Abhijit:: this is Outer surface of tube along Z
      // factor of 2 since fZ is half length
      return 2.*fZ*fRmax*fDphi;
  }

  //  This computes where the random point would be placed
  // 1::rTop, 2::rBot, 3::phiLeft, 4::phiRight, 5::zIn, 6::zOut
  //VECGEOM_CUDA_HEADER_BOTH
  int ChooseSurface() const;

  bool Normal(Vector3D<Precision>const& point, Vector3D<Precision>& normal) const;

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  std::string GetEntityType() const { return "Tube";}

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTube>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#if defined(VECGEOM_USOLIDS)
  std::ostream& StreamInfo(std::ostream &os) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

};

} } // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTUBE_H_
