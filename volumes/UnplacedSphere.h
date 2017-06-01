/// \file UnplacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
#define VECGEOM_VOLUMES_UNPLACEDSPHERE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#include <cmath>
#endif

#include "volumes/Wedge.h"
#include "volumes/ThetaCone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedSphere;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedSphere);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSphere : public VUnplacedVolume, public AlignedBase {

private:
  // Radial and angular dimensions
  Precision fRmin, fRmax, fSPhi, fDPhi, fSTheta, fDTheta;

  // Radial and angular tolerances
  Precision fRminTolerance, mkTolerance, //, kAngTolerance, kRadTolerance,
      fEpsilon;

  // Cached trigonometric values for Phi angle
  Precision sinCPhi, cosCPhi, cosHDPhiOT, cosHDPhiIT, sinSPhi, cosSPhi, sinEPhi, cosEPhi, hDPhi, cPhi, ePhi;

  // Cached trigonometric values for Theta angle
  Precision sinSTheta, cosSTheta, sinETheta, cosETheta, tanSTheta, tanSTheta2, tanETheta, tanETheta2, eTheta;

  Precision fabsTanSTheta, fabsTanETheta;

  // Flags for identification of section, shell or full sphere
  bool fFullPhiSphere, fFullThetaSphere, fFullSphere;

  // Precomputed values computed from parameters
  Precision fCubicVolume, fSurfaceArea;

  // Tolerance compatiable with USolids
  // Precision epsilon;// = 2e-11;
  // Precision frTolerance;//=1e-9;     //radial tolerance;

  // Member variables go here
  // Precision fR,fRTolerance, fRTolI, fRTolO;
  Wedge fPhiWedge; // the Phi bounding of the Sphere
  ThetaCone fThetaCone;

public:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void InitializePhiTrigonometry()
  {
    hDPhi = 0.5 * fDPhi; // half delta phi
    cPhi  = fSPhi + hDPhi;
    ePhi  = fSPhi + fDPhi;

    sinCPhi    = std::sin(cPhi);
    cosCPhi    = std::cos(cPhi);
    cosHDPhiIT = std::cos(hDPhi - 0.5 * kAngTolerance); // inner/outer tol half dphi
    cosHDPhiOT = std::cos(hDPhi + 0.5 * kAngTolerance);
    sinSPhi    = std::sin(fSPhi);
    cosSPhi    = std::cos(fSPhi);
    sinEPhi    = std::sin(ePhi);
    cosEPhi    = std::cos(ePhi);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void InitializeThetaTrigonometry()
  {
    eTheta = fSTheta + fDTheta;

    sinSTheta = std::sin(fSTheta);
    cosSTheta = std::cos(fSTheta);
    sinETheta = std::sin(eTheta);
    cosETheta = std::cos(eTheta);

    tanSTheta     = std::tan(fSTheta);
    fabsTanSTheta = std::fabs(tanSTheta);
    tanSTheta2    = tanSTheta * tanSTheta;
    tanETheta     = std::tan(eTheta);
    fabsTanETheta = std::fabs(tanETheta);
    tanETheta2    = tanETheta * tanETheta;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CheckThetaAngles(Precision sTheta, Precision dTheta)
  {
    if ((sTheta < 0) || (sTheta > kPi)) {
      // std::ostringstream message;
      // message << "sTheta outside 0-PI range." << std::endl
      //       << "Invalid starting Theta angle for solid: " ;//<< GetName();
      // return;
      // UUtils::Exception("USphere::CheckThetaAngles()", "GeomSolids0002",
      //                FatalError, 1, message.str().c_str());
    } else {
      fSTheta = sTheta;
    }
    if (dTheta + sTheta >= kPi) {
      fDTheta = kPi - sTheta;
    } else if (dTheta > 0) {
      fDTheta = dTheta;
    } else {
      /*
        std::ostringstream message;
      message << "Invalid dTheta." << std::endl
              << "Negative delta-Theta (" << dTheta << "), for solid: ";
      return;
       */
      //<< GetName();
      // UUtils::Exception("USphere::CheckThetaAngles()", "GeomSolids0002",
      //                FatalError, 1, message.str().c_str());
    }
    if (fDTheta - fSTheta < kPi) {
      fFullThetaSphere = false;
    } else {
      fFullThetaSphere = true;
    }
    fFullSphere = fFullPhiSphere && fFullThetaSphere;

    InitializeThetaTrigonometry();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CheckSPhiAngle(Precision sPhi)
  {
    // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0

    if (sPhi < 0) {
      fSPhi = 2 * kPi - std::fmod(std::fabs(sPhi), 2 * kPi);
    } else {
      fSPhi = std::fmod(sPhi, 2 * kPi);
    }
    if (fSPhi + fDPhi > 2 * kPi) {
      fSPhi -= 2 * kPi;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CheckDPhiAngle(Precision dPhi)
  {
    if (dPhi >= 2 * kPi - kAngTolerance * 0.5) {

      fDPhi          = 2 * kPi;
      fSPhi          = 0;
      fFullPhiSphere = true;

    } else {
      fFullPhiSphere = false;
      if (dPhi > 0) {
        fDPhi = dPhi;
      } else {
        /*
      std::ostringstream message;
      message << "Invalid dphi." << std::endl
              << "Negative delta-Phi (" << dPhi << "), for solid: ";
      return;
         */

        // << GetName();
        // UUtils::Exception("USphere::CheckDPhiAngle()", "GeomSolids0002",
        //                FatalError, 1, message.str().c_str());
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CheckPhiAngles(Precision sPhi, Precision dPhi)
  {
    CheckDPhiAngle(dPhi);
    // if (!fFullPhiSphere && sPhi) { CheckSPhiAngle(sPhi); }
    if (!fFullPhiSphere) {
      CheckSPhiAngle(sPhi);
    }
    fFullSphere = fFullPhiSphere && fFullThetaSphere;

    InitializePhiTrigonometry();
  }

  // constructor
  // VECCORE_ATT_HOST_DEVICE
  // UnplacedSphere();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Wedge const &GetWedge() const { return fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ThetaCone const &GetThetaCone() const { return fThetaCone; }

  VECCORE_ATT_HOST_DEVICE
  UnplacedSphere(Precision pRmin, Precision pRmax, Precision pSPhi, Precision pDPhi, Precision pSTheta,
                 Precision pDTheta);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInsideRadius() const { return fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadius() const { return fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadius() const { return fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartPhiAngle() const { return fSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaPhiAngle() const { return fDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStartThetaAngle() const { return fSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDeltaThetaAngle() const { return fDTheta; }

  // Functions to get Tolerance
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFRminTolerance() const { return fRminTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetMKTolerance() const { return mkTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetAngTolerance() const { return kAngTolerance; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullSphere() const { return fFullSphere; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullPhiSphere() const { return fFullPhiSphere; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFullThetaSphere() const { return fFullThetaSphere; }

  // All angle related functions
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetHDPhi() const { return hDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCPhi() const { return cPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEPhi() const { return ePhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinCPhi() const { return sinCPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosCPhi() const { return cosCPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSPhi() const { return sinSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSPhi() const { return cosSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinEPhi() const { return sinEPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosEPhi() const { return cosEPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetETheta() const { return eTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinSTheta() const { return sinSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosSTheta() const { return cosSTheta; }

  //****************************************************************
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta() const { return tanSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta() const { return tanETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanSTheta() const { return fabsTanSTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetFabsTanETheta() const { return fabsTanETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanSTheta2() const { return tanSTheta2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanETheta2() const { return tanETheta2; }
  //****************************************************************

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSinETheta() const { return sinETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosETheta() const { return cosETheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiOT() const { return cosHDPhiOT; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetCosHDPhiIT() const { return cosHDPhiIT; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Initialize()
  {
    fCubicVolume = 0.;
    fSurfaceArea = 0.;
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetInsideRadius(Precision newRmin)
  {
    fRmin          = newRmin;
    fRminTolerance = (fRmin) ? std::max(kRadTolerance, fEpsilon * fRmin) : 0;
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetInnerRadius(Precision newRmin) { SetInsideRadius(newRmin); }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetOuterRadius(Precision newRmax)
  {
    fRmax       = newRmax;
    mkTolerance = std::max(kRadTolerance,
                           fEpsilon * fRmax); // RELOOK at kTolerance, may be we will take directly from base/global.h
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetStartPhiAngle(Precision newSPhi, bool compute = true)
  {
    // Flag 'compute' can be used to explicitely avoid recomputation of
    // trigonometry in case SetDeltaPhiAngle() is invoked afterwards

    CheckSPhiAngle(newSPhi);
    fFullPhiSphere = false;
    if (compute) {
      InitializePhiTrigonometry();
    }
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetDeltaPhiAngle(Precision newDPhi)
  {
    CheckPhiAngles(fSPhi, newDPhi);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetStartThetaAngle(Precision newSTheta)
  {
    CheckThetaAngles(newSTheta, fDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetDeltaThetaAngle(Precision newDTheta)
  {
    CheckThetaAngles(fSTheta, newDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }

  // Old access functions
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return GetInsideRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return GetOuterRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return GetStartPhiAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return GetDeltaPhiAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSTheta() const { return GetStartThetaAngle(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDTheta() const { return GetDeltaThetaAngle(); }

  //*****************************************************
  /*
  VECCORE_ATT_HOST_DEVICE
  Precision GetfRTolO() const { return fRTolO; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetfRTolI() const { return fRTolI; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetfRTolerance() const { return fRTolerance; }

  VECCORE_ATT_HOST_DEVICE
  void SetRadius (const Precision r);

  //_____________________________________________________________________________
  */

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

#if !defined(VECCORE_CUDA)
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Capacity() const { return fCubicVolume; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

#ifndef VECCORE_CUDA
  VECCORE_ATT_HOST_DEVICE
#endif
  Vector3D<Precision> SamplePointOnSurface() const;

  std::string GetEntityType() const;
#endif

  void GetParametersList(int aNumber, Precision *aArray) const;

  UnplacedSphere *Clone() const;

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

  VECCORE_ATT_HOST_DEVICE
  void ComputeBBox() const;

  // VECCORE_ATT_HOST_DEVICE
  // Precision sqr(Precision x) {return x*x;};

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  // VECCORE_ATT_HOST_DEVICE
  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                Transformation3D const *const transformation,
                                                const TranslationCode trans_code, const RotationCode rot_code,
                                                VPlacedVolume *const placement = NULL);

#else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);

  __device__ static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                           Transformation3D const *const transformation,
                                                           const TranslationCode trans_code,
                                                           const RotationCode rot_code, const int id,
                                                           VPlacedVolume *const placement = NULL);

#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedSphere>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:
#ifndef VECCORE_CUDA

  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
                                           VPlacedVolume *const placement = NULL) const final
  {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code, placement);
  }

#else

  __device__ virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation,
                                                      const TranslationCode trans_code, const RotationCode rot_code,
                                                      const int id, VPlacedVolume *const placement = NULL) const final
  {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code, id, placement);
  }

#endif
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
