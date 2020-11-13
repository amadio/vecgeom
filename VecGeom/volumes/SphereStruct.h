/*
 * SphereStruct.h
 *
 *  Created on: Jul 10, 2017
 *      Author: rsehgal
 */

#ifndef VECGEOM_SPHERESTRUCT_H_
#define VECGEOM_SPHERESTRUCT_H_

#include "VecGeom/volumes/Wedge_Evolution.h"
//#include "VecGeom/volumes/Wedge.h"
//#include "VecGeom/volumes/ThetaCone.h"
#include "VecGeom/volumes/ThetaCone_Evolution.h"
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct SphereStruct {
  T fRmin;
  T fRmax;
  T fSPhi;
  T fDPhi;
  T fSTheta;
  T fDTheta;

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

  Precision fCubicVolume, fSurfaceArea;

  // data members for Theta and Phi
  evolution::Wedge fPhiWedge;
  ThetaCone fThetaCone;

  Precision kAngTolerance;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void InitializePhiTrigonometry()
  {
    hDPhi = 0.5 * fDPhi; // half delta phi
    cPhi  = fSPhi + hDPhi;
    ePhi  = fSPhi + fDPhi;

    sinCPhi    = std::sin(cPhi);
    cosCPhi    = vecCore::math::Cos(cPhi);
    cosHDPhiIT = vecCore::math::Cos(hDPhi - 0.5 * kAngTolerance); // inner/outer tol half dphi
    cosHDPhiOT = vecCore::math::Cos(hDPhi + 0.5 * kAngTolerance);
    sinSPhi    = std::sin(fSPhi);
    cosSPhi    = vecCore::math::Cos(fSPhi);
    sinEPhi    = std::sin(ePhi);
    cosEPhi    = vecCore::math::Cos(ePhi);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void InitializeThetaTrigonometry()
  {
    eTheta = fSTheta + fDTheta;

    sinSTheta = std::sin(fSTheta);
    cosSTheta = vecCore::math::Cos(fSTheta);
    sinETheta = std::sin(eTheta);
    cosETheta = vecCore::math::Cos(eTheta);

    tanSTheta     = sinSTheta / cosSTheta;
    fabsTanSTheta = std::fabs(tanSTheta);
    tanSTheta2    = tanSTheta * tanSTheta;
    tanETheta     = sinETheta / cosETheta;
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

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetInsideRadius(Precision newRmin)
  {
    fRmin          = newRmin;
    fRminTolerance = (fRmin) ? std::max(kRadTolerance, fEpsilon * fRmin) : 0;
    Initialize();
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
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
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
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
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetDeltaPhiAngle(Precision newDPhi)
  {
    CheckPhiAngles(fSPhi, newDPhi);
    Initialize();
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetStartThetaAngle(Precision newSTheta)
  {
    CheckThetaAngles(newSTheta, fDTheta);
    Initialize();
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetDeltaThetaAngle(Precision newDTheta)
  {
    CheckThetaAngles(fSTheta, newDTheta);
    Initialize();
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Initialize()
  {
    fCubicVolume = 0.;
    fSurfaceArea = 0.;
  }

  // Constructor
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SphereStruct(T pRmin, T pRmax, T pSPhi, T pDPhi, T pSTheta, T pDTheta)
      : fRmin(pRmin), fRmax(pRmax), fSPhi(pSPhi), fDPhi(pDPhi), fSTheta(pSTheta), fDTheta(pDTheta), fRminTolerance(0),
        mkTolerance(0), fEpsilon(kEpsilon), sinCPhi(0), cosCPhi(0), cosHDPhiOT(0), cosHDPhiIT(0), sinSPhi(0),
        cosSPhi(0), sinEPhi(0), cosEPhi(0), hDPhi(0), cPhi(0), ePhi(0), sinSTheta(0), cosSTheta(0), sinETheta(0),
        cosETheta(0), tanSTheta(0), tanSTheta2(0), tanETheta(0), tanETheta2(0), eTheta(0), fFullPhiSphere(true),
        fFullThetaSphere(true), fFullSphere(true), fCubicVolume(0.), fSurfaceArea(0.), fPhiWedge(pDPhi, pSPhi),
        fThetaCone(pSTheta, pDTheta)
  {

    kAngTolerance  = 1e-9;
    fRminTolerance = (fRmin) ? Max(kRadTolerance, fEpsilon * fRmin) : 0;
    mkTolerance    = Max(kRadTolerance, fEpsilon * fRmax);

    CheckPhiAngles(pSPhi, pDPhi);
    CheckThetaAngles(pSTheta, pDTheta);
#ifndef VECCORE_CUDA
    CalcCapacity();
    CalcSurfaceArea();
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SphereStruct() {}

  //#ifndef VECCORE_CUDA
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CalcCapacity()
  {
    if (fCubicVolume != 0.) {
      ;
    } else {
      fCubicVolume = fDPhi * (vecCore::math::Cos(fSTheta) - vecCore::math::Cos(fSTheta + fDTheta)) *
                     (fRmax * fRmax * fRmax - fRmin * fRmin * fRmin) / 3.;
    }
  }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CalcSurfaceArea()
  {

    if (fSurfaceArea != 0.) {
      ;
    } else {
      Precision Rsq = fRmax * fRmax;
      Precision rsq = fRmin * fRmin;

      fSurfaceArea = fDPhi * (rsq + Rsq) * (cosSTheta - cosETheta);
      if (!fFullPhiSphere) {
        fSurfaceArea = fSurfaceArea + fDTheta * (Rsq - rsq);
      }
      if (fSTheta > 0) {
        Precision acos1 = 0.;
        if (fDPhi != kTwoPi)
          acos1 = vecCore::math::ACos((sinSTheta * sinSTheta) * vecCore::math::Cos(fDPhi) + (cosSTheta * cosSTheta));
        if (fDPhi > kPi) {
          fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * (2 * kPi - acos1);
        } else {
          fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * acos1;
        }
      }
      if (eTheta < kPi) {
        Precision acos2 = 0.;
        if (fDPhi != kTwoPi)
          acos2 = vecCore::math::ACos((sinETheta * sinETheta) * vecCore::math::Cos(fDPhi) + (cosETheta * cosETheta));
        if (fDPhi > kPi) {
          fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * (2 * kPi - acos2);
        } else {
          fSurfaceArea = fSurfaceArea + 0.5 * (Rsq - rsq) * acos2;
        }
      }
    }
  }
  //#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_SPHERESTRUCT_H_ */
