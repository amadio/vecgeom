/*
 * TrdStruct.h
 *
 *  Created on: 20.02.2017
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_TRDSTRUCT_H_
#define VECGEOM_VOLUMES_TRDSTRUCT_H_
#include "base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct TrdStruct {
  T fDX1; // Half-length along x at the surface positioned at -dz
  T fDX2; // Half-length along x at the surface positioned at +dz
  T fDY1; // Half-length along y at the surface positioned at -dz
  T fDY2; // Half-length along y at the surface positioned at +dz
  T fDZ;  // Half-length along z axis

  // cached values
  T fX2minusX1;
  T fY2minusY1;
  T fHalfX1plusX2;
  T fHalfY1plusY2;
  T fCalfX, fCalfY;
  T fSecxz, fSecyz;
  T fToleranceX; // Corrected tolerance for Inside checks on X
  T fToleranceY; // Corrected tolerance for Inside checks on Y

  T fFx, fFy;

  VECCORE_ATT_HOST_DEVICE
  TrdStruct() {}

  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x1, const T x2, const T y1, const T z)
      : fDX1(x1), fDX2(x2), fDY1(y1), fDY2(y1), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0),
        fHalfY1plusY2(0), fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x1, const T x2, const T y1, const T y2, const T z)
      : fDX1(x1), fDX2(x2), fDY1(y1), fDY2(y2), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0),
        fHalfY1plusY2(0), fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x1, const T y1, const T z)
      : fDX1(x1), fDX2(x1), fDY1(y1), fDY2(y1), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0),
        fHalfY1plusY2(0), fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAllParameters(T x1, T x2, T y1, T y2, T z)
  {
    fDX1 = x1;
    fDX2 = x2;
    fDY1 = y1;
    fDY2 = y2;
    fDZ  = z;
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {
    fX2minusX1    = fDX2 - fDX1;
    fY2minusY1    = fDY2 - fDY1;
    fHalfX1plusX2 = 0.5 * (fDX1 + fDX2);
    fHalfY1plusY2 = 0.5 * (fDY1 + fDY2);

    fFx    = 0.5 * (fDX1 - fDX2) / fDZ;
    fFy    = 0.5 * (fDY1 - fDY2) / fDZ;
    fSecxz = sqrt(1 + fFx * fFx);
    fSecyz = sqrt(1 + fFy * fFy);

    fCalfX      = 1. / Sqrt(1.0 + fFx * fFx);
    fCalfY      = 1. / Sqrt(1.0 + fFy * fFy);
    fToleranceX = kTolerance * Sqrt(fX2minusX1 * fX2minusX1 + 4 * fDZ * fDZ);
    fToleranceY = kTolerance * Sqrt(fX2minusX1 * fX2minusX1 + 4 * fDZ * fDZ);
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
