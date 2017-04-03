/*
 * ParaboloidStruct.h
 *
 *  Created on: 14.07.2016
 *      Author: Raman Sehgal (raman.sehgal@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_PARABOLOIDSTRUCT_H_
#define VECGEOM_VOLUMES_PARABOLOIDSTRUCT_H_
#include "base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/*
 * A Paraboloid struct without member functions,
 * to encapsulate just the parameters and some
 * other cached values related to Paraboloid that
 * are required in Implementation
 *
 */
template <typename T = double>
struct ParaboloidStruct {
  T fRlo;
  T fRhi;
  T fDz;

  // Values computed from parameters, to be cached
  T fDx;
  T fDy;
  T fA;
  T fInvA;
  T fA2;
  T fB;
  T fB2;
  T fInvB;
  T fK1;
  T fK2;
  T fRlo2;
  T fRhi2;

  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct()
      : fRlo(0.), fRhi(0.), fDz(0.), fA(0.), fInvA(0.), fB(0.), fInvB(0.), fK1(0.), fK2(0.), fRlo2(0.), fRhi2(0.)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct(const T rlo, const T rhi, const T dz)
      : fRlo(rlo), fRhi(rhi), fDz(dz), fA(0.), fInvA(0.), fB(0.), fInvB(0.), fK1(0.), fK2(0.), fRlo2(0.), fRhi2(0.)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {
    fRlo2 = fRlo * fRlo;
    fRhi2 = fRhi * fRhi;
    T dd  = 1. / (fRhi2 - fRlo2);
    fA    = 2. * fDz * dd;
    fB    = -fDz * (fRlo2 + fRhi2) * dd;
    fK1   = (fRhi2 - fRlo2) * 0.5 / fDz;
    fK2   = (fRhi2 + fRlo2) * 0.5;
    fInvA = 1 / fA;
    fInvB = 1 / fB;
    fA2   = fA * fA;
    fB2   = fB * fB;
    ComputeBoundingBox();
  }

  VECCORE_ATT_HOST_DEVICE
  void ComputeBoundingBox()
  {
    fDx = Max(fRhi, fRlo);
    fDy = fDx;
  }
  VECCORE_ATT_HOST_DEVICE
  void SetRloAndRhiAndDz(const T rlo, const T rhi, const T dz)
  {

    if ((rlo < 0) || (rhi < 0) || (dz <= 0)) {

      printf("Error SetRloAndRhiAndDz: invalid dimensions. Check (rlo>=0) (rhi>=0) (dz>0)\n");
      return;
    }
    fRlo = rlo;
    fRhi = rhi;
    fDz  = dz;
    CalculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetRlo(const T rlo) { SetRloAndRhiAndDz(rlo, fRhi, fDz); }

  VECCORE_ATT_HOST_DEVICE
  void SetRhi(const T rhi) { SetRloAndRhiAndDz(fRlo, rhi, fDz); }

  VECCORE_ATT_HOST_DEVICE
  void SetDz(const T dz) { SetRloAndRhiAndDz(fRlo, fRhi, dz); }
};
}
} // end

#endif
