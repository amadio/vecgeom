// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members for the UnplacedParaboloid class
/// @file volumes/ParaboloidStruct.h
/// @author Raman Sehgal

#ifndef VECGEOM_VOLUMES_PARABOLOIDSTRUCT_H_
#define VECGEOM_VOLUMES_PARABOLOIDSTRUCT_H_
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct encapsulating data members of the unplaced paraboloid
template <typename T = double>
struct ParaboloidStruct {
  T fRlo; ///< Radius of the circle at z = -dz
  T fRhi; ///< Radius of the circle at z = +dz
  T fDz;  ///< Half size in z

  // Values computed from parameters, to be cached
  T fDx;   ///< Half size of the bounding box in x
  T fDy;   ///< Half size of the bounding box in y
  T fA;    ///< Parameter a in the equation of paraboloid: z = a * (x^2 + y^2) + b
  T fInvA; ///< Inverted value of a
  T fA2;   ///< Parameter a squared
  T fB;    ///< Parameter b in the equation of paraboloid: z = a * (x^2 + y^2) + b
  T fB2;   ///< Parameter b squared
  T fInvB; ///< Inverted value of b
  T fK1;   ///< Cached value: 0.5 * (Rhi^2 - Rlo^2) / Dz
  T fK2;   ///< Cached value: 0.5 * (Rhi^2 + Rlo^2)
  T fRlo2; ///< Radius of the circle at z = -dz squared
  T fRhi2; ///< Radius of the circle at z = +dz squared

  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct()
      : fRlo(0.), fRhi(0.), fDz(0.), fA(0.), fInvA(0.), fB(0.), fInvB(0.), fK1(0.), fK2(0.), fRlo2(0.), fRhi2(0.)
  {
    CalculateCached();
  }

  /// Constructor
  /// @param rlo Radius of the circle at z = -dz
  /// @param rhi Radius of the circle at z = +dz
  /// @param dz Half size in z
  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct(const T rlo, const T rhi, const T dz)
      : fRlo(rlo), fRhi(rhi), fDz(dz), fA(0.), fInvA(0.), fB(0.), fInvB(0.), fK1(0.), fK2(0.), fRlo2(0.), fRhi2(0.)
  {
    CalculateCached();
  }

  /// Sets data members
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

  /// Sets fDx and fDy
  VECCORE_ATT_HOST_DEVICE
  void ComputeBoundingBox()
  {
    fDx = Max(fRhi, fRlo);
    fDy = fDx;
  }

  /// Sets parameters of the paraboloid
  /// @param rlo Radius of the circle at z = -dz
  /// @param rhi Radius of the circle at z = +dz
  /// @param dz Half size in z
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

  /// Sets the raduis of the circle at z = -dz
  /// @param val Value of the radius
  VECCORE_ATT_HOST_DEVICE
  void SetRlo(const T rlo) { SetRloAndRhiAndDz(rlo, fRhi, fDz); }

  /// Sets the raduis of the circle at z = +dz
  /// @param val Value of the radius
  VECCORE_ATT_HOST_DEVICE
  void SetRhi(const T rhi) { SetRloAndRhiAndDz(fRlo, rhi, fDz); }

  /// Sets the half size in z
  /// @param val Value of the half size in z
  VECCORE_ATT_HOST_DEVICE
  void SetDz(const T dz) { SetRloAndRhiAndDz(fRlo, fRhi, dz); }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
