/*
 * HypeStruct.h
 *
 *  Created on: Feb 22, 2017
 *      Author: rsehgal
 */

#ifndef VOLUMES_HYPESTRUCT_H_
#define VOLUMES_HYPESTRUCT_H_

#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// An Hype struct without member functions to encapsulate just the parameters
template <typename T = double>
struct HypeStruct {
  T fRmin;  // Inner radius
  T fRmax;  // Outer radius
  T fStIn;  // Stereo angle for inner surface
  T fStOut; // Stereo angle for outer surface
  T fDz;    // z-coordinate of the cutting planes

  // Precomputed Values
  Precision fTIn;      // Tangent of the Inner stereo angle
  Precision fTOut;     // Tangent of the Outer stereo angle
  Precision fTIn2;     // Squared value of fTIn
  Precision fTOut2;    // Squared value of fTOut
  Precision fTIn2Inv;  // Inverse of fTIn2
  Precision fTOut2Inv; // Inverse of fTOut2
  Precision fRmin2;    // Squared Inner radius
  Precision fRmax2;    // Squared Outer radius
  Precision fDz2;      // Squared z-coordinate

  Precision fEndInnerRadius2; // Squared endcap Inner Radius
  Precision fEndOuterRadius2; // Squared endcap Outer Radius
  Precision fEndInnerRadius;  // Endcap Inner Radius
  Precision fEndOuterRadius;  // Endcap Outer Radius

  Precision fInSqSide; // side of the square inscribed in the inner circle

  // Volume and Surface Area
  Precision fCubicVolume, fSurfaceArea;
  Precision zToleranceLevel;
  Precision innerRadToleranceLevel, outerRadToleranceLevel;

  VECCORE_ATT_HOST_DEVICE
  HypeStruct() : fRmin(0.), fRmax(0.), fStIn(0.), fStOut(0.), fDz(0.) {}

  VECCORE_ATT_HOST_DEVICE
  HypeStruct(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
             const Precision dz)
      : fRmin(rMin), fRmax(rMax), fStIn(stIn), fStOut(stOut), fDz(dz)
  {
    CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {

    fTIn   = vecCore::math::Tan(fStIn);    // Tangent of the Inner stereo angle  (*kDegToRad);
    fTOut  = vecCore::math::Tan(fStOut);   // Tangent of the Outer stereo angle
    fTIn2  = fTIn * fTIn;   // squared value of fTIn
    fTOut2 = fTOut * fTOut; // squared value of fTOut

    fTIn2Inv  = 1. / fTIn2;
    fTOut2Inv = 1. / fTOut2;

    fRmin2 = fRmin * fRmin;
    fRmax2 = fRmax * fRmax;
    fDz2   = fDz * fDz;

    fEndInnerRadius2       = fTIn2 * fDz2 + fRmin2;
    fEndOuterRadius2       = fTOut2 * fDz2 + fRmax2;
    fEndInnerRadius        = vecCore::math::Sqrt(fEndInnerRadius2);
    fEndOuterRadius        = vecCore::math::Sqrt(fEndOuterRadius2);
    fInSqSide              = vecCore::math::Sqrt(2.) * fRmin;
    zToleranceLevel        = kTolerance * fDz;
    innerRadToleranceLevel = kTolerance * fEndInnerRadius; // GetEndInnerRadius();
    outerRadToleranceLevel = kTolerance * fEndOuterRadius; // GetEndOuterRadius();
    CalcCapacity();
    CalcSurfaceArea();
    // DetectConvexity();
  }

  VECCORE_ATT_HOST_DEVICE
  bool InnerSurfaceExists() const { return (fRmin > 0.) || (fStIn != 0.); }

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity() { fCubicVolume = Volume(true) - Volume(false); }

  VECCORE_ATT_HOST_DEVICE
  Precision Volume(bool outer)
  {
    if (outer)
      return 2 * kPi * fDz * ((fRmax) * (fRmax) + (fDz2 * fTOut2 / 3.));
    else
      return 2 * kPi * fDz * ((fRmin) * (fRmin) + (fDz2 * fTIn2 / 3.));
  }

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea() { fSurfaceArea = Area(true) + Area(false) + AreaEndCaps(); }

  VECCORE_ATT_HOST_DEVICE
  Precision Area(bool outer)
  {
    Precision fT = 0., fR = 0.;
    if (outer) {
      fT = fTOut;
      fR = fRmax;
    } else {
      fT = fTIn;
      fR = fRmin;
    }

    Precision ar = 0.;

    if (fT == 0)
      ar = 4 * kPi * fR * fDz;
    else {
      Precision p = fT * std::sqrt(fT * fT);
      Precision q = p * fDz * std::sqrt(fR * fR + (std::pow(fT, 2.) + std::pow(fT, 4.)) * std::pow(fDz, 2.));
      Precision r = fR * fR * std::asinh(p * fDz / fR);
      ar          = ((q + r) / (2 * p)) * 4 * kPi;
    }
    return ar;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision AreaEndCaps() { return 2 * kPi * (fEndOuterRadius2 - fEndInnerRadius2); }

  VECCORE_ATT_HOST_DEVICE
  void SetParameters(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
                     const Precision dz)
  {
    fRmin  = rMin;
    fStIn  = stIn;
    fRmax  = rMax;
    fStOut = stOut;
    fDz    = dz;
    CalculateCached();
  }
};
}
}
#endif /* VOLUMES_HYPESTRUCT_H_ */
