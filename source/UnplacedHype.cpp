/// \file UnplacedHype.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedHype.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedHype.h"
//#include "volumes/kernel/shapetypes/HypeTypes.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>
#include "base/RNG.h"
#include "base/Global.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::SetParameters(const Precision rMin, const Precision rMax,
                                 const Precision stIn, const Precision stOut,
                                 const Precision dz) {

  // TODO: add eventual check
  fRmin = rMin;
  fStIn = stIn;
  fRmax = rMax;
  fStOut = stOut;
  fDz = dz;
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedHype::UnplacedHype(const Precision rMin, const Precision rMax,
                           const Precision stIn, const Precision stOut,
                           const Precision dz) {

  SetParameters(rMin, rMax, stIn, stOut, dz);

  fTIn = tan(fStIn);      // Tangent of the Inner stereo angle  (*kDegToRad);
  fTOut = tan(fStOut);    // Tangent of the Outer stereo angle
  fTIn2 = fTIn * fTIn;    // squared value of fTIn
  fTOut2 = fTOut * fTOut; // squared value of fTOut

  fTIn2Inv = 1. / fTIn2;
  fTOut2Inv = 1. / fTOut2;

  fRmin2 = fRmin * fRmin;
  fRmax2 = fRmax * fRmax;
  fDz2 = fDz * fDz;

  fEndInnerRadius2 = fTIn2 * fDz2 + fRmin2;
  fEndOuterRadius2 = fTOut2 * fDz2 + fRmax2;
  fEndInnerRadius = Sqrt(fEndInnerRadius2);
  fEndOuterRadius = Sqrt(fEndOuterRadius2);
  fInSqSide = Sqrt(2.) * fRmin;
  zToleranceLevel = kTolerance * fDz;
  innerRadToleranceLevel = kTolerance * GetEndInnerRadius();
  outerRadToleranceLevel = kTolerance * GetEndOuterRadius();
  CalcCapacity();
  CalcSurfaceArea();
  DetectConvexity();
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::DetectConvexity(){
  //Default Convexity set to false
  fGlobalConvexity = false;
  //Logic to calculate the convexity
  if( (fRmin == 0.) && (fStIn == 0.) && (fStOut == 0.) ) //Hype becomes Solid Tube.
    fGlobalConvexity = true;
}


VECGEOM_CUDA_HEADER_BOTH
bool UnplacedHype::InnerSurfaceExists() const { return (fRmin > 0.) || (fStIn != 0.); }

VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::CalcCapacity() { fCubicVolume = Volume(true) - Volume(false); }

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedHype::Volume(bool outer) {
  if (outer)
    return 2 * kPi * fDz * ((fRmax) * (fRmax) + (fDz2 * fTOut2 / 3.));
  else
    return 2 * kPi * fDz * ((fRmin) * (fRmin) + (fDz2 * fTIn2 / 3.));
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::CalcSurfaceArea() { fSurfaceArea = Area(true) + Area(false) + AreaEndCaps(); }

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedHype::Area(bool outer) {
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
    ar = ((q + r) / (2 * p)) * 4 * kPi;
  }
  return ar;
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedHype::AreaEndCaps() { return 2 * kPi * (GetEndOuterRadius2() - GetEndInnerRadius2()); }

#ifndef VECGEOM_NVCC
void UnplacedHype::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const {
  // Returns the full 3D cartesian extent of the solid.
  Precision rMax = GetEndOuterRadius();
  aMin.Set(-rMax, -rMax, -fDz);
  aMax.Set(rMax, rMax, fDz);
}


VECGEOM_CUDA_HEADER_BOTH
std::string UnplacedHype::GetEntityType() const { return "Hyperboloid\n"; }


Vector3D<Precision> UnplacedHype::GetPointOnSurface() const {

  Precision xRand, yRand, zRand, r2, aOne, aTwo, aThree, chose, sinhu;
  Precision phi, cosphi, sinphi, rBar2Out, rBar2In, alpha, t, rOut, rIn2, rOut2;

  // we use the formula of the area of a surface of revolution to compute
  // the areas, using the equation of the hyperbola:
  // x^2 + y^2 = (z*tanphi)^2 + r^2

  rBar2Out = fRmax2;
  alpha = 2. * kPi * rBar2Out * std::cos(fStOut) / fTOut;
  t = fDz * fTOut / (fRmax * std::cos(fStOut));
  t = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aOne = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  rBar2In = fRmin2;
  alpha = 2. * kPi * rBar2In * std::cos(fStIn) / fTIn;
  t = fDz * fTIn / (fRmin * std::cos(fStIn));
  t = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aTwo = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  aThree = kPi * ((fRmax2 + (fDz * fTOut) * (fDz * fTOut) - (fRmin2 + (fDz * fTIn) * (fDz * fTIn))));

  if (fStOut == 0.) {
    aOne = std::fabs(2. * kPi * fRmax * 2. * fDz);
  }
  if (fStIn == 0.) {
    aTwo = std::fabs(2. * kPi * fRmin * 2. * fDz);
  }

  phi = RNG::Instance().uniform(0., 2. * kPi);
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  sinhu = RNG::Instance().uniform(-1. * fDz * fTOut / fRmax, fDz * fTOut / fRmax);

  chose = RNG::Instance().uniform(0., aOne + aTwo + 2. * aThree);
  if (chose >= 0. && chose < aOne) {
    if (fStOut != 0.) {
      zRand = fRmax * sinhu / fTOut;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fRmax * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fRmax * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fRmax * cosphi, fRmax * sinphi, RNG::Instance().uniform(-fDz, fDz)); // RandFlat::shoot
    }
  } else if (chose >= aOne && chose < aOne + aTwo) {
    if (fStIn != 0.) {
      sinhu = RNG::Instance().uniform(-1. * fDz * fTIn / fRmin, fDz * fTIn / fRmin);
      zRand = fRmin * sinhu / fTIn;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fRmin * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fRmin * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fRmin * cosphi, fRmin * sinphi, RNG::Instance().uniform(-1. * fDz, fDz));
    }
  } else if (chose >= aOne + aTwo && chose < aOne + aTwo + aThree) {
    rIn2 = fRmin2 + fTIn2 * fDz * fDz;
    rOut2 = fRmax2 + fTOut2 * fDz * fDz;
    rOut = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2 = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  } else {
    rIn2 = fRmin2 + fTIn2 * fDz * fDz;
    rOut2 = fRmax2 + fTOut2 * fDz * fDz;
    rOut = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2 = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = -1. * fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  }
}
#endif

#if defined(VECGEOM_USOLIDS)
VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::GetParametersList(int, double *aArray) const {
  aArray[0] = GetRmin();
  aArray[1] = GetStIn();
  aArray[2] = GetRmax();
  aArray[3] = GetStOut();
  aArray[4] = GetDz();
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedHype *UnplacedHype::Clone() const { return new UnplacedHype(fRmin, fStIn, fRmax, fStOut, fDz); }

std::ostream &UnplacedHype::StreamInfo(std::ostream &os) const
// Definition taken from
{

  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"

     << " Solid type: VecGeomHype\n"
     << " Parameters: \n"

     << "               Inner radius: " << fRmin << " mm \n"
     << "               Inner Stereo Angle " << fStIn << " rad \n"
     << "               Outer radius: " << fRmax << "mm\n"
     << "               Outer Stereo Angle " << fStOut << " rad \n"
     << "               Half Height: " << fDz << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif


void UnplacedHype::Print() const {
  printf("UnplacedHype {%.2f, %.2f, %.2f, %.2f, %.2f}", fRmin, fRmax, fStIn, fStOut, fDz);
}

void UnplacedHype::Print(std::ostream &os) const {
  os << "UnplacedHype {" << fRmin << ", " << fRmax << ", " << fStIn << ", " << fStOut << ", " << fDz << "}";
}

#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedHype::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedHype<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedHype<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedHype::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedHype>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedHype::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedHype<trans_code, rot_code>(logical_volume,
                                                        transformation, NULL, id);
    return placement;
  }
  return new SpecializedHype<trans_code, rot_code>(logical_volume,
                                                  transformation,NULL, id);
}

__device__
VPlacedVolume* UnplacedHype::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedHype>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedHype>(in_gpu_ptr, fRmin, fRmax, fStIn, fStOut, fDz);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedHype>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedHype>::SizeOf();
template void DevicePtr<cuda::UnplacedHype>::Construct(const Precision rMin, const Precision rMax,
                                                      const Precision stIn, const Precision stOut,
                                                      const Precision dZ) const;

} // End cxx namespace

#endif

} // End global namespace
