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

/*
VECCORE_ATT_HOST_DEVICE
void UnplacedHype::SetParameters(const Precision rMin, const Precision rMax, const Precision stIn,
                                 const Precision stOut, const Precision dz)
{

  // TODO: add eventual check
  fRmin  = rMin;
  fStIn  = stIn;
  fRmax  = rMax;
  fStOut = stOut;
  fDz    = dz;
}
*/

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::DetectConvexity()
{
  // Default Convexity set to false
  fGlobalConvexity = false;
  // Logic to calculate the convexity
  if ((fHype.fRmin == 0.) && (fHype.fStIn == 0.) && (fHype.fStOut == 0.)) // Hype becomes Solid Tube.
    fGlobalConvexity = true;
}

/*
VECCORE_ATT_HOST_DEVICE
bool UnplacedHype::InnerSurfaceExists() const
{
  return (fRmin > 0.) || (fStIn != 0.);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::CalcCapacity()
{
  fCubicVolume = Volume(true) - Volume(false);
}

VECCORE_ATT_HOST_DEVICE
Precision UnplacedHype::Volume(bool outer)
{
  if (outer)
    return 2 * kPi * fDz * ((fRmax) * (fRmax) + (fDz2 * fTOut2 / 3.));
  else
    return 2 * kPi * fDz * ((fRmin) * (fRmin) + (fDz2 * fTIn2 / 3.));
}

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::CalcSurfaceArea()
{
  fSurfaceArea = Area(true) + Area(false) + AreaEndCaps();
}

VECCORE_ATT_HOST_DEVICE
Precision UnplacedHype::Area(bool outer)
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
Precision UnplacedHype::AreaEndCaps()
{
  return 2 * kPi * (GetEndOuterRadius2() - GetEndInnerRadius2());
}
*/

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  Precision rMax = GetEndOuterRadius();
  aMin.Set(-rMax, -rMax, -fHype.fDz);
  aMax.Set(rMax, rMax, fHype.fDz);
}

#ifndef VECCORE_CUDA
VECCORE_ATT_HOST_DEVICE
std::string UnplacedHype::GetEntityType() const
{
  return "Hyperboloid\n";
}
#endif

Vector3D<Precision> UnplacedHype::SamplePointOnSurface() const
{

  Precision xRand, yRand, zRand, r2, aOne, aTwo, aThree, chose, sinhu;
  Precision phi, cosphi, sinphi, rBar2Out, rBar2In, alpha, t, rOut, rIn2, rOut2;

  // we use the formula of the area of a surface of revolution to compute
  // the areas, using the equation of the hyperbola:
  // x^2 + y^2 = (z*tanphi)^2 + r^2

  rBar2Out = fHype.fRmax2;
  alpha    = 2. * kPi * rBar2Out * std::cos(fHype.fStOut) / fHype.fTOut;
  t        = fHype.fDz * fHype.fTOut / (fHype.fRmax * std::cos(fHype.fStOut));
  t        = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aOne     = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  rBar2In = fHype.fRmin2;
  alpha   = 2. * kPi * rBar2In * std::cos(fHype.fStIn) / fHype.fTIn;
  t       = fHype.fDz * fHype.fTIn / (fHype.fRmin * std::cos(fHype.fStIn));
  t       = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aTwo    = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  aThree = kPi * ((fHype.fRmax2 + (fHype.fDz * fHype.fTOut) * (fHype.fDz * fHype.fTOut) -
                   (fHype.fRmin2 + (fHype.fDz * fHype.fTIn) * (fHype.fDz * fHype.fTIn))));

  if (fHype.fStOut == 0.) {
    aOne = std::fabs(2. * kPi * fHype.fRmax * 2. * fHype.fDz);
  }
  if (fHype.fStIn == 0.) {
    aTwo = std::fabs(2. * kPi * fHype.fRmin * 2. * fHype.fDz);
  }

  phi    = RNG::Instance().uniform(0., 2. * kPi);
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  sinhu  = RNG::Instance().uniform(-1. * fHype.fDz * fHype.fTOut / fHype.fRmax, fHype.fDz * fHype.fTOut / fHype.fRmax);

  chose = RNG::Instance().uniform(0., aOne + aTwo + 2. * aThree);
  if (chose >= 0. && chose < aOne) {
    if (fHype.fStOut != 0.) {
      zRand = fHype.fRmax * sinhu / fHype.fTOut;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmax * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmax * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fHype.fRmax * cosphi, fHype.fRmax * sinphi,
                                 RNG::Instance().uniform(-fHype.fDz, fHype.fDz)); // RandFlat::shoot
    }
  } else if (chose >= aOne && chose < aOne + aTwo) {
    if (fHype.fStIn != 0.) {
      sinhu = RNG::Instance().uniform(-1. * fHype.fDz * fHype.fTIn / fHype.fRmin, fHype.fDz * fHype.fTIn / fHype.fRmin);
      zRand = fHype.fRmin * sinhu / fHype.fTIn;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmin * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmin * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fHype.fRmin * cosphi, fHype.fRmin * sinphi,
                                 RNG::Instance().uniform(-1. * fHype.fDz, fHype.fDz));
    }
  } else if (chose >= aOne + aTwo && chose < aOne + aTwo + aThree) {
    rIn2  = fHype.fRmin2 + fHype.fTIn2 * fHype.fDz * fHype.fDz;
    rOut2 = fHype.fRmax2 + fHype.fTOut2 * fHype.fDz * fHype.fDz;
    rOut  = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2    = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = fHype.fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  } else {
    rIn2  = fHype.fRmin2 + fHype.fTIn2 * fHype.fDz * fHype.fDz;
    rOut2 = fHype.fRmax2 + fHype.fTOut2 * fHype.fDz * fHype.fDz;
    rOut  = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2    = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = -1. * fHype.fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  }
}

#if defined(VECGEOM_USOLIDS)
VECCORE_ATT_HOST_DEVICE
void UnplacedHype::GetParametersList(int, double *aArray) const
{
  aArray[0] = fHype.fRmin;  // GetRmin();
  aArray[1] = fHype.fStIn;  // GetStIn();
  aArray[2] = fHype.fRmax;  // GetRmax();
  aArray[3] = fHype.fStOut; // GetStOut();
  aArray[4] = fHype.fDz;    // GetDz();
}

VECCORE_ATT_HOST_DEVICE
UnplacedHype *UnplacedHype::Clone() const
{
  return new UnplacedHype(fHype.fRmin, fHype.fStIn, fHype.fRmax, fHype.fStOut, fHype.fDz);
}

std::ostream &UnplacedHype::StreamInfo(std::ostream &os) const
// Definition taken from
{

  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"

     << " Solid type: VecGeomHype\n"
     << " Parameters: \n"

     << "               Inner radius: " << fHype.fRmin << " mm \n"
     << "               Inner Stereo Angle " << fHype.fStIn << " rad \n"
     << "               Outer radius: " << fHype.fRmax << "mm\n"
     << "               Outer Stereo Angle " << fHype.fStOut << " rad \n"
     << "               Half Height: " << fHype.fDz << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif

void UnplacedHype::Print() const
{
  printf("UnplacedHype {%.2f, %.2f, %.2f, %.2f, %.2f}", fHype.fRmin, fHype.fRmax, fHype.fStIn, fHype.fStOut, fHype.fDz);
}

void UnplacedHype::Print(std::ostream &os) const
{
  os << "UnplacedHype {" << fHype.fRmin << ", " << fHype.fRmax << ", " << fHype.fStIn << ", " << fHype.fStOut << ", "
     << fHype.fDz << "}";
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedHype::SpecializedVolume(LogicalVolume const *const volume,
                                               Transformation3D const *const transformation,
                                               const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                               const int id,
#endif
                                               VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedHype>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                             id,
#endif
                                                             placement);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedHype::Create(LogicalVolume const *const logical_volume,
                                    Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                    const int id,
#endif
                                    VPlacedVolume *const placement)
{
  if (placement) {
#ifdef VECCORE_CUDA
    new (placement) SpecializedHype<transCodeT, rotCodeT>(logical_volume, transformation, id);
#else
    new (placement) SpecializedHype<transCodeT, rotCodeT>(logical_volume, transformation);
    return placement;
#endif
  }

#ifdef VECCORE_CUDA
  return new SpecializedHype<transCodeT, rotCodeT>(logical_volume, transformation, id);
#else
  return new SpecializedHype<transCodeT, rotCodeT>(logical_volume, transformation);
#endif
}

/*
#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedHype::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedHype<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedHype<trans_code, rot_code>(logical_volume, transformation);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume *UnplacedHype::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, const int id,
                                   VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedHype<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedHype<trans_code, rot_code>(logical_volume, transformation, id);
}

#endif
*/

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedHype>(in_gpu_ptr, fHype.fRmin, fHype.fRmax, fHype.fStIn, fHype.fStOut, fHype.fDz);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedHype>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedHype>::SizeOf();
template void DevicePtr<cuda::UnplacedHype>::Construct(const Precision rmin, const Precision rmax, const Precision stIn,
                                                       const Precision stOut, const Precision z) const;

} // End cxx namespace

#endif
} // End global namespace
