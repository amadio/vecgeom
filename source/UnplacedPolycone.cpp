/*
 * UnplacedPolycone.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#include "management/GeoManager.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/UnplacedCone.h"
#include "volumes/PlacedPolycone.h"
#include "volumes/PlacedCone.h"
#include "volumes/SpecializedPolycone.h"
#include "management/VolumeFactory.h"
#include "volumes/utilities/GenerationUtilities.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#endif
#include <iostream>
#include <cstdio>
#include <vector>
#include "base/Vector.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedPolycone::Reset()
{
  double phiStart = fPolycone.fOriginal_parameters->fHStart_angle;
  double *Z, *R1, *R2;
  int num = fPolycone.fOriginal_parameters->fHNum_z_planes; // fOriginalParameters-> NumZPlanes;
  Z       = new double[num];
  R1      = new double[num];
  R2      = new double[num];
  for (int i = 0; i < num; i++) {
    Z[i]  = fPolycone.fOriginal_parameters->fHZ_values[i]; // fOriginalParameters->fZValues[i];
    R1[i] = fPolycone.fOriginal_parameters->fHRmin[i];     // fOriginalParameters->Rmin[i];
    R2[i] = fPolycone.fOriginal_parameters->fHRmax[i];     // fOriginalParameters->Rmax[i];
  }

  fPolycone.Init(phiStart, fPolycone.fOriginal_parameters->fHOpening_angle, num, Z, R1, R2);
  delete[] R1;
  delete[] Z;
  delete[] R2;
}

// Alternative constructor, required for integration with Geant4.
// Input must be such that r[i],z[i] describe the outer,inner or inner,outer envelope of the polycone, after
// connecting all adjacent points, and closing the polygon by connecting last -> first point.
// Hence z[] array must be symmetrical: z[0..Nz] = z[2Nz, 2Nz-1, ..., Nz+1], where 2*Nz = numRz.
VECCORE_ATT_HOST_DEVICE
UnplacedPolycone::UnplacedPolycone(Precision phiStart, // initial phi starting angle
                                   Precision phiTotal, // total phi angle
                                   int numRZ,          // number corners in r,z space (must be an even number)
                                   Precision const *r, // r coordinate of these corners
                                   Precision const *z) // z coordinate of these corners

{
  fPolycone.fStartPhi = phiStart;
  fPolycone.fDeltaPhi = phiStart + phiTotal;
  fPolycone.fNz       = (numRZ / 2);
  //      fPolycone.fSections(),
  fPolycone.fZs = (numRZ / 2);
  fPolycone.fPhiWedge.Set(fPolycone.fDeltaPhi, phiStart);

  // data integrity checks
  int Nz = numRZ / 2;
  assert(numRZ % 2 == 0 && "UnplPolycone ERROR: r[],z[] arrays provided contain odd number of points, please fix.\n");
  for (int i = 0; i < numRZ / 2; ++i) {
    assert(z[i] == z[numRZ - 1 - i] && "UnplPolycone ERROR: z[] array is not symmetrical, please fix.\n");
  }

  // reuse input array as argument, in ascending order
  bool ascendingZ        = true;
  const Precision *zarg  = z;
  const Precision *r1arg = r;
  if (z[0] > z[1]) {
    ascendingZ = false;
    zarg       = z + Nz; // second half of input z[] is ascending due to symmetry already verified
    r1arg      = r + Nz;
  }

  // reorganize remainder of r[] data in ascending-z order
  Precision *r2arg = new Precision[Nz];
  for (int i = 0; i < Nz; ++i)
    r2arg[i] = (ascendingZ ? r[2 * Nz - 1 - i] : r[Nz - 1 - i]);

  // identify which rXarg is rmax and rmin and ensure that Rmax > Rmin for all points provided
  const Precision *rmin = r1arg, *rmax = r2arg;
  if (r1arg[0] > r2arg[0]) {
    rmax = r1arg;
    rmin = r2arg;
  }

  // final data integrity cross-check
  for (int i = 0; i < Nz; ++i) {
    assert(rmax[i] > rmin[i] &&
           "UnplPolycone ERROR: r[] provided has problems of the Rmax < Rmin type, please check!\n");
  }

  // init internal members
  fPolycone.Init(phiStart, phiTotal, Nz, zarg, rmin, rmax);
  delete[] r2arg;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::Print() const
{
  printf("UnplacedPolycone {%.2f, %.2f, %d}\n", fPolycone.fStartPhi, fPolycone.fDeltaPhi, fPolycone.fNz);
  printf("have %zu size Z\n", fPolycone.fZs.size());
  printf("------- z planes follow ---------\n");
  for (size_t p = 0; p < fPolycone.fZs.size(); ++p) {
    printf(" plane %zu at z pos %lf\n", p, fPolycone.fZs[p]);
  }

  printf("have %zu size fSections\n", fPolycone.fSections.size());
  printf("------ sections follow ----------\n");
  for (int s = 0; s < fPolycone.GetNSections(); ++s) {
    printf("## section %d, shift %lf\n", s, fPolycone.fSections[s].fShift);
    fPolycone.fSections[s].fSolid->Print();
    printf("\n");
  }
}
// VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::Print(std::ostream &os) const
{
  os << "UnplacedPolycone output to string not implemented -- calling Print() instead:\n";
  Print();
}

std::ostream &UnplacedPolycone::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Polycone\n"
     << " Parameters: \n"
     << "     N = number of Z-sections: " << fPolycone.fSections.size() << ", # Z-coords=" << fPolycone.fZs.size()
     << "\n"
     << "     z-coordinates:\n";

  uint nz = fPolycone.fZs.size();
  for (uint j = 0; j < (nz - 1) / 5 + 1; ++j) {
    os << "       [ ";
    for (uint i = 0; i < 5; ++i) {
      uint ind = 5 * j + i;
      if (ind < fPolycone.fNz) os << fPolycone.fZs[ind] << "; ";
    }
    os << " ]\n";
  }
  if (fPolycone.fDeltaPhi < kTwoPi) {
    os << "     Wedge starting angles: fSphi=" << fPolycone.fStartPhi * kRadToDeg << "deg, "
       << ", fDphi=" << fPolycone.fDeltaPhi * kRadToDeg << "deg\n";
  }

  size_t nsections = fPolycone.fSections.size();
  os << "\n    # cone sections: " << nsections << "\n";
  for (size_t i = 0; i < nsections; ++i) {
    ConeStruct<double> *subcone = fPolycone.fSections[i].fSolid;
    os << "     cone #" << i << " Rmin1=" << subcone->fRmin1 << " Rmax1=" << subcone->fRmax1
       << " Rmin2=" << subcone->fRmin2 << " Rmax2=" << subcone->fRmax2 << " HalfZ=" << subcone->fDz
       << " from z=" << fPolycone.fZs[i] << " to z=" << fPolycone.fZs[i + 1] << "mm\n";
  }
  os << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedPolycone>();
}

DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const
{

  // idea: reconstruct defining arrays: copy them to GPU; then construct the UnplacedPolycon object from scratch
  // on the GPU
  std::vector<Precision> rmin, z, rmax;
  ReconstructSectionArrays(z, rmin, rmax);

  Precision *z_gpu_ptr    = AllocateOnGpu<Precision>(z.size() * sizeof(Precision));
  Precision *rmin_gpu_ptr = AllocateOnGpu<Precision>(rmin.size() * sizeof(Precision));
  Precision *rmax_gpu_ptr = AllocateOnGpu<Precision>(rmax.size() * sizeof(Precision));

  vecgeom::CopyToGpu(&z[0], z_gpu_ptr, sizeof(Precision) * z.size());
  vecgeom::CopyToGpu(&rmin[0], rmin_gpu_ptr, sizeof(Precision) * rmin.size());
  vecgeom::CopyToGpu(&rmax[0], rmax_gpu_ptr, sizeof(Precision) * rmax.size());

  vecgeom::CopyToGpu(&z[0], z_gpu_ptr, sizeof(Precision) * z.size());
  vecgeom::CopyToGpu(&rmin[0], rmin_gpu_ptr, sizeof(Precision) * rmin.size());
  vecgeom::CopyToGpu(&rmax[0], rmax_gpu_ptr, sizeof(Precision) * rmax.size());

  int s = z.size();

  // attention here z.size() might be different than fNz due to compactification during Reconstruction
  DevicePtr<cuda::VUnplacedVolume> gpupolycon = CopyToGpuImpl<UnplacedPolycone>(
      gpu_ptr, fPolycone.fStartPhi, fPolycone.fDeltaPhi, s, z_gpu_ptr, rmin_gpu_ptr, rmax_gpu_ptr);

  // remove temporary space from GPU
  FreeFromGpu(z_gpu_ptr);
  FreeFromGpu(rmin_gpu_ptr);
  FreeFromGpu(rmax_gpu_ptr);

  return gpupolycon;
}

#endif // VECGEOM_CUDA_INTERFACE

Precision UnplacedPolycone::SurfaceArea() const
{
  const int numPlanes = GetNSections();

  PolyconeSection const &sec0 = GetSection(0);
  Precision totArea = (kPi * (sec0.fSolid->fRmax1 * sec0.fSolid->fRmax1 - sec0.fSolid->fRmin1 * sec0.fSolid->fRmin1));

  for (int i = 0; i < numPlanes; i++) {
    PolyconeSection const &sec = GetSection(i);

    Precision sectionArea =
        (sec.fSolid->fRmin1 + sec.fSolid->fRmin2) *
        std::sqrt((sec.fSolid->fRmin1 - sec.fSolid->fRmin2) * (sec.fSolid->fRmin1 - sec.fSolid->fRmin2) +
                  4. * sec.fSolid->fDz * sec.fSolid->fDz);

    sectionArea += (sec.fSolid->fRmax1 + sec.fSolid->fRmax2) *
                   std::sqrt((sec.fSolid->fRmax1 - sec.fSolid->fRmax2) * (sec.fSolid->fRmax1 - sec.fSolid->fRmax2) +
                             4. * sec.fSolid->fDz * sec.fSolid->fDz);

    sectionArea *= 0.5 * GetDeltaPhi();

    if (GetDeltaPhi() < kTwoPi) {
      sectionArea += std::fabs(2 * sec.fSolid->fDz) *
                     (sec.fSolid->fRmax1 + sec.fSolid->fRmax2 - sec.fSolid->fRmin1 - sec.fSolid->fRmin2);
    }
    totArea += sectionArea;
  }

  PolyconeSection const &secn = GetSection(numPlanes - 1);
  const auto last = kPi * (secn.fSolid->fRmax2 * secn.fSolid->fRmax2 - secn.fSolid->fRmin2 * secn.fSolid->fRmin2);
  totArea += last;

  return totArea;
}

#ifndef VECCORE_CUDA
/////////////////////////////////////////////////////////////////////////
//
// SamplePointOnSurface
//
// GetPointOnCone
//
// Auxiliary method for Get Point On Surface
//

Vector3D<Precision> UnplacedPolycone::GetPointOnCone(Precision fRmin1, Precision fRmax1, Precision fRmin2,
                                                     Precision fRmax2, Precision zOne, Precision zTwo,
                                                     Precision &totArea) const
{
#if (1)
  // declare working variables
  //
  Precision Aone, Atwo, Afive, phi, zRand, fDPhi, cosu, sinu;
  Precision rRand1, rmin, rmax, chose, rone, rtwo, qone, qtwo;
  Precision fDz = (zTwo - zOne) / 2., afDz = std::fabs(fDz);
  Vector3D<Precision> point, offset        = Vector3D<Precision>(0., 0., 0.5 * (zTwo + zOne));
  fDPhi = GetDeltaPhi();
  rone  = (fRmax1 - fRmax2) / (2. * fDz);
  rtwo  = (fRmin1 - fRmin2) / (2. * fDz);
  if (fRmax1 == fRmax2) {
    qone = 0.;
  } else {
    qone = fDz * (fRmax1 + fRmax2) / (fRmax1 - fRmax2);
  }
  if (fRmin1 == fRmin2) {
    qtwo = 0.;
  } else {
    qtwo = fDz * (fRmin1 + fRmin2) / (fRmin1 - fRmin2);
  }
  Aone    = 0.5 * fDPhi * (fRmax2 + fRmax1) * ((fRmin1 - fRmin2) * (fRmin1 - fRmin2) + (zTwo - zOne) * (zTwo - zOne));
  Atwo    = 0.5 * fDPhi * (fRmin2 + fRmin1) * ((fRmax1 - fRmax2) * (fRmax1 - fRmax2) + (zTwo - zOne) * (zTwo - zOne));
  Afive   = fDz * (fRmax1 - fRmin1 + fRmax2 - fRmin2);
  totArea = Aone + Atwo + 2. * Afive;

  phi  = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosu = std::cos(phi);
  sinu = std::sin(phi);

  if (GetDeltaPhi() >= kTwoPi) {
    Afive = 0;
  }
  chose = RNG::Instance().uniform(0., Aone + Atwo + 2. * Afive);
  if ((chose >= 0) && (chose < Aone)) {
    if (fRmax1 != fRmax2) {
      zRand = RNG::Instance().uniform(-1. * afDz, afDz);
      point = Vector3D<Precision>(rone * cosu * (qone - zRand), rone * sinu * (qone - zRand), zRand);
    } else {
      point = Vector3D<Precision>(fRmax1 * cosu, fRmax1 * sinu, RNG::Instance().uniform(-1. * afDz, afDz));
    }
  } else if (chose >= Aone && chose < Aone + Atwo) {
    if (fRmin1 != fRmin2) {
      zRand = RNG::Instance().uniform(-1. * afDz, afDz);
      point = Vector3D<Precision>(rtwo * cosu * (qtwo - zRand), rtwo * sinu * (qtwo - zRand), zRand);

    } else {
      point = Vector3D<Precision>(fRmin1 * cosu, fRmin1 * sinu, RNG::Instance().uniform(-1. * afDz, afDz));
    }
  } else if ((chose >= Aone + Atwo + Afive) && (chose < Aone + Atwo + 2. * Afive)) {
    zRand  = RNG::Instance().uniform(-afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2. * fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2. * fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0., 1.) * (rmax * rmax - rmin * rmin) + rmin * rmin);
    point  = Vector3D<Precision>(rRand1 * std::cos(GetStartPhi()), rRand1 * std::sin(GetStartPhi()), zRand);
  } else {
    zRand  = RNG::Instance().uniform(-1. * afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2. * fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2. * fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0., 1.) * (rmax * rmax - rmin * rmin) + rmin * rmin);
    point  = Vector3D<Precision>(rRand1 * std::cos(GetEndPhi()), rRand1 * std::sin(GetEndPhi()), zRand);
  }

  return point + offset;
#endif
}

//
// GetPointOnTubs
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnTubs(Precision fRMin, Precision fRMax, Precision zOne, Precision zTwo,
                                                     Precision &totArea) const
{

  Precision xRand, yRand, zRand, phi, cosphi, sinphi, chose, aOne, aTwo, aFou, rRand, fDz, fSPhi, fDPhi;
  fDz   = std::fabs(0.5 * (zTwo - zOne));
  fSPhi = GetStartPhi();
  fDPhi = GetDeltaPhi();

  aOne    = 2. * fDz * fDPhi * fRMax;
  aTwo    = 2. * fDz * fDPhi * fRMin;
  aFou    = 2. * fDz * (fRMax - fRMin);
  totArea = aOne + aTwo + 2. * aFou;
  phi     = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi  = std::cos(phi);
  sinphi  = std::sin(phi);
  rRand   = fRMin + (fRMax - fRMin) * std::sqrt(RNG::Instance().uniform(0., 1.));

  if (GetDeltaPhi() >= 2 * kPi) aFou = 0;

  chose = RNG::Instance().uniform(0., aOne + aTwo + 2. * aFou);
  if ((chose >= 0) && (chose < aOne)) {
    xRand = fRMax * cosphi;
    yRand = fRMax * sinphi;
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  } else if ((chose >= aOne) && (chose < aOne + aTwo)) {
    xRand = fRMin * cosphi;
    yRand = fRMin * sinphi;
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  } else if ((chose >= aOne + aTwo) && (chose < aOne + aTwo + aFou)) {
    xRand = rRand * std::cos(fSPhi + fDPhi);
    yRand = rRand * std::sin(fSPhi + fDPhi);
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  }

  // else

  xRand = rRand * std::cos(fSPhi + fDPhi);
  yRand = rRand * std::sin(fSPhi + fDPhi);
  zRand = RNG::Instance().uniform(-1. * fDz, fDz);
  return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
}

//
// GetPointOnRing
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnRing(Precision fRMin1, Precision fRMax1, Precision fRMin2,
                                                     Precision fRMax2, Precision zOne) const
{

  Precision xRand, yRand, phi, cosphi, sinphi, rRand1, rRand2, A1, Atot, rCh;
  phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);

  if (fRMin1 == fRMin2) {
    rRand1 = fRMin1;
    A1     = 0.;
  } else {
    rRand1 = RNG::Instance().uniform(fRMin1, fRMin2);
    A1     = std::fabs(fRMin2 * fRMin2 - fRMin1 * fRMin1);
  }
  if (fRMax1 == fRMax2) {
    rRand2 = fRMax1;
    Atot   = A1;
  } else {
    rRand2 = RNG::Instance().uniform(fRMax1, fRMax2);
    Atot   = A1 + std::fabs(fRMax2 * fRMax2 - fRMax1 * fRMax1);
  }
  rCh = RNG::Instance().uniform(0., Atot);

  if (rCh > A1) {
    rRand1 = rRand2;
  }

  xRand = rRand1 * cosphi;
  yRand = rRand1 * sinphi;

  return Vector3D<Precision>(xRand, yRand, zOne);
}

//
// GetPointOnCut
//
// Auxiliary method for Get Point On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnCut(Precision fRMin1, Precision fRMax1, Precision fRMin2,
                                                    Precision fRMax2, Precision zOne, Precision zTwo,
                                                    Precision &totArea) const
{

  if (zOne == zTwo) {
    return GetPointOnRing(fRMin1, fRMax1, fRMin2, fRMax2, zOne);
  }
  if ((fRMin1 == fRMin2) && (fRMax1 == fRMax2)) {
    return GetPointOnTubs(fRMin1, fRMax1, zOne, zTwo, totArea);
  }
  return GetPointOnCone(fRMin1, fRMax1, fRMin2, fRMax2, zOne, zTwo, totArea);
}

//
// SamplePointOnSurface
//
Vector3D<Precision> UnplacedPolycone::SamplePointOnSurface() const
{

  Precision Area = 0, totArea = 0, Achose1 = 0, Achose2 = 0, phi, cosphi, sinphi, rRand;
  int i         = 0;
  int numPlanes = GetNSections();

  phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  std::vector<Precision> areas;
  PolyconeSection const &sec0 = GetSection(0);
  areas.push_back(kPi * (sec0.fSolid->fRmax1 * sec0.fSolid->fRmax1 - sec0.fSolid->fRmin1 * sec0.fSolid->fRmin1));
  rRand =
      sec0.fSolid->fRmin1 + ((sec0.fSolid->fRmax1 - sec0.fSolid->fRmin1) * std::sqrt(RNG::Instance().uniform(0., 1.)));

  areas.push_back(kPi * (sec0.fSolid->fRmax1 * sec0.fSolid->fRmax1 - sec0.fSolid->fRmin1 * sec0.fSolid->fRmin1));

  for (i = 0; i < numPlanes; i++) {
    PolyconeSection const &sec = GetSection(i);
    Area                       = (sec.fSolid->fRmin1 + sec.fSolid->fRmin2) *
           std::sqrt((sec.fSolid->fRmin1 - sec.fSolid->fRmin2) * (sec.fSolid->fRmin1 - sec.fSolid->fRmin2) +
                     4. * sec.fSolid->fDz * sec.fSolid->fDz);

    Area += (sec.fSolid->fRmax1 + sec.fSolid->fRmax2) *
            std::sqrt((sec.fSolid->fRmax1 - sec.fSolid->fRmax2) * (sec.fSolid->fRmax1 - sec.fSolid->fRmax2) +
                      4. * sec.fSolid->fDz * sec.fSolid->fDz);

    Area *= 0.5 * GetDeltaPhi();

    if (GetDeltaPhi() < kTwoPi) {
      Area += std::fabs(2 * sec.fSolid->fDz) *
              (sec.fSolid->fRmax1 + sec.fSolid->fRmax2 - sec.fSolid->fRmin1 - sec.fSolid->fRmin2);
    }

    areas.push_back(Area);
    totArea += Area;
  }
  PolyconeSection const &secn = GetSection(numPlanes - 1);
  areas.push_back(kPi * (secn.fSolid->fRmax2 * secn.fSolid->fRmax2 - secn.fSolid->fRmin2 * secn.fSolid->fRmin2));

  totArea += (areas[0] + areas[numPlanes + 1]);
  Precision chose = RNG::Instance().uniform(0., totArea);

  if ((chose >= 0.) && (chose < areas[0])) {
    return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, fPolycone.fZs[0]);
  }

  for (i = 0; i < numPlanes; i++) {
    Achose1 += areas[i];
    Achose2 = (Achose1 + areas[i + 1]);
    if (chose >= Achose1 && chose < Achose2) {
      PolyconeSection const &sec = GetSection(i);
      return GetPointOnCut(sec.fSolid->fRmin1, sec.fSolid->fRmax1, sec.fSolid->fRmin2, sec.fSolid->fRmax2,
                           fPolycone.fZs[i], fPolycone.fZs[i + 1], Area);
    }
  }

  rRand =
      secn.fSolid->fRmin2 + ((secn.fSolid->fRmax2 - secn.fSolid->fRmin2) * std::sqrt(RNG::Instance().uniform(0., 1.)));

  return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, fPolycone.fZs[numPlanes]);
}

bool UnplacedPolycone::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  bool valid = true;
  int index  = GetSectionIndex(point.z() - kTolerance);

  if (index < 0) {
    valid                 = false;
    if (index == -1) norm = Vector3D<Precision>(0., 0., -1.);
    if (index == -2) norm = Vector3D<Precision>(0., 0., 1.);
    return valid;
  }
  PolyconeSection const &sec = GetSection(index);
  valid                      = sec.fSolid->Normal(point - Vector3D<Precision>(0, 0, sec.fShift), norm);

  // if point is within tolerance of a Z-plane between 2 sections, get normal from other section too
  if (size_t(index + 1) < fPolycone.fSections.size() && std::abs(point.z() - fPolycone.fZs[index + 1]) < kTolerance) {
    PolyconeSection const &sec2 = GetSection(index + 1);
    bool valid2                 = false;
    Vector3D<Precision> norm2;
    valid2 = sec2.fSolid->Normal(point - Vector3D<Precision>(0, 0, sec2.fShift), norm2);

    if (!valid && valid2) {
      norm  = norm2;
      valid = valid2;
    }

    // if both valid && valid2 true, norm and norm2 should be added...
    if (valid && valid2) {

      // discover exiting direction by moving point a bit (it would be good to have track direction here)
      // if(sec.fSolid->Contains(point + kTolerance*10*norm - Vector3D<Precision>(0, 0, sec.fShift))){

      //}
      bool c2;
      using CI = ConeImplementation<ConeTypes::UniversalCone>;
      // CI::Contains<Precision,false>(*sec2.fSolid,
      //                            point + kTolerance * 10 * norm2 - Vector3D<Precision>(0, 0, sec2.fShift), c2);
      CI::Contains(*sec2.fSolid, point + kTolerance * 10 * norm2 - Vector3D<Precision>(0, 0, sec2.fShift), c2);
      if (c2) {
        norm = norm2;
      } else {
        norm = norm + norm2;
        // but we might be in the interior of the polycone, and norm2=(0,0,-1) and norm1=(0,0,1) --> norm=(0,0,0)
        // quick fix:  set a normal pointing to the input point, but setting its z=0 (radial)
        if (norm.Mag2() < kTolerance) norm = Vector3D<Precision>(point.x(), point.y(), 0);
      }
    }
  }
  if (valid) norm /= norm.Mag();
  return valid;
}

#if (0)
// Simplest Extent defintion that does not take PHI into consideration
void UnplacedPolycone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{

  int i          = 0;
  Precision maxR = 0;

  for (i = 0; i < GetNSections(); i++) {
    PolyconeSection const &sec          = GetSection(i);
    if (maxR < sec.fSolid->fRmax1) maxR = sec.fSolid->fRmax1;
    if (maxR < sec.fSolid->fRmax2) maxR = sec.fSolid->fRmax2;
  }

  aMin.x() = -maxR;
  aMin.y() = -maxR;
  //  aMin.z() = fZs[0];
  aMax.x() = maxR;
  aMax.y() = maxR;
  if (fZs[0] > fZs[GetNSections()]) {
    aMax.z() = fZs[0];
    aMin.z() = fZs[GetNSections()];
  } else {
    aMin.z() = fZs[0];
    aMax.z() = fZs[GetNSections()];
  }
}
#endif

#if (1)
// Improved Extent definition that also takes PHI into consideration
void UnplacedPolycone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  /* Algorithm:
   * 1. Get the Extent in Z direction and set
   * 	aMax.z and aMin.z
   *
   * 2. For X and Y direction use Extent of Cone
   * 	and set aMax.x aMax.y, aMin.x and aMin.y
   *
   */
  Precision maxR = 0, minR = kInfLength;
  Precision fSPhi = fPolycone.fStartPhi;
  Precision fDPhi = fPolycone.fDeltaPhi;

  for (int i = 0; i < GetNSections(); i++) {
    PolyconeSection const &sec = GetSection(i);
    maxR                       = Max(maxR, Max(sec.fSolid->_frmax1, sec.fSolid->_frmax2));
    minR                       = Min(minR, Min(sec.fSolid->_frmin1, sec.fSolid->_frmin2));
  }

  if (fPolycone.fZs[0] > fPolycone.fZs[GetNSections()]) {
    aMax.z() = fPolycone.fZs[0];
    aMin.z() = fPolycone.fZs[GetNSections()];
  } else {
    aMin.z() = fPolycone.fZs[0];
    aMax.z() = fPolycone.fZs[GetNSections()];
  }

  // Using Cone to get Extent in X and Y Direction
  double minz = aMin.z();
  double maxz = aMax.z();
  SUnplacedCone<ConeTypes::UniversalCone> tempCone(minR, maxR, minR, maxR, 1, fSPhi, fDPhi);
  tempCone.BaseType_t::Extent(aMin, aMax);
  aMin.z() = minz;
  aMax.z() = maxz;

  return;
}
#endif

#endif // !VECCORE_CUDA

#if (0)

bool UnplacedPolycone::CheckContinuityInRmax(const Vector<Precision> &rOuter)
{

  bool continuous  = true;
  unsigned int len = rOuter.size();
  if (len > 2) {
    for (unsigned int j = 1; j < len;) {
      if (j != (len - 1)) continuous &= (rOuter[j] == rOuter[j + 1]);
      j = j + 2;
    }
  }
  return continuous;
}

bool UnplacedPolycone::CheckContinuity(const double rOuter[], const double rInner[], const double zPlane[],
                                       Vector<Precision> &newROuter, Vector<Precision> &newRInner,
                                       Vector<Precision> &newZPlane)
{

  Vector<Precision> rOut, rIn;
  Vector<Precision> zPl;
  rOut.push_back(rOuter[0]);
  rIn.push_back(rInner[0]);
  zPl.push_back(zPlane[0]);
  for (unsigned int j = 1; j < fNz; j++) {

    if (j == fNz - 1) {
      rOut.push_back(rOuter[j]);
      rIn.push_back(rInner[j]);
      zPl.push_back(zPlane[j]);
    } else {
      if ((zPlane[j] != zPlane[j + 1]) || (rOuter[j] != rOuter[j + 1])) {
        rOut.push_back(rOuter[j]);
        rOut.push_back(rOuter[j]);

        zPl.push_back(zPlane[j]);
        zPl.push_back(zPlane[j]);

        rIn.push_back(rInner[j]);
        rIn.push_back(rInner[j]);

      } else {
        rOut.push_back(rOuter[j]);
        zPl.push_back(zPlane[j]);
        rIn.push_back(rInner[j]);
      }
    }
  }

  if (rOut.size() % 2 != 0) {
    // fNz is odd, the adding of the last item did not happen in the loop.
    rOut.push_back(rOut[rOut.size() - 1]);
    rIn.push_back(rIn[rIn.size() - 1]);
    zPl.push_back(zPl[zPl.size() - 1]);
  }

  /* Creating a new temporary Reduced polycone with desired data elements,
  *  which makes sure that denominator will never be zero (hence avoiding FPE(division by zero)),
  *  while calculating slope.
  *
  *  This will be the minimum polycone,i.e. no extra section which
  *  affect its shape
  */

  for (size_t j = 0; j < rOut.size();) {

    if (zPl[j] != zPl[j + 1]) {
      newZPlane.push_back(zPl[j]);
      newZPlane.push_back(zPl[j + 1]);
      newROuter.push_back(rOut[j]);
      newROuter.push_back(rOut[j + 1]);
      newRInner.push_back(rIn[j]);
      newRInner.push_back(rIn[j + 1]);
    }

    j = j + 2;
  }
  // Minimum polycone construction over

  // Checking Slope continuity and Rmax Continuity
  bool contRmax  = CheckContinuityInRmax(newROuter);
  bool contSlope = CheckContinuityInSlope(newROuter, newZPlane);

  // If both are true then the polycone can be convex
  // but still final convexity depends on Inner Radius also.
  return (contRmax && contSlope);
}

/* Cleaner CheckContinuityInSlope.
 * Because of new design, it will not get the case of FPE exception
 * (division by zero)
 */
bool UnplacedPolycone::CheckContinuityInSlope(const Vector<Precision> &rOuter, const Vector<Precision> &zPlane)
{

  bool continuous      = true;
  Precision startSlope = kInfLength;

  // Doing the actual slope calculation here, and checking continuity,
  for (size_t j = 0; j < rOuter.size(); j = j + 2) {
    Precision currentSlope = (rOuter[j + 1] - rOuter[j]) / (zPlane[j + 1] - zPlane[j]);
    continuous &= (currentSlope <= startSlope);
    startSlope = currentSlope;
  }
  return continuous;
}
#endif

VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::DetectConvexity()
{
  // Default safe convexity value
  fGlobalConvexity = false;

  if (fPolycone.fConvexityPossible) {
    if (fPolycone.fEqualRmax && (fPolycone.fDeltaPhi <= kPi || fPolycone.fDeltaPhi == kTwoPi))
      // In this case, Polycone become solid Cylinder, No need to check anything else, 100% convex
      fGlobalConvexity = true;
    else {
      if (fPolycone.fDeltaPhi <= kPi || fPolycone.fDeltaPhi == kTwoPi) {
        fGlobalConvexity = fPolycone.fContinuityOverAll;
      }
    }
  }

  // return convexity;
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedPolycone::Create(LogicalVolume const *const logical_volume,
                                        Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedPolycone<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedPolycone<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedPolycone::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
                                                   const TranslationCode trans_code, const RotationCode rot_code,
                                                   VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedPolycone>(volume, transformation, trans_code, rot_code,
                                                                 placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedPolycone::Create(LogicalVolume const *const logical_volume,
                                        Transformation3D const *const transformation, const int id,
                                        VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedPolycone<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedPolycone<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedPolycone::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
                                                   const TranslationCode trans_code, const RotationCode rot_code,
                                                   const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedPolycone>(volume, transformation, trans_code, rot_code, id,
                                                                 placement);
}

#endif

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedPolycone>::SizeOf();
template void DevicePtr<cuda::UnplacedPolycone>::Construct(Precision, Precision, int, Precision *, Precision *,
                                                           Precision *) const;

} // End cxx namespace

#endif

} // end global namespace
