/*
 * UnplacedCone.cpp
 *
 *  Created on: Jun 18, 2014
 *      Author: swenzel
 */

#include "volumes/UnplacedCone.h"
#include "volumes/SpecializedCone.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "volumes/utilities/GenerationUtilities.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#endif
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
void UnplacedCone::Print() const
{
  printf("UnplacedCone {rmin1 %.2f, rmax1 %.2f, rmin2 %.2f, "
         "rmax2 %.2f, dz %.2f, phistart %.2f, deltaphi %.2f}",
         fCone.fRmin1, fCone.fRmax1, fCone.fRmin2, fCone.fRmax2, fCone.fDz, fCone.fSPhi, fCone.fDPhi);
}

void UnplacedCone::Print(std::ostream &os) const
{
  os << "UnplacedCone; please implement Print to outstream\n";
}

VECCORE_ATT_HOST_DEVICE
void UnplacedCone::DetectConvexity()
{

  // Default safe convexity value
  fGlobalConvexity = false;

  // Logic to calculate the convexity
  if (fCone.fRmin1 == 0. && fCone.fRmin2 == 0.) { // Implies Solid cone
    if (fCone.fDPhi <= kPi || fCone.fDPhi == kTwoPi) fGlobalConvexity = true;
  }
}

#if !defined(VECCORE_CUDA)
#if (0)
// Simplest Extent definition, that does not take PHI into consideration
void UnplacedCone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  Precision max = fCone.fRmax1 > fCone.fRmax2 ? fCone.fRmax1 : fCone.fRmax2;
  aMin          = Vector3D<Precision>(-max, -max, -fDz);
  aMax          = Vector3D<Precision>(max, max, fCone.fDz);
}
#endif

#if (1)
// Improved Extent definition, that takes PHI also into consideration
void UnplacedCone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // most general case

  Precision max = fCone.fRmax1 > fCone.fRmax2 ? fCone.fRmax1 : fCone.fRmax2;
  Precision min = fCone.fRmin1 > fCone.fRmin2 ? fCone.fRmin2 : fCone.fRmin1;

  aMin = Vector3D<Precision>(-max, -max, -fCone.fDz);
  aMax = Vector3D<Precision>(max, max, fCone.fDz);

  /* Below logic borrowed from Tube.
  **
  ** But it would be great, if its possible to directly call Extent of Tube.
  ** because in that case we can avoid code replication.
  */

  if (fCone.fDPhi == kTwoPi) return;

  // check how many of phi=90, 180, 270, 360deg are outside this tube
  auto Rin       = 0.5 * (max + min);
  bool phi0out   = !fCone.fPhiWedge.Contains(Vector3D<Precision>(Rin, 0, 0));
  bool phi90out  = !fCone.fPhiWedge.Contains(Vector3D<Precision>(0, Rin, 0));
  bool phi180out = !fCone.fPhiWedge.Contains(Vector3D<Precision>(-Rin, 0, 0));
  bool phi270out = !fCone.fPhiWedge.Contains(Vector3D<Precision>(0, -Rin, 0));

  // if none of those 4 phis is outside, largest box still required
  if (!(phi0out || phi90out || phi180out || phi270out)) return;

  // some extent(s) of box will be reduced
  // --> think of 4 points A,B,C,D such that A,B are at Rmin, C,D at Rmax
  //     and A,C at startPhi (fCone.fSPhi), B,D at endPhi (fCone.fSPhi+fDphi)
  auto Cx = max * cos(fCone.fSPhi);
  auto Dx = max * cos(fCone.fSPhi + fCone.fDPhi);
  auto Cy = max * sin(fCone.fSPhi);
  auto Dy = max * sin(fCone.fSPhi + fCone.fDPhi);

  // then rewrite box sides whenever each one of those phis are not contained in the tube section
  if (phi0out) aMax.x()   = Max(Cx, Dx);
  if (phi90out) aMax.y()  = Max(Cy, Dy);
  if (phi180out) aMin.x() = Min(Cx, Dx);
  if (phi270out) aMin.y() = Min(Cy, Dy);

  if (fCone.fDPhi >= kPi) return;

  auto Ax = min * cos(fCone.fSPhi);
  auto Bx = min * cos(fCone.fSPhi + fCone.fDPhi);
  auto Ay = min * sin(fCone.fSPhi);
  auto By = min * sin(fCone.fSPhi + fCone.fDPhi);

  Precision temp;
  temp     = Max(Ax, Bx);
  aMax.x() = temp > aMax.x() ? temp : aMax.x();

  temp     = Max(Ay, By);
  aMax.y() = temp > aMax.y() ? temp : aMax.y();

  temp     = Min(Ax, Bx);
  aMin.x() = temp < aMin.x() ? temp : aMin.x();

  temp     = Min(Ay, By);
  aMin.y() = temp < aMin.y() ? temp : aMin.y();

  return;
}
#endif

bool UnplacedCone::Normal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const
{
  int noSurfaces = 0;
  Precision rho, pPhi;
  Precision distZ, distRMin, distRMax;
  Precision distSPhi = kInfLength, distEPhi = kInfLength;
  Precision pRMin, widRMin;
  Precision pRMax, widRMax;

  const double kHalfTolerance = 0.5 * kTolerance;

  Vector3D<Precision> sumnorm(0., 0., 0.), nZ = Vector3D<Precision>(0., 0., 1.);
  Vector3D<Precision> nR, nr(0., 0., 0.), nPs, nPe;
  norm = sumnorm;

  // do not use an extra fabs here -- negative/positive distZ tells us when point is outside or inside
  distZ = std::fabs(p.z()) - fCone.fDz;
  rho   = std::sqrt(p.x() * p.x() + p.y() * p.y());

  pRMin    = rho - p.z() * fCone.fTanRMin;
  widRMin  = fCone.fRmin2 - fCone.fDz * fCone.fTanRMin;
  distRMin = (pRMin - widRMin) / fCone.fSecRMin;

  pRMax    = rho - p.z() * fCone.fTanRMax;
  widRMax  = fCone.fRmax2 - fCone.fDz * fCone.fTanRMax;
  distRMax = (pRMax - widRMax) / fCone.fSecRMax;

  bool inside = distZ < kTolerance && distRMax < kTolerance;
  if (fCone.fRmin1 || fCone.fRmin2) inside &= distRMin > -kTolerance;

  distZ    = std::fabs(distZ);
  distRMax = std::fabs(distRMax);
  distRMin = std::fabs(distRMin);

  // keep track of nearest normal, needed in case point is not on a surface
  double distNearest              = distZ;
  Vector3D<Precision> normNearest = nZ;
  if (p.z() < 0.) normNearest.Set(0, 0, -1.);

  if (!IsFullPhi()) {
    if (rho) { // Protected against (0,0,z)
      pPhi = std::atan2(p.y(), p.x());

      if (pPhi < fCone.fSPhi - kHalfTolerance)
        pPhi += 2 * kPi;
      else if (pPhi > fCone.fSPhi + fCone.fDPhi + kHalfTolerance)
        pPhi -= 2 * kPi;

      distSPhi = rho * (pPhi - fCone.fSPhi);
      distEPhi = rho * (pPhi - fCone.fSPhi - fCone.fDPhi);
      inside   = inside && (distSPhi > -kTolerance) && (distEPhi < kTolerance);
      distSPhi = std::abs(distSPhi);
      distEPhi = std::abs(distEPhi);
    }

    else if (!(fCone.fRmin1) || !(fCone.fRmin2)) {
      distSPhi = 0.;
      distEPhi = 0.;
    }
    nPs = Vector3D<Precision>(std::sin(fCone.fSPhi), -std::cos(fCone.fSPhi), 0);
    nPe = Vector3D<Precision>(-std::sin(fCone.fSPhi + fCone.fDPhi), std::cos(fCone.fSPhi + fCone.fDPhi), 0);
  }

  if (rho > kHalfTolerance) {
    nR = Vector3D<Precision>(p.x() / rho / fCone.fSecRMax, p.y() / rho / fCone.fSecRMax,
                             -fCone.fTanRMax / fCone.fSecRMax);
    if (fCone.fRmin1 || fCone.fRmin2) {
      nr = Vector3D<Precision>(-p.x() / rho / fCone.fSecRMin, -p.y() / rho / fCone.fSecRMin,
                               fCone.fTanRMin / fCone.fSecRMin);
    }
  }

  if (inside && distZ <= kHalfTolerance) {
    noSurfaces++;
    if (p.z() >= 0.)
      sumnorm += nZ;
    else
      sumnorm.Set(0, 0, -1.);
  }

  if (inside && distRMax <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += nR;
  } else if (noSurfaces == 0 && distRMax < distNearest) {
    distNearest = distRMax;
    normNearest = nR;
  }

  if (fCone.fRmin1 || fCone.fRmin2) {
    if (inside && distRMin <= kHalfTolerance) {
      noSurfaces++;
      sumnorm += nr;
    } else if (noSurfaces == 0 && distRMin < distNearest) {
      distNearest = distRMin;
      normNearest = nr;
    }
  }

  if (!IsFullPhi()) {
    if (inside && distSPhi <= kHalfTolerance) {
      noSurfaces++;
      sumnorm += nPs;
    } else if (noSurfaces == 0 && distSPhi < distNearest) {
      distNearest = distSPhi;
      normNearest = nPs;
    }
    if (inside && distEPhi <= kHalfTolerance) {
      noSurfaces++;
      sumnorm += nPe;
    } else if (noSurfaces == 0 && distEPhi < distNearest) {
      // No more check on distNearest, no need to assign to it.
      // distNearest = distEPhi;
      normNearest = nPe;
    }
  }
  // Final checks
  if (noSurfaces == 0)
    norm = normNearest;
  else if (noSurfaces == 1)
    norm = sumnorm;
  else
    norm = sumnorm.Unit();
  return noSurfaces != 0;
}

template <bool top>
bool UnplacedCone::IsOnZPlane(Vector3D<Precision> const &point) const
{
  if (top) {
    return (point.z() < (fCone.fDz + kTolerance)) && (point.z() > (fCone.fDz - kTolerance));
  } else {
    return (point.z() < (-fCone.fDz + kTolerance)) && (point.z() > (-fCone.fDz - kTolerance));
  }
}

template <bool start>
bool UnplacedCone::IsOnPhiWedge(Vector3D<Precision> const &point) const
{
  if (start) {
    // return GetWedge().IsOnSurfaceGeneric<kScalar>(GetWedge().GetAlong1(), GetWedge().GetNormal1(), point);
    return GetWedge().IsOnSurfaceGeneric(GetWedge().GetAlong1(), GetWedge().GetNormal1(), point);
  } else {
    // return GetWedge().IsOnSurfaceGeneric<kScalar>(GetWedge().GetAlong2(), GetWedge().GetNormal2(), point);
    return GetWedge().IsOnSurfaceGeneric(GetWedge().GetAlong2(), GetWedge().GetNormal2(), point);
  }
}

template <bool inner>
Precision UnplacedCone::GetRadiusOfConeAtPoint(Precision const pointZ) const
{
  if (inner) {
    return GetInnerSlope() * pointZ + GetInnerOffset();

  } else {
    return GetOuterSlope() * pointZ + GetOuterOffset();
  }
}

template <bool inner>
bool UnplacedCone::IsOnConicalSurface(Vector3D<Precision> const &point) const
{

  Precision rho      = point.Perp2();
  Precision coneRad  = GetRadiusOfConeAtPoint<inner>(point.z());
  Precision coneRad2 = coneRad * coneRad;
  return (rho >= (coneRad2 - kTolerance * coneRad)) && (rho <= (coneRad2 + kTolerance * coneRad)) &&
         (Abs(point.z()) < (GetDz() + kTolerance));
}

bool UnplacedCone::IsOnEdge(Vector3D<Precision> &point) const
{
  int count = 0;
  if (IsOnZPlane<true>(point) || IsOnZPlane<false>(point)) count++;
  if (IsOnPhiWedge<true>(point)) count++;
  if (IsOnPhiWedge<false>(point)) count++;
  if (IsOnConicalSurface<true>(point)) count++;
  if (IsOnConicalSurface<false>(point)) count++;

  return count > 1;
}

Vector3D<Precision> UnplacedCone::SamplePointOnSurface() const
{
  // implementation taken from UCons; not verified
  //
  Vector3D<Precision> retPt;
  do {

    double Aone, Atwo, Athree, Afour, Afive, slin, slout, phi;
    double zRand, cosu, sinu, rRand1, rRand2, chose, rone, rtwo, qone, qtwo;
    rone = (fCone.fRmax1 - fCone.fRmax2) / (2. * fCone.fDz);
    rtwo = (fCone.fRmin1 - fCone.fRmin2) / (2. * fCone.fDz);
    qone = 0.;
    qtwo = 0.;
    if (fCone.fRmax1 != fCone.fRmax2) {
      qone = fCone.fDz * (fCone.fRmax1 + fCone.fRmax2) / (fCone.fRmax1 - fCone.fRmax2);
    }
    if (fCone.fRmin1 != fCone.fRmin2) {
      qtwo = fCone.fDz * (fCone.fRmin1 + fCone.fRmin2) / (fCone.fRmin1 - fCone.fRmin2);
    }
    slin   = Sqrt((fCone.fRmin1 - fCone.fRmin2) * (fCone.fRmin1 - fCone.fRmin2) + 4. * fCone.fDz * fCone.fDz);
    slout  = Sqrt((fCone.fRmax1 - fCone.fRmax2) * (fCone.fRmax1 - fCone.fRmax2) + 4. * fCone.fDz * fCone.fDz);
    Aone   = 0.5 * fCone.fDPhi * (fCone.fRmax2 + fCone.fRmax1) * slout;
    Atwo   = 0.5 * fCone.fDPhi * (fCone.fRmin2 + fCone.fRmin1) * slin;
    Athree = 0.5 * fCone.fDPhi * (fCone.fRmax1 * fCone.fRmax1 - fCone.fRmin1 * fCone.fRmin1);
    Afour  = 0.5 * fCone.fDPhi * (fCone.fRmax2 * fCone.fRmax2 - fCone.fRmin2 * fCone.fRmin2);
    Afive  = fCone.fDz * (fCone.fRmax1 - fCone.fRmin1 + fCone.fRmax2 - fCone.fRmin2);

    phi    = RNG::Instance().uniform(fCone.fSPhi, fCone.fSPhi + fCone.fDPhi);
    cosu   = std::cos(phi);
    sinu   = std::sin(phi);
    rRand1 = volumeUtilities::GetRadiusInRing(fCone.fRmin1, fCone.fRmin2);
    rRand2 = volumeUtilities::GetRadiusInRing(fCone.fRmax1, fCone.fRmax2);

    if ((fCone.fSPhi == 0.) && IsFullPhi()) {
      Afive = 0.;
    }
    chose = RNG::Instance().uniform(0., Aone + Atwo + Athree + Afour + 2. * Afive);

    if ((chose >= 0.) && (chose < Aone)) {
      if (fCone.fRmin1 != fCone.fRmin2) {
        zRand = RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz);
        retPt.Set(rtwo * cosu * (qtwo - zRand), rtwo * sinu * (qtwo - zRand), zRand);
      } else {
        retPt.Set(fCone.fRmin1 * cosu, fCone.fRmin2 * sinu, RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz));
      }
    } else if ((chose >= Aone) && (chose <= Aone + Atwo)) {
      if (fCone.fRmax1 != fCone.fRmax2) {
        zRand = RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz);
        retPt.Set(rone * cosu * (qone - zRand), rone * sinu * (qone - zRand), zRand);
      } else {
        retPt.Set(fCone.fRmax1 * cosu, fCone.fRmax2 * sinu, RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz));
      }
    } else if ((chose >= Aone + Atwo) && (chose < Aone + Atwo + Athree)) {
      retPt.Set(rRand1 * cosu, rRand1 * sinu, -1 * fCone.fDz);
    } else if ((chose >= Aone + Atwo + Athree) && (chose < Aone + Atwo + Athree + Afour)) {
      retPt.Set(rRand2 * cosu, rRand2 * sinu, fCone.fDz);
    } else if ((chose >= Aone + Atwo + Athree + Afour) && (chose < Aone + Atwo + Athree + Afour + Afive)) {
      zRand  = RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz);
      rRand1 = RNG::Instance().uniform(
          fCone.fRmin2 - ((zRand - fCone.fDz) / (2. * fCone.fDz)) * (fCone.fRmin1 - fCone.fRmin2),
          fCone.fRmax2 - ((zRand - fCone.fDz) / (2. * fCone.fDz)) * (fCone.fRmax1 - fCone.fRmax2));
      retPt.Set(rRand1 * std::cos(fCone.fSPhi), rRand1 * std::sin(fCone.fSPhi), zRand);
    } else {
      zRand  = RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz);
      rRand1 = RNG::Instance().uniform(
          fCone.fRmin2 - ((zRand - fCone.fDz) / (2. * fCone.fDz)) * (fCone.fRmin1 - fCone.fRmin2),
          fCone.fRmax2 - ((zRand - fCone.fDz) / (2. * fCone.fDz)) * (fCone.fRmax1 - fCone.fRmax2));
      retPt.Set(rRand1 * std::cos(fCone.fSPhi + fCone.fDPhi), rRand1 * std::sin(fCone.fSPhi + fCone.fDPhi), zRand);
    }
  } while (IsOnEdge(retPt));

  return retPt;
}
#endif // VECCORE_CUDA

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedCone::Create(LogicalVolume const *const logical_volume,
                                    Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                    const int id,
#endif
                                    VPlacedVolume *const placement)
{

  using namespace ConeTypes;
  __attribute__((unused)) const UnplacedCone &cone =
      static_cast<const UnplacedCone &>(*(logical_volume->GetUnplacedVolume()));

#ifdef VECCORE_CUDA
#define RETURN_SPECIALIZATION(coneTypeT)                                                   \
  return CreateSpecializedWithPlacement<SpecializedCone<transCodeT, rotCodeT, coneTypeT>>( \
      logical_volume, transformation, id, placement)
#else
#define RETURN_SPECIALIZATION(coneTypeT)                                                                  \
  return CreateSpecializedWithPlacement<SpecializedCone<transCodeT, rotCodeT, coneTypeT>>(logical_volume, \
                                                                                          transformation, placement)
#endif

#ifdef GENERATE_CONE_SPECIALIZATIONS
  if (cone.GetRmin1() <= 0 && cone.GetRmin2() <= 0) {
    if (cone.GetDPhi() >= 2 * M_PI) RETURN_SPECIALIZATION(NonHollowCone);
    if (cone.GetDPhi() == M_PI) RETURN_SPECIALIZATION(NonHollowConeWithPiSector); // == M_PI ???

    if (cone.GetDPhi() < M_PI) RETURN_SPECIALIZATION(NonHollowConeWithSmallerThanPiSector);
    if (cone.GetDPhi() > M_PI) RETURN_SPECIALIZATION(NonHollowConeWithBiggerThanPiSector);
  } else if (cone.GetRmin1() > 0 || cone.GetRmin2() > 0) {
    if (cone.GetDPhi() >= 2 * M_PI) RETURN_SPECIALIZATION(HollowCone);
    if (cone.GetDPhi() == M_PI) RETURN_SPECIALIZATION(HollowConeWithPiSector); // == M_PI ???
    if (cone.GetDPhi() < M_PI) RETURN_SPECIALIZATION(HollowConeWithSmallerThanPiSector);
    if (cone.GetDPhi() > M_PI) RETURN_SPECIALIZATION(HollowConeWithBiggerThanPiSector);
  }
#endif

  RETURN_SPECIALIZATION(UniversalCone);

#undef RETURN_SPECIALIZATION
}

#if defined(VECGEOM_USOLIDS)
VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedCone::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Cone\n"
     << " Parameters: \n"
     << "     Cone Radii Rmin1, Rmax1: " << fCone.fRmin1 << "mm, " << fCone.fRmax1 << "mm\n"
     << "                Rmin2, Rmax2: " << fCone.fRmin2 << "mm, " << fCone.fRmax2 << "mm\n"
     << "     Half-length Z = " << fCone.fDz << "mm\n";
  if (fCone.fDPhi < kTwoPi) {
    os << "     Wedge starting angles:fCone.fSPhi=" << fCone.fSPhi * kRadToDeg << "deg, "
       << ",fCone.fDPhi=" << fCone.fDPhi * kRadToDeg << "deg\n";
  }
  os << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif

// this is repetitive code:

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedCone::SpecializedVolume(LogicalVolume const *const volume,
                                               Transformation3D const *const transformation,
                                               const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                               const int id,
#endif
                                               VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedCone>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                             id,
#endif
                                                             placement);
}

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedCone>(in_gpu_ptr, GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(),
                                     GetDPhi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedCone>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedCone>::SizeOf();
template void DevicePtr<cuda::UnplacedCone>::Construct(const Precision rmin1, const Precision rmax1,
                                                       const Precision rmin2, const Precision rmax2, const Precision z,
                                                       const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End namespace vecgeom
