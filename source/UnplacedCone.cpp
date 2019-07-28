/*
 * UnplacedCone.cpp
 *
 *  Created on: Jun 18, 2014
 *      Author: swenzel
 */

#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedTube.h"
#include "volumes/SpecializedCone.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "volumes/utilities/GenerationUtilities.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#include "volumes/UnplacedImplAs.h"
#endif
#ifdef VECGEOM_ROOT
#include "TGeoCone.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Cons.hh"
#endif

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedCone::ConvertToRoot(char const *label) const
{
  if (GetDPhi() == 2. * M_PI) {
    return new TGeoCone(label, GetDz(), GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2());
  } else {
    return new TGeoConeSeg(label, GetDz(), GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetSPhi() * kRadToDeg,
                           (GetSPhi() + GetDPhi()) * kRadToDeg);
  }
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedCone::ConvertToGeant4(char const *label) const
{
  return new G4Cons(label, GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(), GetDPhi());
}
#endif
#endif

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

#ifndef VECCORE_CUDA
SolidMesh *UnplacedCone::CreateMesh3D(Transformation3D const &trans, const size_t nFaces) const
{

  typedef Vector3D<double> Vec_t;
  SolidMesh *sm = new SolidMesh();

  size_t nSegments = 0;
  if (GetRmin1() == 0. && GetRmin2() == 0.) {
    nSegments = nFaces / 3;
    sm->ResetMesh(4 * (nSegments + 1), 3 * nSegments + 2);
  } else {
    nSegments = nFaces / 4;
    sm->ResetMesh(4 * (nSegments + 1), 4 * nSegments + 2);
  }

  Vec_t *vertices = new Vec_t[4 * (nSegments + 1)];

  double cos, sin;
  double angle = GetSPhi();
  double step  = GetDPhi() / nSegments;
  size_t idx0  = 0;
  size_t idx1  = nSegments + 1;
  size_t idx2  = 2 * (nSegments + 1);
  size_t idx3  = 3 * (nSegments + 1);
  for (size_t i = 0; i <= nSegments; ++i, angle += step) {
    cos              = std::cos(angle);
    sin              = std::sin(angle);
    vertices[idx0++] = Vec_t(GetRmin1() * cos, GetRmin1() * sin, -GetDz()); // bottom inner
    vertices[idx1++] = Vec_t(GetRmax1() * cos, GetRmax1() * sin, -GetDz()); // bottom outer
    vertices[idx2++] = Vec_t(GetRmin2() * cos, GetRmin2() * sin, GetDz());  // top inner
    vertices[idx3++] = Vec_t(GetRmax2() * cos, GetRmax2() * sin, GetDz());  // top outer
  }
  sm->SetVertices(vertices, 4 * (nSegments + 1));

  delete[] vertices;
  sm->TransformVertices(trans);

  for (size_t j = 0, k = j + nSegments + 1; j < nSegments; j++, k++) {
    sm->AddPolygon(4, {k + 1, k, j, j + 1}, true); // bottom surface
  }

  for (size_t j = 0, k = 2 * (nSegments + 1), l = k + nSegments + 1; j < nSegments; j++, k++, l++) {
    sm->AddPolygon(4, {l, l + 1, k + 1, k}, true); // top surface
  }

  for (size_t j = 0, k = (nSegments + 1), l = k + 2 * (nSegments + 1); j < nSegments; j++, k++, l++) {
    sm->AddPolygon(4, {k, k + 1, l + 1, l}, true); // lateral outer surface
  }

  for (size_t j = 0, k = j + 2 * (nSegments + 1); j < nSegments; j++, k++) {
    sm->AddPolygon(4, {k, k + 1, j + 1, j}, true); // lateral inner  surface
  }

  if (GetDPhi() != 2 * kPi) {
    sm->AddPolygon(4, {0, nSegments + 1, 3 * (nSegments + 1), 2 * (nSegments + 1)}, true);
    sm->AddPolygon(
        4, {2 * (nSegments + 1) + nSegments, 3 * (nSegments + 1) + nSegments, nSegments + 1 + nSegments, 0 + nSegments},
        true);
  }

  sm->InitPolygons();

  return sm;
}
#endif

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
VECCORE_ATT_HOST_DEVICE
void UnplacedCone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // most general case

  Precision max = fCone._frmax1 > fCone._frmax2 ? fCone._frmax1 : fCone._frmax2;
  Precision min = fCone._frmin1 > fCone._frmin2 ? fCone._frmin2 : fCone._frmin1;

  aMin = Vector3D<Precision>(-max, -max, -fCone.fDz);
  aMax = Vector3D<Precision>(max, max, fCone.fDz);

  /* Below logic borrowed from Tube.
  **
  ** But it would be great, if it's possible to directly call Extent of Tube.
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
  if (phi0out) aMax.x() = Max(Cx, Dx);
  if (phi90out) aMax.y() = Max(Cy, Dy);
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

VECCORE_ATT_HOST_DEVICE
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

#if (0) // Old buggy definition as pointed in Jira-433
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
#endif

#if (1) // New Simplified and accurate definition to sample points on the surface of Cone
Vector3D<Precision> UnplacedCone::SamplePointOnSurface() const
{
  // implementation taken from UCons; not verified
  //
  double rad = 0.;
  Vector3D<Precision> retPt;
  double zRand      = RNG::Instance().uniform(-1. * fCone.fDz, fCone.fDz);
  int surfSelection = 0;
  if (!fCone.fRmin1 && !fCone.fRmin2) {
    if (fCone.fDPhi < vecgeom::kTwoPi)
      surfSelection = (int)RNG::Instance().uniform(2., 7.);
    else
      surfSelection = (int)RNG::Instance().uniform(2., 5.);

  } else {
    if (fCone.fDPhi < vecgeom::kTwoPi)
      surfSelection = (int)RNG::Instance().uniform(1., 7.);
    else
      surfSelection = (int)RNG::Instance().uniform(1., 5.);
  }

  if (surfSelection == 1) {
    // Generate point on inner Conical surface
    rad       = GetRadiusOfConeAtPoint<true>(zRand);
    retPt.z() = zRand;
  }
  if (surfSelection == 2) {
    // Generate point on outer Conical surface
    rad       = GetRadiusOfConeAtPoint<false>(zRand);
    retPt.z() = zRand;
  }
  if (surfSelection == 3) {
    // Generate point on top Z plane
    rad       = RNG::Instance().uniform(fCone.fRmin2, fCone.fRmax2);
    retPt.z() = fCone.fDz;
  }
  if (surfSelection == 4) {
    // Generate point on bottom Z plane
    rad       = RNG::Instance().uniform(fCone.fRmin1, fCone.fRmax1);
    retPt.z() = -fCone.fDz;
  }

  if (surfSelection == 5) {
    // Generate point on startPhi Surface
    double rmin = 0.;
    if (fCone.fRmin1 || fCone.fRmin2) rmin = GetRadiusOfConeAtPoint<true>(zRand);
    double rmax   = GetRadiusOfConeAtPoint<false>(zRand);
    double rinter = RNG::Instance().uniform(rmin, rmax);
    retPt.Set(rinter * std::cos(fCone.fSPhi), rinter * std::sin(fCone.fSPhi), zRand);
    return retPt;
  }
  if (surfSelection == 6) {
    // Generate point on endPhi Surface
    double rmin = 0.;
    if (fCone.fRmin1 || fCone.fRmin2) rmin = GetRadiusOfConeAtPoint<true>(zRand);
    double rmax   = GetRadiusOfConeAtPoint<false>(zRand);
    double rinter = RNG::Instance().uniform(rmin, rmax);
    retPt.Set(rinter * std::cos(fCone.fSPhi + fCone.fDPhi), rinter * std::sin(fCone.fSPhi + fCone.fDPhi), zRand);
    return retPt;
  }

  double theta = RNG::Instance().uniform(fCone.fSPhi, fCone.fSPhi + fCone.fDPhi);
  retPt.x()    = rad * std::cos(theta);
  retPt.y()    = rad * std::sin(theta);
  return retPt;
}
#endif

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

// this is repetitive code:

template <>
UnplacedCone *Maker<UnplacedCone>::MakeInstance(const Precision &rmin1, const Precision &rmax1, const Precision &rmin2,
                                                const Precision &rmax2, const Precision &dz, const Precision &phimin,
                                                const Precision &deltaphi)
{
// #ifdef GENERATE_CONE_SPECIALIZATIONS
#ifndef VECGEOM_NO_SPECIALIZATION
  if (rmin1 <= 0 && rmin2 <= 0) {
    if (deltaphi >= 2 * M_PI) {
      // NonHollowCone becomes NonHollowTube
      if (rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::NonHollowCone>, SUnplacedTube<TubeTypes::NonHollowTube>>(
            rmin1, rmax1, dz, phimin, deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::NonHollowCone>(rmin1, rmax1, rmin2, rmax2, dz, phimin, deltaphi);
      }
    }
    if (deltaphi == M_PI) {
      // NonHollowConeWithPiSector becomes NonHollowTubeWithPiSector
      if (rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::NonHollowConeWithPiSector>,
                                   SUnplacedTube<TubeTypes::NonHollowTubeWithPiSector>>(rmin1, rmax1, dz, phimin,
                                                                                        deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::NonHollowConeWithPiSector>(rmin1, rmax1, rmin2, rmax2, dz, phimin,
                                                                       deltaphi); // == M_PI ???
      }
    }
    if (deltaphi < M_PI) {
      // NonHollowConeWithSmallerThanPiSector becomes NonHollowTubeWithSmallerThanPiSector
      if (rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::NonHollowConeWithSmallerThanPiSector>,
                                   SUnplacedTube<TubeTypes::NonHollowTubeWithSmallerThanPiSector>>(rmin1, rmax1, dz,
                                                                                                   phimin, deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::NonHollowConeWithSmallerThanPiSector>(rmin1, rmax1, rmin2, rmax2, dz,
                                                                                  phimin, deltaphi);
      }
    }
    if (deltaphi > M_PI) {
      // NonHollowConeWithBiggerThanPiSector becomes NonHollowTubeWithBiggerThanPiSector
      if (rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::NonHollowConeWithBiggerThanPiSector>,
                                   SUnplacedTube<TubeTypes::NonHollowTubeWithBiggerThanPiSector>>(rmin1, rmax1, dz,
                                                                                                  phimin, deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::NonHollowConeWithBiggerThanPiSector>(rmin1, rmax1, rmin2, rmax2, dz, phimin,
                                                                                 deltaphi);
      }
    }
  } else if (rmin1 > 0 || rmin2 > 0) {
    if (deltaphi >= 2 * M_PI) {
      // HollowCone becomes HollowTube
      if (rmin1 == rmin2 && rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::HollowCone>, SUnplacedTube<TubeTypes::HollowTube>>(
            rmin1, rmax1, dz, phimin, deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::HollowCone>(rmin1, rmax1, rmin2, rmax2, dz, phimin, deltaphi);
      }
    }
    // HollowConeWithPiSector becomes HollowTubeWithPiSector
    if (deltaphi == M_PI) {
      if (rmin1 == rmin2 && rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::HollowConeWithPiSector>,
                                   SUnplacedTube<TubeTypes::HollowTubeWithPiSector>>(rmin1, rmax1, dz, phimin,
                                                                                     deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::HollowConeWithPiSector>(rmin1, rmax1, rmin2, rmax2, dz, phimin,
                                                                    deltaphi); // == M_PI ???
      }
    }
    if (deltaphi < M_PI) {
      // HollowConeWithSmallerThanPiSector becomes HollowTubeWithSmallerThanPiSector
      if (rmin1 == rmin2 && rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::HollowConeWithSmallerThanPiSector>,
                                   SUnplacedTube<TubeTypes::HollowTubeWithSmallerThanPiSector>>(rmin1, rmax1, dz,
                                                                                                phimin, deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::HollowConeWithSmallerThanPiSector>(rmin1, rmax1, rmin2, rmax2, dz, phimin,
                                                                               deltaphi);
      }
    }
    if (deltaphi > M_PI) {
      // HollowConeWithBiggerThanPiSector becomes HollowTubeWithBiggerThanPiSector
      if (rmin1 == rmin2 && rmax1 == rmax2) {
        return new SUnplacedImplAs<SUnplacedCone<ConeTypes::HollowConeWithBiggerThanPiSector>,
                                   SUnplacedTube<TubeTypes::HollowTubeWithBiggerThanPiSector>>(rmin1, rmax1, dz, phimin,
                                                                                               deltaphi);
      } else {
        return new SUnplacedCone<ConeTypes::HollowConeWithBiggerThanPiSector>(rmin1, rmax1, rmin2, rmax2, dz, phimin,
                                                                              deltaphi);
      }
    }
  }
  // this should never happen...
  return nullptr;
#else
  // if no specialization, return the most general case
  return new SUnplacedCone<ConeTypes::UniversalCone>(rmin1, rmax1, rmin2, rmax2, dz, phimin, deltaphi);
#endif
}

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<SUnplacedCone<ConeTypes::UniversalCone>>(in_gpu_ptr, GetRmin1(), GetRmax1(), GetRmin2(),
                                                                GetRmax2(), GetDz(), GetSPhi(), GetDPhi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedCone::CopyToGpu() const
{
  return CopyToGpuImpl<SUnplacedCone<ConeTypes::UniversalCone>>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::SUnplacedCone<cuda::ConeTypes::UniversalCone>>::SizeOf();
template void DevicePtr<cuda::SUnplacedCone<cuda::ConeTypes::UniversalCone>>::Construct(
    const Precision rmin1, const Precision rmax1, const Precision rmin2, const Precision rmax2, const Precision z,
    const Precision sphi, const Precision dphi) const;

} // namespace cxx

#endif

} // End namespace vecgeom
