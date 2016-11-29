/// \file UnplacedTrapezoid.cpp
/// \author Guilherme Lima (lima@fnal.gov)
//
// 140407 G. Lima    - based on equivalent box code
// 160722 G. Lima    Revision + moving to new backend structure

#include "volumes/UnplacedTrapezoid.h"
#include "backend/Backend.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedTrapezoid.h"
#include "base/RNG.h"
#include <cstdio>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Vec3D = Vector3D<Precision>;
#ifdef VECGEOM_PLANESHELL_DISABLE
using TrapSidePlane = TrapezoidStruct<double>::TrapSidePlane;
#endif

VECGEOM_CUDA_HEADER_BOTH
UnplacedTrapezoid::UnplacedTrapezoid(TrapCorners const corners) : fTrap()
{
  // fill data members
  fromCornersToParameters(corners);

  // check planarity of all four sides
  // TODO: this needs a proper logger treatment as per geantv conventions
  bool good = MakePlanes(corners);
  if (!good) printf("***** WARNING in Trapezoid constructor: corners provided fail coplanarity tests.");

  fGlobalConvexity = true;
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedTrapezoid::UnplacedTrapezoid(double dx, double dy, double dz, double)
    : fTrap(dz, 0., 0., dy, dx, dx, 0., dy, dx, dx, 0.)
{
// TODO: this needs a proper logger treatment as per geantv conventions
#ifndef VECGEOM_NVCC
  fprintf(stderr, "*** ERROR: STEP-based trapezoid constructor called, but not implemented ***");
#endif
  assert(false);
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedTrapezoid::UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox)
    : fTrap(zbox, 0., 0., ybox, xbox, xbox, 0., ybox, xbox, xbox, 0.)
{
  // validity check
  // TODO: this needs a proper logger treatment as per geantv conventions
  if (xbox <= 0 || ybox <= 0 || zbox <= 0) {
    printf("UnplacedTrapezoid(xbox,...) - GeomSolids0002, Fatal Exception: Invalid input length parameters for Solid: "
           "UnplacedTrapezoid\n");
    printf("\t X=%f, Y=%f, Z=%f", xbox, ybox, zbox);
  }

  MakePlanes();
  fGlobalConvexity = true;
}

// VECGEOM_CUDA_HEADER_BOTH
// void UnplacedTrapezoid::CalcCapacity()
Precision UnplacedTrapezoid::Capacity() const
{
  // cubic approximation used in Geant4
  Precision vol = fTrap.fDz * ((fTrap.fDx1 + fTrap.fDx2 + fTrap.fDx3 + fTrap.fDx4) * (fTrap.fDy1 + fTrap.fDy2) +
                               (fTrap.fDx4 + fTrap.fDx3 - fTrap.fDx2 - fTrap.fDx1) * (fTrap.fDy2 - fTrap.fDy1) / 3.0);

  /*
  // GL: leaving this here for future reference
    // accurate volume calculation
    TrapCorners pt;
    this->fromParametersToCorners(pt);

    // more precise, hopefully correct version (to be checked)
    Precision BmZm = pt[1].x() - pt[0].x();
    Precision BpZm = pt[3].x() - pt[2].x();
    Precision BmZp = pt[5].x() - pt[4].x();
    Precision BpZp = pt[7].x() - pt[6].x();
    Precision xCorr = (BpZp-BpZm + BmZp-BmZm) / (BpZm+BmZm);

    Precision ymZm = pt[0].y();
    Precision ypZm = pt[2].y();
    Precision ymZp = pt[4].y();
    Precision ypZp = pt[6].y();
    Precision yCorr = (ypZp-ypZm - (ymZp-ymZm)) / (ypZm-ymZm);

    Precision volume = 4*fDz*fDy1*(fDx1+fDx2) * ( 1.0 + (xCorr + yCorr)/2.0 + xCorr*yCorr/3.0 );
  */

  return vol;
}

/*
Precision UnplacedTrapezoid::SurfaceArea() const
{

  Vec3D ba(fDx1 - fDx2 + fTanAlpha1 * 2 * fDy1, 2 * fDy1, 0);
  Vec3D bc(2 * fDz * fTthetaCphi - (fDx4 - fDx2) + fTanAlpha2 * fDy2 - fTanAlpha1 * fDy1,
           2 * fDz * fTthetaSphi + fDy2 - fDy1, 2 * fDz);
  Vec3D dc(-fDx4 + fDx3 + 2 * fTanAlpha2 * fDy2, 2 * fDy2, 0);
  Vec3D da(-2 * fDz * fTthetaCphi - (fDx1 - fDx3) - fTanAlpha1 * fDy1 + fTanAlpha2 * fDy2,
           -2 * fDz * fTthetaSphi - fDy1 + fDy2, -2 * fDz);

  Vec3D ef(fDx2 - fDx1 + 2 * fTanAlpha1 * fDy1, 2 * fDy1, 0);
  Vec3D eh(2 * fDz * fTthetaCphi + fDx3 - fDx1 + fTanAlpha1 * fDy1 - fTanAlpha2 * fDy2,
           2 * fDz * fTthetaSphi - fDy2 + fDy1, 2 * fDz);
  Vec3D gh(fDx3 - fDx4 - 2 * fTanAlpha2 * fDy2, -2 * fDy2, 0);
  Vec3D gf(-2 * fDz * fTthetaCphi + fDx2 - fDx4 + fTanAlpha1 * fDy1 - fTanAlpha2 * fDy2,
           -2 * fDz * fTthetaSphi + fDy1 - fDy2, -2 * fDz);

  Vec3D cr;
  cr             = ba.Cross(bc);
  Precision babc = cr.Mag();
  cr             = dc.Cross(da);
  Precision dcda = cr.Mag();
  cr             = ef.Cross(eh);
  Precision efeh = cr.Mag();
  cr             = gh.Cross(gf);
  Precision ghgf = cr.Mag();

  Precision surfArea = 2 * fDy1 * (fDx1 + fDx2) + 2 * fDy2 * (fDx3 + fDx4) +
                       (fDx1 + fDx3) * Sqrt(4 * fDz * fDz + Pow(fDy2 - fDy1 - 2 * fDz * fTthetaSphi, 2.0)) +
                       (fDx2 + fDx4) * Sqrt(4 * fDz * fDz + Pow(fDy2 - fDy1 + 2 * fDz * fTthetaSphi, 2.0)) +
                       0.5 * (babc + dcda + efeh + ghgf);

  return surfArea;
}
*/
Precision UnplacedTrapezoid::SurfaceArea() const
{
  const TrapezoidStruct<double> &t = fTrap;
  Vec3D ba(t.fDx1 - t.fDx2 + t.fTanAlpha1 * 2 * t.fDy1, 2 * t.fDy1, 0);
  Vec3D bc(2 * t.fDz * t.fTthetaCphi - (t.fDx4 - t.fDx2) + t.fTanAlpha2 * t.fDy2 - t.fTanAlpha1 * t.fDy1,
           2 * t.fDz * t.fTthetaSphi + t.fDy2 - t.fDy1, 2 * t.fDz);
  Vec3D dc(-t.fDx4 + t.fDx3 + 2 * t.fTanAlpha2 * t.fDy2, 2 * t.fDy2, 0);
  Vec3D da(-2 * t.fDz * t.fTthetaCphi - (t.fDx1 - t.fDx3) - t.fTanAlpha1 * t.fDy1 + t.fTanAlpha2 * t.fDy2,
           -2 * t.fDz * t.fTthetaSphi - t.fDy1 + t.fDy2, -2 * t.fDz);

  Vec3D ef(t.fDx2 - t.fDx1 + 2 * t.fTanAlpha1 * t.fDy1, 2 * t.fDy1, 0);
  Vec3D eh(2 * t.fDz * t.fTthetaCphi + t.fDx3 - t.fDx1 + t.fTanAlpha1 * t.fDy1 - t.fTanAlpha2 * t.fDy2,
           2 * t.fDz * t.fTthetaSphi - t.fDy2 + t.fDy1, 2 * t.fDz);
  Vec3D gh(t.fDx3 - t.fDx4 - 2 * t.fTanAlpha2 * t.fDy2, -2 * t.fDy2, 0);
  Vec3D gf(-2 * t.fDz * t.fTthetaCphi + t.fDx2 - t.fDx4 + t.fTanAlpha1 * t.fDy1 - t.fTanAlpha2 * t.fDy2,
           -2 * t.fDz * t.fTthetaSphi + t.fDy1 - t.fDy2, -2 * t.fDz);

  Vec3D cr;
  cr             = ba.Cross(bc);
  Precision babc = cr.Mag();
  cr             = dc.Cross(da);
  Precision dcda = cr.Mag();
  cr             = ef.Cross(eh);
  Precision efeh = cr.Mag();
  cr             = gh.Cross(gf);
  Precision ghgf = cr.Mag();

  Precision surfArea =
      2 * t.fDy1 * (t.fDx1 + t.fDx2) + 2 * t.fDy2 * (t.fDx3 + t.fDx4) +
      (t.fDx1 + t.fDx3) * std::sqrt(4 * t.fDz * t.fDz + Pow(t.fDy2 - t.fDy1 - 2 * t.fDz * t.fTthetaSphi, 2.0)) +
      (t.fDx2 + t.fDx4) * std::sqrt(4 * t.fDz * t.fDz + Pow(t.fDy2 - t.fDy1 + 2 * t.fDz * t.fTthetaSphi, 2.0)) +
      0.5 * (babc + dcda + efeh + ghgf);

  return surfArea;
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrapezoid::Extent(Vec3D &aMin, Vec3D &aMax) const
{
  aMin.z() = -fTrap.fDz;
  aMax.z() = fTrap.fDz;

  TrapCorners pt;
  this->fromParametersToCorners(pt);

  Precision ext01 = Max(pt[0].x(), pt[1].x());
  Precision ext23 = Max(pt[2].x(), pt[3].x());
  Precision ext45 = Max(pt[4].x(), pt[5].x());
  Precision ext67 = Max(pt[6].x(), pt[7].x());
  Precision extA  = Max(ext01, ext23);
  Precision extB  = Max(ext45, ext67);
  aMax.x()        = Max(extA, extB);

  ext01    = Min(pt[0].x(), pt[1].x());
  ext23    = Min(pt[2].x(), pt[3].x());
  ext45    = Min(pt[4].x(), pt[5].x());
  ext67    = Min(pt[6].x(), pt[7].x());
  extA     = Min(ext01, ext23);
  extB     = Min(ext45, ext67);
  aMin.x() = Min(extA, extB);

  ext01    = Max(pt[0].y(), pt[1].y());
  ext23    = Max(pt[2].y(), pt[3].y());
  ext45    = Max(pt[4].y(), pt[5].y());
  ext67    = Max(pt[6].y(), pt[7].y());
  extA     = Max(ext01, ext23);
  extB     = Max(ext45, ext67);
  aMax.y() = Max(extA, extB);

  ext01    = Min(pt[0].y(), pt[1].y());
  ext23    = Min(pt[2].y(), pt[3].y());
  ext45    = Min(pt[4].y(), pt[5].y());
  ext67    = Min(pt[6].y(), pt[7].y());
  extA     = Min(ext01, ext23);
  extB     = Min(ext45, ext67);
  aMin.y() = Min(extA, extB);
}

Vector3D<Precision> UnplacedTrapezoid::GetPointOnSurface() const
{
  TrapCorners pt;
  this->fromParametersToCorners(pt);

  // make sure we provide the points in a clockwise fashion
  Precision chose = RNG::Instance().uniform() * SurfaceArea();

  Precision sumArea = 0.0;
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[0])) {
    return GetPointOnPlane(pt[0], pt[1], pt[5], pt[4]);
  }

  sumArea += fTrap.sideAreas[0];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[1])) {
    return GetPointOnPlane(pt[2], pt[6], pt[7], pt[3]);
  }

  sumArea += fTrap.sideAreas[1];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[2])) {
    return GetPointOnPlane(pt[0], pt[4], pt[6], pt[2]);
  }

  sumArea += fTrap.sideAreas[2];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[3])) {
    return GetPointOnPlane(pt[1], pt[3], pt[7], pt[5]);
  }

  sumArea += fTrap.sideAreas[3];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[4])) {
    return GetPointOnPlane(pt[0], pt[1], pt[3], pt[2]);
  }

  sumArea += fTrap.sideAreas[4];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[5])) {
    return GetPointOnPlane(pt[4], pt[6], pt[7], pt[5]);
  }

  // should never get here...
  return Vec3D(0., 0., 0.);
}

Vec3D UnplacedTrapezoid::GetPointOnPlane(Vec3D const &p0, Vec3D const &p1, Vec3D const &p2, Vec3D const &p3) const
{
  Precision lambda1, lambda2, chose, aOne, aTwo;
  Vec3D t, u, v, w, Area, normal;

  t = p1 - p0;
  u = p2 - p1;
  v = p3 - p2;
  w = p0 - p3;

  Area = Vec3D(w.y() * v.z() - w.z() * v.y(), w.z() * v.x() - w.x() * v.z(), w.x() * v.y() - w.y() * v.x());

  aOne = 0.5 * Area.Mag();

  Area = Vec3D(t.y() * u.z() - t.z() * u.y(), t.z() * u.x() - t.x() * u.z(), t.x() * u.y() - t.y() * u.x());

  aTwo = 0.5 * Area.Mag();

  chose = RNG::Instance().uniform(0., aOne + aTwo);

  if ((chose >= 0.) && (chose < aOne)) {
    lambda1 = RNG::Instance().uniform(0., 1.);
    lambda2 = RNG::Instance().uniform(0., lambda1);
    return (p2 + lambda1 * v + lambda2 * w);
  }

  // else

  lambda1 = RNG::Instance().uniform(0., 1.);
  lambda2 = RNG::Instance().uniform(0., lambda1);
  return (p0 + lambda1 * t + lambda2 * u);
}

#if defined(VECGEOM_USOLIDS)
/*
VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrapezoid::GetParametersList(int, double *aArray) const
{
  aArray[0] = GetRadius();
}
*/

VECGEOM_CUDA_HEADER_BOTH
UnplacedTrapezoid *UnplacedTrapezoid::Clone() const
{
  return new UnplacedTrapezoid(*this);
}
#endif

void UnplacedTrapezoid::Print() const
{
  // Note: units printed out chosen such that same numbers can be used as arguments to full constructor
  printf("UnplacedTrapezoid {%.3fmm, %.3frad, %.3frad, %.3fmm, %.3fmm, %.3fmm, %.3frad, %.3fmm, %.3fmm, %.3fmm, "
         "%.3frad}\n",
         fTrap.fDz, fTrap.fTheta, fTrap.fPhi, fTrap.fDy1, fTrap.fDx1, fTrap.fDx2, this->alpha1(), fTrap.fDy2,
         fTrap.fDx3, fTrap.fDx4, this->alpha2());
}

void UnplacedTrapezoid::Print(std::ostream &os) const
{
  // Note: units printed out chosen such that same numbers can be used as arguments to full constructor
  os << "UnplacedTrapezoid { " << fTrap.fDz << "mm, " << fTrap.fTheta << "rad, " << fTrap.fPhi << "rad, " << fTrap.fDy1
     << "mm, " << fTrap.fDx1 << "mm, " << fTrap.fDx2 << "mm, " << this->alpha1() << "rad, " << fTrap.fDy2 << "mm, "
     << fTrap.fDx3 << "mm, " << fTrap.fDx4 << "mm, " << this->alpha2() << "rad }\n";
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrapezoid::fromParametersToCorners(TrapCorners pt) const
{
  const TrapezoidStruct<double> &t = fTrap;

  // hopefully the compiler will optimize the repeated multiplications ... to be checked!
  double dxdyDy1 = t.fTanAlpha1 * t.fDy1;
  double dxdyDy2 = t.fTanAlpha2 * t.fDy2;
  double dxdzDz  = t.fTthetaCphi * t.fDz;
  double dydzDz  = t.fTthetaSphi * t.fDz;

  pt[0] = Vec3D(-dxdzDz - dxdyDy1 - t.fDx1, -dydzDz - t.fDy1, -t.fDz);
  pt[1] = Vec3D(-dxdzDz - dxdyDy1 + t.fDx1, -dydzDz - t.fDy1, -t.fDz);
  pt[2] = Vec3D(-dxdzDz + dxdyDy1 - t.fDx2, -dydzDz + t.fDy1, -t.fDz);
  pt[3] = Vec3D(-dxdzDz + dxdyDy1 + t.fDx2, -dydzDz + t.fDy1, -t.fDz);
  pt[4] = Vec3D(+dxdzDz - dxdyDy2 - t.fDx3, +dydzDz - t.fDy2, +t.fDz);
  pt[5] = Vec3D(+dxdzDz - dxdyDy2 + t.fDx3, +dydzDz - t.fDy2, +t.fDz);
  pt[6] = Vec3D(+dxdzDz + dxdyDy2 - t.fDx4, +dydzDz + t.fDy2, +t.fDz);
  pt[7] = Vec3D(+dxdzDz + dxdyDy2 + t.fDx4, +dydzDz + t.fDy2, +t.fDz);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrapezoid::fromCornersToParameters(TrapCorners const pt)
{
  fTrap.fDz         = pt[7].z();
  Precision DzRecip = 1.0 / fTrap.fDz;

  fTrap.fDy1       = 0.50 * (pt[2].y() - pt[0].y());
  fTrap.fDx1       = 0.50 * (pt[1].x() - pt[0].x());
  fTrap.fDx2       = 0.50 * (pt[3].x() - pt[2].x());
  fTrap.fTanAlpha1 = 0.25 * (pt[2].x() + pt[3].x() - pt[1].x() - pt[0].x()) / fTrap.fDy1;

  fTrap.fDy2       = 0.50 * (pt[6].y() - pt[4].y());
  fTrap.fDx3       = 0.50 * (pt[5].x() - pt[4].x());
  fTrap.fDx4       = 0.50 * (pt[7].x() - pt[6].x());
  fTrap.fTanAlpha2 = 0.25 * (pt[6].x() + pt[7].x() - pt[5].x() - pt[4].x()) / fTrap.fDy2;

  fTrap.fTthetaCphi = (pt[4].x() + fTrap.fDy2 * fTrap.fTanAlpha2 + fTrap.fDx3) * DzRecip;
  fTrap.fTthetaSphi = (pt[4].y() + fTrap.fDy2) * DzRecip;

  fTrap.fTheta = atan(sqrt(fTrap.fTthetaSphi * fTrap.fTthetaSphi + fTrap.fTthetaCphi * fTrap.fTthetaCphi));
  fTrap.fPhi   = atan2(fTrap.fTthetaSphi, fTrap.fTthetaCphi);
}

//////////////////////////////////////////////////////////////////////////////
//
// Calculate the coefficients of the plane p1->p2->p3->p4->p1
// where the ThreeVectors 1-4 are in clockwise order when viewed from
// "inside" of the plane (i.e. opposite to normal vector, which points outwards).
//
// Return true if the ThreeVectors are coplanar + set coefficients
//        false if ThreeVectors are not coplanar
//
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedTrapezoid::MakeAPlane(const Vec3D &p1, const Vec3D &p2, const Vec3D &p3, const Vec3D &p4,
#ifndef VECGEOM_PLANESHELL_DISABLE
                                   unsigned int iplane)
#else
                                   TrapSidePlane &plane)
#endif
{
  bool good;
  Precision a, b, c, norm;
  Vec3D v12, v13, v14, Vcross;

  v12    = p2 - p1;
  v13    = p3 - p1;
  v14    = p4 - p1;
  Vcross = v12.Cross(v13);

  // check coplanarity
  // if (std::fabs( v14.Dot(Vcross)/(Vcross.Mag()*v14.Mag()) ) > kTolerance)  {
  if (std::fabs(v14.Dot(Vcross) / (Vcross.Mag() * v14.Mag())) > 1.0e-7) {
    printf("*** ERROR (UnplacedTrapezoid): coplanarity test failure by %e.\n",
           std::fabs(v14.Dot(Vcross) / (Vcross.Mag() * v14.Mag())));
    printf("\tcorner 1: (%f; %f; %f)\n", p1.x(), p1.y(), p1.z());
    printf("\tcorner 2: (%f; %f; %f)\n", p2.x(), p2.y(), p2.z());
    printf("\tcorner 3: (%f; %f; %f)\n", p3.x(), p3.y(), p3.z());
    printf("\tcorner 4: (%f; %f; %f)\n", p4.x(), p4.y(), p4.z());
    good = false;
  }

  // cms.gdml does contain some bad trap corners... go ahead and try to build them anyway
  //  else {

  // a,b,c correspond to the x/y/z components of the
  // normal vector to the plane

  // Let create diagonals 3-1 and 4-2 than (3-1)x(4-2) provides
  // vector perpendicular to the plane directed to outside !!!
  // and a,b,c, = f(1,2,3,4) external relative to trapezoid normal

  //??? can these be optimized?
  a = +(p3.y() - p1.y()) * (p4.z() - p2.z()) - (p4.y() - p2.y()) * (p3.z() - p1.z());

  b = -(p3.x() - p1.x()) * (p4.z() - p2.z()) + (p4.x() - p2.x()) * (p3.z() - p1.z());

  c = +(p3.x() - p1.x()) * (p4.y() - p2.y()) - (p4.x() - p2.x()) * (p3.y() - p1.y());

  norm = 1.0 / std::sqrt(a * a + b * b + c * c); // normalization factor, always positive

#ifndef VECGEOM_PLANESHELL_DISABLE
  a *= norm;
  b *= norm;
  c *= norm;

  // Calculate fD: p1 is in plane so fD = -n.p1.Vect()
  Precision d = -(a * p1.x() + b * p1.y() + c * p1.z());

  fTrap.fPlanes.Set(iplane, a, b, c, d);
#else
  plane.fA = a * norm;
  plane.fB = b * norm;
  plane.fC = c * norm;

  // Calculate fD: p1 is in plane so fD = -n.p1.Vect()
  plane.fD = -(plane.fA * p1.x() + plane.fB * p1.y() + plane.fC * p1.z());

  unsigned int iplane = (&plane - fTrap.fPlanes); // pointer arithmetics used here
#endif

  fTrap.sideAreas[iplane] = 0.5 * (Vcross.Mag() + v13.Cross(v14).Mag());
  fTrap.normals[iplane]   = (Vcross + v13.Cross(v14)).Normalized();

  // well, at least for now, always return TRUE even though points are not coplanar!!!
  good = true;
  return good;
}

VECGEOM_CUDA_HEADER_BOTH
bool UnplacedTrapezoid::MakePlanes()
{
  TrapCorners pt;
  fromParametersToCorners(pt);
  return MakePlanes(pt);
}

VECGEOM_CUDA_HEADER_BOTH
bool UnplacedTrapezoid::MakePlanes(TrapCorners const pt)
{

  // Checking coplanarity of all four side faces
  bool good = true;

// Bottom side with normal approx. -Y
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakeAPlane(pt[0], pt[1], pt[5], pt[4], 0);
#else
  good                = MakeAPlane(pt[0], pt[1], pt[5], pt[4], fTrap.fPlanes[0]);
#endif
  if (!good) printf("***** GeomSolids0002 - Face at ~-Y not planar for Solid: UnplacedTrapezoid\n");

// Top side with normal approx. +Y
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakeAPlane(pt[2], pt[6], pt[7], pt[3], 1);
#else
  good = MakeAPlane(pt[2], pt[6], pt[7], pt[3], fTrap.fPlanes[1]);
#endif
  if (!good) printf("***** GeomSolids0002 - Face at ~+Y not planar for Solid: UnplacedTrapezoid\n");

// Front side with normal approx. -X
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakeAPlane(pt[0], pt[4], pt[6], pt[2], 2);
#else
  good = MakeAPlane(pt[0], pt[4], pt[6], pt[2], fTrap.fPlanes[2]);
#endif
  if (!good) printf("***** GeomSolids0002 - Face at ~-X not planar for Solid: UnplacedTrapezoid\n");

// Back side with normal approx. +X
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakeAPlane(pt[1], pt[3], pt[7], pt[5], 3);
#else
  good = MakeAPlane(pt[1], pt[3], pt[7], pt[5], fTrap.fPlanes[3]);
#endif
  if (!good) printf("***** GeomSolids0002 - Face at ~+X not planar for Solid: UnplacedTrapezoid\n");

  // include areas for -Z,+Z surfaces
  fTrap.sideAreas[4] = 2 * (fTrap.fDx1 + fTrap.fDx2) * fTrap.fDy1;
  fTrap.sideAreas[5] = 2 * (fTrap.fDx3 + fTrap.fDx4) * fTrap.fDy2;
  fTrap.normals[4]   = Vec3D(0, 0, -1);
  fTrap.normals[5]   = Vec3D(0, 0, 1);

  return good;
}

//===================== specialization stuff
#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedTrapezoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTrapezoid<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedTrapezoid<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedTrapezoid::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation,
                                                    const TranslationCode trans_code, const RotationCode rot_code,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTrapezoid>(volume, transformation, trans_code, rot_code,
                                                                  placement);
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume *UnplacedTrapezoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, const int id,
                                         VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTrapezoid<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedTrapezoid<trans_code, rot_code>(logical_volume, transformation, id);
}

__device__ VPlacedVolume *UnplacedTrapezoid::SpecializedVolume(LogicalVolume const *const volume,
                                                               Transformation3D const *const transformation,
                                                               const TranslationCode trans_code,
                                                               const RotationCode rot_code, const int id,
                                                               VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTrapezoid>(volume, transformation, trans_code, rot_code, id,
                                                                  placement);
}

#endif

//========== CUDA stuff
#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrapezoid::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedTrapezoid>(in_gpu_ptr, fTrap.fDz, fTrap.fTheta, fTrap.fPhi, fTrap.fDy1, fTrap.fDx1,
                                          fTrap.fDx2, this->alpha1(), fTrap.fDy2, fTrap.fDx3, fTrap.fDx4,
                                          this->alpha2());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrapezoid::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTrapezoid>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {
template size_t DevicePtr<cuda::UnplacedTrapezoid>::SizeOf();

template void DevicePtr<cuda::UnplacedTrapezoid>::Construct(const Precision dz, const Precision theta,
                                                            const Precision phi, const Precision dy1,
                                                            const Precision dx1, const Precision dx2,
                                                            const Precision tanAlpha1, const Precision dy2,
                                                            const Precision dx3, const Precision dx4,
                                                            const Precision tanAlpha2) const;

} // End cxx namespace

#endif

} // End global namespace
