//
// File:    TestTet.cpp
// Purpose: unit tests for the Tet
// author Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include <assert.h>
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Tet.h"
#include "ApproxEqual.h"

#include <cmath>

bool testvecgeom = false;

template <class Tet_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestTet()
{
  // int verbose = 0;

  Vec_t p0(0., 0., 2.), p1(0., 0., 0.), p2(2., 0., 0.), p3(0., 2., 0.);
  Tet_t tet("TestTet", p0, p1, p2, p3);

  // Check surfce area and volume
  //
  std::cout << "=== Check Getters, SurfaceArea(), Capacity(), Extent()" << std::endl;

  Vec_t v0, v1, v2, v3;
  tet.GetVertices(v0, v1, v2, v3);
  assert(v0 == p0);
  assert(v1 == p1);
  assert(v2 == p2);
  assert(v3 == p3);

  double sqrt2 = std::sqrt(2.), sqrt3 = std::sqrt(3.);
  double sx = 2., sy = 2., sz = 2., sxyz = 2. * sqrt3;
  double area = tet.SurfaceArea();
  std::cout << "Area : " << area << std::endl;
  assert(area == sx + sy + sz + sxyz);

  double vol = tet.Capacity();
  std::cout << "Volume : " << vol << std::endl;
  assert(vol == 4. / 3.);

  Vec_t bmin, bmax;
  tet.Extent(bmin, bmax);
  std::cout << "Extent : " << bmin << ", " << bmax << std::endl;
  assert(bmin == Vec_t(0., 0., 0.));
  assert(bmax == Vec_t(2., 2., 2.));

  // Check Inside()
  //
  std::cout << "=== Check Inside()" << std::endl;
  double kin  = 0.999;
  double kout = 1.001;

  Vec_t pc = (p0 + p1 + p2 + p3) / 4.; // center of tetrahedron

  Vec_t pf0 = (p0 + p1 + p2) / 3.; // centers of faces
  Vec_t pf1 = (p1 + p2 + p3) / 3.;
  Vec_t pf2 = (p2 + p3 + p0) / 3.;
  Vec_t pf3 = (p3 + p0 + p1) / 3.;

  Vec_t pe01 = (p0 + p1) / 2.; // centers of edges
  Vec_t pe02 = (p0 + p2) / 2.;
  Vec_t pe03 = (p0 + p3) / 2.;
  Vec_t pe12 = (p1 + p2) / 2.;
  Vec_t pe13 = (p1 + p3) / 2.;
  Vec_t pe23 = (p2 + p3) / 2.;

  assert(tet.Inside(p0) == vecgeom::kSurface);
  assert(tet.Inside(p1) == vecgeom::kSurface);
  assert(tet.Inside(p2) == vecgeom::kSurface);
  assert(tet.Inside(p3) == vecgeom::kSurface);

  assert(tet.Inside(pf0) == vecgeom::kSurface);
  assert(tet.Inside(pf1) == vecgeom::kSurface);
  assert(tet.Inside(pf2) == vecgeom::kSurface);
  assert(tet.Inside(pf3) == vecgeom::kSurface);

  assert(tet.Inside(pe01) == vecgeom::kSurface);
  assert(tet.Inside(pe02) == vecgeom::kSurface);
  assert(tet.Inside(pe03) == vecgeom::kSurface);
  assert(tet.Inside(pe12) == vecgeom::kSurface);
  assert(tet.Inside(pe13) == vecgeom::kSurface);
  assert(tet.Inside(pe23) == vecgeom::kSurface);

  assert(tet.Inside(pc) == vecgeom::kInside);
  assert(tet.Inside(pc + (p0 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (p0 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (p1 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (p2 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (p3 - pc) * kin) == vecgeom::kInside);

  assert(tet.Inside(pc + (pf0 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pf1 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pf2 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pf3 - pc) * kin) == vecgeom::kInside);

  assert(tet.Inside(pc + (pe01 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pe02 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pe03 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pe12 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pe13 - pc) * kin) == vecgeom::kInside);
  assert(tet.Inside(pc + (pe23 - pc) * kin) == vecgeom::kInside);

  assert(tet.Inside(pc + (p0 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (p0 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (p1 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (p2 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (p3 - pc) * kout) == vecgeom::kOutside);

  assert(tet.Inside(pc + (pf0 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pf1 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pf2 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pf3 - pc) * kout) == vecgeom::kOutside);

  assert(tet.Inside(pc + (pe01 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pe02 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pe03 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pe12 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pe13 - pc) * kout) == vecgeom::kOutside);
  assert(tet.Inside(pc + (pe23 - pc) * kout) == vecgeom::kOutside);

  // Check Normal()
  //
  std::cout << "=== Check Normal()" << std::endl;
  Vec_t norm;
  bool validity = false;
  validity      = tet.Normal(pf0, norm);
  assert(norm == Vec_t(0., -1., 0.) && validity);
  validity = tet.Normal(pf1, norm);
  assert(norm == Vec_t(0., 0., -1.) && validity);
  validity = tet.Normal(pf2, norm);
  assert(norm == Vec_t(1., 1., 1.) / sqrt3 && validity);
  validity = tet.Normal(pf3, norm);
  assert(norm == Vec_t(-1., 0., 0.) && validity);

  validity = tet.Normal(pe01, norm);
  assert(norm == Vec_t(-1., -1., 0.) / sqrt2 && validity);
  validity = tet.Normal(pe12, norm);
  assert(norm == Vec_t(0., -1., -1.) / sqrt2 && validity);
  validity = tet.Normal(pe13, norm);
  assert(norm == Vec_t(-1., 0., -1.) / sqrt2 && validity);
  validity = tet.Normal(p1, norm);
  assert(norm == Vec_t(-1., -1., -1.) / sqrt3 && validity);

  // Check ApproxSurfaceNormal()
  //
  std::cout << "=== Check ApproxSurfaceNormal()" << std::endl;
  validity = true;
  validity = tet.Normal(pf0 + Vec_t(0., -1., 0.), norm);
  assert(norm == Vec_t(0., -1., 0.) && !validity);
  validity = tet.Normal(pf1 + Vec_t(0., 0., -1.), norm);
  assert(norm == Vec_t(0., 0., -1.) && !validity);
  validity = tet.Normal(pf2 + Vec_t(1., 1., 1.), norm);
  assert(norm == Vec_t(1., 1., 1.) / sqrt3 && !validity);
  validity = tet.Normal(pf3 + Vec_t(-1., 0., 0.), norm);
  assert(norm == Vec_t(-1., 0., 0.) && !validity);

  validity = tet.Normal(pf0 + Vec_t(.0, .1, .0), norm);
  assert(norm == Vec_t(0., -1., 0.) && !validity);
  validity = tet.Normal(pf1 + Vec_t(.0, .0, .1), norm);
  assert(norm == Vec_t(0., 0., -1.) && !validity);
  validity = tet.Normal(pf2 + Vec_t(-.1, -.1, -.1), norm);
  assert(norm == Vec_t(1., 1., 1.) / sqrt3 && !validity);
  validity = tet.Normal(pf3 + Vec_t(.1, .0, .0), norm);
  assert(norm == Vec_t(-1., 0., 0.) && !validity);

  // Check SafetyToIn()
  //
  std::cout << "=== Check SafetyToIn()" << std::endl;
  assert(ApproxEqual(tet.SafetyToIn(pf0 + Vec_t(0., -1., 0.)), 1.));
  assert(ApproxEqual(tet.SafetyToIn(pf1 + Vec_t(0., 0., -1.)), 1.));
  assert(ApproxEqual(tet.SafetyToIn(pf2 + Vec_t(1., 1., 1.) / sqrt3), 1.));
  assert(ApproxEqual(tet.SafetyToIn(pf3 + Vec_t(-1., 0., 0.)), 1.));

  assert(tet.SafetyToIn(pf0) == 0.);
  assert(tet.SafetyToIn(pf1) == 0.);
  assert(tet.SafetyToIn(pf2) == 0.);
  assert(tet.SafetyToIn(pf3) == 0.);

  assert(ApproxEqual(tet.SafetyToIn(pf0 + Vec_t(.0, .1, .0)), -0.1));
  assert(ApproxEqual(tet.SafetyToIn(pf1 + Vec_t(.0, .0, .1)), -0.1));
  assert(ApproxEqual(tet.SafetyToIn(pf2 + Vec_t(-.1, -.1, -.1) / sqrt3), -0.1));
  assert(ApproxEqual(tet.SafetyToIn(pf3 + Vec_t(.1, .0, .0)), -0.1));

  // Check SafetyToOut()
  //
  std::cout << "=== Check SafetyToOut()" << std::endl;
  assert(ApproxEqual(tet.SafetyToOut(pf0 + Vec_t(0., -1., 0.)), -1.));
  assert(ApproxEqual(tet.SafetyToOut(pf1 + Vec_t(0., 0., -1.)), -1.));
  assert(ApproxEqual(tet.SafetyToOut(pf2 + Vec_t(1., 1., 1.) / sqrt3), -1.));
  assert(ApproxEqual(tet.SafetyToOut(pf3 + Vec_t(-1., 0., 0.)), -1.));

  assert(tet.SafetyToOut(pf0) == 0.);
  assert(tet.SafetyToOut(pf1) == 0.);
  assert(tet.SafetyToOut(pf2) == 0.);
  assert(tet.SafetyToOut(pf3) == 0.);

  assert(ApproxEqual(tet.SafetyToOut(pf0 + Vec_t(.0, .1, .0)), 0.1));
  assert(ApproxEqual(tet.SafetyToOut(pf1 + Vec_t(.0, .0, .1)), 0.1));
  assert(ApproxEqual(tet.SafetyToOut(pf2 + Vec_t(-.1, -.1, -.1) / sqrt3), 0.1));
  assert(ApproxEqual(tet.SafetyToOut(pf3 + Vec_t(.1, .0, .0)), 0.1));

  // Check DistanceToIn()
  //
  std::cout << "=== Check DistanceToIn()" << std::endl;
  Vec_t pnt(0.5, 0.5, -0.5), dir(0., 0., 1.);
  assert(tet.DistanceToIn(pnt, dir) == 0.5);
  assert(tet.DistanceToIn(pnt, -dir) == vecgeom::kInfLength);

  pnt = Vec_t(0.5, 0.5, -0.5 * vecgeom::kHalfTolerance);
  assert(tet.DistanceToIn(pnt, dir) == 0.5 * vecgeom::kHalfTolerance);
  assert(tet.DistanceToIn(pnt, -dir) == vecgeom::kInfLength);

  pnt = Vec_t(0.5, 0.5, 0.0);
  assert(tet.DistanceToIn(pnt, dir) == 0.0);
  assert(tet.DistanceToIn(pnt, -dir) == vecgeom::kInfLength);

  pnt = Vec_t(0.5, 0.5, 0.5 * vecgeom::kHalfTolerance);
  assert(tet.DistanceToIn(pnt, dir) == -0.5 * vecgeom::kHalfTolerance);
  assert(tet.DistanceToIn(pnt, -dir) == vecgeom::kInfLength);

  pnt = Vec_t(0.5, 0.5, 0.5);
  assert(tet.DistanceToIn(pnt, dir) == -0.5);
  assert(ApproxEqual(tet.DistanceToIn(pnt, -dir), -0.5));

  Vec_t pntIn, pntTolIn, pntOut, pntTolOut;
  pnt = Vec_t(0.4, 0.4, 0.0);
  pntIn.Set(pnt.x(), pnt.y(), 0.5 * vecgeom::kHalfTolerance);
  pntTolIn.Set(pnt.x(), pnt.y(), vecgeom::kHalfTolerance);
  pntOut.Set(pnt.x(), pnt.y(), -0.5 * vecgeom::kHalfTolerance);
  pntTolOut.Set(pnt.x(), pnt.y(), -vecgeom::kHalfTolerance);

  Vec_t dirIn, dirOut;
  dir = Vec_t(1., 0., 0.);
  dir.Normalize();
  dirIn = Vec_t(0.89 - pnt.x(), 0.0, 0.5 * vecgeom::kHalfTolerance);
  dirIn.Normalize();
  dirOut = Vec_t(0.89 - pnt.x(), 0.0, -0.5 * vecgeom::kHalfTolerance);
  dirOut.Normalize();

  std::cout << std::setprecision(16) << std::endl;
  std::cout << "   distToIn(pntTolOut,dirOut) = " << tet.DistanceToIn(pntTolOut, dirOut) << std::endl;
  std::cout << "   distToIn(pntTolOut,dir) = " << tet.DistanceToIn(pntTolOut, dir) << std::endl;
  std::cout << "   distToIn(pntTolOut,dirIn) = " << tet.DistanceToIn(pntTolOut, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToIn(pntOut,dirOut) = " << tet.DistanceToIn(pntOut, dirOut) << std::endl;
  std::cout << "   distToIn(pntOut,dir) = " << tet.DistanceToIn(pntOut, dir) << std::endl;
  std::cout << "   distToIn(pntOut,dirIn) = " << tet.DistanceToIn(pntOut, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToIn(pnt,dirOut) = " << tet.DistanceToIn(pnt, dirOut) << std::endl;
  std::cout << "   distToIn(pnt,dir) = " << tet.DistanceToIn(pnt, dir) << std::endl;
  std::cout << "   distToIn(pnt,dirIn) = " << tet.DistanceToIn(pnt, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToIn(pntIn,dirOut) = " << tet.DistanceToIn(pntIn, dirOut) << std::endl;
  std::cout << "   distToIn(pntIn,dir) = " << tet.DistanceToIn(pntIn, dir) << std::endl;
  std::cout << "   distToIn(pntIn,dirIn) = " << tet.DistanceToIn(pntIn, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToIn(pntTolIn,dirOut) = " << tet.DistanceToIn(pntTolIn, dirOut) << std::endl;
  std::cout << "   distToIn(pntTolIn,dir) = " << tet.DistanceToIn(pntTolIn, dir) << std::endl;
  std::cout << "   distToIn(pntTolIn,dirIn) = " << tet.DistanceToIn(pntTolIn, dirIn) << std::endl;
  std::cout << std::endl;

  // Check DistanceToOut()
  //
  std::cout << "=== Check DistanceToOut()" << std::endl;
  pnt = Vec_t(0.5, 0.5, -0.5);
  dir = Vec_t(0., 0., 1.);
  assert(tet.DistanceToOut(pnt, dir) == -1);
  assert(tet.DistanceToOut(pnt, -dir) == -1);

  pnt = Vec_t(0.5, 0.5, -0.5 * vecgeom::kHalfTolerance);
  assert(ApproxEqual(tet.DistanceToOut(pnt, dir), 1.));
  assert(tet.DistanceToOut(pnt, -dir) == -0.5 * vecgeom::kHalfTolerance);

  pnt = Vec_t(0.5, 0.5, 0.0);
  assert(ApproxEqual(tet.DistanceToOut(pnt, dir), 1.));
  assert(tet.DistanceToOut(pnt, -dir) == 0.0);

  pnt = Vec_t(0.5, 0.5, 0.5 * vecgeom::kHalfTolerance);
  assert(ApproxEqual(tet.DistanceToOut(pnt, dir), 1.));
  assert(tet.DistanceToOut(pnt, -dir) == 0.5 * vecgeom::kHalfTolerance);

  pnt = Vec_t(0.5, 0.5, 0.5);
  assert(ApproxEqual(tet.DistanceToOut(pnt, dir), 0.5));
  assert(tet.DistanceToOut(pnt, -dir) == 0.5);

  pnt = Vec_t(0.4, 0.4, 0.0);
  pntIn.Set(pnt.x(), pnt.y(), 0.5 * vecgeom::kHalfTolerance);
  pntOut.Set(pnt.x(), pnt.y(), -0.5 * vecgeom::kHalfTolerance);

  dir = Vec_t(1., 0., 0.);
  dir.Normalize();
  dirIn = Vec_t(0.9 - pnt.x(), 0.0, 0.5 * vecgeom::kHalfTolerance);
  dirIn.Normalize();
  dirOut = Vec_t(0.9 - pnt.x(), 0.0, -0.5 * vecgeom::kHalfTolerance);
  dirOut.Normalize();

  std::cout << "   distToOut(pntTolOut,dirOut) = " << tet.DistanceToOut(pntTolOut, dirOut) << std::endl;
  std::cout << "   distToOut(pntTolOut,dir) = " << tet.DistanceToOut(pntTolOut, dir) << std::endl;
  std::cout << "   distToOut(pntTolOut,dirIn) = " << tet.DistanceToOut(pntTolOut, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToOut(pntOut,dirOut) = " << tet.DistanceToOut(pntOut, dirOut) << std::endl;
  std::cout << "   distToOut(pntOut,dir) = " << tet.DistanceToOut(pntOut, dir) << std::endl;
  std::cout << "   distToOut(pntOut,dirIn) = " << tet.DistanceToOut(pntOut, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToOut(pnt,dirOut) = " << tet.DistanceToOut(pnt, dirOut) << std::endl;
  std::cout << "   distToOut(pnt,dir) = " << tet.DistanceToOut(pnt, dir) << std::endl;
  std::cout << "   distToOut(pnt,dirIn) = " << tet.DistanceToOut(pnt, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToOut(pntIn,dirOut) = " << tet.DistanceToOut(pntIn, dirOut) << std::endl;
  std::cout << "   distToOut(pntIn,dir) = " << tet.DistanceToOut(pntIn, dir) << std::endl;
  std::cout << "   distToOut(pntIn,dirIn) = " << tet.DistanceToOut(pntIn, dirIn) << std::endl;
  std::cout << std::endl;

  std::cout << "   distToOut(pntTolIn,dirOut) = " << tet.DistanceToOut(pntTolIn, dirOut) << std::endl;
  std::cout << "   distToOut(pntTolIn,dir) = " << tet.DistanceToOut(pntTolIn, dir) << std::endl;
  std::cout << "   distToOut(pntTolIn,dirIn) = " << tet.DistanceToOut(pntTolIn, dirIn) << std::endl;
  std::cout << std::endl;

  // Check SamplePointOnSurface()
  //
  std::cout << "=== SamplePointOnSurface()" << std::endl;
  int nx = 0, ny = 0, nz = 0, nxyz = 0, nfactor = 10000, ntot = area * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t rndPoint = tet.GetUnplacedVolume()->SamplePointOnSurface();
    assert(tet.Inside(rndPoint) == vecgeom::EInside::kSurface);
    if (rndPoint.x() == 0.)
      ++nx;
    else if (rndPoint.y() == 0.)
      ++ny;
    else if (rndPoint.z() == 0.)
      ++nz;
    else
      ++nxyz;
  }
  std::cout << "sx,sy,sz,sxyz = " << sx << ", \t" << sy << ", \t" << sz << ", \t" << sxyz << std::endl;
  std::cout << "nx,ny,nz,nxyz = " << nx << ", \t" << ny << ", \t" << nz << ", \t" << nxyz << std::endl;
  assert(std::abs(nx - sx * nfactor) < 2. * std::sqrt(ntot));
  assert(std::abs(ny - sy * nfactor) < 2. * std::sqrt(ntot));
  assert(std::abs(nz - sz * nfactor) < 2. * std::sqrt(ntot));
  assert(std::abs(nxyz - sxyz * nfactor) < 2. * std::sqrt(ntot));

  return true;
}

int main(int argc, char *argv[])
{
  assert(TestTet<vecgeom::SimpleTet>());
  std::cout << "VecGeomTet passed\n";

  return 0;
}
