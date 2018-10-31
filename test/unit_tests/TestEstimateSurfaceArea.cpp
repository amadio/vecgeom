//
// File:    TestEstimateSurfaceArea.cpp
// Purpose: test and benchmark the algorithm in VUnplacedVolume::EstimateSurfaceArea()
// Author:  Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

// ensure asserts are compiled in
#undef NDEBUG

#include <cmath>
#include <ctime>
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Trd.h"
#include "volumes/Tube.h"
#include "volumes/Cone.h"
#include "volumes/ScaledShape.h"
#include "volumes/BooleanVolume.h"

using namespace vecgeom;
using Vec3D_t = Vector3D<Precision>;

double error(double real, double expected)
{
  double err = (expected == 0 && real != 0) ? 100 : 0;
  if (expected != 0) {
    err = (int(10000. * (real - expected) / expected)) / 100.;
  }
  return err;
};

/*
/////////////////////////////////////////////////////////////////////////////////
//
// Former algorithm for the surface area estimation. The algorithm does not
// work properly when Safety is underestimated, for example, in case of
// Scaled solids. It has been replaced with the six-point algorithm
// in October 2018 (Geant4 10.05)
//
double EstimateArea_Safety(const VPlacedVolume *solid, int nStat)
{
  double ell = -1.;
  Vec3D_t p;
  Vec3D_t minCorner;
  Vec3D_t maxCorner;
  Vec3D_t delta;

  // min max extents of pSolid along X,Y,Z
  solid->Extent(minCorner, maxCorner);

  // limits
  delta = maxCorner - minCorner;

  if (ell <= 0.) // Automatic definition of skin thickness
  {
    Precision minval = delta.x();
    if (delta.y() < delta.x()) {
      minval = delta.y();
    }
    if (delta.z() < minval) {
      minval = delta.z();
    }
    ell = .01 * minval;
  }

  Precision dd = 2 * ell;
  minCorner.x() -= ell;
  minCorner.y() -= ell;
  minCorner.z() -= ell;
  delta.x() += dd;
  delta.y() += dd;
  delta.z() += dd;

  int inside = 0;
  for (int i = 0; i < nStat; ++i) {
    p = minCorner + Vec3D_t(delta.x() * RNG::Instance().uniform(), delta.y() * RNG::Instance().uniform(),
                            delta.z() * RNG::Instance().uniform());
    if (solid->Contains(p)) {
      if (solid->SafetyToOut(p) < ell) {
        inside++;
      }
    } else {
      if (solid->SafetyToIn(p) < ell) {
        inside++;
      }
    }
  }
  // @@ The conformal correction can be upgraded
  return delta.x() * delta.y() * delta.z() * inside / dd / nStat;
}
*/

/////////////////////////////////////////////////////////////////////////////////
//
// Benchmark the surface area estimation algorithm(s)
//
void BenchEstimateArea(const VPlacedVolume *solid, double area)
{
  clock_t t;
  double a1e5 = 0;
  double a1e6 = 0;
  double a1e7 = 0;

  /*
  // Test EstimateArea_Safety()
  //
  std::cout << "\n=== Old algorithm based on Safety (replaced in Geant4 10.05)" << std::endl;
  t    = clock();
  a1e5 = EstimateArea_Safety(solid, 100000);
  a1e6 = EstimateArea_Safety(solid, 1000000);
  a1e7 = EstimateArea_Safety(solid, 10000000);
  std::cout << "N = 10^5   Estimated Area = " << a1e5 << " (" << error(a1e5, area) << "%) " << std::endl;
  std::cout << "N = 10^6   Estimated Area = " << a1e6 << " (" << error(a1e6, area) << "%) " << std::endl;
  std::cout << "N = 10^7   Estimated Area = " << a1e7 << " (" << error(a1e7, area) << "%) " << std::endl;
  t = clock() - t;
  std::cout << "   Time: " << (double)t / CLOCKS_PER_SEC << " sec" << std::endl;
  */

  // Test G4VSolid::EstimateSurfaceArea()
  //
  std::cout << "\n=== Current VUnplacedVolume::EstimateSurfaceArea()" << std::endl;
  t    = clock();
  a1e5 = (solid->GetUnplacedVolume())->EstimateSurfaceArea(100000);
  a1e6 = (solid->GetUnplacedVolume())->EstimateSurfaceArea(1000000);
  a1e7 = (solid->GetUnplacedVolume())->EstimateSurfaceArea(10000000);
  std::cout << "N = 10^5   Estimated Area = " << a1e5 << " (" << error(a1e5, area) << "%) " << std::endl;
  std::cout << "N = 10^6   Estimated Area = " << a1e6 << " (" << error(a1e6, area) << "%) " << std::endl;
  std::cout << "N = 10^7   Estimated Area = " << a1e7 << " (" << error(a1e7, area) << "%) " << std::endl;
  t = clock() - t;
  std::cout << "   Time: " << (double)t / CLOCKS_PER_SEC << " sec" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestScaledTrd()
{
  double dx1 = 200, dx2 = 100, dy1 = 200, dy2 = 100, dz = 800;
  VPlacedVolume *trd = new SimpleTrd("Trd", dx1, dx2, dy1, dy2, dz);

  double sx = 3, sy = 2, sz = 0.2;
  VPlacedVolume *solid = new SimpleScaledShape("Scaled Trd", trd, sx, sy, sz);

  dx1 *= sx;
  dx2 *= sx;
  dy1 *= sy;
  dy2 *= sy;
  dz *= sz;
  double area = 4 * (dx1 * dy1 + dx2 * dy2) + 2 * (dy1 + dy2) * std::hypot(dx1 - dx2, 2 * dz) +
                2 * (dx1 + dx2) * std::hypot(dy1 - dy2, 2 * dz);
  std::cout << "\n*********************************************************"
            << "\n*** Test Scaled Trd - Scale factors (" << sx << ", " << sy << ", " << sz << ")"
            << " *** Surface area : " << area << "\n***" << std::endl;
  std::cout << "Trd->SurfaceArea()    : " << trd->SurfaceArea() << std::endl;
  std::cout << "solid->SurfaceAarea() : " << solid->SurfaceArea() << std::endl;

  BenchEstimateArea(solid, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestScaledCone()
{
  double rminus = 50, rplus = 10, dz = 50;
  VPlacedVolume *cone = new SimpleCone("Cone", 0, rminus, 0, rplus, dz, 0, kTwoPi);

  double sx = 10, sy = 5, sz = 2;
  VPlacedVolume *solid = new SimpleScaledShape("Scaled Cone", cone, sx, sy, sz);

  double area = 887358;
  std::cout << "\n*********************************************************"
            << "\n*** Test Scaled Cone - Scale factors (" << sx << ", " << sy << ", " << sz << ")"
            << " *** Surface area : " << area << "\n***" << std::endl;
  std::cout << "Cone->SurfaceArea()   : " << cone->SurfaceArea() << std::endl;
  std::cout << "solid->SurfaceAarea() : " << solid->SurfaceArea() << std::endl;

  BenchEstimateArea(solid, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestUnionSolid()
{
  double rbig = 100, zbig = 100;
  GenericUnplacedTube big(0, rbig, zbig, 0, kTwoPi);
  LogicalVolume lbig("Big Tube", &big);
  auto pbig = lbig.Place();

  double rsml = 25, zsml = 170;
  GenericUnplacedTube small(0, rsml, zsml, 0, kTwoPi);
  LogicalVolume lsmall("Small Tube", &small);

  double offset = 60;
  auto psmall0  = lsmall.Place();
  auto psmall1  = lsmall.Place(new Transformation3D(offset, 0, 0));
  auto psmall2  = lsmall.Place(new Transformation3D(-offset, 0, 0));
  auto psmall3  = lsmall.Place(new Transformation3D(0, offset, 0));
  auto psmall4  = lsmall.Place(new Transformation3D(0, -offset, 0));

  UnplacedBooleanVolume<kUnion> union1(kUnion, pbig, psmall0);
  LogicalVolume lsolid1("Union_1", &union1);
  auto psolid1 = lsolid1.Place();

  UnplacedBooleanVolume<kUnion> union2(kUnion, psolid1, psmall1);
  LogicalVolume lsolid2("Union_2", &union2);
  auto psolid2 = lsolid2.Place();

  UnplacedBooleanVolume<kUnion> union3(kUnion, psolid2, psmall2);
  LogicalVolume lsolid3("Union_3", &union3);
  auto psolid3 = lsolid3.Place();

  UnplacedBooleanVolume<kUnion> union4(kUnion, psolid3, psmall3);
  LogicalVolume lsolid4("Union_4", &union4);
  auto psolid4 = lsolid4.Place();

  UnplacedBooleanVolume<kUnion> union5(kUnion, psolid4, psmall4);
  LogicalVolume lsolid5("Union_5", &union5);
  auto psolid5 = lsolid5.Place();

  double sbig = 2 * zbig * kTwoPi * rbig + 2 * kPi * rbig * rbig;
  double sadd = 2 * (zsml - zbig) * kTwoPi * rsml;
  double area = sbig + 5 * sadd;
  std::cout << "\n*********************************************************"
            << "\n*** Test Union Solid - Disk plus 5 cylinders *** Surface area : " << area << "\n***" << std::endl;
  std::cout << "Disk area         : " << sbig << std::endl;
  std::cout << "Increment area    : " << sadd << std::endl;
  std::cout << "Real surface area : " << area << std::endl;
  std::cout << "solid->SurfaceAarea() : " << psolid5->SurfaceArea() << std::endl;

  BenchEstimateArea(psolid5, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestSubtractionSolid()
{
  double rbig = 100, zbig = 100;
  GenericUnplacedTube big(0, rbig, zbig, 0, kTwoPi);
  LogicalVolume lbig("Big Tube", &big);
  auto pbig = lbig.Place();

  double rsml = 25, zsml = 170;
  GenericUnplacedTube small(0, rsml, zsml, 0, kTwoPi);
  LogicalVolume lsmall("Small Tube", &small);

  double offset = 60;
  auto psmall0  = lsmall.Place();
  auto psmall1  = lsmall.Place(new Transformation3D(offset, 0, 0));
  auto psmall2  = lsmall.Place(new Transformation3D(-offset, 0, 0));
  auto psmall3  = lsmall.Place(new Transformation3D(0, offset, 0));
  auto psmall4  = lsmall.Place(new Transformation3D(0, -offset, 0));

  UnplacedBooleanVolume<kSubtraction> union1(kSubtraction, pbig, psmall0);
  LogicalVolume lsolid1("Subtraction_1", &union1);
  auto psolid1 = lsolid1.Place();

  UnplacedBooleanVolume<kSubtraction> union2(kSubtraction, psolid1, psmall1);
  LogicalVolume lsolid2("Subtraction_2", &union2);
  auto psolid2 = lsolid2.Place();

  UnplacedBooleanVolume<kSubtraction> union3(kSubtraction, psolid2, psmall2);
  LogicalVolume lsolid3("Subtraction_3", &union3);
  auto psolid3 = lsolid3.Place();

  UnplacedBooleanVolume<kSubtraction> union4(kSubtraction, psolid3, psmall3);
  LogicalVolume lsolid4("Subtraction_4", &union4);
  auto psolid4 = lsolid4.Place();

  UnplacedBooleanVolume<kSubtraction> union5(kSubtraction, psolid4, psmall4);
  LogicalVolume lsolid5("Subtraction_5", &union5);
  auto psolid5 = lsolid5.Place();

  double sbig  = 2 * zbig * kTwoPi * rbig + 2 * kPi * rbig * rbig;
  double shole = 2 * zbig * kTwoPi * rsml - 2 * kPi * rsml * rsml;
  double area  = sbig + 5 * shole;
  std::cout << "\n*********************************************************"
            << "\n*** Test Subtraction Solid - Disk with 5 holes *** Surface area : " << area << "\n***" << std::endl;
  std::cout << "Disk area         : " << sbig << std::endl;
  std::cout << "Increment area    : " << shole << std::endl;
  std::cout << "Real surface area : " << area << std::endl;
  std::cout << "solid->SurfaceAarea() : " << psolid5->SurfaceArea() << std::endl;

  BenchEstimateArea(psolid5, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestIntersectionSolid()
{
  double dx = 100, dy1 = 50, dy2 = 80, dz = 500;
  UnplacedBox box1(dx, dy1, dz);
  LogicalVolume lbox1("box1", &box1);
  auto pbox1 = lbox1.Place();

  UnplacedBox box2(dx, dy2, dz);
  LogicalVolume lbox2("box2", &box2);
  double ang = 60;
  auto pbox2 = lbox2.Place(new Transformation3D(0, 0, 0, 0, ang, 0));

  UnplacedBooleanVolume<kIntersection> solid(kIntersection, pbox1, pbox2);
  LogicalVolume lsolid("Intersection", &solid);
  auto psolid = lsolid.Place();

  double deg  = kPi / 180;
  double side = 2 * dx / Sin(ang * deg);
  double area = 4 * dx * side + 8 * side * Min(dy1, dy2);
  std::cout << "\n*********************************************************"
            << "\n*** Test Intersection Solid - two intersecting bars *** Surface area : " << area << "\n***"
            << std::endl;
  std::cout << "solid->SurfaceAarea() : " << psolid->SurfaceArea() << std::endl;

  BenchEstimateArea(psolid, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
void TestSubtractionNull()
{
  double dbig = 100;
  UnplacedBox big(dbig, dbig, dbig);
  LogicalVolume lbig("Big Box", &big);
  auto pbig = lbig.Place();

  double dsml = 10;
  UnplacedBox sml(dsml, dsml, dsml);
  LogicalVolume lsml("Sml Box", &sml);
  auto psml = lsml.Place();

  UnplacedBooleanVolume<kSubtraction> solid(kSubtraction, psml, pbig);
  LogicalVolume lsolid("Null", &solid);
  auto psolid = lsolid.Place();

  double area = 0;
  std::cout << "\n*********************************************************"
            << "\n*** Test Subtraction Solid - Null object *** Surface area: " << area << "\n***" << std::endl;
  std::cout << "Big box area      : " << pbig->SurfaceArea() << std::endl;
  std::cout << "Small box area    : " << psml->SurfaceArea() << std::endl;
  std::cout << "solid->SurfaceAarea() : " << psolid->SurfaceArea() << std::endl;

  BenchEstimateArea(psolid, area);
}

/////////////////////////////////////////////////////////////////////////////////
//
int main()
{
  TestScaledTrd();
  TestScaledCone();
  TestUnionSolid();
  TestSubtractionSolid();
  TestIntersectionSolid();
  TestSubtractionNull();
  return 0;
}
