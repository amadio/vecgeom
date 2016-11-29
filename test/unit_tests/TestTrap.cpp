//
// File:    TestTrap.cpp
// Purpose: Unit tests for the trapezoid
//

//.. ensure asserts are compiled in
#undef NDEBUG

#include "base/Vector3D.h"
#include "VecCore/VecMath.h"
#include "volumes/Box.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTrap.hh"
#include "UVector3.hh"
#endif

#include "volumes/Trapezoid.h"

bool testvecgeom = false;
double degToRad  = vecCore::math::ATan(1.0) / 45.;

template <typename Constants, class Trap_t>
bool TestTrap()
{
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(20, 0, 0), ponyside(0, 30, 0), ponzside(0, 0, 40);
  Vec_t ponmxside(-20, 0, 0), ponmyside(0, -30, 0), ponmzside(0, 0, -40);
  Vec_t ponzsidey(0, 25, 40), ponmzsidey(0, 20, -40);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100), pbig(100, 100, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Vec_t vxmz(1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));
  Vec_t vymz(0, 1 / std::sqrt(2.0), -1 / std::sqrt(2.0));
  Vec_t vmxmz(-1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));
  Vec_t vmymz(0, -1 / std::sqrt(2.0), -1 / std::sqrt(2.0));
  Vec_t vxz(1 / std::sqrt(2.0), 0, 1 / std::sqrt(2.0));
  Vec_t vyz(0, 1 / std::sqrt(2.0), 1 / std::sqrt(2.0));

  double Dist, dist, vol, volCheck;
  Vec_t normal, norm;
  bool valid, convex;

  double cosa = 4 / std::sqrt(17.), sina = 1 / std::sqrt(17.);

  Vec_t trapvert[8] = {Vec_t(-10.0, -20.0, -40.0), Vec_t(+10.0, -20.0, -40.0), Vec_t(-10.0, +20.0, -40.0),
                       Vec_t(+10.0, +20.0, -40.0), Vec_t(-30.0, -40.0, +40.0), Vec_t(+30.0, -40.0, +40.0),
                       Vec_t(-30.0, +40.0, +40.0), Vec_t(+30.0, +40.0, +40.0)};

  Trap_t trap1("Test Boxlike #1", 40, 0, 0, 30, 20, 20, 0, 30, 20, 20, 0); // box:20,30,40

  //    Trap_t trap2("Test Trdlike #2",40,0,0,20,10,10,0,40,30,30,0);

  Trap_t trap2("Test Trdlike #2", trapvert);

  Trap_t trap3("trap3", 50, 0, 0, 50, 50, 50, UUtils::kPi / 4, 50, 50, 50, UUtils::kPi / 4);
  Trap_t trap4("trap4", 50, 0, 0, 50, 50, 50, -UUtils::kPi / 4, 50, 50, 50, -UUtils::kPi / 4);

  // confirm Trd calculations
  Trap_t trap5("Trd2", 20, 20, 30, 30, 40);
  // std::cout<<" Trd-like capacity: "<< trap5.Capacity() <<"\n";
  // std::cout<<" Trd-like surface area: "<< trap5.SurfaceArea() <<"\n";

  Vec_t Corners[8];
  Corners[0] = Vec_t(-3., -3., -3.);
  Corners[1] = Vec_t(3., -3., -3.);
  Corners[2] = Vec_t(-3., 3., -3.);
  Corners[3] = Vec_t(3., 3., -3.);
  Corners[4] = Vec_t(-3., -3., 3.);
  Corners[5] = Vec_t(3., -3., 3.);
  Corners[6] = Vec_t(-3., 3., 3.);
  Corners[7] = Vec_t(3., 3., 3.);

  Trap_t tempTrap("temp trap", Corners);

  // Check cubic volume

  vol      = trap1.Capacity();
  volCheck = 8 * 20 * 30 * 40;
  assert(ApproxEqual(vol, volCheck));

  vol      = trap4.Capacity();
  volCheck = 8 * 50 * 50 * 50;
  assert(ApproxEqual(vol, volCheck));

  vol      = trap3.Capacity();
  volCheck = 8 * 50 * 50 * 50;
  assert(ApproxEqual(vol, volCheck));

  vol      = trap2.Capacity();
  volCheck = 2 * 40. * ((20. + 40.) * (10. + 30.) + (30. - 10.) * (40. - 20.) / 3.);
  assert(ApproxEqual(vol, volCheck));

  // Check surface area

  vol      = trap1.SurfaceArea();
  volCheck = 2 * (40 * 60 + 80 * 60 + 80 * 40);
  assert(ApproxEqual(vol, volCheck));

  vol      = trap2.SurfaceArea();
  volCheck = 4 * (10 * 20 + 30 * 40) +
             2 * ((20 + 40) * std::sqrt(4 * 40 * 40 + (30 - 10) * (30 - 10)) +
                  (30 + 10) * std::sqrt(4 * 40 * 40 + (40 - 20) * (40 - 20)));
  assert(ApproxEqual(vol, volCheck));

  // std::cout<<"Trd Surface Area : " << trap5.SurfaceArea()<<std::endl;
  assert(trap5.SurfaceArea() == 20800);

  // vecgeom::cxx::SimpleTrapezoid const* ptrap1 = dynamic_cast<vecgeom::cxx::SimpleTrapezoid*>(&trap1);
  // if(ptrap1 != NULL) {
  // ptrap1->Print();
  // // ptrap1->PrintType(std::cout);
  // // ptrap1->PrintUnplacedType(std::cout);
  // // ptrap1->PrintImplementationType(std::cout);
  // }

  // Check Inside
  assert(trap1.Inside(pzero) == vecgeom::EInside::kInside);
  assert(trap1.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(trap1.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(trap1.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(trap1.Inside(ponzside) == vecgeom::EInside::kSurface);

  assert(trap2.Inside(pzero) == vecgeom::EInside::kInside);
  assert(trap2.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(trap2.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(trap2.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(trap2.Inside(ponzside) == vecgeom::EInside::kSurface);

  // test GetPointOnSurface()
  vecgeom::Vector3D<vecgeom::Precision> ponsurf;
  for (int i = 0; i < 1000000; ++i) {
    ponsurf = trap1.GetPointOnSurface();
    assert(trap1.Inside(ponsurf) == vecgeom::EInside::kSurface);
    assert(trap1.Normal(ponsurf, normal) && ApproxEqual(normal.Mag2(), 1.0));
  }
  for (int i = 0; i < 1000000; ++i) {
    ponsurf = trap2.GetPointOnSurface();
    assert(trap2.Inside(ponsurf) == vecgeom::EInside::kSurface);
    assert(trap2.Normal(ponsurf, normal) && ApproxEqual(normal.Mag2(), 1.0));
  }

  // Check Surface Normal

  valid = trap1.Normal(ponxside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(1., 0., 0.)));
  valid = trap1.Normal(ponmxside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(-1., 0., 0.)));
  valid = trap1.Normal(ponyside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., 1., 0.)));
  valid = trap1.Normal(ponmyside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., -1., 0.)));
  valid = trap1.Normal(ponzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., 0., 1.)));
  valid = trap1.Normal(ponmzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., 0., -1.)));
  valid = trap1.Normal(ponzsidey, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., 0., 1.)));
  valid = trap1.Normal(ponmzsidey, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0., 0., -1.)));

  // Normals on Edges

  Vec_t edgeXY(20.0, 30., 0.0);
  Vec_t edgemXmY(-20.0, -30., 0.0);
  Vec_t edgeXmY(20.0, -30., 0.0);
  Vec_t edgemXY(-20.0, 30., 0.0);
  Vec_t edgeXZ(20.0, 0.0, 40.0);
  Vec_t edgemXmZ(-20.0, 0.0, -40.0);
  Vec_t edgeXmZ(20.0, 0.0, -40.0);
  Vec_t edgemXZ(-20.0, 0.0, 40.0);
  Vec_t edgeYZ(0.0, 30.0, 40.0);
  Vec_t edgemYmZ(0.0, -30.0, -40.0);
  Vec_t edgeYmZ(0.0, 30.0, -40.0);
  Vec_t edgemYZ(0.0, -30.0, 40.0);

  // double invSqrt2 = 1.0 / std::sqrt(2.0);
  // double invSqrt3 = 1.0 / std::sqrt(3.0);

  valid = trap1.Normal(edgeXY, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt2, invSqrt2, 0.0)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgemXmY, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt2, -invSqrt2, 0.0)) && valid);
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeXmY, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt2, -invSqrt2, 0.0)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXY, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt2, invSqrt2, 0.0)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgeXZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt2, 0.0, invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, -invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeXmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt2, 0.0, -invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemXZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt2, 0.0, invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  valid = trap1.Normal(edgeYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(0.0, invSqrt2, invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(0.0, -invSqrt2, -invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgeYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(0.0, invSqrt2, -invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(edgemYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(0.0, -invSqrt2, invSqrt2)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  // Normals on corners

  Vec_t cornerXYZ(20.0, 30., 40.0);
  Vec_t cornermXYZ(-20.0, 30., 40.0);
  Vec_t cornerXmYZ(20.0, -30., 40.0);
  Vec_t cornermXmYZ(-20.0, -30., 40.0);
  Vec_t cornerXYmZ(20.0, 30., -40.0);
  Vec_t cornermXYmZ(-20.0, 30., -40.0);
  Vec_t cornerXmYmZ(20.0, -30., -40.0);
  Vec_t cornermXmYmZ(-20.0, -30., -40.0);

  valid = trap1.Normal(cornerXYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXmYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXmYZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt3, invSqrt3, -invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt3, invSqrt3, -invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornerXmYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(invSqrt3, -invSqrt3, -invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));
  valid = trap1.Normal(cornermXmYmZ, normal);
  // assert(valid && ApproxEqual(normal, Vec_t(-invSqrt3, -invSqrt3, -invSqrt3)));
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  valid = trap2.Normal(ponxside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(cosa, 0, -sina)));
  valid = trap2.Normal(ponmxside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(-cosa, 0, -sina)));
  valid = trap2.Normal(ponyside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, cosa, -sina)));
  valid = trap2.Normal(ponmyside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, -cosa, -sina)));
  valid = trap2.Normal(ponzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trap2.Normal(ponmzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, 0, -1)));
  valid = trap2.Normal(ponzsidey, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = trap2.Normal(ponmzsidey, normal);
  // std::cout << " Normal at " << ponmzsidey << " is " << normal
  //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::endl;
  // assert(valid && ApproxEqual(normal, Vec_t(0, 0.615412, -0.788205))); // (0,cosa,-sina) ?
  assert(valid && ApproxEqual(normal.Mag2(), 1.0));

  // SafetyFromInside(P)

  Dist = trap1.SafetyFromInside(pzero);
  assert(ApproxEqual(Dist, 20));
  Dist = trap1.SafetyFromInside(vx);
  assert(ApproxEqual(Dist, 19));
  Dist = trap1.SafetyFromInside(vy);
  assert(ApproxEqual(Dist, 20));
  Dist = trap1.SafetyFromInside(vz);
  assert(ApproxEqual(Dist, 20));

  Dist = trap2.SafetyFromInside(pzero);
  assert(ApproxEqual(Dist, 20 * cosa));
  Dist = trap2.SafetyFromInside(vx);
  assert(ApproxEqual(Dist, 19 * cosa));
  Dist = trap2.SafetyFromInside(vy);
  assert(ApproxEqual(Dist, 20 * cosa));
  Dist = trap2.SafetyFromInside(vz);
  assert(ApproxEqual(Dist, 20 * cosa + sina));

  // DistanceToOut(P,V)

  Dist = trap1.DistanceToOut(pzero, vx, norm, convex);
  assert(ApproxEqual(Dist, 20) && ApproxEqual(norm, vx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vmx, norm, convex);
  assert(ApproxEqual(Dist, 20) && ApproxEqual(norm, vmx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vy, norm, convex);
  assert(ApproxEqual(Dist, 30) && ApproxEqual(norm, vy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vmy, norm, convex);
  assert(ApproxEqual(Dist, 30) && ApproxEqual(norm, vmy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vz, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vz));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vmz, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vmz));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(pzero, vxy, norm, convex);
  assert(ApproxEqual(Dist, std::sqrt(800.)) && ApproxEqual(norm, vx));
  if (!testvecgeom) assert(convex);

  Dist = trap1.DistanceToOut(ponxside, vx, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmxside, vmx, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vmx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponyside, vy, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmyside, vmy, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vmy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponzside, vz, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vz));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmzside, vmz, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vmz));
  if (!testvecgeom) assert(convex);

  Dist = trap1.DistanceToOut(ponxside, vmx, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vmx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmxside, vx, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vx));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponyside, vmy, norm, convex);
  assert(ApproxEqual(Dist, 60) && ApproxEqual(norm, vmy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmyside, vy, norm, convex);
  assert(ApproxEqual(Dist, 60) && ApproxEqual(norm, vy));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponzside, vmz, norm, convex);
  assert(ApproxEqual(Dist, 80) && ApproxEqual(norm, vmz));
  if (!testvecgeom) assert(convex);
  Dist = trap1.DistanceToOut(ponmzside, vz, norm, convex);
  assert(ApproxEqual(Dist, 80) && ApproxEqual(norm, vz));
  if (!testvecgeom) assert(convex);

  Dist = trap2.DistanceToOut(pzero, vx, norm, convex);
  assert(ApproxEqual(Dist, 20) && ApproxEqual(norm, Vec_t(cosa, 0, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vmx, norm, convex);
  assert(ApproxEqual(Dist, 20) && ApproxEqual(norm, Vec_t(-cosa, 0, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vy, norm, convex);
  assert(ApproxEqual(Dist, 30) && ApproxEqual(norm, Vec_t(0, cosa, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vmy, norm, convex);
  assert(ApproxEqual(Dist, 30) && ApproxEqual(norm, Vec_t(0, -cosa, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vz, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vz));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vmz, norm, convex);
  assert(ApproxEqual(Dist, 40) && ApproxEqual(norm, vmz));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(pzero, vxy, norm, convex);
  assert(ApproxEqual(Dist, std::sqrt(800.)));
  if (!testvecgeom) assert(convex);

  Dist = trap2.DistanceToOut(ponxside, vx, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, Vec_t(cosa, 0, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(ponmxside, vmx, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, Vec_t(-cosa, 0, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(ponyside, vy, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, Vec_t(0, cosa, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(ponmyside, vmy, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, Vec_t(0, -cosa, -sina)));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(ponzside, vz, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vz));
  if (!testvecgeom) assert(convex);
  Dist = trap2.DistanceToOut(ponmzside, vmz, norm, convex);
  assert(ApproxEqual(Dist, 0) && ApproxEqual(norm, vmz));
  if (!testvecgeom) assert(convex);

  // SafetyFromOutside(P)

  Dist = trap1.SafetyFromOutside(pbig);
  //  std::cout<<"trap1.SafetyFromOutside(pbig) = "<<Dist<<std::endl;
  assert(ApproxEqual(Dist, 80));

  Dist = trap1.SafetyFromOutside(pbigx);
  assert(ApproxEqual(Dist, 80));

  Dist = trap1.SafetyFromOutside(pbigmx);
  assert(ApproxEqual(Dist, 80));

  Dist = trap1.SafetyFromOutside(pbigy);
  assert(ApproxEqual(Dist, 70));

  Dist = trap1.SafetyFromOutside(pbigmy);
  assert(ApproxEqual(Dist, 70));

  Dist = trap1.SafetyFromOutside(pbigz);
  assert(ApproxEqual(Dist, 60));

  Dist = trap1.SafetyFromOutside(pbigmz);
  assert(ApproxEqual(Dist, 60));

  Dist = trap2.SafetyFromOutside(pbigx);
  assert(ApproxEqual(Dist, 80 * cosa));
  Dist = trap2.SafetyFromOutside(pbigmx);
  assert(ApproxEqual(Dist, 80 * cosa));
  Dist = trap2.SafetyFromOutside(pbigy);
  assert(ApproxEqual(Dist, 70 * cosa));
  Dist = trap2.SafetyFromOutside(pbigmy);
  assert(ApproxEqual(Dist, 70 * cosa));
  Dist = trap2.SafetyFromOutside(pbigz);
  assert(ApproxEqual(Dist, 60));
  Dist = trap2.SafetyFromOutside(pbigmz);
  assert(ApproxEqual(Dist, 60));

  //=== add test cases to reproduce a crash in Geant4: negative SafetyFromInside() is not acceptable
  // std::cout <<"trap1.S2O(): Line "<< __LINE__ <<", p="<< testp <<", saf2out=" << Dist <<"\n";

  Vec_t testp;
  double testValue = 0.11;
  testp            = ponxside + testValue * vx;
  Dist             = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  if (Dist > 0) std::cout << "trap1.S2O(): Line " << __LINE__ << ", p=" << testp << ", saf2out=" << Dist << "\n";
  assert(Dist <= 0);

  testp = ponxside - testValue * vx;
  Dist  = trap1.SafetyFromOutside(testp);
  if (Dist > 0) std::cout << "trap1.S2I(): Line " << __LINE__ << ", p=" << testp << ", saf2in=" << Dist << "\n";
  assert(Dist <= 0);
  Dist = trap1.SafetyFromInside(testp);
  assert(ApproxEqual(Dist, testValue));

  testp = ponmxside + testValue * vx;
  Dist  = trap1.SafetyFromOutside(testp);
  if (Dist > 0) std::cout << "trap1.S2I(): Line " << __LINE__ << ", p=" << testp << ", saf2in=" << Dist << "\n";
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist >= 0);

  testp = ponmxside - testValue * vx;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist <= 0);

  testp = ponyside + testValue * vy;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist <= 0);

  testp = ponyside - testValue * vy;
  Dist  = trap1.SafetyFromOutside(testp);
  Dist  = trap1.SafetyFromInside(testp);
  assert(ApproxEqual(Dist, testValue));

  testp = ponmyside + testValue * vy;
  Dist  = trap1.SafetyFromOutside(testp);
  Dist  = trap1.SafetyFromInside(testp);
  assert(ApproxEqual(Dist, testValue));

  testp = ponmyside - testValue * vy;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist <= 0);

  testp = ponzside + testValue * vz;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist <= 0);

  testp = ponzside - testValue * vz;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(Dist <= 0);
  Dist = trap1.SafetyFromInside(testp);
  assert(ApproxEqual(Dist, testValue));

  testp = ponmzside + testValue * vz;
  Dist  = trap1.SafetyFromOutside(testp);
  // std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
  assert(Dist <= 0.);
  Dist = trap1.SafetyFromInside(testp);
  assert(ApproxEqual(Dist, testValue));

  testp = ponmzside - testValue * vz;
  Dist  = trap1.SafetyFromOutside(testp);
  assert(ApproxEqual(Dist, testValue));
  Dist = trap1.SafetyFromInside(testp);
  assert(Dist <= 0);

  // DistanceToIn(P,V)

  Dist = trap1.DistanceToIn(pbigx, vmx);
  assert(ApproxEqual(Dist, 80));
  Dist = trap1.DistanceToIn(pbigmx, vx);
  assert(ApproxEqual(Dist, 80));
  Dist = trap1.DistanceToIn(pbigy, vmy);
  assert(ApproxEqual(Dist, 70));
  Dist = trap1.DistanceToIn(pbigmy, vy);
  assert(ApproxEqual(Dist, 70));
  Dist = trap1.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual(Dist, 60));
  Dist = trap1.DistanceToIn(pbigmz, vz);
  assert(ApproxEqual(Dist, 60));
  Dist = trap1.DistanceToIn(pbigx, vxy);
  assert(ApproxEqual(Dist, Constants::kInfLength));
  Dist = trap1.DistanceToIn(pbigmx, vxy);
  assert(ApproxEqual(Dist, Constants::kInfLength));

  Dist = trap2.DistanceToIn(pbigx, vmx);
  assert(ApproxEqual(Dist, 80));
  Dist = trap2.DistanceToIn(pbigmx, vx);
  assert(ApproxEqual(Dist, 80));
  Dist = trap2.DistanceToIn(pbigy, vmy);
  assert(ApproxEqual(Dist, 70));
  Dist = trap2.DistanceToIn(pbigmy, vy);
  assert(ApproxEqual(Dist, 70));
  Dist = trap2.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual(Dist, 60));
  Dist = trap2.DistanceToIn(pbigmz, vz);
  assert(ApproxEqual(Dist, 60));
  Dist = trap2.DistanceToIn(pbigx, vxy);
  assert(ApproxEqual(Dist, Constants::kInfLength));
  Dist = trap2.DistanceToIn(pbigmx, vxy);
  assert(ApproxEqual(Dist, Constants::kInfLength));

  dist = trap3.DistanceToIn(Vec_t(50, -50, 0), vy);
  assert(ApproxEqual(dist, 50));

  dist = trap3.DistanceToIn(Vec_t(50, -50, 0), vmy);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap4.DistanceToIn(Vec_t(50, 50, 0), vy);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap4.DistanceToIn(Vec_t(50, 50, 0), vmy);
  assert(ApproxEqual(dist, 50));

  dist = trap1.DistanceToIn(Vec_t(0, 60, 0), vxmy);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 50, 0), vxmy);
  std::cout << "trap1.DistanceToIn(Vec_t(0,50,0),vxmy) = " << dist << " and vxmy=" << vxmy << std::endl;
  // assert(ApproxEqual(dist,sqrt(800.)));  // A bug in UTrap!!!  Just keep printout above as a reminder

  dist = trap1.DistanceToIn(Vec_t(0, 40, 0), vxmy);
  assert(ApproxEqual(dist, 10.0 * std::sqrt(2.0)));

  dist = trap1.DistanceToIn(Vec_t(0, 40, 50), vxmy);
  assert(ApproxEqual(dist, Constants::kInfLength));

  // Parallel to side planes

  dist = trap1.DistanceToIn(Vec_t(40, 60, 0), vmx);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap1.DistanceToIn(Vec_t(40, 60, 0), vmy);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap1.DistanceToIn(Vec_t(40, 60, 50), vmz);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 50), vymz);
  assert(ApproxEqual(dist, 10.0 * std::sqrt(2.0)));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 80), vymz);
  assert(ApproxEqual(dist, Constants::kInfLength));

  dist = trap1.DistanceToIn(Vec_t(0, 0, 70), vymz);
  std::cout << "trap1.DistanceToIn(Vec_t(0,0,70),vymz) = " << dist << ", vymz=" << vymz << std::endl;
  // assert(ApproxEqual(dist,30.0*sqrt(2.0)));  // A bug in UTrap!!!  Just keep printout above as a reminder

  // CalculateExtent

  Vec_t minExtent, maxExtent;
  trap1.Extent(minExtent, maxExtent);
  assert(ApproxEqual(minExtent, Vec_t(-20, -30, -40)));
  assert(ApproxEqual(maxExtent, Vec_t(20, 30, 40)));
  trap2.Extent(minExtent, maxExtent);
  assert(ApproxEqual(minExtent, Vec_t(-30, -40, -40)));
  assert(ApproxEqual(maxExtent, Vec_t(30, 40, 40)));

  return true;
}

void TestVECGEOM375()
{
  // unit test coming from Issue VECGEOM-375
  // Note: angles in rad, lengths in mm!
  vecgeom::UnplacedTrapezoid t(/*pDz=*/1522, /*pTheta=*/1.03516586152568 * degToRad,
                               /* double pPhi=*/-90.4997606158588 * degToRad, /*pDy1=*/147.5,
                               /*pDx1=*/11.0548515319824, /*pDx2=*/13.62808513641355,
                               /*pTanAlpha1=*/0.4997657953434789 * degToRad, /*pDy2 = */ 92.5,
                               /*pDx3=*/11.0548515319824, /*pDx4=*/12.6685743331909,
                               /*pTanAlpha2=*/0.499766062147744 * degToRad);

  vecgeom::LogicalVolume logvol("", &t);
  logvol.Place();
  // t.StreamInfo(std::cout);
  // t.Print(std::cout);

  using Vec_t = vecgeom::Vector3D<double>;
  Vec_t ponsurf, normal;
  // check normal for a million points
  for (int i = 0; i < 1000000; ++i) {
    ponsurf = t.GetPointOnSurface();
    if (t.Inside(ponsurf) != vecgeom::EInside::kSurface) {
      double dist = vecCore::math::Max(t.SafetyToIn(ponsurf), t.SafetyToOut(ponsurf));
      // if triggered, a quick fix is to relax the value of trapSurfaceTolerance in TrapezoidImplementation.h
      std::cout << "*** Not on surface: i=" << i << ", ponsurf=" << ponsurf << " for dist=" << dist << "\n";
    }
    assert(t.Inside(ponsurf) == vecgeom::EInside::kSurface);
    assert(t.Normal(ponsurf, normal) && ApproxEqual(normal.Mag2(), 1.0));
  }
}

void TestVECGEOM353()
{
  // unit test coming from Issue VECGEOM-353
  vecgeom::UnplacedTrapezoid t(/*pDz=*/0.5, /*pTheta=*/0, /* double pPhi=*/0, /*pDy1=*/3,
                               /*pDx1=*/14.237500000000001, /* double pDx2 =*/12.592500000000001,
                               /*pTanAlpha1 =*/0.274166666666666, /*pDy2 = */ 3,
                               /*pDx3 = */ 14.237500000000001, /*pDx4 = */ 12.592500000000001,
                               /*pTanAlpha2 = */ 0.274166666666666);

  vecgeom::LogicalVolume l("", &t);
  auto p      = l.Place();
  using Vec_t = vecgeom::Vector3D<double>;
  Vec_t point(-16.483749999999997, -6.4512999999999989, 0.00000099999999999999995);
  assert(!p->Contains(point));
  auto dist = p->DistanceToIn(point, Vec_t(1., 0., 0.));
  assert(dist == vecgeom::kInfLength);
}

void TestVECGEOM393()
{
  using namespace vecgeom;
  // unit test coming from Issue VECGEOM-393
  const double &deg = degToRad;
  vecgeom::UnplacedTrapezoid trap(60, 20 * deg, 5 * deg, 40, 30, 40, 10 * deg, 16, 10, 14, 10 * deg);

  Vector3D<Precision> extMin, extMax;
  trap.Extent(extMin, extMax);
  assert(ApproxEqual(extMin, Vector3D<double>(-58.80819229, -41.90332577, -60.)));
  assert(ApproxEqual(extMax, Vector3D<double>(38.57634475, 38.09667423, 60.)));
}

#ifdef VECGEOM_USOLIDS
struct USOLIDSCONSTANTS {
  static constexpr double kInfLength = DBL_MAX;
};
#endif
struct VECGEOMCONSTANTS {
  static constexpr double kInfLength = vecgeom::kInfLength;
};

int main(int argc, char *argv[])
{

  if (argc < 2) {
    std::cerr << "need to give argument: --usolids or --vecgeom\n";
    return 1;
  }

  if (!strcmp(argv[1], "--usolids")) {
#ifndef VECGEOM_USOLIDS
    std::cerr << "VECGEOM_USOLIDS was not defined\n";
    return 2;
#else
#ifndef VECGEOM_REPLACE_USOLIDS
    TestTrap<USOLIDSCONSTANTS, UTrap>();
    std::cout << "UTrap passed (but notice discrepancies above, where asserts have been disabled!)\n";
#else
    testvecgeom = true; // needed to avoid testing convexity when vecgeom is used
    TestTrap<VECGEOMCONSTANTS, UTrap>();
    std::cout << "UTrap --> VecGeom trap passed\n";
#endif
#endif
  }

  else if (!strcmp(argv[1], "--vecgeom")) {
    testvecgeom = true; // needed to avoid testing convexity when vecgeom is used
    TestVECGEOM375();
    TestVECGEOM353();
    TestVECGEOM393();
    TestTrap<VECGEOMCONSTANTS, VECGEOM_NAMESPACE::SimpleTrapezoid>();

    std::cout << "VecGeom Trap passed.\n";
  }

  else {
    std::cerr << "argument needs to be either of: --usolids or --vecgeom\n";
    return 1;
  }

  return 0;
}
