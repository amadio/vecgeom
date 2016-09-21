#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTrap.hh"

#include "base/Vector3D.h"
#include "volumes/Trapezoid.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

// using Trap_t = UTrap;
using Trap_t    = vecgeom::SimpleTrapezoid;
using Precision = vecgeom::Precision;
using Vec_t     = vecgeom::Vector3D<vecgeom::Precision>;

int main(int argc, char *argv[])
{
  int errCode = 0;
  VUSolid *solid;
  if (argc > 1) {
    if (strcmp(argv[1], "vec") == 0) {
      solid = new Trap_t("test_VecGeomTrap", 40, 0, 0, 30, 20, 20, 0, 30, 20, 20, 0);
    } else {
      solid = new UTrap("test_USolidsTrap", 40, 0, 0, 30, 20, 20, 0, 30, 20, 20, 0);
    }
  } else {
    // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
    vecgeom::TrapCorners xyz;
    Precision xoffset = 9;
    Precision yoffset = -6;

    // define corner points
    // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
    xyz[0] = Vec_t(-2 + xoffset, -5 + yoffset, -15);
    xyz[1] = Vec_t(2 + xoffset, -5 + yoffset, -15);
    xyz[2] = Vec_t(-3 + xoffset, 5 + yoffset, -15);
    xyz[3] = Vec_t(3 + xoffset, 5 + yoffset, -15);
    xyz[4] = Vec_t(-4 - xoffset, -10 - yoffset, 15);
    xyz[5] = Vec_t(4 - xoffset, -10 - yoffset, 15);
    xyz[6] = Vec_t(-6 - xoffset, 10 - yoffset, 15);
    xyz[7] = Vec_t(6 - xoffset, 10 - yoffset, 15);

    // create trapezoid
    solid = new Trap_t("test_slantedTrap", xyz);
  }
  ((vecgeom::VPlacedVolume *)solid)->GetUnplacedVolume()->Print(std::cout);

  ShapeTester tester;

  if (argc > 2) {
    if (strcmp(argv[2], "vis") == 0) {
#ifdef VECGEOM_ROOT
      TApplication theApp("App", 0, 0);
      errCode = tester.Run(solid);
      theApp.Run();
#endif
    }
  } else {
    // tester.SetMethod("Consistency");
    errCode = tester.Run(solid);
  }
  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return 0;
}
