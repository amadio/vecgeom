#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UGenericTrap.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/GenTrap.h"

#include "../benchmark/ArgParser.h"
#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimpleGenTrap GenTrap_t;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "Usage: shape_testGenTrap -test <#>:\n"
                 "       0 - twisted\n"
                 "       1 - planar\n"
                 "       2 - one face triangle\n"
                 "       3 - one face line\n"
                 "       4 - one face point\n"
                 "       5 - one face line, other triangle\n";
  }
  OPTION_INT(test, 0);
  using namespace vecgeom;
  // 4 different vertices, twisted
  Precision verticesx0[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy0[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};
  // 4 different vertices, planar
  Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};
  // 3 different vertices
  Precision verticesx2[8] = {-3, -3, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy2[8] = {-2.5, -2.5, 2.5, -3, -2, 2, 2, -2};
  // 2 different vertices
  Precision verticesx3[8] = {-3, -3, 2.5, 2.5, -2, -2, 2, 2};
  Precision verticesy3[8] = {-2.5, -2.5, -3, -3, -2, 2, 2, -2};
  // 1 vertex (pyramid)
  Precision verticesx4[8] = {-3, -3, -3, -3, -2, -2, 2, 2};
  Precision verticesy4[8] = {-2.5, -2.5, -2.5, -2.5, -2, 2, 2, -2};
  // 2 vertex bottom, 3 vertices top
  Precision verticesx5[8] = {-3, -3, 2.5, 2.5, -2, -2, 2, 2};
  Precision verticesy5[8] = {-2.5, -2.5, -3, -3, -2, -2, 2, -2};

  GenTrap_t *solid = 0;
  switch (test) {
  case 0:
    // 4 different vertices, twisted
    std::cout << "Testing twisted trapezoid\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx0, verticesy0, 5);
    break;
  case 1:
    // 4 different vertices, planar
    std::cout << "Testing planar trapezoid\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx1, verticesy1, 5);
    break;
  case 2:
    // 3 different vertices
    std::cout << "Testing trapezoid with one face triangle\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx2, verticesy2, 5);
    break;
  case 3:
    // 2 different vertices
    std::cout << "Testing trapezoid with one face line degenerated\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx3, verticesy3, 5);
    break;
  case 4:
    // 1 vertex (pyramid)
    std::cout << "Testing trapezoid with one face point degenerated (pyramid)\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx4, verticesy4, 5);
    break;
  case 5:
    // 2 vertex bottom, 3 vertices top
    std::cout << "Testing trapezoid with line on one face and triangle on other\n";
    solid = new GenTrap_t("test_VecGeomGenTrap", verticesx5, verticesy5, 5);
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  int errCode = 0;

  ShapeTester tester;
  // tester.SetSolidTolerance(1.e-7);
  //  tester.EnableDebugger(true);

  if (argc > 3) {
    if (strcmp(argv[3], "vis") == 0) {
#ifdef VECGEOM_ROOT
      TApplication theApp("App", 0, 0);
      errCode = tester.Run(solid);
      theApp.Run();
#endif
    }
  } else {
    errCode = tester.Run(solid);
    //    tester.Run(solid, "debug");
  }
  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return 0;
}
