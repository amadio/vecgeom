/// \file shape_testParallelepiped.h
/// \author: Mihaela Gheata (mihaela.gheata@cern.ch)
#include "ShapeTester.h"
#include "VUSolid.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Parallelepiped.h"

#include "../benchmark/ArgParser.h"
#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimpleParallelepiped Para_t;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "Usage: shape_testParallelepiped -test <#>:\n"
                 "       0 - alpha, theta, phi\n"
                 "       1 - alpha=0 theta=0 phi=0 (box)\n";
  }
  OPTION_INT(test, 0);
  using namespace vecgeom;

  Para_t *solid = 0;
  Precision dx = 10.;
  Precision dy = 7;
  Precision dz = 15;
  Precision alpha = 30.;
  Precision theta = 30.;
  Precision phi = 45.;

  switch (test) {
  case 0:
    // Non-zero alpha, theta, phi
    std::cout << "Testing general parallelepiped\n";
    solid = new Para_t("test_VecGeomPara", dx, dy, dz, alpha, theta, phi);
    break;
  case 1:
    // Parallelepiped degenerated to box
    std::cout << "Testing box-like parallelepiped\n";
    solid = new Para_t("test_VecGeomPara", dx, dy, dz, 0, 0, 0);
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  int errCode = 0;

  ShapeTester tester;
  tester.SetSolidTolerance(1.e-7);
  tester.SetTestBoundaryErrors(true);
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
