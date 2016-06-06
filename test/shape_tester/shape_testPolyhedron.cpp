/// \file shape_testPolyhedron.h
/// \author: Mihaela Gheata (mihaela.gheata@cern.ch)
#include "ShapeTester.h"
#include "VUSolid.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Polyhedron.h"

#include "../benchmark/ArgParser.h"
#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimplePolyhedron Polyhedron_t;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "Usage: shape_testPolyhedron -test <#>:\n"
                 "       0 - case 0\n"
                 "       1 - case 1\n";
  }
  OPTION_INT(test, 0);
  using namespace vecgeom;

  Polyhedron_t *solid = 0;
  Precision phiStart = 0., deltaPhi = 120.;
  int sides = 4;
  constexpr int nPlanes = 5;
  double zPlanes[nPlanes] = {-2, -1, 1, 1, 2};
  double rInner[nPlanes] = {0, 1, 0.5, 1, 0};
  double rOuter[nPlanes] = {1, 2, 2, 2.5, 1};

  switch (test) {
  case 0:
    // Non-zero alpha, theta, phi
    std::cout << "Testing polyhedron #0\n";
    solid = new Polyhedron_t("test_VecGeomPolyhedron", phiStart, deltaPhi, sides, nPlanes, zPlanes, rInner, rOuter);
    break;
  case 1:
    // Polyhedron degenerated to box
    std::cout << "NOT implemented polyhedron #1\n";
    return 1;
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  int errCode = 0;

  ShapeTester tester;
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(false);
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
