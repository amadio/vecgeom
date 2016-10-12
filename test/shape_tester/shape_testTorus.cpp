#include <VecCore/VecCore>
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTorus.hh"

#include "volumes/SpecializedTorus2.h"
#include "../benchmark/ArgParser.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

// typedef UBox Box_t;
typedef vecgeom::SimpleTorus2 Torus_t;

int main(int argc, char *argv[])
{
  OPTION_INT(test, 0);
  OPTION_INT(vis, 0);
  if (test < 0 || test > 3) {
    std::cout << "+++ Invalid test number. Range is 0-3. +++\n";
    return 1;
  }

  int errCode    = 0;
  double rmin[4] = {0., 5., 5., 1.};
  double rmax[4] = {10., 10., 10., 5.};
  double rtor[4] = {30., 30., 30., 100.};
  double sphi[4] = {0., 0., 0.25 * M_PI, 0.};
  double dphi[4] = {2. * M_PI, 2. * M_PI, 0.75 * M_PI, 2. * M_PI};

  const char *message[] = {"simple torus (no rmin, no phi cut)", "torus tube (rmin, no phi cut)",
                           "general torus (rmin, phi cut)", "ring wire (large rtor/rmax aspect ratio)"};

  if (argc == 1) {
    std::cout << "Usage: shape_testTorus -test <#>:\n";
    std::cout << "       0 - " << message[0] << "\n";
    std::cout << "       1 - " << message[1] << "\n";
    std::cout << "       2 - " << message[2] << "\n";
    std::cout << "       3 - " << message[3] << "\n";
  }

  VUSolid *torus = new Torus_t("test_VecGeomTorus", rmin[test], rmax[test], rtor[test], sphi[test], dphi[test]);

  ShapeTester tester;
  tester.SetTestBoundaryErrors(false);
  tester.SetSolidTolerance(1.e-7);
  std::cout << "### TESTING: " << message[test] << " ###\n";

  if (vis) {
#ifdef VECGEOM_ROOT
    TApplication theApp("App", 0, 0);
    tester.EnableDebugger(true);
    errCode = tester.Run(torus);
    theApp.Run();
#endif
  } else {
    errCode = tester.Run(torus);
  }
  std::cout << "Final Error count for " << message[test] << " name: " << torus->GetName() << "*** = " << errCode
            << std::endl;
  std::cout << "=========================================================\n";
  return 0;
}
