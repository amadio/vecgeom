#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTrd.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Trd.h"

#include "../benchmark/ArgParser.h"
#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

// typedef UCons Cone_t;
typedef vecgeom::SimpleTrd Trd_t;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "Usage: shape_testTrd -test <#>:\n"
                 "       0 - Constant dx/dy (box)\n"
                 "       1 - Variable dx constant dy\n"
                 "       2 - Increasing dx increasing dy\n"
                 "       3 - Decreasing dx decreasing dy\n"
                 "       4 - BaBar thin dx large dy\n"
                 "       5 - USolids\n";
  }
  OPTION_INT(test, 0);
  int errCode = 0;
  VUSolid *solid = 0;

  switch (test) {
  case 0:
    // Constant dx/dy (box)
    solid = new Trd_t("test_VecGeomTrdBox", 20., 20., 30., 30., 40.);
    break;
  case 1:
    // Variable dx constant dy
    solid = new Trd_t("test_VecGeomTrdConstY", 20., 30., 30., 30., 40.);
    break;
  case 2:
    // Increasing dx increasing dy
    solid = new Trd_t("test_VecGeomTrdIncreasingXY", 10., 20., 20., 40., 40.);
    break;
  case 3:
    // Decreasing dx decreasing dy
    solid = new Trd_t("test_VecGeomTrdDecreasingXY", 30., 10., 40., 30., 40.);
    break;
  case 4:
    // BaBar thin dx large dy
    solid = new Trd_t("test_VecGeomTrd", 0.15, 0.15, 24.707, 24.707, 7.);
    break;
  case 5:
    // USolids
    solid = new UTrd("test_USolidsTrd", 5., 8., 6., 9., 5.);
    solid->StreamInfo(std::cout);
    break;
  default:
    std::cout << "Unknown test case.\n";
    return 1;
  }

  ShapeTester tester;
  tester.SetTestBoundaryErrors(false);

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
  }
  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return 0;
}
