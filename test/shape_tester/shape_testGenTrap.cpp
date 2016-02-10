#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UGenericTrap.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/GenTrap.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimpleGenTrap GenTrap_t;

int main(int argc, char *argv[]) {
  using namespace vecgeom;
  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  int errCode = 0;
  GenTrap_t *solid = new GenTrap_t("test_VecGeomGenTrap", verticesx, verticesy, 5);

  ShapeTester tester;
  //  tester.EnableDebugger(true);

  if (argc > 1) {
    if (strcmp(argv[1], "vis") == 0) {
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
