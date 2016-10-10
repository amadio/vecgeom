#include <VecCore/VecCore>
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTorus.hh"

#include "volumes/SpecializedTorus2.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

// typedef UBox Box_t;
typedef vecgeom::SimpleTorus2 Torus_t;

int main(int argc, char *argv[])
{
  int errCode = 0;
  double rmin = 5.;
  double rmax = 10;
  double rtor = 50;
  double sphi = 0.;
  double dphi = 2. * M_PI;
  ;
  VUSolid *torus;
  if (argc > 1) {
    if (strcmp(argv[1], "vec") == 0) {
      torus = new Torus_t("test_VecGeomTorus", rmin, rmax, rtor, sphi, dphi);
    } else {
      torus = new UTorus("test_USolidsTorus", rmin, rmax, rtor, sphi, dphi);
    }
  } else {
    torus = new Torus_t("test_VecGeomTorus", rmin, rmax, rtor, sphi, dphi);
  }

  ShapeTester tester;

  if (argc > 2) {
    if (strcmp(argv[2], "vis") == 0) {
#ifdef VECGEOM_ROOT
      TApplication theApp("App", 0, 0);
      errCode = tester.Run(torus);
      theApp.Run();
#endif
    }
  } else {
    // tester.SetMethod("Consistency");
    // tester.SetMethod("XRayProfile");
    errCode = tester.Run(torus);
  }
  std::cout << "Final Error count for Shape *** " << torus->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================\n";
  return 0;
}
