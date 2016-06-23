#include "ShapeTester.h"
#include "VUSolid.hh"
//#include "UTubs.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Hype.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

#define PI 3.14159265358979323846

typedef vecgeom::SimpleHype Hype_t;

int main(int argc, char *argv[])
{

  int errCode   = 0;
  VUSolid *hype = new Hype_t("test_VecGeomHype", 5., 20, PI / 6, PI / 3, 50);
  ShapeTester tester;
  tester.SetSolidTolerance(1e-7);
  if (argc > 1) {
    tester.Run(hype, argv[1]);
  } else {
    tester.Run(hype);
  }

  std::cout << "Final Error count for Shape *** " << hype->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return 0;
}
