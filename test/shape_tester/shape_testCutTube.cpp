#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VecGeom/volumes/CutTube.h"
typedef vecgeom::SimpleCutTube CutTube_t;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testCutTube -test <#>:\n"
                 "       0 - rmin = 0, dphi = 2*kPi\n"
                 "       1 - rmin > 0, dphi = 2*kPi\n"
                 "       2 - rmin = 0, dphi < 2*kPi\n"
                 "       3 - rmin > 0, dphi < 2*kPi\n"
              << std::endl;
  }
  using namespace vecgeom;

  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 3);

  double rmin = 0.;
  double rmax = 5.;
  double dz   = 10.;
  double sphi = 0.;
  double dphi = 2 * kPi;
  double thb  = 3 * kPi / 4;
  double phib = kPi / 3;
  double tht  = kPi / 4;
  double phit = 2 * kPi / 3;

  switch (type) {
  case 0:
    break;
  case 1:
    rmin = 3.;
    break;
  case 2:
    dphi = 2 * kPi / 3;
    break;
  case 3:
    rmin = 3.;
    dphi = 2 * kPi / 3;
    break;
  default:
    printf("Unknown test\n");
    return 1;
  }
  Vector3D<double> nbottom(std::sin(thb) * std::cos(phib), std::sin(thb) * std::sin(phib), std::cos(thb));
  Vector3D<double> ntop(std::sin(tht) * std::cos(phit), std::sin(tht) * std::sin(phit), std::cos(tht));

  CutTube_t *cuttube = new CutTube_t("test_VecGeomCutTube", rmin, rmax, dz, sphi, dphi, nbottom, ntop);
  cuttube->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(cuttube);

  std::cout << "Final Error count for Shape *** " << cuttube->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (cuttube) delete cuttube;
  return 0;
}
