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

  Precision rmin = 0.;
  Precision rmax = 5.;
  Precision dz   = 10.;
  Precision sphi = 0.;
  Precision dphi = 2. * kPi;
  Precision thb  = 3. * kPi / 4.;
  Precision phib = kPi / 3.;
  Precision tht  = kPi / 4.;
  Precision phit = 2. * kPi / 3.;

  switch (type) {
  case 0:
    break;
  case 1:
    rmin = 3.;
    break;
  case 2:
    dphi = 2. * kPi / 3.;
    break;
  case 3:
    rmin = 3.;
    dphi = 2. * kPi / 3.;
    break;
  default:
    printf("Unknown test\n");
    return 1;
  }
  Vector3D<Precision> nbottom(std::sin(thb) * std::cos(phib), std::sin(thb) * std::sin(phib), std::cos(thb));
  Vector3D<Precision> ntop(std::sin(tht) * std::cos(phit), std::sin(tht) * std::sin(phit), std::cos(tht));

  CutTube_t *cuttube = new CutTube_t("test_VecGeomCutTube", rmin, rmax, dz, sphi, dphi, nbottom, ntop);
  cuttube->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  //tester.SetTestBoundaryErrors(true);
  tester.SetSolidTolerance(vecgeom::kTolerance);

  int errCode = tester.Run(cuttube);

  std::cout << "Final Error count for Shape *** " << cuttube->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (cuttube) delete cuttube;
  return 0;
}
