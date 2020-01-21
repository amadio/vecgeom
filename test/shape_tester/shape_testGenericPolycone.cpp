#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VecGeom/volumes/GenericPolycone.h"

typedef vecgeom::SimpleGenericPolycone Poly_t;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  using namespace vecgeom;

  double sphi         = 0.;
  double dphi         = vecgeom::kTwoPi;
  const int numRZ1    = 10;
  double polycone_r[] = {1, 5, 3, 4, 9, 9, 3, 3, 2, 1};
  double polycone_z[] = {0, 1, 2, 3, 0, 5, 4, 3, 2, 1};
  auto poly2          = new Poly_t("GenericPoly", sphi, dphi, numRZ1, polycone_r, polycone_z);
  // GenericPolycone_t Simple("GenericPolycone", startPhi, deltaPhi, numRz,r,z);

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(1.e-7);
  tester.SetTestBoundaryErrors(false);
  int errCode = tester.Run(poly2);

  std::cout << "Final Error count for Shape *** " << poly2->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (poly2) delete poly2;
  return 0;
}
