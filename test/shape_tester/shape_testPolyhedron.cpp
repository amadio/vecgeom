/// \file shape_testPolyhedron.h
/// \author: Mihaela Gheata (mihaela.gheata@cern.ch)

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include "VecGeom/volumes/Polyhedron.h"
typedef vecgeom::SimplePolyhedron Polyhedron_t;
using vecgeom::Precision;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testPolyhedron -test <#>:\n"
                 "       0 - case 0\n"
                 "       1 - case 1\n";
  }
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 0);
  using namespace vecgeom;

  Polyhedron_t *solid = 0;

  switch (type) {
  case 0: {
    // Non-zero alpha, theta, phi
    std::cout << "Testing polyhedron #0\n";
    Precision phiStart = 0., deltaPhi = 120. * kDegToRad;
    int sides                  = 4;
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-2, -1, 1, 1, 2};
    Precision rInner[nPlanes]  = {0, 1, 0.5, 1, 0};
    Precision rOuter[nPlanes]  = {1, 2, 2, 2.5, 1};
    solid = new Polyhedron_t("test_VecGeomPolyhedron", phiStart, deltaPhi, sides, nPlanes, zPlanes, rInner, rOuter);
  } break;
  case 1:
    // Polyhedron degenerated to box
    std::cout << "NOT implemented polyhedron #1\n";
    return 1;
    break;
  default:
    std::cout << "Unknown test case.\n";
  }

  if (!solid) return 0;

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
  return 0;
}
