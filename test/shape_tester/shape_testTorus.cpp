#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "volumes/PlacedVolume.h"

#ifdef VECGEOM_USOLIDS
#include "UTorus.hh"
#endif
#include "volumes/SpecializedTorus2.h"

using VGTorus = vecgeom::SimpleTorus2;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  OPTION_INT(type, 0);

  if (type < 0 || type > 3) {
    std::cout << "+++ Invalid type number. Range is 0-3. +++\n";
    return 1;
  }

  double rmin[4] = {0., 5., 5., 1.};
  double rmax[4] = {10., 10., 10., 5.};
  double rtor[4] = {30., 30., 30., 100.};
  double sphi[4] = {0., 0., 0.25 * M_PI, 0.};
  double dphi[4] = {2. * M_PI, 2. * M_PI, 0.75 * M_PI, 2. * M_PI};

  const char *message[] = {"simple torus (no rmin, no phi cut)", "torus tube (rmin, no phi cut)",
                           "general torus (rmin, phi cut)", "ring wire (large rtor/rmax aspect ratio)"};

  if (argc == 1) {
    std::cout << "Usage: shape_testTorus -type <#>:\n";
    std::cout << "       0 - " << message[0] << "\n";
    std::cout << "       1 - " << message[1] << "\n";
    std::cout << "       2 - " << message[2] << "\n";
    std::cout << "       3 - " << message[3] << "\n";
  }

  std::cout << "### TESTING: " << message[type] << " ###\n";

  if (usolids) {
#ifndef VECGEOM_USOLIDS
    std::cerr << "\n*** ERROR: library built with -DUSOLIDS=OFF and user selected '-usolids true'!\n Aborting...\n\n";
    return 1;
#else
    // USolids conventions
    auto torus = new UTorus("testTorus", rmin[type], rmax[type], rtor[type], sphi[type], dphi[type]);
    torus->StreamInfo(std::cout);
    return runTester<VUSolid>(torus, npoints, usolids, debug, stat);
#endif
  }

  else {
    // VecGeom conventions
    auto torus = new VGTorus("testTorus", rmin[type], rmax[type], rtor[type], sphi[type], dphi[type]);
    torus->Print();
    return runTester<vecgeom::VPlacedVolume>(torus, npoints, usolids, debug, stat);
  }
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat)
{

  ShapeTester<ImplT> tester;
  tester.setConventionsMode(usolids);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  // tester.SetTestBoundaryErrors(false);
  // tester.SetSolidTolerance(1.e-7);
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================\n";

  if (shape) delete shape;
  return errcode;
}
