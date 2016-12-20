#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "volumes/PlacedVolume.h"

#ifdef VECGEOM_USOLIDS
#include "UTubs.hh"
#endif
#include "volumes/Tube.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTube        = vecgeom::SimpleTube;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  OPTION_DOUBLE(dz, 1.);
  OPTION_DOUBLE(rmax, 6.);
  OPTION_DOUBLE(rmin, 2.);
  OPTION_DOUBLE(sphi, 0.);
  OPTION_DOUBLE(dphi, vecgeom::kTwoPi);

  if (usolids) {
#ifndef VECGEOM_USOLIDS
    std::cerr << "\n*** ERROR: library built with -DUSOLIDS=OFF and user selected '-usolids true'!\n Aborting...\n\n";
    return 1;
#else
    // USolids conventions
    auto tube = new UTubs("usolidsTube", dz, rmax, rmin, sphi, dphi);
    tube->StreamInfo(std::cout);
    return runTester<VUSolid>(tube, npoints, usolids, debug, stat);
#endif
  }

  else {
    // VecGeom conventions
    auto tube = new VGTube("vecgeomTube", dz, rmax, rmin, sphi, dphi);
    tube->Print();
    return runTester<VPlacedVolume>(tube, npoints, usolids, debug, stat);
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
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errcode;
}
