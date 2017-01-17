/// \file shape_testOrb.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"

#ifdef VECGEOM_USOLIDS
#include "UOrb.hh"
#endif
#include "volumes/Orb.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGOrb         = vecgeom::SimpleOrb;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  OPTION_DOUBLE(radius, 1.);

  if (usolids) {
#ifndef VECGEOM_USOLIDS
    std::cerr << "\n*** ERROR: library built with -DUSOLIDS=OFF and user selected '-usolids true'!\n Aborting...\n\n";
    return 1;
#else
    // USolids conventions
    auto orb = new UOrb("usolidsOrb", radius);
    orb->StreamInfo(std::cout);
    return runTester<VUSolid>(orb, npoints, usolids, debug, stat);
#endif
  }

  else {
    // VecGeom conventions
    auto orb = new VGOrb("vecgeomOrb", radius);
    orb->Print();
    return runTester<VPlacedVolume>(orb, npoints, usolids, debug, stat);
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
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errCode;
}
