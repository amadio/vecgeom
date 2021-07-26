// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief ShapeTester for the Orb.
/// \file test/shape_tester/shape_testOrb.cpp
/// \author Raman Sehgal

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VecGeom/volumes/Orb.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGOrb         = vecgeom::SimpleOrb;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(radius, 1.);

  auto orb = new VGOrb("vecgeomOrb", radius);
  orb->Print();
  return runTester<VPlacedVolume>(orb, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errCode;
}
