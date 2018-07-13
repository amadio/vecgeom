/// \file shape_testTet.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "volumes/Tet.h"
#include "base/Vector3D.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTet         = vecgeom::SimpleTet;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  vecgeom::Vector3D<double> p0(0., 0., 2.), p1(0., 0., 0.), p2(2., 0., 0.), p3(0., 2., 0.);

  auto tet = new VGTet("vecgeomTet", p0, p1, p2, p3);
  tet->Print();
  return runTester<VPlacedVolume>(tet, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errCode;
}
