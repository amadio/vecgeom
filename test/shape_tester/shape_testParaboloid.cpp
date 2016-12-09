/// \file shape_testParaboloid.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"

#include "UParaboloid.hh"
#include "volumes/Paraboloid.h"

typedef vecgeom::SimpleParaboloid Paraboloid_t;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  Paraboloid_t const *para = new Paraboloid_t("test_para", 6., 10., 10.);
  para->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setConventionsMode(usolids);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(true);
  int errCode = tester.Run(para);

  std::cout << "Final Error count for Shape *** " << para->GetName() << "*** = " << errCode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================" << std::endl;

  if (para) delete para;
  return 0;
}
