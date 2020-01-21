#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Cone.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGCone        = vecgeom::SimpleCone;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(rmin1, 4.);
  OPTION_DOUBLE(rmax1, 6.);
  OPTION_DOUBLE(rmin2, 5.);
  OPTION_DOUBLE(rmax2, 7.);
  OPTION_DOUBLE(dz, 2.);
  OPTION_DOUBLE(sphi, 0.);
  OPTION_DOUBLE(dphi, vecgeom::kTwoPi);

  auto cone = new VGCone("vecgeomCone", rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi);
  cone->Print();
  return runTester<VPlacedVolume>(cone, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{

  ShapeTester<ImplT> tester;
  tester.SetSolidTolerance(1.e-7);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errcode;
}
