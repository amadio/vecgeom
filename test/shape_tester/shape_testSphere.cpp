#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "volumes/PlacedVolume.h"

#include "USphere.hh"
#include "volumes/Sphere.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGSphere      = vecgeom::SimpleSphere;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  OPTION_DOUBLE(rmin, 15.);
  OPTION_DOUBLE(rmax, 20.);
  OPTION_DOUBLE(sphi, 0.);
  OPTION_DOUBLE(dphi, vecgeom::kTwoPi / 3.);
  OPTION_DOUBLE(stheta, 0.);
  OPTION_DOUBLE(dtheta, vecgeom::kTwoPi);

  if (usolids) {
    auto sphere = new USphere("usolidsSphere", rmin, rmax, sphi, dphi, stheta, dtheta);
    sphere->StreamInfo(std::cout);
    return runTester<VUSolid>(sphere, npoints, usolids, debug, stat);
  } else {
    auto sphere = new VGSphere("vecgeomSphere", rmin, rmax, sphi, dphi, stheta, dtheta);
    sphere->Print();
    return runTester<VPlacedVolume>(sphere, npoints, usolids, debug, stat);
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
