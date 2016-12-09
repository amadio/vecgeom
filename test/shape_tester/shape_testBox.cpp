#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "volumes/PlacedVolume.h"

#include "UBox.hh"
#include "volumes/Box.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGBox         = vecgeom::SimpleBox;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  OPTION_DOUBLE(dx, 10.);
  OPTION_DOUBLE(dy, 15.);
  OPTION_DOUBLE(dz, 20.);

  if (usolids) {
    auto box = new UBox("usolidsBox", dx, dy, dz);
    box->StreamInfo(std::cout);
    return runTester<VUSolid>(box, npoints, usolids, debug, stat);
  } else {
    auto box = new VGBox("vecgeomBox", dx, dy, dz);
    box->Print();
    return runTester<VPlacedVolume>(box, npoints, usolids, debug, stat);
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
