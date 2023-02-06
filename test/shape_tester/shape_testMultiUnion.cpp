#ifndef VECGEOM_ENABLE_CUDA

#include "test/benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include <iostream>
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Transformation3D.h"

using namespace vecgeom;
using MultiUnion_t = vecgeom::SimpleMultiUnion;
#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  if (argc == 1) {
    std::cout << "Usage: shape_testMultiUnion <-nsolids N> <-npoints P> <-debug D>\n"
                 "       N - number of box components\n"
                 "       D - debug mode\n";
  }
  OPTION_INT(npoints, 1000);
  OPTION_INT(nsolids, 100);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  MultiUnion_t *solid = 0;
  solid               = new MultiUnion_t("test_VecGeomMultiUnion");

  UnplacedMultiUnion &multiunion = (UnplacedMultiUnion &)(*solid->GetUnplacedVolume());
  std::cout << "Testing multi-union of " << nsolids << " boxes\n";

  constexpr Precision size = 10.;

  Precision sized = size * std::pow(0.5 / nsolids, 1. / 3.);
  for (int i = 0; i < nsolids; ++i) {
    Vector3D<Precision> pos(RNG::Instance().uniform(-size, size), RNG::Instance().uniform(-size, size),
                         RNG::Instance().uniform(-size, size));
    Precision sizernd = RNG::Instance().uniform(0.8 * sized, 1.2 * sized);
    Transformation3D trans(pos.x(), pos.y(), pos.z(), RNG::Instance().uniform(-180, 180),
                           RNG::Instance().uniform(-180, 180), RNG::Instance().uniform(-180, 180));
    trans.SetProperties();
    UnplacedBox *box = new UnplacedBox(sizernd, sizernd, sizernd);
    multiunion.AddNode(box, trans);
  }
  multiunion.Close();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
  return errCode;
#endif
  return 0;
}
