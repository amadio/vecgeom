#include "VecGeom/base/Config.h"

#ifndef VECGEOM_ENABLE_CUDA

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>
#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(nsolids, 100);

  constexpr double size = 10.;

  UnplacedBox worldUnplaced = UnplacedBox(2 * size, 2 * size, 2 * size);

  UnplacedMultiUnion multiunion;
  double sized = size * std::pow(0.5 / nsolids, 1. / 3.);
  for (size_t i = 0; i < nsolids; ++i) {
    Vector3D<double> pos(RNG::Instance().uniform(-size, size), RNG::Instance().uniform(-size, size),
                         RNG::Instance().uniform(-size, size));
    double sizernd = RNG::Instance().uniform(0.8 * sized, 1.2 * sized);
    Transformation3D trans(pos.x(), pos.y(), pos.z(), RNG::Instance().uniform(-180, 180),
                           RNG::Instance().uniform(-180, 180), RNG::Instance().uniform(-180, 180));
    trans.SetProperties();
    UnplacedBox *box = new UnplacedBox(sizernd, sizernd, sizernd);
    multiunion.AddNode(box, trans);
  }
  multiunion.Close();

  std::cout << "Benchmarking multi-union solid having " << nsolids << " random boxes\n";
  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume lunion("multiunion", &multiunion);

  Transformation3D placement(0, 0, 0);
  const VPlacedVolume *placedMunion = world.PlaceDaughter("multi-union", &lunion, &placement);
  (void)placedMunion;

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.SetPoolMultiplier(1);
  //  tester.RunToInBenchmark();
  //  tester.RunToOutBenchmark();
  return tester.RunBenchmark();
#else
  return 0;
#endif
}
