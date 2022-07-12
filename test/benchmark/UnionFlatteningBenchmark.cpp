// Purpose: Simple Unit test for flattening Boolean volumes
#include "VecGeom/base/Config.h"

#ifndef VECGEOM_ENABLE_CUDA

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>
#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/volumes/BooleanVolume.h"

using namespace vecgeom;

#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(opt, 0);
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_INT(nsolids, 100);

  constexpr double size = 10.;

  UnplacedBox worldUnplaced = UnplacedBox(2 * size, 2 * size, 2 * size);

  double sized                = size * std::pow(0.5 / nsolids, 1. / 3.);
  VPlacedVolume **placedBoxes = new VPlacedVolume *[nsolids];
  for (int i = 0; i < nsolids; ++i) {
    Vector3D<double> pos(RNG::Instance().uniform(-size, size), RNG::Instance().uniform(-size, size),
                         RNG::Instance().uniform(-size, size));
    double sizernd = RNG::Instance().uniform(0.8 * sized, 1.2 * sized);

    Transformation3D *trans =
        (i == 0) ? new Transformation3D()
                 : new Transformation3D(pos.x(), pos.y(), pos.z(), RNG::Instance().uniform(-180, 180),
                                        RNG::Instance().uniform(-180, 180), RNG::Instance().uniform(-180, 180));
    trans->SetProperties();
    UnplacedBox *box = new UnplacedBox(sizernd, sizernd, sizernd);
    std::string name = "box";
    name += std::to_string(i);
    LogicalVolume *lbox = new LogicalVolume(name.c_str(), box);
    placedBoxes[i]      = lbox->Place(trans);
  }

  std::cout << "Benchmarking flattened union of " << nsolids << " random boxes\n";
  LogicalVolume world("world", &worldUnplaced);
  VPlacedVolume *last  = placedBoxes[0];
  LogicalVolume *lbool = nullptr;
  for (int i = 1; i < nsolids; i++) {
    UnplacedBooleanVolume<kUnion> *pair = new UnplacedBooleanVolume<kUnion>(kUnion, last, placedBoxes[i]);
    if (opt && i == nsolids - 1)
      lbool = new LogicalVolume(pair->Flatten());
    else
      lbool = new LogicalVolume(pair);
    last    = lbool->Place();
  }

  Transformation3D placement(0, 0, 0);
  const VPlacedVolume *placedBool = world.PlaceDaughter("bool-union", lbool, &placement);
  (void)placedBool;

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
