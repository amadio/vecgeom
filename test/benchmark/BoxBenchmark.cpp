#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(dx, 1.);
  OPTION_DOUBLE(dy, 2.);
  OPTION_DOUBLE(dz, 3.);

  UnplacedBox worldUnplaced = UnplacedBox(dx * 4, dy * 4, dz * 4);
  UnplacedBox boxUnplaced   = UnplacedBox(dx, dy, dz);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume box("box", &boxUnplaced);

  Transformation3D placement(0.1, 0, 0);
  world.PlaceDaughter("box", &box, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetPoolMultiplier(1);
  return tester.RunBenchmark();
}
