// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for Trd
/// @file test/benchmark/TrdParallelepiped.cpp

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Trd.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

double dmax(double d1, double d2)
{
  if (d1 > d2) return d1;
  return d2;
}

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 50000);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(dx1, 3.);
  OPTION_DOUBLE(dx2, 4.);
  OPTION_DOUBLE(dy1, 5.);
  OPTION_DOUBLE(dy2, 6.);
  OPTION_DOUBLE(dz, 7.);

  UnplacedBox worldUnplaced = UnplacedBox(dmax(dx1, dx2) * 4, dmax(dy1, dy2) * 4, dz * 4);
  auto trdUnplaced          = GeoManager::MakeInstance<UnplacedTrd>(dx1, dx2, dy1, dy2, dz);
  // UnplacedTrd(dx1, dx2, dy1, dy2, dz);

  LogicalVolume world("world", &worldUnplaced);
  // LogicalVolume trd("trdLogicalVolume", &trdUnplaced);
  LogicalVolume trd("trdLogicalVolume", trdUnplaced);
  Transformation3D placement(5., 5., 5.);
  world.PlaceDaughter("trdPlaced", &trd, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetPoolMultiplier(1);
  return tester.RunBenchmark();
}
