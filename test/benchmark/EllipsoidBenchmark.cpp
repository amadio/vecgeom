// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for Ellipsoid shape
/// @file test/benchmark/EllipsoidBenchmark.cpp
/// @author Evgueni Tcherniaev

#include "volumes/LogicalVolume.h"
#include "volumes/Ellipsoid.h"
#include "volumes/PlacedBox.h"
#include "volumes/UnplacedBox.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(dx, 3);
  OPTION_DOUBLE(dy, 4);
  OPTION_DOUBLE(dz, 5);
  OPTION_DOUBLE(zbottom, -4.5);
  OPTION_DOUBLE(ztop, 3.5);

  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  auto ellipsoidUnplaced    = GeoManager::MakeInstance<UnplacedEllipsoid>(dx, dy, dz, zbottom, ztop);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume ellipsoid("ellipsoid", ellipsoidUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("ellipsoid", &ellipsoid, &placement);

  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetTolerance(1e-7);
  tester.SetVerbosity(2);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
