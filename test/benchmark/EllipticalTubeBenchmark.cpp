// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for the Elliptical Tube
/// @file test/benchmark/EllipticalTubeBenchmark.cpp
/// @author Raman Sehgal

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/EllipticalTube.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(dx, 5);
  OPTION_DOUBLE(dy, 10);
  OPTION_DOUBLE(dz, 7);

  UnplacedBox worldUnplaced   = UnplacedBox(100., 100., 100.);
  auto ellipticalTubeUnplaced = GeoManager::MakeInstance<UnplacedEllipticalTube>(dx, dy, dz);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume ellipticalTube("ellipticalTube", ellipticalTubeUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("ellipticalTube", &ellipticalTube, &placement);

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
