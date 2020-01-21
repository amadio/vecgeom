// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for the Elliptical Cone.
/// @file test/benchmark/EllipticalConeBenchmark.cpp
/// @author Raman Sehgal, Evgueni Tcherniaev

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/EllipticalCone.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(dx, 0.5);
  OPTION_DOUBLE(dy, 0.4);
  OPTION_DOUBLE(h, 20);
  OPTION_DOUBLE(zcut, 10);

  UnplacedBox worldUnplaced   = UnplacedBox(100., 100., 100.);
  auto ellipticalConeUnplaced = GeoManager::MakeInstance<UnplacedEllipticalCone>(dx, dy, h, zcut);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume ellipticalCone("ellipticalCone", ellipticalConeUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("ellipticalCone", &ellipticalCone, &placement);

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
