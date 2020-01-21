// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for Paraboloid
/// @file test/benchmark/ParaboloidScriptBenchmark.cpp
/// @author Marilena Bandieramonte

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Paraboloid.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 1024);
  OPTION_DOUBLE(rlo, 3.);
  OPTION_DOUBLE(rhi, 5.);
  OPTION_DOUBLE(dz, 7.);

  UnplacedBox worldUnplaced             = UnplacedBox(rhi * 4, rhi * 4, dz * 4);
  UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(rlo, rhi, dz); // rlo=3. - rhi=5. dz=7
  LogicalVolume world("MBworld", &worldUnplaced);
  LogicalVolume paraboloid("paraboloid", &paraboloidUnplaced);
  world.PlaceDaughter(&paraboloid, &Transformation3D::kIdentity);
  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);
  std::cout << "World set\n";

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
