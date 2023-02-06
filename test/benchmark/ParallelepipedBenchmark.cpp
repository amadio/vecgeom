// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for Paralleliped.
/// @file test/benchmark/BenchmarktParallelepiped.cpp

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Parallelepiped.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 16384);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(dx, 20.);
  OPTION_DOUBLE(dy, 30.);
  OPTION_DOUBLE(dz, 40.);
  OPTION_DOUBLE(alpha, 30. / 180. * kPi);
  OPTION_DOUBLE(theta, 15. / 180. * kPi);
  OPTION_DOUBLE(phi, 30. / 180. * kPi);

  UnplacedBox worldUnplaced = UnplacedBox(dx * 4, dy * 4, dz * 4);
  UnplacedParallelepiped paraUnplaced(dx, dy, dz, alpha, theta, phi);
  LogicalVolume world("w0rld", &worldUnplaced);
  LogicalVolume para("p4r4", &paraUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&para, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
