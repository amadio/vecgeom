// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Benchmark for Tetrahedron
/// @file test/benchmark/TetBenchmark.cpp
/// @author Raman Sehgal

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tet.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  // OPTION_DOUBLE(r, 3.);
  double r = 25;
  Vector3D<double> p0(0., 0., 2.), p1(0., 0., 0.), p2(2., 0., 0.), p3(0., 2., 0.);
  UnplacedBox worldUnplaced = UnplacedBox(r * 4, r * 4, r * 4);
  UnplacedTet tetUnplaced   = UnplacedTet(p0, p1, p2, p3);
  LogicalVolume world("w0rld", &worldUnplaced);
  LogicalVolume tet("p4r4", &tetUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&tet, &placement);
  // world.PlaceDaughter(&tet, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(1);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
