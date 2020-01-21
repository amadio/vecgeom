// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Benchmark for the Orb.
/// \file test/benchmark/OrbBenchmark.cpp
/// \author Raman Sehgal

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Orb.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(r, 3.);

  UnplacedBox worldUnplaced = UnplacedBox(r * 4, r * 4, r * 4);
  UnplacedOrb orbUnplaced   = UnplacedOrb(r);
  LogicalVolume world("w0rld", &worldUnplaced);
  LogicalVolume orb("p4r4", &orbUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&orb, &placement);
  // world.PlaceDaughter(&orb, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
