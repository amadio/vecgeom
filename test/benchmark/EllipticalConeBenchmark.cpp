/*
 * EllipticalConeBenchmark.cpp
 *
 *  Created on: 15-Mar-2019
 *      Author: Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)
 */

#include "volumes/LogicalVolume.h"
#include "volumes/EllipticalCone.h"
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
