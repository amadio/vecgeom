/*
 * ConeBenchmark.cpp
 *
 *  Created on: Jun 19, 2014
 *      Author: swenzel
 */

#include "volumes/LogicalVolume.h"
#include "volumes/Cone.h"
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
  OPTION_DOUBLE(rmin1, 5);
  OPTION_DOUBLE(rmax1, 10);
  OPTION_DOUBLE(rmin2, 7);
  OPTION_DOUBLE(rmax2, 15);
  OPTION_DOUBLE(dz, 10);
  OPTION_DOUBLE(sphi, 0);
  OPTION_DOUBLE(dphi, kTwoPi);

  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  auto coneUnplaced         = GeoManager::MakeInstance<UnplacedCone>(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume cone("cone", coneUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("cone", &cone, &placement);

  // now the cone is placed; how do we get it back?
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
