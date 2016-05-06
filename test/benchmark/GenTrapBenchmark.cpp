/*
 * GenTrapBenchmark.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 *      Modified: mihaela.gheata@cern.ch
 */
#include "volumes/LogicalVolume.h"
#include "volumes/GenTrap.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Global.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints, 32768);
  OPTION_INT(nrep, 4);

  UnplacedBox worldUnplaced(10., 10., 20.);

  // twisted
  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  UnplacedGenTrap trapUnplaced(verticesx, verticesy, 10);
  trapUnplaced.Print();

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume trap("gentrap", &trapUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("gentrap", &trap, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(nrep);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.RunBenchmark();
}
