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
#include "base/Vector3D.h"
#include "base/Global.h"
#include <vector>

using namespace vecgeom;

int main()
{
  UnplacedBox worldUnplaced(10., 10., 10.);

  // no twist
  Precision verticesx[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

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
  tester.SetRepetitions(1);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetTolerance(1E-8);
  tester.SetPointCount(1000000);
  tester.SetToInBias(0.8);
  return tester.RunBenchmark();
}
