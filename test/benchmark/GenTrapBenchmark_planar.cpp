/*
 * GenTrapBenchmark.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#include "volumes/LogicalVolume.h"
#include "volumes/GenTrap.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "base/Vector3D.h"
#include "base/Global.h"
#include <vector>

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);

  std::vector<Vector3D<Precision>> vertexlist;
  // no twist
  Precision verticesx[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

  UnplacedGenTrap trapUnplaced1(verticesx, verticesy, 10);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume trap = LogicalVolume("gentrap", &trapUnplaced1);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("gentrap", &trap, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  //  tester.SetRepetitions(1);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetPointCount(1 << 10);
  tester.RunInsideBenchmark();
  tester.RunToInBenchmark();
}
