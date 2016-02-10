#include "volumes/LogicalVolume.h"
#include "volumes/GenTrap.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "base/Vector3D.h"
#include "base/Global.h"
#include <vector>

using namespace vecgeom;

int main() {
  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 20.);

  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  UnplacedGenTrap trapUnplaced1(verticesx, verticesy, 10);
  trapUnplaced1.Print();

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume trap = LogicalVolume("gentrap", &trapUnplaced1);

  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("gentrap", &trap, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(1);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetPointCount(1000000);
  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  tester.SetToInBias(0.8);
  tester.RunToInBenchmark();
  // tester.RunBenchmark();
}
