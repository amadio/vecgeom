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

  std::vector<Vector3D<Precision> > vertexlist;
  // no twist
  vertexlist.push_back( Vector3D<Precision>(-3,-2.5, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-2.5, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3, 2.5, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 2.5,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-2,-2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-2, 2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 2, 2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 2,-2, 0 ) );

  UnplacedGenTrap trapUnplaced1( &vertexlist[0], 10 );
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
