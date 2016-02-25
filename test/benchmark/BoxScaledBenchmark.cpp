/// \file BoxScaledBenchmark.cpp
/// \author Mhaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Tube.h"
#include "volumes/ScaledShape.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,10024);
  OPTION_INT(nrep, 4);

  OPTION_DOUBLE(dx,1.);
  OPTION_DOUBLE(dy,2.);
  OPTION_DOUBLE(dz,3.);
  

  OPTION_DOUBLE(sx,2.);
  OPTION_DOUBLE(sy,2.);
  OPTION_DOUBLE(sz,2.);

  Transformation3D placement = Transformation3D(0.1, 0, 0);
  UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);
  UnplacedBox boxUnplaced = UnplacedBox(dx, dy, dz);
  UnplacedScaledShape scaledUnplaced(&boxUnplaced, sx, sy, sz);
  
  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume scaled = LogicalVolume("scaledbOX", &scaledUnplaced);
  world.PlaceDaughter("pScaledBox", &scaled, &placement);
//  world.PlaceDaughter("pScaledTube", &scaled, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
//  tester.RunInsideBenchmark();
//  tester.RunToOutBenchmark();
  tester.RunBenchmark();

 return 0;
}
