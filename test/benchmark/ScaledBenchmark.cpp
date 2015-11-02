/// \file ScaledBenchmark.cpp
/// \author Mhaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Tube.h"
#include "volumes/ScaledShape.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "management/CppExporter.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);

  OPTION_DOUBLE(rmin,0);
  OPTION_DOUBLE(rmax,5);
  OPTION_DOUBLE(dz,10);
  OPTION_DOUBLE(sphi,0);
  OPTION_DOUBLE(dphi,kTwoPi);

  OPTION_DOUBLE(sx,0.5);
  OPTION_DOUBLE(sy,1.3);
  OPTION_DOUBLE(sz,1.);

  Transformation3D placement = Transformation3D(1, 1, 1);
  UnplacedBox worldUnplaced = UnplacedBox(10,10,10);
  UnplacedTube tubeUnplaced = UnplacedTube(rmin, rmax, dz, sphi, dphi);
  UnplacedScaledShape scaledUnplaced(&tubeUnplaced, sx, sy, sz);
  
  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume scaled = LogicalVolume("scaledTube", &scaledUnplaced);
  world.PlaceDaughter("pScaledTube", &scaled, &placement);
//  world.PlaceDaughter("pScaledTube", &scaled, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);
  GeomCppExporter::Instance().DumpGeometry( std::cout );

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
