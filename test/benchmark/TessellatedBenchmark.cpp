#ifndef VECGEOM_ENABLE_CUDA

#include "volumes/LogicalVolume.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Stopwatch.h"
#include <iostream>
#include "volumes/Tessellated.h"
#include "test/core/TessellatedOrb.h"

using namespace vecgeom;

#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(ngrid, 100);
  constexpr double r = 10.;

  UnplacedBox worldUnplaced = UnplacedBox(20., 20., 20.);

  UnplacedTessellated tsl = UnplacedTessellated();
  // Create the tessellated solid
  size_t nfacets = TessellatedOrb(r, ngrid, tsl);
  tsl.Close();
  std::cout << "Benchmarking tessellated sphere having " << nfacets << " facets\n";

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume tessellated("tessellated", &tsl);

  Transformation3D placement(0, 0, 0);
  const VPlacedVolume *placedTsl = world.PlaceDaughter("tessellated", &tessellated, &placement);
  (void)placedTsl;

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.SetPoolMultiplier(1);

  //  tester.RunToInBenchmark();
  //  tester.RunToOutBenchmark();
  return tester.RunBenchmark();
#else
  return 0;
#endif
}
