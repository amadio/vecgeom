#ifndef VECGEOM_ENABLE_CUDA

#include "volumes/LogicalVolume.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Stopwatch.h"
#include <iostream>
#include "volumes/Extruded.h"

using namespace vecgeom;

#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_INT(nvert, 8);
  OPTION_INT(nsect, 2);
  OPTION_BOOL(convex,'t');
  
  double rmin = 10.;
  double rmax = 20.;
  
  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nvert];

  double phi = 2.* kPi/nvert;
  double r;
  for (int i=0; i<nvert; ++i) {
    r = rmax;
    if (i%2 > 0 && !convex) r = rmin;
    vertices[i].x = r * vecCore::math::Cos(i*phi);
    vertices[i].y = r * vecCore::math::Sin(i*phi);     
  }
  for (int i=0; i<nsect; ++i) {
    sections[i].fOrigin.Set(0,0,-20. + i*40./(nsect-1));
    sections[i].fScale = 1;
  }
    
  UnplacedBox worldUnplaced = UnplacedBox(20., 20., 20.);

  UnplacedExtruded xtru(nvert, vertices, nsect, sections);
  std::cout << "Benchmarking extruded polygon having " << vertices << " vertices and " << nsect << " sections\n";

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume xtruVol("xtru", &xtru);

  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("extruded", &xtruVol, &placement);

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
