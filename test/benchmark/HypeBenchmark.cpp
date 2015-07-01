/// \file HypeBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Hype.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

#define PI 3.141592653589793
using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,10);
  OPTION_DOUBLE(rmin,10.);
  OPTION_DOUBLE(rmax,20.);
  OPTION_DOUBLE(sin,PI/6.);
  OPTION_DOUBLE(sout,PI/4.);
  OPTION_DOUBLE(dz,50);


  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, dz*4);
  UnplacedHype hypeUnplaced = UnplacedHype(rmin,rmax,sin,sout,dz);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume hype = LogicalVolume("p4r4", &hypeUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  //world.PlaceDaughter(&hype, &placement);
  world.PlaceDaughter(&hype, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());

  //tester.SetTolerance(1e-5);
  tester.SetVerbosity(3);
  //if(strcmp(argv[1],"show")==0)
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);

  tester.RunInsideBenchmark();
  tester.RunToOutBenchmark();
  //tester.RunToInBenchmark();
  //tester.RunBenchmark();


  return 0;
}