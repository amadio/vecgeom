/// \file HypeBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

//.. Geant4 giving discrepant results for the Hype shape
#undef VECGEOM_GEANT4

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Hype.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#define PI 3.141592653589793
using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 10);
  OPTION_DOUBLE(rmin, 10.);
  OPTION_DOUBLE(rmax, 20.);
  OPTION_DOUBLE(sin, PI / 5.);
  OPTION_DOUBLE(sout, PI / 3.);
  OPTION_DOUBLE(dz, 50);

  UnplacedBox worldUnplaced = UnplacedBox(rmax * 4, rmax * 4, dz * 4);
  auto hypeUnplaced         = GeoManager::MakeInstance<UnplacedHype>(rmin, rmax, sin, sout, dz);
  LogicalVolume world("w0rld", &worldUnplaced);
  LogicalVolume hype("p4r4", hypeUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  // world.PlaceDaughter(&hype, &placement);
  world.PlaceDaughter(&hype, &Transformation3D::kIdentity);
  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);
  Benchmarker tester(GeoManager::Instance().GetWorld());
  // tester.SetTolerance(1e-5);
  tester.SetVerbosity(1);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
