/// \file SphereBenchmark.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Sphere.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 10);
  OPTION_DOUBLE(rmin, 15.);
  OPTION_DOUBLE(rmax, 20.);
  OPTION_DOUBLE(sphi, 0.);
  // OPTION_DOUBLE(dphi, 2 * kPi / 3.);
  OPTION_DOUBLE(dphi, 2 * kPi);
  // OPTION_DOUBLE(stheta, kPi / 4.);
  // OPTION_DOUBLE(dtheta, kPi / 6.);
  OPTION_DOUBLE(stheta, 0.);
  OPTION_DOUBLE(dtheta, kPi);

  UnplacedBox worldUnplaced = UnplacedBox(rmax * 4, rmax * 4, rmax * 4);
  auto sphereUnplaced       = GeoManager::MakeInstance<UnplacedSphere>(rmin, rmax, sphi, dphi, stheta, dtheta);

  LogicalVolume world("w0rld", &worldUnplaced);
  LogicalVolume sphere("p4r4", sphereUnplaced);
  Transformation3D placement = Transformation3D(5, 5, 5);
  world.PlaceDaughter(&sphere, &placement);
  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetTolerance(1e-6);
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
