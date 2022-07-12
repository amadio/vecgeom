#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/CutTube.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int benchmark(double rmin, double rmax, double dz, double sphi, double dphi, double thb, double phib, double tht,
              double phit, int npoints, int nrep)
{
  Vector3D<double> nbottom(std::sin(thb) * std::cos(phib), std::sin(thb) * std::sin(phib), std::cos(thb));
  Vector3D<double> ntop(std::sin(tht) * std::cos(phit), std::sin(tht) * std::sin(phit), std::cos(tht));
  UnplacedBox worldUnplaced = UnplacedBox(rmax * 4, rmax * 4, dz * 4);
  UnplacedCutTube ctubUnplaced(rmin, rmax, dz, sphi, dphi, nbottom, ntop);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume ctube("ctube", &ctubUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("ctube", &ctube, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetTolerance(1.e-6);
  tester.SetVerbosity(2);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(rmin, 3);
  OPTION_DOUBLE(rmax, 5);
  OPTION_DOUBLE(dz, 10);
  OPTION_DOUBLE(sphi, 0);
  OPTION_DOUBLE(dphi, 2 * kPi / 3);
  OPTION_DOUBLE(thb, 3 * kPi / 4);
  OPTION_DOUBLE(phib, kPi / 3);
  OPTION_DOUBLE(tht, kPi / 4);
  OPTION_DOUBLE(phit, 2 * kPi / 3);

  return benchmark(rmin, rmax, dz, sphi, dphi, thb, phib, tht, phit, npoints, nrep);
}
