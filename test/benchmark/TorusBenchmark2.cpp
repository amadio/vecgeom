#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Torus2.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Global.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(drmin, 1.2);
  OPTION_DOUBLE(drmax, 3.1);
  OPTION_DOUBLE(drtor, 5.);
  OPTION_DOUBLE(dsphi, 0.);
  OPTION_DOUBLE(ddphi, kTwoPi);

  UnplacedBox worldUnplaced((drtor + drmax) * 2, (drtor + drmax) * 2, (drtor + drmax) * 2);
  UnplacedTorus2 torusUnplaced(drmin, drmax, drtor, dsphi, ddphi);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume torus("torus", &torusUnplaced);

  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("torus", &torus, &placement);
  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetTolerance(10. * vecgeom::kTolerance);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
