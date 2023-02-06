#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/kernel/shapetypes/TubeTypes.h"
#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeomTest/Benchmarker.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 1024);

  UnplacedBox worldUnplaced(10., 10., 10.);
  LogicalVolume world("world", &worldUnplaced);

  // components for boolean solid
  UnplacedBox motherbox(5., 5., 5.);
  UnplacedBox box2(2, 2., 10.);
  // translation for boolean solid right shape ( it should now stick outside )
  Transformation3D translation(-2.5, 0, 3.5);

  VPlacedVolume *placedbox2      = (new LogicalVolume("", &box2))->Place(&translation);
  VPlacedVolume *placedmotherbox = (new LogicalVolume("", &motherbox))->Place();

  // now make the unplaced boolean solid
  UnplacedBooleanVolume<kUnion> booleansolid(kUnion, placedmotherbox, placedbox2);
  LogicalVolume booleanlogical("booleanL", &booleansolid);

  // place the boolean volume into the world

  // placement of boolean solid
  Transformation3D placement(5, 5, 5);

  // add this boolean solid to the world
  world.PlaceDaughter(&booleanlogical, &placement);
  GeoManager::Instance().SetWorldAndClose(world.Place());

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
