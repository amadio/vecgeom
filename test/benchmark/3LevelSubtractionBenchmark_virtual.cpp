#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/kernel/shapetypes/TubeTypes.h"
#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeomTest/Benchmarker.h"

using namespace vecgeom;

int main()
{
  UnplacedBox worldUnplaced(10., 10., 10.);
  LogicalVolume world("world", &worldUnplaced);

  // components for boolean solid
  UnplacedBox motherbox(5., 5., 5.);
  GenericUnplacedTube subtractedtube(0.5, 2., 2., 0, 2. * M_PI);

  // translation for boolean solid right shape
  Transformation3D translation(-2.5, 0, 0);

  // we will also subtract another small box
  UnplacedBox subtractedbox(1, 1, 1);
  Transformation3D translation2(4, 4, 4);

  VPlacedVolume *placedsubtractedtube = (new LogicalVolume("", &subtractedtube))->Place(&translation);
  VPlacedVolume *placedmotherbox      = (new LogicalVolume("", &motherbox))->Place();

  VPlacedVolume *placedsubtractedbox = (new LogicalVolume("", &subtractedbox))->Place(&translation2);

  // now make the unplaced boolean solid
  UnplacedBooleanVolume<kSubtraction> booleansolid(kSubtraction, placedmotherbox, placedsubtractedtube);
  LogicalVolume booleanlogical("booleanL", &booleansolid);

  UnplacedBooleanVolume<kSubtraction> booleansolid2(kSubtraction, booleanlogical.Place(), placedsubtractedbox);

  LogicalVolume booleanlogical2("booleanL2", &booleansolid2);

  // place the boolean volume into the world

  // placement of boolean solid
  Transformation3D placement(5, 5, 5);

  // add this boolean solid to the world
  world.PlaceDaughter(&booleanlogical2, &placement);

  GeoManager::Instance().SetWorldAndClose(world.Place());

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1 << 10);
  return tester.RunBenchmark();
}
