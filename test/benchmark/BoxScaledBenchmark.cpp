/// \file BoxScaledBenchmark.cpp
/// \author Mhaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/ScaledShape.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);

  OPTION_DOUBLE(dx, 1.);
  OPTION_DOUBLE(dy, 2.);
  OPTION_DOUBLE(dz, 3.);

  OPTION_DOUBLE(sx, 2.);
  OPTION_DOUBLE(sy, 2.);
  OPTION_DOUBLE(sz, 2.);

  Transformation3D placement = Transformation3D(0.1, 0, 0);
  UnplacedBox worldUnplaced  = UnplacedBox(dx * 4, dy * 4, dz * 4);
  UnplacedBox boxUnplaced    = UnplacedBox(dx, dy, dz);
  UnplacedScaledShape scaledUnplaced(&boxUnplaced, sx, sy, sz);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume scaled("scaledbOX", &scaledUnplaced);
  world.PlaceDaughter("pScaledBox", &scaled, &placement);
  //  world.PlaceDaughter("pScaledTube", &scaled, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
