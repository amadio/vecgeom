/// \file ScaledBenchmark.cpp
/// \author Mhaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/ScaledShape.h"
#include "VecGeomBenchmark/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/management/CppExporter.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);

  OPTION_DOUBLE(rmin, 0);
  OPTION_DOUBLE(rmax, 5);
  OPTION_DOUBLE(dz, 10);
  OPTION_DOUBLE(sphi, 0);
  OPTION_DOUBLE(dphi, kTwoPi);

  OPTION_DOUBLE(sx, 0.5);
  OPTION_DOUBLE(sy, 1.3);
  OPTION_DOUBLE(sz, 1.);

  Transformation3D placement = Transformation3D(1, 1, 1);
  UnplacedBox worldUnplaced  = UnplacedBox(10, 10, 10);
  GenericUnplacedTube tubeUnplaced(rmin, rmax, dz, sphi, dphi);
  UnplacedScaledShape scaledUnplaced(&tubeUnplaced, sx, sy, sz);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume scaled("scaledTube", &scaledUnplaced);
  world.PlaceDaughter("pScaledTube", &scaled, &placement);
  //  world.PlaceDaughter("pScaledTube", &scaled, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);
  GeomCppExporter::Instance().DumpGeometry(std::cout);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
