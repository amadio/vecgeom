/*
 * GenericPolyconeBenchmark.cpp
 *
 *  Created on: 15-Jan-2019
 *      Author: Raman Sehgal (raman.sehgal@cern.ch)
 */

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/GenericPolycone.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);
  OPTION_DOUBLE(sphi, 0.);
  OPTION_DOUBLE(dphi, kTwoPi);

  UnplacedBox worldUnplaced = UnplacedBox(50., 50., 50.);
  LogicalVolume world("world", &worldUnplaced);

  /* GenericPolycone as specified in jira issue 425 */
  const int numRZ1    = 10;
  double polycone_r[] = {1, 5, 3, 4, 9, 9, 3, 3, 2, 1};
  double polycone_z[] = {0, 1, 2, 3, 0, 5, 4, 3, 2, 1};
  auto genericPolyconeUnplaced =
      GeoManager::MakeInstance<UnplacedGenericPolycone>(sphi, dphi, numRZ1, polycone_r, polycone_z);

  LogicalVolume genericPolycone("genericPolycone", genericPolyconeUnplaced);

  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("genericPolycone", &genericPolycone, &placement);

  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetTolerance(1e-4);
  tester.SetVerbosity(2);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  // tester.SetVerbosity(3);
  int testerCode = tester.RunBenchmark();

  std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  std::cout << "Returned Tester code : " << testerCode << std::endl;
  std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  return testerCode;
}
