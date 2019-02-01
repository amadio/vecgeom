/*
 * EllipticalConeBenchmark.cpp
 *
 *  Created on: 15-Jan-2019
 *      Author: Raman Sehgal (raman.sehgal@cern.ch)
 */

#include "volumes/LogicalVolume.h"
#include "volumes/EllipticalCone.h"
#include "volumes/PlacedBox.h"
#include "volumes/UnplacedBox.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_INT(nrep, 1);

  /*
   * Set the required Parameters of EllipticalCone
   * as mentioned for Cone below
   */
  /*OPTION_DOUBLE(dx, 5);
  OPTION_DOUBLE(dy, 10);
  OPTION_DOUBLE(dz, 7);*/

  UnplacedBox worldUnplaced   = UnplacedBox(100., 100., 100.);
  /*
   * Add the required elliptical cone parameter as done for Cone below
   */
  //auto ellipticalConeUnplaced = GeoManager::MakeInstance<UnplacedEllipticalCone>(dx, dy, dz);

  /* Currently there is only default constructor present so i am doing as below, but later
   * this should be removed
   */
  auto ellipticalConeUnplaced = GeoManager::MakeInstance<UnplacedEllipticalCone>();


  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume ellipticalCone("ellipticalCone", ellipticalConeUnplaced);

  Transformation3D placement(5, 0, 0);
  world.PlaceDaughter("ellipticalCone", &ellipticalCone, &placement);

  VPlacedVolume *worldPlaced = world.Place();
  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetTolerance(1e-7);
  tester.SetVerbosity(2);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  return tester.RunBenchmark();
}
