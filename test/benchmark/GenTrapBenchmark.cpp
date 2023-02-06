/*
 * GenTrapBenchmark.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 *      Modified: mihaela.gheata@cern.ch
 */
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/GenTrap.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Global.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_INT(type, 0);

  UnplacedBox worldUnplaced(10., 10., 10.);

  // twisted
  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  // no twist
  Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
  Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

  UnplacedGenTrap trapUnplaced(verticesx, verticesy, 10);
  UnplacedGenTrap trapUnplaced1(verticesx1, verticesy1, 10);
  UnplacedGenTrap *trapPtr = nullptr;
  switch (type) {
  case 0:
    std::cout << "________________________________________________\n"
                 " Testing twisted trapezoid for npoints = "
              << npoints << "\n________________________________________________" << std::endl;
    trapPtr = &trapUnplaced;
    break;
  case 1:
    std::cout << "________________________\n= Testing planar trapezoid for npoint s= " << npoints
              << "=\n________________________" << std::endl;
    trapPtr = &trapUnplaced1;
    break;
  default:
    std::cout << "Unknown trapezoid type" << std::endl;
    return 1;
  }
  trapPtr->Print();

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume trap("gentrap", trapPtr);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("gentrap", &trap, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(nrep);
  tester.SetPoolMultiplier(1); // set this if we want to compare results
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  return tester.RunBenchmark();
}
