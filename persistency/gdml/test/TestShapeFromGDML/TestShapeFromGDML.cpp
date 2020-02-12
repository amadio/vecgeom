//!    \file TestShapeFromGDML
//!    \brief reads a gdml file to VecGeom
//!
//!    \authors Author:  mihaela.gheata@cern.ch
//!

#include <iostream>
#include "Frontend.h"
#include "VecGeom/management/CppExporter.h"
#include "VecGeom/management/GeoManager.h"
#include "../../../../test/benchmark/ArgParser.h"
#include "../../../../test/shape_tester/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestShapeFromGDML <filename>.gdml [npoints]\n"
            << std::endl;
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{

  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";

  if (shape) delete shape;
  return errcode;
}
} // namespace

int main(int argc, char *argv[])
{
  if (argc < 2) {
    usage();
    return 1;
  }
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  auto const filename = std::string(argv[1]);
  std::cout << "Now loading VecGeom geometry from " << filename << std::endl;
  auto const loaded = vgdml::Frontend::Load(filename);
  std::cout << "Have depth " << vecgeom::GeoManager::Instance().getMaxDepth() << std::endl;

  auto &geoManager = vecgeom::GeoManager::Instance();
  std::vector<vecgeom::LogicalVolume *> v1;
  geoManager.GetAllLogicalVolumes(v1);
  std::cout << "Have logical volumes " << v1.size() << std::endl;
  std::vector<vecgeom::VPlacedVolume *> v2;
  geoManager.getAllPlacedVolumes(v2);
  std::cout << "Have placed volumes " << v2.size() << std::endl;
  if (!loaded) return 1;

  vecgeom::VPlacedVolume const *vol = geoManager.GetWorld()->GetDaughters().operator[](0);
  return runTester<vecgeom::VPlacedVolume>(vol, npoints, debug, stat);
}
