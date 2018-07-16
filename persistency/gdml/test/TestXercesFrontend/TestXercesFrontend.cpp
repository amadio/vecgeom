//!    \file TestXercesFrontendd.cpp
//!    \brief reads a gdml file to VecGeom
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include <iostream>
#include "Frontend.h"
#include "management/CppExporter.h"
#include "management/GeoManager.h"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestXercesFrontend <filename>.gdml \n"
               "      will use TestXercesFrontend.gdml by default\n"
            << std::endl;
}
} // namespace

int main(int argC, char *argV[])
{
  if (argC != 2) {
    usage();
  }
  auto const filename = std::string((argC > 1) ? argV[1] : "TestXercesFrontend.gdml");
  std::cout << "Now loading VecGeom geometry" << std::endl;
  auto const loaded = vgdml::Frontend::Load(filename);
  std::cout << "Loading VecGeom geometry done" << std::endl;
  std::cout << "TestXercesFrontend loaded with result \"" << (loaded ? "true" : "false") << "\"" << std::endl;
  std::cout << "Have depth " << vecgeom::GeoManager::Instance().getMaxDepth() << std::endl;
  std::vector<vecgeom::LogicalVolume *> v1;
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(v1);
  std::cout << "Have logical volumes " << v1.size() << std::endl;
  std::cout << "Have registered volumes (including virtual) "
            << vecgeom::GeoManager::Instance().GetRegisteredVolumesCount() << std::endl;
  std::vector<vecgeom::VPlacedVolume *> v2;
  vecgeom::GeoManager::Instance().getAllPlacedVolumes(v2);
  std::cout << "Have placed volumes " << v2.size() << std::endl;
  return !loaded;
}
