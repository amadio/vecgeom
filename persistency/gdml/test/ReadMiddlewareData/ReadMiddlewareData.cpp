//!    \file TestMiddlewareData.cpp
//!    \brief reads a gdml file to VecGeom and demonstrates how to access material, region info and material-cut couple
//!    associations
//!
//!    \authors Author:  Andrei Gheata <andrei.gheata@cern.ch>
//!

#include <iostream>
#include "Frontend.h"
#include "Middleware.h"
#include "VecGeom/management/GeoManager.h"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    ReadMiddlewareData <filename>.gdml \n"
            << std::endl;
}
} // namespace

int main(int argC, char *argV[])
{
  if (argC != 2) {
    usage();
  }
  auto const filename = std::string((argC > 1) ? argV[1] : "cms2018.gdml");
  std::cout << "Now loading VecGeom geometry ... " << std::endl;
  auto const loadedMiddleware = vgdml::Frontend::Load(filename, false, 1); // mm unit is 1
  std::cout << "Geometry loaded with result: \"" << (loadedMiddleware ? "true" : "false") << "\"" << std::endl;
  if (!loadedMiddleware) return 1;

  auto const &aMiddleware = *loadedMiddleware;
  // Different maps can be retrieved from the middleware object
  auto materialMap    = aMiddleware.GetMaterialMap();
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();

  // Region cuts have length units incorporated, one needs to divide the value by the VecGeom millimeter unit
  double mmunit     = vecgeom::GeoManager::GetMillimeterUnit();
  auto const &lvmap = vecgeom::GeoManager::Instance().GetLogicalVolumesMap();

  // Iterate the material map and print all related info. See MaterialInfo.h for content.
  for (const auto &kv : volumeMatMap) {
    // The key is the logical volume id. Look-it up in the map of logicak volumes from GeoManager
    // Note: we could search also by material name using the materialMap
    auto lvpair = lvmap.find(kv.first);
    if (lvpair == lvmap.end()) {
      std::cout << "Error in ReadMiddlewareData: Cannot find logical volume with id = " << kv.first << std::endl;
      return 1;
    }
    auto lv                    = lvpair->second;
    vgdml::Material const &mat = kv.second;

    std::cout << lv->GetName() << " (id = " << kv.first << "): material " << mat.name << std::endl;
    if (mat.attributes.size()) std::cout << "  attributes:\n";
    for (const auto &attv : mat.attributes)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.fractions.size()) std::cout << "  fractions:\n";
    for (const auto &attv : mat.fractions)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.components.size()) std::cout << "  components:\n";
    for (const auto &attv : mat.components)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
  }

  return 0;
}
