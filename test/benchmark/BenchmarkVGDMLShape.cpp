#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include <iostream>

#ifdef VECGEOM_GDML
#include "Frontend.h"
#endif

using namespace vecgeom;

void print_usage()
{
  std::cout << "____________________________________________________________________________________\nUsage:  "
               "BenchmarkVGDMLShape -gdml_name <filename> -vol_name NAME "
               "[-npoints N] [-nrep R] [-tolerance T]\n\n        # use either a logical or a physical volume name \n"
               "        Example: BenchmarkVGDMLShape -gdml_name cms2018.gdml -pvol_name ESPM0x7f4a8f13de00 -npoints "
               "100000\n____________________________________________________________________________________\n";
}

int main(int argc, char *argv[])
{
  if (argc == 1) {
    print_usage();
    return 1;
  }
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_STRING(vol_name, "");
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(tolerance, 1.e-8);

  if (vol_name.size() == 0) {
    print_usage();
    std::cout << "You must provide a volume name\n";
  }

#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#else
  // Try to open the input file
  OPTION_INT(max_depth, 0);
  GeoManager::Instance().SetTransformationCacheDepth(max_depth);
  auto load = vgdml::Frontend::Load(gdml_name.c_str(), false);
  if (!load) return 2;
#endif

  VPlacedVolume *pvol = nullptr;
  std::vector<VPlacedVolume *> volumes;
  GeoManager::Instance().getAllPlacedVolumes(volumes);
  for (auto volume : volumes) {
    if ((vol_name != volume->GetName()) && (vol_name != volume->GetLogicalVolume()->GetName())) continue;
    pvol = volume;
    break;
  }

  if (!pvol) {
    std::cout << "### Did not find a volume named " << vol_name << "\n";
    return 2;
  }

  Benchmarker tester(pvol, true);
  tester.SetTolerance(tolerance);
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetPoolMultiplier(1);

  std::cout << "Testing logical volume " << pvol->GetLogicalVolume()->GetName() << " ...\n";
  return tester.RunBenchmark();
}
