/*
 * NavigationSpecializerTest.cpp
 *
 *  Created on: 11.09.2015
 *      Author: swenzel
 */

#include "source/NavigationSpecializer.h"
#include "VecGeom/management/RootGeoManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace vecgeom;

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " geometryfile.root volumename [--loopunroll] [--basenav BasicNavigator]\n";
    return 1;
  }

  vecgeom::RootGeoManager::Instance().LoadRootGeometry(argv[1]);

  // we assume a naming convention as in NavigationKernelBenchmarker
  std::stringstream instatestream;
  instatestream << "states_" << argv[1] << "_" << argv[2] << ".bin";
  std::stringstream outstatestream;
  outstatestream << "outstates_" << argv[1] << "_" << argv[2] << "_simple.bin";

  vecgeom::NavigationSpecializer specializer(instatestream.str(), outstatestream.str());
  for (auto i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--basenav") == 0) {
      std::cerr << "setting a basenav\n";
      specializer.SetBaseNavigator(argv[i + 1]);
    }
  }

  for (auto i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--loopunroll") == 0) specializer.EnableLoopUnrolling();
  }

  std::ofstream outputfile;
  std::stringstream outputname;
  // TODO: think about making this an command line option
  outputname << argv[2] << "Navigator.h";
  outputfile.open(outputname.str());
  specializer.ProduceSpecializedNavigator(vecgeom::GeoManager::Instance().FindLogicalVolume(argv[2]), outputfile);
  outputfile.close();
  return 0;
}
