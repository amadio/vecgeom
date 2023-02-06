/*
 * BenchmarkShapeFromROOTFile.cpp
 *
 *  Created on: Jan 26, 2015
 *      Author: swenzel
 */

#include "VecGeomTest/RootGeoManager.h"

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "VecGeomTest/Visualizer.h"
#include <string>
#include <cmath>
#include <iostream>

using namespace vecgeom;

// benchmarking any available shape (logical volume) found in a ROOT file
// usage: BenchmarkShapeFromROOTFile detector.root logicalvolumename
// logicalvolumename should not contain trailing pointer information

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << "geom.root logicalvolumename [--npoints n] [--nrep rep] [--verbosity 1|2|3|4]\n";
  }

  TGeoManager::Import(argv[1]);
  std::string testvolume(argv[2]);

  size_t npoints(10000);
  int verbosity(3);
  int repetitions(1);
  for (auto i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--npoints")) {
      if (i + 1 < argc) {
        npoints = atoi(argv[i + 1]);
      }
    }
    if (!strcmp(argv[i], "--nrep")) {
      if (i + 1 < argc) {
        repetitions = atoi(argv[i + 1]);
      }
    }
    if (!strcmp(argv[i], "--verbosity")) {
      if (i + 1 < argc) {
        verbosity = atoi(argv[i + 1]);
      }
    }
  }

  int found               = 0;
  TGeoVolume *foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  for (auto i = 0; i < vlist->GetEntries(); ++i) {
    TGeoVolume *vol = reinterpret_cast<TGeoVolume *>(vlist->At(i));
    std::string fullname(vol->GetName());

    std::size_t founds = fullname.compare(testvolume);
    if (founds == 0) {
      found++;
      foundvolume = vol;

      std::cerr << "found matching volume " << fullname << " of type " << vol->GetShape()->ClassName() << "\n";
    }
  }

  std::cerr << "volume found " << testvolume << " " << found << " times \n";
  // now get the VecGeom shape and benchmark it
  if (foundvolume) {
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " << foundvolume->GetShape()->Capacity() << "\n";
    LogicalVolume *converted = RootGeoManager::Instance().Convert(foundvolume);

    converted->Print();

    // get the bounding box
    TGeoBBox *bbox = dynamic_cast<TGeoBBox *>(foundvolume->GetShape());
    double bx, by, bz;
    double const *offset;
    if (bbox) {
      bx = bbox->GetDX();
      by = bbox->GetDY();
      bz = bbox->GetDZ();
      bbox->InspectShape();
      offset = bbox->GetOrigin();

      UnplacedBox worldUnplaced((bx + fabs(offset[0])) * 4, (by + fabs(offset[1])) * 4, (bz + fabs(offset[2])) * 4);
      LogicalVolume world("world", &worldUnplaced);
      // for the moment at origin
      Transformation3D placement(0, 0, 0);
      /*VPlacedVolume const *  vol =*/
      world.PlaceDaughter("testshape", converted, &placement);

      GeoManager::Instance().SetWorldAndClose(world.Place());

      Benchmarker tester(GeoManager::Instance().GetWorld());
      tester.SetTolerance(1E-4);
      tester.SetVerbosity(verbosity);
      tester.SetPoolMultiplier(1);
      tester.SetRepetitions(repetitions);
      tester.SetPointCount(npoints);

      tester.CompareMetaInformation();

      int returncodeIns   = tester.RunInsideBenchmark();
      int returncodeToIn  = tester.RunToInBenchmark();
      int returncodeToOut = tester.RunToOutBenchmark();

      // run boundary tests
      int rtrncodeInBdr   = tester.RunToInFromBoundaryBenchmark();
      int rtrncodeInBdrE  = tester.RunToInFromBoundaryExitingBenchmark();
      int rtrncodeOutBdr  = tester.RunToOutFromBoundaryBenchmark();
      int rtrncodeOutBdrE = tester.RunToOutFromBoundaryExitingBenchmark();

      return returncodeIns + returncodeToIn + returncodeToOut + rtrncodeInBdr + rtrncodeInBdrE + rtrncodeOutBdrE +
             rtrncodeOutBdr;
    }
    return 1;
  } else {
    std::cerr << " NO SUCH VOLUME [" << testvolume << "] FOUND ... EXITING \n";
    return 1;
  }
}
