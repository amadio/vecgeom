#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"

#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"

using namespace vecgeom;

// benchmarking any available shape (logical volume) found in a ROOT file
// usage: shape_testFromROOTFile detector.root logicalvolumename [option]
//   option : [--usolids(default)|--vecgeom] [--vis]
// logicalvolumename should not contain trailing pointer information

bool usolids = false; // test VecGeom by default

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  if (argc < 3) {
    std::cerr << "***** Error: need to give root geometry file and logical volume name\n";
    std::cerr << "Ex: ./shape_testFromROOTFile ExN03.root World\n";
    exit(1);
  }

  TGeoManager::Import(argv[1]);
  std::string testvolume(argv[2]);

  for (auto i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--usolids")) {
#ifndef VECGEOM_REPLACE_USOLIDS
      usolids = true;
#else
      std::cerr << "\n*** --usolids choice in a USolids --> VecGeom mode.  Testing VecGeom shapes instead!\n";
      std::cerr << "Press <Enter> to continue...";
      std::cin.ignore();
      std::cerr << "\n";
      usolids = false;
#endif
    } else if (!strcmp(argv[i], "--vecgeom")) {
      usolids = false;
    }
  }

  int errCode             = 0;
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

  std::cerr << "volume found " << found << " times \n";
  foundvolume->GetShape()->InspectShape();
  std::cerr << "volume capacity " << foundvolume->GetShape()->Capacity() << "\n";

  // now get the shape and benchmark it
  if (foundvolume) {
    LogicalVolume *converted = RootGeoManager::Instance().Convert(foundvolume);

    int errCode = 0;
    if (usolids) {
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
      VUSolid *shape = converted->Place()->ConvertToUSolids()->Clone();
#else
      VUSolid *shape = converted->Place();
#endif
      std::cerr << "\n==============Shape StreamInfo ========= \n";
      shape->StreamInfo(std::cerr);

      std::cerr << "\n=========Using USolids=========\n\n";
      ShapeTester<VUSolid> tester;
      tester.setConventionsMode(usolids);
      tester.setDebug(debug);
      tester.setStat(stat);
      tester.SetMaxPoints(npoints);
      tester.SetSolidTolerance(1.e-9);
      tester.SetTestBoundaryErrors(true);
      errCode = tester.Run(shape);
      std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << " ("
                << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
      std::cout << "=========================================================" << std::endl;
    }

    else { // !usolids
      VPlacedVolume *shape = converted->Place();
      std::cerr << "\n=========Using VecGeom=========\n\n";
      ShapeTester<VUSolid> tester;
      tester.setConventionsMode(usolids);
      tester.setDebug(debug);
      tester.setStat(stat);
      tester.SetMaxPoints(npoints);
      tester.SetSolidTolerance(1.e-9);
      tester.SetTestBoundaryErrors(true);
      errCode = tester.Run(shape);
      std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errCode << " ("
                << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
      std::cout << "=========================================================" << std::endl;
    }
    return errCode;
  } else {
    std::cerr << " NO SUCH VOLUME [" << testvolume << "] FOUND ... EXITING \n";

    errCode = 2048; // errCode: 1000 0000 0000
    return errCode;
  }
}
