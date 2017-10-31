#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#ifdef VECGEOM_USOLIDS
#include "VUSolid.hh"
#endif

#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"

using namespace vecgeom;

// benchmarking any available shape (logical volume) found in a ROOT file
// usage: shape_testFromROOTFile detector.root logicalvolumename [option]
//   option : [--usolids(default)|--vecgeom]
// logicalvolumename should not contain trailing pointer information

bool usolids = false; // test VecGeom by default

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

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
  if (!foundvolume) {
    std::cerr << " NO SUCH VOLUME [" << testvolume << "] FOUND ... EXITING \n";
    errCode = 2048; // errCode: 1000 0000 0000
    return errCode;
  }

  LogicalVolume *converted = RootGeoManager::Instance().Convert(foundvolume);

  if (usolids) {
#ifndef VECGEOM_USOLIDS
    std::cerr << "\n*** ERROR: library built with -DUSOLIDS=OFF and user selected '-usolids true'!\n Aborting...\n\n";
    return 1;
#else
    // enforce USolids conventions
    VUSolid *shape = converted->Place(); //->ConvertToUSolids()->Clone();
    std::cerr << "\n==============Shape StreamInfo ========= \n";
    shape->StreamInfo(std::cerr);

    std::cerr << "\n=========Using USolids=========\n\n";
    return runTester<VUSolid>(shape, npoints, usolids, debug, stat);
#endif
  }

  else { // !usolids
    VPlacedVolume *shape = converted->Place();
    std::cerr << "\n=========Using VecGeom=========\n\n";
    shape->Print();
    return runTester<VPlacedVolume>(shape, npoints, usolids, debug, stat);
  }
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setConventionsMode(usolids);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetTestBoundaryErrors(false);
  tester.SetMaxPoints(npoints);
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================\n";

  if (shape) delete shape;
  return errcode;
}
