#include <stdlib.h>
#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"

#include "TApplication.h"
#include "TCanvas.h"
#include "TView.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "VecGeom/management/RootGeoManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"

using namespace vecgeom;

// debugging any available shape (logical volume) found in a ROOT file
// usage: shape_debugFromROOTFile detector.root logicalvolumename x y z vx vy vz
// logicalvolumename should not contain trailing pointer information

double runTester(VPlacedVolume const *shape, Vector3D<double> const &point, Vector3D<double> const &dir)
{
  const char *sInside[3] = {"kInside", "kSurface", "kOutside"};
  double distance;
  auto inside = shape->Inside(point);
  std::cout << "Inside = " << sInside[inside - 1] << std::endl;
  distance                                = shape->SafetyToIn(point);
  if (distance < kHalfTolerance) distance = 0.0;
  std::cout << "SafetyFromOutside = " << distance << std::endl;
  distance                                = shape->SafetyToOut(point);
  if (distance < kHalfTolerance) distance = 0.0;
  std::cout << "SafetyFromInside = " << distance << std::endl;
  distance                                = shape->DistanceToIn(point, dir);
  if (distance < kHalfTolerance) distance = 0.0;
  std::cout << "DistanceToIn = " << distance << std::endl;
  distance                                = shape->DistanceToOut(point, dir);
  if (distance < kHalfTolerance) distance = 0.0;
  std::cout << "DistanceToOut = " << distance << std::endl;

  if (shape) delete shape;
  return distance;
}

void DrawArrow(Vector3D<double> const &point, Vector3D<double> const &dir, double size, double dout, short color)
{
  TPolyLine3D *pl = new TPolyLine3D(2);
  pl->SetLineColor(color);
  pl->SetNextPoint(point[0], point[1], point[2]);
  pl->SetNextPoint(point[0] + size * dir[0], point[1] + size * dir[1], point[2] + size * dir[2]);
  TPolyMarker3D *pm1 = new TPolyMarker3D(2);
  TPolyMarker3D *pm2 = new TPolyMarker3D(1);
  pm1->SetNextPoint(point[0], point[1], point[2]);
  pm1->SetNextPoint(point[0] + dout * dir[0], point[1] + dout * dir[1], point[2] + dout * dir[2]);
  pm2->SetNextPoint(point[0] + size * dir[0], point[1] + size * dir[1], point[2] + size * dir[2]);
  pm1->SetMarkerColor(2);
  pm2->SetMarkerColor(color);
  pm1->SetMarkerStyle(6);
  pm2->SetMarkerStyle(26);
  pm2->SetMarkerSize(0.4);
  pl->Draw();
  pm1->Draw();
  pm2->Draw();
}

TGeoNode *FindDaughter(TGeoVolume *vol, std::string &name, int &index)
{
  int nd          = vol->GetNdaughters();
  TGeoNode *found = nullptr;
  for (int i = 0; i < nd; ++i) {
    TGeoNode *node = vol->GetNode(i);
    if (name == node->GetVolume()->GetName()) {
      found = node;
      index = i;
      found->GetVolume()->SetVisibility(true);
      found->GetVolume()->SetTransparency(0);
      found->GetVolume()->SetLineColor(kRed);
      break;
    }
  }
  if (found) {
    for (int i = 0; i < nd; ++i) {
      if (vol->GetNode(i) != found) vol->GetNode(i)->GetVolume()->SetVisibility(false);
    }
  }
  return found;
}

int main(int argc, char *argv[])
{

  if (argc < 9) {
    std::cerr << "***** Error: need to give root geometry file, logical volume name. point and direction coordinates\n";
    std::cerr << "Ex: ./shape_testFromROOTFile ExN03.root mother daughter 1.3 2.5 3.6 0 0 1\n";
    exit(1);
  }

  TGeoManager::Import(argv[1]);
  if (!gGeoManager) return 1;
  std::string testvolume(argv[2]);
  std::string testdaughter(argv[3]);
  // Local point/direction (mm)
  double point[3];
  point[0] = 0.1 * atof(argv[4]);
  point[1] = 0.1 * atof(argv[5]);
  point[2] = 0.1 * atof(argv[6]);
  double lpoint[3];
  memcpy(lpoint, point, 3 * sizeof(double));
  double direction[3];
  direction[0] = atof(argv[7]);
  direction[1] = atof(argv[8]);
  direction[2] = atof(argv[9]);
  double ldir[3];
  memcpy(ldir, direction, 3 * sizeof(double));

  bool daughter                        = false;
  if (testdaughter != "void") daughter = true;

  int found                 = 0;
  TGeoVolume *foundvolume   = NULL;
  TGeoVolume *founddaughter = NULL;
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
  if (!foundvolume) {
    std::cerr << "Cannot find volume: " << testvolume << std::endl;
    return 1;
  }
  foundvolume->GetShape()->InspectShape();
  int index = -1;
  if (daughter) {
    TGeoNode *dnode = FindDaughter(foundvolume, testdaughter, index);
    if (!dnode) {
      std::cerr << "Cannot find daughter " << testdaughter << " of volume " << testvolume << std::endl;
      return 1;
    }
    founddaughter = dnode->GetVolume();

    std::cerr << "daughter found\n" << std::endl;
    founddaughter->GetShape()->InspectShape();
    dnode->GetMatrix()->MasterToLocal(point, lpoint);
    dnode->GetMatrix()->MasterToLocalVect(direction, ldir);
  }

  /*
    double master[3] = {-70.37002915950117,-193.7782403798211,-382.7815168187679};
    double masterdir[3] = {0.9910804931325202, 0.06730730580739153, 0.1150181842890534};
    TGeoNode *node = gGeoManager->FindNode(master[0], master[1], master[2]);
    if (node && (node->GetVolume() == foundvolume)) {
      std::cout << " == Path is correct\n";
      if (daughter) gGeoManager->CdDown(index);
      std::cout << "touchable: " << gGeoManager->GetPath() << std::endl;
      gGeoManager->GetCurrentMatrix()->MasterToLocal(master, lpoint);
      gGeoManager->GetCurrentMatrix()->MasterToLocalVect(masterdir, ldir);
    }
  */

  LogicalVolume *converted =
      (daughter) ? RootGeoManager::Instance().Convert(founddaughter) : RootGeoManager::Instance().Convert(foundvolume);

  VPlacedVolume *shape = converted->Place(); // VPlacedVolume
  std::cerr << "\n=========Using VecGeom=========\n\n";
  shape->Print();
  Vector3D<double> amin, amax;
  shape->Extent(amin, amax);
  double size = 0.2 * (amax - amin).Mag();

  Vector3D<double> lp(lpoint[0], lpoint[1], lpoint[2]);
  Vector3D<double> ld(ldir[0], ldir[1], ldir[2]);
  printf("local point: %.16f %.16f %.16f  local dir: %.16f %.16f %.16f\n", lp[0], lp[1], lp[2], ld[0], ld[1], ld[2]);

  double dout       = runTester(shape, lp, ld);
  TApplication *app = new TApplication("VecGeom Visualizer", nullptr, nullptr);
  TCanvas *c        = new TCanvas(foundvolume->GetName(), "", 1200, 800);
  gGeoManager->SetTopVisible();
  gGeoManager->SetVisLevel(1);
  foundvolume->SetTransparency(40);
  foundvolume->SetVisContainers();
  foundvolume->SetLineColor(kBlue);
  if (!daughter)
    foundvolume->DrawOnly();
  else
    foundvolume->Draw();
  DrawArrow(lp, ld, size, dout, kMagenta);
  c->GetView()->ShowAxis();
  app->Run();
  return 0;
}
