//
// File:    shapeDebug.cpp
// Purpose: Similar to CompareDistances, user provide 6 floats at input:
//  a point (x,y,z) and a direction (vx,vy,vz), and the shape, point and
//  track is drawn using ROOT, from original point to next intersection with shape,
//  and all distances and safeties are compared with ROOT.
//  Note: ROOT is required for visualization.
//        Geant4 is also used when available, but they are not mandatory.
//
#include "VecGeom/management/RootGeoManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "utilities/Visualizer.h"
#include <string>
#include <cmath>
#include <iostream>

#ifdef VECGEOM_GEANT4
#include "G4ThreeVector.hh"
#include "G4VSolid.hh"
#endif

using namespace vecgeom;

int main(int argc, char *argv[])
{

  if (argc < 7) {
    std::cerr << "Fixed shape in source code - user needs to give a local point + local dir\n";
    std::cerr << "example: " << argv[0] << " 10.0 0.8 -3.5 1 0 0\n";
    return 1;
  }

  double px   = atof(argv[1]);
  double py   = atof(argv[2]);
  double pz   = atof(argv[3]);
  double dirx = atof(argv[4]);
  double diry = atof(argv[5]);
  double dirz = atof(argv[6]);

  TGeoShape *testShape = new TGeoBBox(10, 15, 20);
  // TGeoShape *testShape = new TGeoTubeSeg(3, 6, 2, 0, 360. * 0.6);
  // TGeoShape *testShape = new TGeoTrap(3, 6, 2, 0, 360. * 0.6);
  testShape->InspectShape();
  std::cerr << "volume capacity " << testShape->Capacity() << "\n";

  // now get the VecGeom shape and benchmark it
  if (testShape) {
    LogicalVolume *converted     = RootGeoManager::Instance().Convert(new TGeoVolume("testVolume", testShape));
    VPlacedVolume *vecgeomplaced = converted->Place();
    Vector3D<Precision> point(px, py, pz);
    Vector3D<Precision> dir(dirx, diry, dirz);
    // normalize direction vector
    dir = dir.Unit();

    SOA3D<Precision> pointcontainer(4);
    pointcontainer.resize(4);
    SOA3D<Precision> dircontainer(4);
    dircontainer.resize(4);
    Precision *output = new Precision[4];
    Precision *steps  = new Precision[4];

    if (argc > 9) {
      pointcontainer.set(0, point);
      dircontainer.set(0, dir.x(), dir.y(), dir.z());
      double px2   = atof(argv[7]);
      double py2   = atof(argv[8]);
      double pz2   = atof(argv[9]);
      double dirx2 = atof(argv[10]);
      double diry2 = atof(argv[11]);
      double dirz2 = atof(argv[12]);
      pointcontainer.set(1, px2, py2, pz2);
      dircontainer.set(1, dirx2, diry2, dirz2);
      pointcontainer.set(2, point);
      dircontainer.set(2, dir.x(), dir.y(), dir.z());
      pointcontainer.set(3, px2, py2, pz2);
      dircontainer.set(3, dirx2, diry2, dirz2);

      for (auto i = 0; i < 4; ++i) {
        steps[i] = vecgeom::kInfLength;
      }

    } else {
      for (auto i = 0; i < 4; ++i) {
        pointcontainer.set(i, point);
        dircontainer.set(i, dir.x(), dir.y(), dir.z());
        steps[i] = vecgeom::kInfLength;
      }
    }
    if (!dir.IsNormalized()) {
      std::cerr << "** Attention: Direction is not normalized **\n";
      std::cerr << "** Direction differs from 1 by "
                << std::sqrt(dir.x() * dir.x() + dir.y() * dir.y() + dir.z() * dir.z()) - 1. << "\n";
    }
    double dist;
    std::cout << "VecGeom Capacity " << vecgeomplaced->Capacity() << "\n";
    std::cout << "VecGeom CONTAINS " << vecgeomplaced->Contains(point) << "\n";
    std::cout << "VecGeom INSIDE " << vecgeomplaced->Inside(point) << "\n";
    dist = vecgeomplaced->DistanceToIn(point, dir);
    std::cout << "VecGeom DI " << dist << "\n";
    if (dist >= vecgeom::kTolerance && dist < vecgeom::kInfLength) {
      std::cout << "VecGeom INSIDE(p=p+dist*dir) " << vecgeomplaced->Inside(point + dir * dist) << "\n";
      if (vecgeomplaced->Inside(point + dir * dist) == vecgeom::kOutside)
        std::cout << "VecGeom Distance seems to be too big  DI(p=p+dist*dir,-dir) "
                  << vecgeomplaced->DistanceToIn(point + dir * dist, -dir) << "\n";
      if (vecgeomplaced->Inside(point + dir * dist) == vecgeom::kInside)
        std::cout << "VecGeom Distance seems to be too small DO(p=p+dist*dir,dir) "
                  << vecgeomplaced->DistanceToOut(point + dir * dist, dir) << "\n";
    }
    vecgeomplaced->DistanceToIn(pointcontainer, dircontainer, steps, output);
    std::cout << "VecGeom DI-V " << output[0] << "\n";
    std::cout << "VecGeom DO " << vecgeomplaced->DistanceToOut(point, dir) << "\n";
    vecgeomplaced->DistanceToOut(pointcontainer, dircontainer, steps, output);
    std::cout << "VecGeom DO-V " << output[0] << "\n";

    std::cout << "VecGeom SI " << vecgeomplaced->SafetyToIn(point) << "\n";
    std::cout << "VecGeom SO " << vecgeomplaced->SafetyToOut(point) << "\n";

    std::cout << "ROOT Capacity " << testShape->Capacity() << "\n";
    std::cout << "ROOT CONTAINS " << testShape->Contains(&Vector3D<double>(point)[0]) << "\n";
    std::cout << "ROOT DI " << testShape->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0])
              << "\n";
    std::cout << "ROOT DO " << testShape->DistFromInside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0])
              << "\n";
    std::cout << "ROOT SI " << testShape->Safety(&Vector3D<double>(point)[0], false) << "\n";
    std::cout << "ROOT SO " << testShape->Safety(&Vector3D<double>(point)[0], true) << "\n";

    TGeoShape const *rootback = vecgeomplaced->ConvertToRoot();
    if (rootback) {
      std::cout << "ROOTBACKCONV CONTAINS " << rootback->Contains(&Vector3D<double>(point)[0]) << "\n";
      std::cout << "ROOTBACKCONV DI "
                << rootback->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]) << "\n";
      std::cout << "ROOTBACKCONV DO "
                << rootback->DistFromInside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]) << "\n";
      std::cout << "ROOTBACKCONV SI " << rootback->Safety(&Vector3D<double>(point)[0], false) << "\n";
      std::cout << "ROOTBACKCONV SO " << rootback->Safety(&Vector3D<double>(point)[0], true) << "\n";
    } else {
      std::cerr << "ROOT backconversion failed\n";
    }

#ifdef VECGEOM_GEANT4
    G4ThreeVector g4p(point.x(), point.y(), point.z());
    G4ThreeVector g4d(dir.x(), dir.y(), dir.z());

    G4VSolid const *g4solid = vecgeomplaced->ConvertToGeant4();
    if (g4solid != NULL) {
      std::cout << "G4 CONTAINS " << g4solid->Inside(g4p) << "\n";
      std::cout << "G4 DI " << g4solid->DistanceToIn(g4p, g4d) << "\n";
      std::cout << "G4 DO " << g4solid->DistanceToOut(g4p, g4d) << "\n";
      std::cout << "G4 SI " << g4solid->DistanceToIn(g4p) << "\n";
      std::cout << "G4 SO " << g4solid->DistanceToOut(g4p) << "\n";
    } else {
      std::cerr << "G4 conversion failed\n";
    }
#endif

    double step = 0;
    //    if( testShape->Contains( &point[0] ) ){
    //      step = testShape->DistFromInside( &point[0], &dir[0] );
    //    }
    //    else {
    //      step = testShape->DistFromOutside( &point[0], &dir[0] );
    //    }

    // modified to show problem in DistanceToIn()
    step = testShape->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]);
    Visualizer visualizer;
    visualizer.AddVolume(*vecgeomplaced);
    visualizer.AddPoint(point);
    visualizer.AddLine(point, point + step * dir);
    visualizer.Show();
  } else {
    std::cerr << " Error: problems constructing volume [" << testShape << "] ... EXITING\n";
    return 1;
  }
}
