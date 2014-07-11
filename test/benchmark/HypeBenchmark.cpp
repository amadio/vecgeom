/// \file HypeBenchmark.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Hype.h"
#include "volumes/Parallelepiped.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {
    
    UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
    UnplacedHype hypeUnplaced = UnplacedHype(2., 45., 5., 45., 7.);
    
    LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
    LogicalVolume hype = LogicalVolume("hype", &hypeUnplaced);
    world.PlaceDaughter(&hype, &Transformation3D::kIdentity);
    
    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().set_world(worldPlaced);
    
    Benchmarker tester(GeoManager::Instance().world());
    tester.SetVerbosity(3);
    tester.SetPointCount(128);
    tester.RunBenchmark();
    return 0;
}