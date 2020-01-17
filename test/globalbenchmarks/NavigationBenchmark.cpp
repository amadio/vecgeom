/*
 *  File: NavigationBenchmark.cpp
 *
 *  Created on: Oct 25, 2014
 *      Author: swenzel, lima
 */

#undef VERBOSE
//#define VERBOSE

#ifdef VECGEOM_ROOT
#include "VecGeom/management/RootGeoManager.h"
#include "utilities/Visualizer.h"
#endif

#ifdef VECGEOM_GEANT4
#include "VecGeom/management/G4GeoManager.h"
#include "G4ThreeVector.hh"
// #include "G4TouchableHistoryHandle.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4GeometryManager.hh"
#endif

#include "VecGeom/benchmarking/NavigationBenchmarker.h"
#include "test/benchmark/ArgParser.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Orb.h"
#include "VecGeom/volumes/Trapezoid.h"

using namespace VECGEOM_NAMESPACE;

VPlacedVolume *SetupGeometry()
{

  UnplacedBox *worldUnplaced      = new UnplacedBox(10, 10, 10);
  UnplacedTrapezoid *trapUnplaced = new UnplacedTrapezoid(4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0);
  UnplacedBox *boxUnplaced        = new UnplacedBox(2, 2, 2);
  UnplacedOrb *orbUnplaced        = new UnplacedOrb(2.8);

  LogicalVolume *world = new LogicalVolume("world", worldUnplaced);
  LogicalVolume *trap  = new LogicalVolume("trap", trapUnplaced);
  LogicalVolume *box   = new LogicalVolume("box", boxUnplaced);
  LogicalVolume *orb   = new LogicalVolume("orb", orbUnplaced);

  Transformation3D *ident = new Transformation3D(0, 0, 0, 0, 0, 0);
  orb->PlaceDaughter("orb1", box, ident);
  trap->PlaceDaughter("box1", orb, ident);

  Transformation3D *placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-5, 5, 5, 0, 0, 0);   // 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);   // 0, 45,  0);
  Transformation3D *placement4 = new Transformation3D(5, 5, -5, 0, 0, 0);   // 0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5, 5, 0, 0, 0);  // 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5, 5, -5, 0, 0, 0);  // 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D(5, -5, -5, 0, 0, 0);  // 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 0, 0, 0); // 45, 45, 45);

  world->PlaceDaughter("trap1", trap, placement1);
  world->PlaceDaughter("trap2", trap, placement2);
  world->PlaceDaughter("trap3", trap, placement3);
  world->PlaceDaughter("trap4", trap, placement4);
  world->PlaceDaughter("trap5", trap, placement5);
  world->PlaceDaughter("trap6", trap, placement6);
  world->PlaceDaughter("trap7", trap, placement7);
  world->PlaceDaughter("trap8", trap, placement8);

  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();

  // cleanup
  delete ident;
  delete placement1;
  delete placement2;
  delete placement3;
  delete placement4;
  delete placement5;
  delete placement6;
  delete placement7;
  delete placement8;
  return w;
}

int main(int argc, char *argv[])
{
  OPTION_INT(ntracks, 1024);
  OPTION_INT(nreps, 3);
  OPTION_STRING(geometry, "navBench");
  OPTION_STRING(logvol, "world");
  OPTION_DOUBLE(bias, 0.8f);
#ifdef VECGEOM_ROOT
  OPTION_BOOL(vis, false);
#endif

  // default values used above are always printed.  If help true, stop now, so user will know which options
  // are available, and what the default values are.
  OPTION_BOOL(help, false);
  if (help) return 0;

#ifdef VECGEOM_ENABLE_CUDA
  // If CUDA enabled, then GPU hardware is required!
  int nDevice;
  cudaGetDeviceCount(&nDevice);

  if(nDevice > 0) {
    cudaDeviceReset();
  }
  else {
    std::cout << "\n ***** No Cuda Capable Device!!! *****\n" << std::endl;
    return 0;
  }
#else
  std::cerr<<"... VECGEOM_ENABLE_CUDA not defined at compilation?!?\n";
#endif

  if (geometry.compare("navBench") == 0) {
    SetupGeometry();

#ifdef VERBOSE
    //.. print geometry details
    for (auto &element : GeoManager::Instance().GetLogicalVolumesMap()) {
      auto *lvol = element.second;
      // lvol->SetLevelLocator(nullptr); // force-disable locators to test default GlobalLocator implementations
      std::cerr << "SetupBoxGeom(): logVol=" << lvol << ", name=" << lvol->GetName()
                << ", locator=" << (lvol->GetLevelLocator() ? lvol->GetLevelLocator()->GetName() : "NULL") << "\n";
    }

    std::vector<VPlacedVolume *> v1;
    GeoManager::Instance().getAllPlacedVolumes(v1);
    for (auto &plvol : v1) {
      std::cerr << "placedVol=" << plvol << ", name=" << plvol->GetName() << ", world=" << world << ", <"
                << world->GetName() << ", " << GeoManager::Instance().GetWorld() << ">\n";
    }

    //.. more details
    world->PrintContent();
    std::cerr << "\n";
#endif

#ifdef VECGEOM_ROOT
    // Exporting to ROOT file
    RootGeoManager::Instance().ExportToROOTGeometry(GeoManager::Instance().GetWorld(), "navBench.root");
    RootGeoManager::Instance().Clear();
#endif
  }

// Now try to read back in.  This is needed to make comparisons to VecGeom easily,
// since it builds VecGeom geometry based on the ROOT geometry and its TGeoNodes.
#ifdef VECGEOM_ROOT
  auto rootgeom = geometry + ".root";
  RootGeoManager::Instance().set_verbose(0);
  RootGeoManager::Instance().LoadRootGeometry(rootgeom.c_str());
#endif

#ifdef VECGEOM_GEANT4
  auto g4geom = geometry + ".gdml";
  G4GeoManager::Instance().LoadG4Geometry(g4geom.c_str());
#endif

// Visualization
#ifdef VECGEOM_ROOT
  if (vis) { // note that visualization block returns, excluding the rest of benchmark
    Visualizer visualizer;
    const VPlacedVolume *world = GeoManager::Instance().GetWorld();
    visualizer.AddVolume(*world);

    Vector<Daughter> const *daughters = world->GetLogicalVolume()->GetDaughtersp();
    for (size_t i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *daughter = (*daughters)[i];
      Transformation3D const &trf1  = *(daughter->GetTransformation());
      visualizer.AddVolume(*daughter, trf1);

      Vector<Daughter> const* daughters2 = daughter->GetLogicalVolume()->GetDaughtersp();
      for (int ii=0; ii < (int)daughters2->size(); ++ii) {
	VPlacedVolume const* daughter2 = (*daughters2)[ii];
	Transformation3D const& trf2 = *(daughter2->GetTransformation());
	Transformation3D comb = trf1;
	comb.MultiplyFromRight(trf2);
	visualizer.AddVolume(*daughter2, comb);
      }
    }

    std::vector<VPlacedVolume *> v1;
    GeoManager::Instance().getAllPlacedVolumes(v1);
    for (auto &plvol : v1) {
      std::cerr << "placedVol=" << plvol << ", name=" << plvol->GetName() << ", world=" << world << ", <"
                << world->GetName() << ", " << GeoManager::Instance().GetWorld() << ">\n";
    }

    // visualizer.Show();
    return 0;
  }
#endif

  std::cout << "\n*** Validating VecGeom navigation...\n";

  const LogicalVolume *startVolume = GeoManager::Instance().GetWorld()->GetLogicalVolume();
  if (logvol.compare("world") != 0) {
    startVolume = GeoManager::Instance().FindLogicalVolume(logvol.c_str());
  }

#ifdef VERBOSE
  std::cout << "NavigationBenchmark: logvol=<" << logvol << ">, startVolume=<"
            << (startVolume ? startVolume->GetLabel() : "NULL") << ">\n";
  if (startVolume) std::cout << *startVolume << "\n";
#endif

  // no more than about 1000 points used for validation
  int np = Min(ntracks, 1024);

  // prepare tracks to be used for benchmarking
  SOA3D<Precision> points(np);
  SOA3D<Precision> dirs(np);

  Vector3D<Precision> samplingVolume(10, 10, 10);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // run validation on subsample of np tracks
  Precision *maxSteps = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  for (int i = 0; i < np; ++i)
    maxSteps[i] = 10. * RNG::Instance().uniform();

  // Must be validated before being benchmarked
  bool ok = validateVecGeomNavigation(np, points, dirs, maxSteps);
  if (!ok) {
    std::cout << "VecGeom validation failed." << std::endl;
    //return 1;
  }
  else {
    std::cout << "VecGeom validation passed." << std::endl;
  }

  // on mic.fnal.gov CPUs, loop execution takes ~70sec for ntracks=10M
  while (np <= ntracks) {
    std::cout << "\n*** Running navigation benchmarks with ntracks=" << np << " and nreps=" << nreps << ".\n";
    runNavigationBenchmarks(startVolume, np, nreps, maxSteps, bias);
    np *= 8;
  }

  // cleanup
  vecCore::AlignedFree(maxSteps);
#ifdef VECGEOM_ROOT
  RootGeoManager::Instance().Clear();
#endif
  return 0;
}
