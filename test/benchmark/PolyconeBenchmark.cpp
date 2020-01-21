#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Global.h"
#include "ArgParser.h"
#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#endif

using namespace vecgeom;

// #define VISUALIZER

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(phistart, 0.);
  OPTION_DOUBLE(phidelta, kTwoPi);

  int Nz        = 4;
  double rmin[] = {0.1, 0., 0., 0.2};
  double rmax[] = {1., 2., 2., 1.5};
  double z[]    = {-1, -0.5, 0.5, 10};

  UnplacedBox worldUnplaced(5, 5, 15);
  auto pconUnplaced = GeoManager::MakeInstance<UnplacedPolycone>(phistart, phidelta, Nz, z, rmin, rmax);

  pconUnplaced->Print();

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume pcon("pcon", pconUnplaced);

  Transformation3D placement(0, 0, 0);
#if defined(VECGEOM_ROOT) and defined(VISUALIZER)
  VPlacedVolume const *vol =
#endif
      world.PlaceDaughter("pcon", &pcon, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetTolerance(1E-7);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetInsideBias(0.5);

  auto errcode = tester.RunBenchmark();

#ifdef VISUALIZER
  Visualizer visualizer;
  visualizer.AddVolume(*vol);
  if (tester.GetProblematicContainPoints().size() > 0) {
    for (auto v : tester.GetProblematicContainPoints()) {
      visualizer.AddPoint(v);

      // for debugging purpose
      std::cerr << " " << vol->Contains(v) << "\n";
      std::cout << v << "\n";
    }
    visualizer.Show();
  }
#endif

  return errcode;
}
