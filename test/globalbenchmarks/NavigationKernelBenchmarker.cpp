#include "volumes/utilities/VolumeUtilities.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/RNG.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/GlobalLocator.h"
#include "navigation/NavStatePool.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "base/Stopwatch.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NewSimpleNavigator.h"
#include "navigation/SimpleABBoxNavigator.h"
//#include "navigation/ZDCEMAbsorberNavigator.h"
#include "navigation/SimpleABBoxLevelLocator.h"
#include "navigation/HybridLevelLocator.h"
#include "navigation/HybridNavigator2.h"
#include "management/HybridManager2.h"

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#include <iostream>

#define CALLGRIND_ENABLED
#ifdef CALLGRIND_ENABLED
#include <valgrind/callgrind.h>
#endif

#ifdef CALLGRIND_ENABLED
#define RUNBENCH(NAME) \
    CALLGRIND_START_INSTRUMENTATION; \
    NAME; \
    CALLGRIND_STOP_INSTRUMENTATION; \
    CALLGRIND_DUMP_STATS
#else
   #define RUNBENCH(NAME) \
    NAME
#endif


using namespace vecgeom;


template <typename T>
__attribute__((noinline))
void benchNavigator(SOA3D<Precision> const & points,
                    SOA3D<Precision> const & dirs,
                    NavStatePool &inpool) {
  Precision *steps = new Precision[points.size()];
  NavigationState *newstate = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  Stopwatch timer;
  VNavigator *se = T::Instance();
  size_t hittargetchecksum=0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    steps[i] = se->ComputeStepAndPropagatedState(points[i], dirs[i], vecgeom::kInfinity, *inpool[i], *newstate);
    //   std::cerr << "** " << newstate->Top()->GetLabel() << "\n";
    hittargetchecksum+=(size_t)newstate->Top();
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
  }
  delete[] steps;
  std::cerr << "accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum << "\n";
  NavigationState::ReleaseInstance(newstate);
}

template <typename T>
__attribute__((noinline))
void benchVectorNavigator(SOA3D<Precision> const & points,
                          SOA3D<Precision> const & dirs,
                          NavStatePool &inpool, NavStatePool &outpool) {
  Precision *step_max = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  for (decltype(points.size()) i = 0; i < points.size(); ++i)
    step_max[i] = vecgeom::kInfinity;
  Precision *steps = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  Stopwatch timer;
  VNavigator *se = T::Instance();
  timer.Start();
  se->ComputeStepsAndPropagatedStates(points, dirs, step_max, inpool, outpool, steps);
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    // std::cerr << "---- " << steps[i] << "\n";
    accum += steps[i];
  }
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << "\n";
  _mm_free(steps);
  _mm_free(step_max);
}


void benchmarkOldNavigator(SOA3D<Precision> const & points,
                           SOA3D<Precision> const & dirs,
                           NavStatePool &inpool ){
  Precision *steps = new Precision[points.size()];
  NavigationState *newstate = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  Stopwatch timer;
  SimpleNavigator nav;
  size_t hittargetchecksum=0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    nav.FindNextBoundaryAndStep(points[i], dirs[i], *inpool[i], *newstate, vecgeom::kInfinity, steps[i]);
    //    std::cerr << "** " << newstate->Top()->GetLabel() << "\n";
    hittargetchecksum+=(size_t) newstate->Top();
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
  }
  delete[] steps;
  std::cerr << "accum  OldSimpleNavigator " << accum << " target checksum " << hittargetchecksum << "\n";
  NavigationState::ReleaseInstance(newstate);
}


void benchDifferentNavigators(SOA3D<Precision> const &points,
                              SOA3D<Precision> const &dirs,
                              NavStatePool &pool, NavStatePool &outpool){
  std::cerr << "##\n";
  //    RUNBENCH( benchNavigator<ZDCEMAbsorberNavigator>(points, dirs, pool) );
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<NewSimpleNavigator<false>>(points, dirs, pool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<SimpleABBoxNavigator<false>>(points, dirs, pool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<HybridNavigator<false>>(points, dirs, pool));
  std::cerr << "##\n";
  RUNBENCH(benchmarkOldNavigator(points, dirs, pool));
}

int main( int argc, char * argv[] )
{
  // read in detector passed as argument
  if (argc > 1) {
    RootGeoManager::Instance().set_verbose(3);
    RootGeoManager::Instance().LoadRootGeometry(std::string(argv[1]));
  } else {
    std::cerr << "please give a ROOT geometry file\n";
    return 1;
  }

  // setup data structures
  int npoints = 50000;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> localpoints(npoints);
  SOA3D<Precision> directions(npoints);
  NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth());
  NavStatePool statepoolout(npoints, GeoManager::Instance().getMaxDepth());

  // setup test points
  TGeoBBox const *rootbbox = dynamic_cast<TGeoBBox const *>(gGeoManager->GetTopVolume()->GetShape());
  Vector3D<Precision> bbox(rootbbox->GetDX(), rootbbox->GetDY(), rootbbox->GetDZ());



  // play with some heuristics to init level locators
  for (auto &element : GeoManager::Instance().GetLogicalVolumesMap()) {
    LogicalVolume *lvol = element.second;
    if (lvol->GetDaughtersp()->size() > 8) {
        //HybridManager2::Instance().InitStructure(lvol);
        lvol->SetLevelLocator(SimpleABBoxLevelLocator::GetInstance());
        //lvol->SetLevelLocator(HybridLevelLocator::GetInstance());
    }
  }



  std::string volname(argv[2]);
  volumeUtilities::FillGlobalPointsAndDirectionsForLogicalVolume<SOA3D<Precision>>(
      GeoManager::Instance().FindLogicalVolume(volname.c_str()), localpoints, points, directions, 0.4, npoints);
  std::cerr << "\n points filled\n";
  SimpleNavigator nav;
  for (unsigned int i = 0; i < points.size(); ++i) {
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], *(statepool[i]), true);
    if (statepool[i]->Top()->GetLogicalVolume() != GeoManager::Instance().FindLogicalVolume(volname.c_str())) {
      //
      std::cerr << "problem : point " << i << " probably in overlapping region \n";
      points.set(i, points[i - 1]);
      statepool[i - 1]->CopyTo(statepool[i]);
    }
  }
  HybridManager2::Instance().InitStructure(GeoManager::Instance().FindLogicalVolume(volname.c_str()));

  std::cerr << "located ...\n";
  benchDifferentNavigators(points, directions, statepool, statepoolout);
  return 0;
}
