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
#include "navigation/SimpleABBoxLevelLocator.h"
#include "navigation/HybridLevelLocator.h"
#include "navigation/HybridNavigator2.h"
#include "management/HybridManager2.h"
//#define BENCH_GENERATED_NAVIGATOR
#ifdef BENCH_GENERATED_NAVIGATOR
#include "navigation/GeneratedNavigator.h"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#include <iostream>
#include <set>
#include <sstream>

#undef NDEBUG
#include <cassert>

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

void NavStateUnitTest() {
  NavigationState *state1 = NavigationState::MakeInstance(10);
  NavigationState *state2 = NavigationState::MakeInstance(10);
  // test - 0 ( one empty path )
  state1->Clear();
  state2->Clear();
  state2->PushIndexType(1);
  assert(state1->Distance(*state2) == 1);

  // test - 1 ( equal paths )
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  assert(state1->RelativePath(*state2).compare("") == 0);
  assert(state1->Distance(*state2) == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 2
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(2);
  assert(state1->RelativePath(*state2).compare("/down/2") == 0);
  assert(state1->Distance(*state2) == 1);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 3
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(2);
  state2->PushIndexType(4);
  std::cerr << state1->RelativePath(*state2) << "\n";
  std::cerr << state1->Distance(*state2) << "\n";
  assert(state1->RelativePath(*state2).compare("/down/2/down/4") == 0);
  assert(state1->Distance(*state2) == 2);


  // test - 4
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state2->PushIndexType(1);
  std::cerr << "HUHU " << state1->Distance(*state2) << "\n";
  assert(state1->Distance(*state2) == 2);
  assert(state1->RelativePath(*state2).compare("/up/up") == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 5
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state2->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(5);
  state2->PushIndexType(1);
  std::cerr << state1->RelativePath(*state2) << "\n";
  assert(state1->RelativePath(*state2).compare("/up/horiz/3/down/1") == 0);
  assert(state1->Distance(*state2) == 4);

  // test - 6
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state1->PushIndexType(3);

  state2->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(5);
  state2->PushIndexType(1);
  state2->PushIndexType(1);
  std::cerr << state1->RelativePath(*state2) << "\n";
  assert(state1->RelativePath(*state2).compare("/up/up/horiz/3/down/1/down/1") == 0);
  assert(state1->Distance(*state2) == 6);
}

void analyseOutStates( NavStatePool & inpool, NavStatePool const & outpool ){
    std::set<VPlacedVolume const *> pset;
    std::set<LogicalVolume const *> lset;
    std::set<std::string > pathset;
    std::set<std::string > crossset;
    std::set<std::string > diffset;
    std::set<std::string > matrices;
    for (auto j = decltype(outpool.capacity()){0}; j < outpool.capacity(); ++j) {
      std::stringstream pathstringstream2;
      auto *navstate = outpool[j];
      navstate->printValueSequence(pathstringstream2);
      pset.insert(navstate->Top());
      lset.insert(navstate->Top()->GetLogicalVolume());
      pathset.insert(pathstringstream2.str());

      std::stringstream pathstringstream1;
      auto *instate = inpool[j];
      instate->printValueSequence(pathstringstream1);
      crossset.insert(pathstringstream1.str() + " -- " + pathstringstream2.str() );
      diffset.insert( instate->RelativePath(*navstate) );

      Transformation3D g;
      Transformation3D g2;
      Transformation3D invg2;

      instate->TopMatrix(g);
      navstate->TopMatrix(g2);
      g.SetProperties();
      g2.SetProperties();
      g2.Inverse(invg2);
      invg2.MultiplyFromRight(g);
      invg2.FixZeroes();
      std::stringstream matrixstream;
      invg2.Print(matrixstream);
      matrices.insert(matrixstream.str());
    }

    std::cerr << " size of diffset " << diffset.size() << "\n";
    std::cerr << " size of matrixset " << matrices.size() << "\n";
    std::cerr << " size of target pset " << pset.size() << "\n";
    std::cerr << " size of target lset " << lset.size() << "\n";
    std::cerr << " size of target state set " << pathset.size() << "\n";
    std::cerr << " total combinations " << crossset.size() << "\n";
    std::cerr << " normalized per input state " << crossset.size()/(1.*pathset.size()) << "\n";

    for(auto & s : crossset ){
      std::cerr << s << "\n";
    }
    for(auto & s : diffset ){
      std::cerr << s << "\n";
    }
    for(auto & s : matrices ){
         std::cerr << s << "\n";
       }
}

template <typename T>
__attribute__((noinline))
void benchNavigator(SOA3D<Precision> const & points,
                    SOA3D<Precision> const & dirs,
                    NavStatePool const &inpool, NavStatePool & outpool) {
  Precision *steps = new Precision[points.size()];
 // NavigationState *newstate = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  Stopwatch timer;
  VNavigator *se = T::Instance();
  size_t hittargetchecksum=0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    steps[i] = se->ComputeStepAndPropagatedState(points[i], dirs[i], vecgeom::kInfinity, *inpool[i], *outpool[i]);
    //   std::cerr << "** " << newstate->Top()->GetLabel() << "\n";
    // hittargetchecksum+=(size_t)outpool[i]->Top();
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
  }
  delete[] steps;
  std::cerr << "accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum << "\n";
 // NavigationState::ReleaseInstance(newstate);
}

template <typename T>
__attribute__((noinline))
void benchVectorNavigator(SOA3D<Precision> const & __restrict__ points,
                          SOA3D<Precision> const & __restrict__ dirs,
                          NavStatePool const & __restrict__ inpool, NavStatePool & __restrict__ outpool) {
  Precision *step_max = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  for (decltype(points.size()) i = 0; i < points.size(); ++i)
    step_max[i] = vecgeom::kInfinity;
  Precision *steps = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  Stopwatch timer;
  VNavigator *se = T::Instance();
  NavigationState const ** inpoolarray;
  NavigationState ** outpoolarray;
  inpool.ToPlainPointerArray(inpoolarray);
  outpool.ToPlainPointerArray(outpoolarray);
  timer.Start();
  se->ComputeStepsAndPropagatedStates(points, dirs, step_max, inpoolarray, outpoolarray, steps);
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  size_t hittargetchecksum=0L;
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    // std::cerr << "---- " << steps[i] << "\n";
    accum += steps[i];
    hittargetchecksum+=(size_t) outpool[i]->Top();
  }
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum << "\n";
  _mm_free(steps);
  _mm_free(step_max);
  delete[] inpoolarray;
  delete[] outpoolarray;
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
  RUNBENCH(benchNavigator<NewSimpleNavigator<false>>(points, dirs, pool, outpool));
  outpool.ToFile("simplenavoutpool.bin");
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<NewSimpleNavigator<true>>(points, dirs, pool,outpool));
  std::cerr << "##\n";
#ifdef BENCH_GENERATED_NAVIGATOR
  RUNBENCH(benchNavigator<GeneratedNavigator>(points, dirs, pool, outpool));
  outpool.ToFile("generatedoutpool.bin");
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<GeneratedNavigator>(points, dirs, pool, outpool));
  std::cerr << "##\n";
#endif
  RUNBENCH(benchVectorNavigator<NewSimpleNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<NewSimpleNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<SimpleABBoxNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<SimpleABBoxNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<SimpleABBoxNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<SimpleABBoxNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<HybridNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<HybridNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<HybridNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<HybridNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchmarkOldNavigator(points, dirs, pool));
  analyseOutStates(pool, outpool);
}

int main( int argc, char * argv[] )
{
    NavStateUnitTest();
  // read in detector passed as argument
  if (argc > 1) {
    RootGeoManager::Instance().set_verbose(3);
    RootGeoManager::Instance().LoadRootGeometry(std::string(argv[1]));
  } else {
    std::cerr << "please give a ROOT geometry file\n";
    return 1;
  }
  // setup data structures
  int npoints = 500000;
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
  HybridManager2::Instance().InitStructure(GeoManager::Instance().FindLogicalVolume(volname.c_str()));

  bool usecached=false;
  if( argc >= 4 && strcmp(argv[3], "--usecache") == 0 ) usecached=true;

  if (!usecached) {
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
    std::cerr << "located ...\n";

    points.ToFile("points.bin");
    directions.ToFile("directions.bin");
    statepool.ToFile("states.bin");
  } else {
    std::cerr << " loading points from cache \n";
    points.FromFile("points.bin");
    directions.FromFile("directions.bin");
    statepool.FromFile("states.bin");
  }
  benchDifferentNavigators(points, directions, statepool, statepoolout);
  return 0;
}
