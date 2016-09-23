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

//#define CALLGRIND_ENABLED
#ifdef CALLGRIND_ENABLED
#include <valgrind/callgrind.h>
#endif

#ifdef CALLGRIND_ENABLED
#define RUNBENCH(NAME)             \
  CALLGRIND_START_INSTRUMENTATION; \
  NAME;                            \
  CALLGRIND_STOP_INSTRUMENTATION;  \
  CALLGRIND_DUMP_STATS
#else
#define RUNBENCH(NAME) NAME
#endif

using namespace vecgeom;

bool gBenchVecInterface = false;
bool gAnalyseOutStates  = false;

void analyseOutStates(NavStatePool &inpool, NavStatePool const &outpool)
{
  std::set<VPlacedVolume const *> pset;
  std::set<LogicalVolume const *> lset;
  std::set<std::string> pathset;
  std::set<std::string> crossset;
  std::set<std::string> diffset;
  std::set<std::string> matrices;
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
    crossset.insert(pathstringstream1.str() + " -- " + pathstringstream2.str());
    diffset.insert(instate->RelativePath(*navstate));

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
  std::cerr << " normalized per input state " << crossset.size() / (1. * pathset.size()) << "\n";

  for (auto &s : crossset) {
    std::cerr << s << "\n";
  }
  for (auto &s : diffset) {
    std::cerr << s << "\n";
  }
  for (auto &s : matrices) {
    std::cerr << s << "\n";
  }
}

#ifdef VECGEOM_ROOT
__attribute__((noinline)) void benchmarkROOTNavigator(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs)
{

  TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
  auto nPoints           = points.size();
  TGeoBranchArray *instates[nPoints];
  TGeoBranchArray *outstates[nPoints];
  Precision *steps = new Precision[points.size()];
  // we don't have the input state container in ROOT form
  // we generate them but do not take this into account for the timing measurement
  for (size_t i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    rootnav->ResetState();
    rootnav->FindNode(pos.x(), pos.y(), pos.z());
    instates[i]  = TGeoBranchArray::MakeInstance(GeoManager::Instance().getMaxDepth());
    outstates[i] = TGeoBranchArray::MakeInstance(GeoManager::Instance().getMaxDepth());
    instates[i]->InitFromNavigator(rootnav);
  }
#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  Stopwatch timer;
  timer.Start();
  for (size_t i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];

    instates[i]->UpdateNavigator(rootnav);

    rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
    rootnav->SetCurrentDirection(dir.x(), dir.y(), dir.z());
    volatile TGeoNode *node = rootnav->FindNextBoundaryAndStep(kInfLength);
    (void)node;
    steps[i] = rootnav->GetStep();
    // safe output states ( for fair comparison with VecGeom )
    outstates[i]->InitFromNavigator(rootnav);
  }
  timer.Stop();
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
  }
  delete[] steps;

  // cleanup states
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    delete instates[i];
    delete outstates[i];
  }

  std::cerr << "accum  TGeo " << accum << " target checksum \n";
}
#endif

template <typename T>
__attribute__((noinline)) void benchNavigator(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs,
                                              NavStatePool const &inpool, NavStatePool &outpool)
{
  Precision *steps = new Precision[points.size()];
  // NavigationState *newstate = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  Stopwatch timer;
  VNavigator *se           = T::Instance();
  size_t hittargetchecksum = 0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    steps[i] = se->ComputeStepAndPropagatedState(points[i], dirs[i], vecgeom::kInfLength, *inpool[i], *outpool[i]);
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
__attribute__((noinline)) void benchVectorNavigator(SOA3D<Precision> const &__restrict__ points,
                                                    SOA3D<Precision> const &__restrict__ dirs,
                                                    NavStatePool const &__restrict__ inpool,
                                                    NavStatePool &__restrict__ outpool)
{
  Precision *step_max = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  for (decltype(points.size()) i = 0; i < points.size(); ++i)
    step_max[i]                  = vecgeom::kInfLength;
  Precision *steps               = (double *)_mm_malloc(sizeof(double) * points.size(), 32);
  Stopwatch timer;
  VNavigator *se = T::Instance();
  NavigationState const **inpoolarray;
  NavigationState **outpoolarray;
  inpool.ToPlainPointerArray(inpoolarray);
  outpool.ToPlainPointerArray(outpoolarray);
  timer.Start();
  se->ComputeStepsAndPropagatedStates(points, dirs, step_max, inpoolarray, outpoolarray, steps);
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  size_t hittargetchecksum = 0L;
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    // std::cerr << "---- " << steps[i] << "\n";
    accum += steps[i];
    hittargetchecksum += (size_t)outpool[i]->Top();
  }
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum
            << "\n";
  _mm_free(steps);
  _mm_free(step_max);
  delete[] inpoolarray;
  delete[] outpoolarray;
}

void benchmarkOldNavigator(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs, NavStatePool &inpool)
{
  Precision *steps          = new Precision[points.size()];
  NavigationState *newstate = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  Stopwatch timer;
  SimpleNavigator nav;
  size_t hittargetchecksum = 0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    nav.FindNextBoundaryAndStep(points[i], dirs[i], *inpool[i], *newstate, vecgeom::kInfLength, steps[i]);
    //    std::cerr << "** " << newstate->Top()->GetLabel() << "\n";
    hittargetchecksum += (size_t)newstate->Top();
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

void benchDifferentNavigators(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs, NavStatePool &pool,
                              NavStatePool &outpool)
{
  std::cerr << "##\n";
  //    RUNBENCH( benchNavigator<ZDCEMAbsorberNavigator>(points, dirs, pool) );
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<NewSimpleNavigator<false>>(points, dirs, pool, outpool));
  outpool.ToFile("simplenavoutpool.bin");
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<NewSimpleNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
#ifdef BENCH_GENERATED_NAVIGATOR
  RUNBENCH(benchNavigator<GeneratedNavigator>(points, dirs, pool, outpool));
  outpool.ToFile("generatedoutpool.bin");
  std::cerr << "##\n";
  RUNBENCH(benchVectorNavigator<GeneratedNavigator>(points, dirs, pool, outpool));
  std::cerr << "##\n";
#endif
  if (gBenchVecInterface) {
    RUNBENCH(benchVectorNavigator<NewSimpleNavigator<false>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
    RUNBENCH(benchVectorNavigator<NewSimpleNavigator<true>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
  }
  RUNBENCH(benchNavigator<SimpleABBoxNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<SimpleABBoxNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  if (gBenchVecInterface) {
    RUNBENCH(benchVectorNavigator<SimpleABBoxNavigator<false>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
    RUNBENCH(benchVectorNavigator<SimpleABBoxNavigator<true>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
  }
  RUNBENCH(benchNavigator<HybridNavigator<false>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  RUNBENCH(benchNavigator<HybridNavigator<true>>(points, dirs, pool, outpool));
  std::cerr << "##\n";
  if (gBenchVecInterface) {
    RUNBENCH(benchVectorNavigator<HybridNavigator<false>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
    RUNBENCH(benchVectorNavigator<HybridNavigator<true>>(points, dirs, pool, outpool));
    std::cerr << "##\n";
  }
  RUNBENCH(benchmarkOldNavigator(points, dirs, pool));
  std::cerr << "##\n";
#ifdef VECGEOM_ROOT
  benchmarkROOTNavigator(points, dirs);
#endif
  if (gAnalyseOutStates) {
    analyseOutStates(pool, outpool);
  }
}

int main(int argc, char *argv[])
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
  int npoints = 500000;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> localpoints(npoints);
  SOA3D<Precision> directions(npoints);
  NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth());
  NavStatePool statepoolout(npoints, GeoManager::Instance().getMaxDepth());

  // play with some heuristics to init level locators
  for (auto &element : GeoManager::Instance().GetLogicalVolumesMap()) {
    LogicalVolume *lvol = element.second;
    if (lvol->GetDaughtersp()->size() > 8) {
      // HybridManager2::Instance().InitStructure(lvol);
      lvol->SetLevelLocator(SimpleABBoxLevelLocator::GetInstance());
      // lvol->SetLevelLocator(HybridLevelLocator::GetInstance());
    }
  }

  std::string volname(argv[2]);
  auto lvol = GeoManager::Instance().FindLogicalVolume(volname.c_str());
  HybridManager2::Instance().InitStructure(lvol);

  // some output on volume
  std::cerr << "NavigationKernelBenchmarker run on " << argv[2] << " having " << lvol->GetDaughters().size()
            << " daughters \n";

  bool usecached = false;

  for (auto i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--usecache")) usecached = true;
    // benchmark vector interface?
    if (!strcmp(argv[i], "--vecbench")) gBenchVecInterface = true;
    // analyse state transitions
    if (!strcmp(argv[i], "--statetrans")) gAnalyseOutStates = true;
  }

  if (argc >= 4 && strcmp(argv[3], "--usecache") == 0) usecached = true;

  std::stringstream pstream;
  pstream << "points_" << argv[1] << "_" << argv[2] << ".bin";
  std::stringstream dstream;
  dstream << "directions_" << argv[1] << "_" << argv[2] << ".bin";
  std::stringstream statestream;
  statestream << "states_" << argv[1] << "_" << argv[2] << ".bin";
  if (!usecached) {
    volumeUtilities::FillGlobalPointsAndDirectionsForLogicalVolume<SOA3D<Precision>>(lvol, localpoints, points,
                                                                                     directions, 0.4, npoints);
    std::cerr << "\n points filled\n";
    for (unsigned int i = 0; i < points.size(); ++i) {
      GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], *(statepool[i]), true);
      if (statepool[i]->Top()->GetLogicalVolume() != lvol) {
        //
        std::cerr << "problem : point " << i << " probably in overlapping region \n";
        points.set(i, points[i - 1]);
        statepool[i - 1]->CopyTo(statepool[i]);
      }
    }
    std::cerr << "located ...\n";

    points.ToFile(pstream.str());
    directions.ToFile(dstream.str());
    statepool.ToFile(statestream.str());
  } else {
    std::cerr << " loading points from cache \n";
    points.FromFile(pstream.str());
    directions.FromFile(dstream.str());
    statepool.FromFile(statestream.str());
  }
  benchDifferentNavigators(points, directions, statepool, statepoolout);
  return 0;
}
