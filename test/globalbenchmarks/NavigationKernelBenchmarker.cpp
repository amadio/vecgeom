#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/navigation/NavStatePool.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/RootGeoManager.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/SimpleABBoxLevelLocator.h"
#include "VecGeom/navigation/HybridLevelLocator.h"
#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/management/HybridManager2.h"
#ifdef VECGEOM_EMBREE
#include "VecGeom/navigation/EmbreeNavigator.h"
#endif
//#define BENCH_GENERATED_NAVIGATOR
#ifdef BENCH_GENERATED_NAVIGATOR
#include "VecGeom/navigation/GeneratedNavigator.h"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "VecGeom/management/G4GeoManager.h"
#endif

#include <iostream>
#include <set>
#include <sstream>
#include <dlfcn.h>

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
bool gBenchWithSafety   = false;
bool gSpecializedLib    = false;
Precision gMAXSTEP      = vecgeom::kInfLength; // global variable to configure max step asked in ComputeStep

std::string gSpecLibName;
VNavigator const *gSpecializedNavigator;

void InitNavigators()
{
  for (auto &lvol : GeoManager::Instance().GetLogicalVolumesMap()) {
    if (lvol.second->GetDaughtersp()->size() < 4) {
      lvol.second->SetNavigator(NewSimpleNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 5) {
      lvol.second->SetNavigator(SimpleABBoxNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 10) {
      lvol.second->SetNavigator(HybridNavigator<>::Instance());
      HybridManager2::Instance().InitStructure((lvol.second));
    }
    if (lvol.second->GetDaughtersp()->size() >= 10) {
      lvol.second->SetNavigator(HybridNavigator<>::Instance());
      HybridManager2::Instance().InitStructure((lvol.second));
    }

    if (lvol.second->ContainsAssembly()) {
      lvol.second->SetLevelLocator(SimpleAssemblyAwareABBoxLevelLocator::GetInstance());
    } else {
      lvol.second->SetLevelLocator(SimpleABBoxLevelLocator::GetInstance());
    }
  }
}

void InitSpecializedNavigators(std::string libname)
{
  void *handle;
  handle = dlopen(libname.c_str(), RTLD_NOW);
  if (!handle) {
    std::cerr << "Error loading navigator shared lib: " << dlerror() << "\n";
    std::cerr << "doing nothing ... \n";
    return;
  }

  // the create detector "function type":
  typedef void (*InitFunc_t)();

  // find entry symbol correct symbol in lib
  // TODO: get rid of hard coded name (which might not be portable)
  InitFunc_t init = (InitFunc_t)dlsym(handle, "_Z25InitSpecializedNavigatorsv");

  if (init != nullptr) {
    // call the init function which is going to set specific navigators as
    // compiled in by the user
    init();
  } else {
    std::cerr << "Init specialized navigators from shared lib failed; symbol not found\n";
  }
}

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
template <bool WithSafety = false, bool WithReloc = true>
__attribute__((noinline)) void benchmarkROOTNavigator(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs)
{

  TGeoNavigator *rootnav      = ::gGeoManager->GetCurrentNavigator();
  auto nPoints                = points.size();
  TGeoBranchArray **instates  = new TGeoBranchArray *[nPoints];
  TGeoBranchArray **outstates = new TGeoBranchArray *[nPoints];
  Precision *steps            = new Precision[points.size()];
  Precision *safeties;
  if (WithSafety) {
    safeties = new Precision[points.size()];
  }
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
    rootnav->ResetState();
    instates[i]->UpdateNavigator(rootnav);

    rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
    rootnav->SetCurrentDirection(dir.x(), dir.y(), dir.z());
    if (WithSafety) {
      safeties[i] = rootnav->Safety(true);
    }
    if (WithReloc) {
      volatile TGeoNode *node = rootnav->FindNextBoundaryAndStep(gMAXSTEP);
      (void)node;
    } else {
      volatile TGeoNode *node = rootnav->FindNextBoundary(gMAXSTEP);
      (void)node;
    }
    steps[i] = rootnav->GetStep();
    if (WithReloc) {
      // save output states ( for fair comparison with VecGeom )
      outstates[i]->InitFromNavigator(rootnav);
    }
  }
  timer.Stop();
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  std::cerr << timer.Elapsed() << "\n";
  Precision accum(0.);
  Precision saccum(0.);
  size_t hittargetchecksum = 0L;
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
    if (WithSafety) {
      saccum += safeties[i];
    }
    // target checksum via the table held from RootGeoManager
    std::cerr << "i/size=" << i << '/' << points.size() << "\n";
    hittargetchecksum +=
        (size_t)RootGeoManager::Instance().Lookup(outstates[i]->GetNode(outstates[i]->GetLevel()))->id();
  }
  delete[] steps;
  if (WithSafety) {
    delete[] safeties;
  }

  // cleanup states
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    delete instates[i];
    delete outstates[i];
  }
  delete[] instates;
  delete[] outstates;

  std::cerr << "accum  TGeo " << accum << " target checksum " << hittargetchecksum << "\n";
  if (WithSafety) {
    std::cerr << "safety accum  TGeo " << saccum << "\n";
  }
}

#ifdef VECGEOM_GEANT4
template <bool WithSafety = false>
__attribute__((noinline)) void benchmarkG4Navigator(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs)
{
  G4VPhysicalVolume *world(vecgeom::G4GeoManager::Instance().GetG4GeometryFromROOT());
  if (world != nullptr) G4GeoManager::Instance().LoadG4Geometry(world);

  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm = 10.; // cm --> mm conversion
  G4Navigator &g4nav = *(G4GeoManager::Instance().GetNavigator());
  // G4TouchableHistory **g4history = new G4TouchableHistory *[nPoints];

  Precision *steps = new Precision[points.size()];

  // get a time estimate to just to locate points
  // (The reason is that the G4Navigator has a huge internal state and it is not foreseen to do
  //  multi-track processing with a basked of G4TouchableHistories as states)
  Stopwatch timer;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    G4ThreeVector g4pos(points[i].x() * cm, points[i].y() * cm, points[i].z() * cm);
    G4ThreeVector g4dir(dirs[i].x(), dirs[i].y(), dirs[i].z());
    // false --> locate from top
    g4nav.LocateGlobalPointAndSetup(g4pos, &g4dir, false);
  }
  Precision timeForLocate = (Precision)timer.Stop();

#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    G4ThreeVector g4pos(points[i].x() * cm, points[i].y() * cm, points[i].z() * cm);
    G4ThreeVector g4dir(dirs[i].x(), dirs[i].y(), dirs[i].z());
    G4double maxStep = gMAXSTEP;

    // false --> locate from top
    G4VPhysicalVolume const *vol = g4nav.LocateGlobalPointAndSetup(g4pos, &g4dir, false);
    (void)vol;
    G4double safety = 0.0;
    steps[i]        = g4nav.ComputeStep(g4pos, g4dir, maxStep, safety);

    G4ThreeVector nextPos = g4pos + (steps[i] + 1.0e-6) * g4dir;
    // TODO: save touchable history array - returnable?  symmetrize with ROOT/VECGEOM benchmark
    g4nav.SetGeometricallyLimitedStep();

    volatile G4VPhysicalVolume const *nextvol = g4nav.LocateGlobalPointAndSetup(nextPos);
    (void)nextvol;
  }
  timer.Stop();
  std::cerr << (Precision)timer.Elapsed() - timeForLocate << "\n";
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  // cleanup
  // delete[] g4history;
  //_mm_free(maxSteps);
  Precision accum{0.};
  size_t hittargetchecksum{0L};
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
    if (WithSafety) {
      // saccum += safeties[i];
    }
    // target checksum via the table held from RootGeoManager
    // hittargetchecksum +=
    //    (size_t)RootGeoManager::Instance().Lookup(outstates[i]->GetNode(outstates[i]->GetLevel()))->id();
  }
  std::cerr << "accum  G4 " << accum / cm << " target checksum " << hittargetchecksum << "\n";
}
#endif // end if G4
#endif // end if ROOT

template <bool WithSafety = true>
__attribute__((noinline)) void benchNavigator(VNavigator const *se, SOA3D<Precision> const &points,
                                              SOA3D<Precision> const &dirs, NavStatePool const &inpool,
                                              NavStatePool &outpool)
{
  Precision *steps = new Precision[points.size()];
  Precision *safeties;
  if (WithSafety) safeties = new Precision[points.size()];
  Stopwatch timer;
  size_t hittargetchecksum = 0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    if (WithSafety) {
      steps[i] = se->ComputeStepAndSafetyAndPropagatedState(points[i], dirs[i], gMAXSTEP, *inpool[i], *outpool[i], true,
                                                            safeties[i]);
    } else {
      steps[i] = se->ComputeStepAndPropagatedState(points[i], dirs[i], gMAXSTEP, *inpool[i], *outpool[i]);
    }
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  Precision accum(0.), saccum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
    if (WithSafety) {
      saccum += safeties[i];
    }
    if (outpool[i]->Top()) hittargetchecksum += (size_t)outpool[i]->Top()->id();
  }
  delete[] steps;
  std::cerr << "accum  " << se->GetName() << " " << accum << " target checksum " << hittargetchecksum << "\n";
  if (WithSafety) {
    std::cerr << "saccum  " << se->GetName() << " " << saccum << "\n";
  }
}

// version benchmarking navigation without relocation
template <bool WithSafety = true>
__attribute__((noinline)) void benchNavigatorNoReloc(VNavigator const *se, SOA3D<Precision> const &points,
                                                     SOA3D<Precision> const &dirs, NavStatePool const &inpool,
                                                     NavStatePool &outpool)
{
  Precision *steps = new Precision[points.size()];
  Precision *safeties;
  if (WithSafety) safeties = new Precision[points.size()];
  Stopwatch timer;
  size_t hittargetchecksum = 0L;
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    if (WithSafety) {
      steps[i] = se->ComputeStepAndSafety(points[i], dirs[i], gMAXSTEP, *const_cast<NavigationState *>(inpool[i]), true,
                                          safeties[i]);
    } else {
      steps[i] = se->ComputeStep(points[i], dirs[i], gMAXSTEP, *inpool[i], *outpool[i]);
    }
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  Precision accum(0.), saccum(0.);
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    accum += steps[i];
    if (WithSafety) {
      saccum += safeties[i];
    }
    if (outpool[i]->Top()) hittargetchecksum += (size_t)outpool[i]->Top()->id();
  }
  delete[] steps;
  std::cerr << "accum  " << se->GetName() << " " << accum << " target checksum " << hittargetchecksum << "\n";
  if (WithSafety) {
    std::cerr << "saccum  " << se->GetName() << " " << saccum << "\n";
  }
}

template <typename T, bool WithSafety = true>
__attribute__((noinline)) void benchVectorNavigator(SOA3D<Precision> const &__restrict__ points,
                                                    SOA3D<Precision> const &__restrict__ dirs,
                                                    NavStatePool const &__restrict__ inpool,
                                                    NavStatePool &__restrict__ outpool)
{
  Precision *step_max = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    step_max[i] = gMAXSTEP;
  }
  Precision *steps    = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
  Precision *safeties = nullptr;
  bool *calcs;
  if (WithSafety) {
    safeties = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
    calcs    = (bool *)vecCore::AlignedAlloc(32, sizeof(bool) * points.size());
    for (decltype(points.size()) i = 0; i < points.size(); ++i)
      calcs[i] = true;
  }

  Stopwatch timer;
  VNavigator *se = T::Instance();
  NavigationState const **inpoolarray;
  NavigationState **outpoolarray;
  inpool.ToPlainPointerArray(inpoolarray);
  outpool.ToPlainPointerArray(outpoolarray);
  timer.Start();
  if (WithSafety) {
    se->ComputeStepsAndSafetiesAndPropagatedStates(points, dirs, step_max, inpoolarray, outpoolarray, steps, calcs,
                                                   safeties);
  } else {
    se->ComputeStepsAndPropagatedStates(points, dirs, step_max, inpoolarray, outpoolarray, steps);
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  Precision accum(0.);
  Precision saccum(0.);
  size_t hittargetchecksum = 0L;
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    // std::cerr << "---- " << steps[i] << "\n";
    accum += steps[i];
    if (WithSafety) {
      saccum += safeties[i];
    }
    if (outpool[i]->Top()) hittargetchecksum += (size_t)outpool[i]->Top()->id();
  }
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum
            << "\n";
  if (WithSafety) {
    std::cerr << "VECTOR saccum  " << T::GetClassName() << " " << saccum << "\n";
    vecCore::AlignedFree(safeties);
  }
  vecCore::AlignedFree(steps);
  vecCore::AlignedFree(step_max);
  delete[] inpoolarray;
  delete[] outpoolarray;
}

template <typename T, bool WithSafety = true>
__attribute__((noinline)) void benchVectorNavigatorNoReloc(SOA3D<Precision> const &__restrict__ points,
                                                           SOA3D<Precision> const &__restrict__ dirs,
                                                           NavStatePool const &__restrict__ inpool,
                                                           NavStatePool &__restrict__ outpool)
{
  Precision *step_max = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    step_max[i] = gMAXSTEP;
  }
  Precision *steps    = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
  Precision *safeties = nullptr;
  bool *calcs;

  if (WithSafety) {
    safeties = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * points.size());
    calcs    = (bool *)vecCore::AlignedAlloc(32, sizeof(bool) * points.size());
    for (decltype(points.size()) i = 0; i < points.size(); ++i)
      calcs[i] = true;
  }

  Stopwatch timer;
  VNavigator *se = T::Instance();
  NavigationState const **inpoolarray;
  NavigationState **outpoolarray;
  inpool.ToPlainPointerArray(inpoolarray);
  outpool.ToPlainPointerArray(outpoolarray);
  timer.Start();
  if (WithSafety) {
    se->ComputeStepsAndSafeties(points, dirs, step_max, inpoolarray, steps, calcs, safeties);
  } else {
    // nothing to do
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  Precision accum(0.);
  Precision saccum(0.);
  size_t hittargetchecksum = 0L;
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    // std::cerr << "---- " << steps[i] << "\n";
    accum += steps[i];
    if (WithSafety) {
      saccum += safeties[i];
    }
    if (outpool[i]->Top()) hittargetchecksum += (size_t)outpool[i]->Top()->id();
  }
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << " target checksum " << hittargetchecksum
            << "\n";
  if (WithSafety) {
    std::cerr << "VECTOR saccum  " << T::GetClassName() << " " << saccum << "\n";
    vecCore::AlignedFree(safeties);
  }
  vecCore::AlignedFree(steps);
  vecCore::AlignedFree(step_max);
  delete[] inpoolarray;
  delete[] outpoolarray;
}

template <bool WithSafety = false>
void benchDifferentNavigators(SOA3D<Precision> const &points, SOA3D<Precision> const &dirs, NavStatePool &pool,
                              NavStatePool &outpool, std::string outfilenamebase)
{
  std::cerr << "##\n";
  if (gSpecializedLib) {
    RUNBENCH((benchNavigator<WithSafety>(gSpecializedNavigator, points, dirs, pool, outpool)));
  }
  std::cerr << "##\n";
  RUNBENCH((benchNavigator<WithSafety>(NewSimpleNavigator<false>::Instance(), points, dirs, pool, outpool)));
  std::stringstream str;
  str << outfilenamebase << "_simple.bin";
  outpool.ToFile(str.str());
  std::cerr << "##\n";
  RUNBENCH((benchNavigator<WithSafety>(NewSimpleNavigator<true>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";

#ifdef VECGEOM_ROOT
  benchmarkROOTNavigator<WithSafety>(points, dirs);
#ifdef VECGEOM_GEANT4
  benchmarkG4Navigator<WithSafety>(points, dirs);
#endif // VECGEOM_GEANT4
#endif

#ifdef BENCH_GENERATED_NAVIGATOR
  RUNBENCH((benchNavigator<GeneratedNavigator, WithSafety>(points, dirs, pool, outpool)));
  outpool.ToFile("generatedoutpool.bin");
  std::cerr << "##\n";
  RUNBENCH((benchVectorNavigator<GeneratedNavigator, WithSafety>(points, dirs, pool, outpool)));
  std::cerr << "##\n";
#endif
  if (gBenchVecInterface) {
    RUNBENCH((benchVectorNavigator<NewSimpleNavigator<false>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
    RUNBENCH((benchVectorNavigator<NewSimpleNavigator<true>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
  }
  RUNBENCH((benchNavigator<WithSafety>(SimpleABBoxNavigator<false>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  RUNBENCH((benchNavigator<WithSafety>(SimpleABBoxNavigator<false>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  if (gBenchVecInterface) {
    RUNBENCH((benchVectorNavigator<SimpleABBoxNavigator<false>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
    RUNBENCH((benchVectorNavigator<SimpleABBoxNavigator<true>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
  }
  RUNBENCH((benchNavigator<WithSafety>(HybridNavigator<false>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  RUNBENCH((benchNavigator<WithSafety>(HybridNavigator<true>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  if (gBenchVecInterface) {
    RUNBENCH((benchVectorNavigator<HybridNavigator<false>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
    RUNBENCH((benchVectorNavigator<HybridNavigator<true>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
  }

#ifdef VECGEOM_EMBREE
  // Embree
  RUNBENCH((benchNavigator<WithSafety>(EmbreeNavigator<false>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  RUNBENCH((benchNavigator<WithSafety>(EmbreeNavigator<true>::Instance(), points, dirs, pool, outpool)));
  std::cerr << "##\n";
  if (gBenchVecInterface) {
    RUNBENCH((benchVectorNavigator<EmbreeNavigator<false>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
    RUNBENCH((benchVectorNavigator<EmbreeNavigator<true>, WithSafety>(points, dirs, pool, outpool)));
    std::cerr << "##\n";
  }
#endif

#ifdef VECGEOM_ROOT
  benchmarkROOTNavigator<WithSafety>(points, dirs);
#ifdef VECGEOM_GEANT4
  benchmarkG4Navigator<WithSafety>(points, dirs);
#endif // VECGEOM_GEANT4
#endif

  std::cerr << "## -- TESTING WITHOUT RELOC\n";
  // testing interfaces without relocation
  RUNBENCH((benchNavigatorNoReloc<WithSafety>(NewSimpleNavigator<false>::Instance(), points, dirs, pool, outpool)));
  // the vector navigator
  RUNBENCH((benchVectorNavigatorNoReloc<NewSimpleNavigator<false>, WithSafety>(points, dirs, pool, outpool)));
  // testing interfaces without relocation
  RUNBENCH((benchNavigatorNoReloc<WithSafety>(SimpleABBoxNavigator<false>::Instance(), points, dirs, pool, outpool)));
  // the vector navigator
  RUNBENCH((benchVectorNavigatorNoReloc<SimpleABBoxNavigator<false>, WithSafety>(points, dirs, pool, outpool)));

#ifdef VECGEOM_ROOT
  benchmarkROOTNavigator<true, false>(points, dirs);
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

  // play with some heuristics to init level locators
  for (auto &element : GeoManager::Instance().GetLogicalVolumesMap()) {
    LogicalVolume *lvol = element.second;
    if (lvol->GetDaughtersp()->size() > 8) {
      // HybridManager2::Instance().InitStructure(lvol);
      lvol->SetLevelLocator(SimpleABBoxLevelLocator::GetInstance());
      // lvol->SetLevelLocator(HybridLevelLocator::GetInstance());
    }
  }
  InitNavigators();

  std::string volname(argv[2]);
  auto lvol = GeoManager::Instance().FindLogicalVolume(volname.c_str());
  HybridManager2::Instance().InitStructure(lvol);
#ifdef VECGEOM_EMBREE
  EmbreeManager::Instance().InitStructure(lvol);
#endif

  // some output on volume
  std::cerr << "NavigationKernelBenchmarker run on " << argv[2] << " having " << lvol->GetDaughters().size()
            << " daughters \n";

  bool usecached = false;
  int npoints    = 500000;
  for (auto i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--usecache")) usecached = true;
    // benchmark vector interface?
    if (!strcmp(argv[i], "--vecbench")) gBenchVecInterface = true;
    // analyse state transitions
    if (!strcmp(argv[i], "--statetrans")) gAnalyseOutStates = true;
    // whether to benchmark navigation with safeties
    if (!strcmp(argv[i], "--withsafety")) gBenchWithSafety = true;
    // to set the maxstep number
    if (!strcmp(argv[i], "--maxstep")) {
      if (i + 1 < argc) {
        gMAXSTEP = atof(argv[i + 1]);
        std::cout << "setting maxstep to " << gMAXSTEP << "\n";
      }
    }
    // to set the number of tracks
    if (!strcmp(argv[i], "--ntracks")) {
      if (i + 1 < argc) {
        npoints = atoi(argv[i + 1]);
        std::cout << "setting npoints to " << npoints << "\n";
      }
    }
    if (!strcmp(argv[i], "--navlib")) {
      if (i + 1 < argc) {
        gSpecializedLib = true;
        gSpecLibName    = argv[i + 1];
        InitSpecializedNavigators(gSpecLibName);
        gSpecializedNavigator = lvol->GetNavigator();
        std::cerr << gSpecializedNavigator->GetName() << "\n";
      }
    }
  }

  // setup data structures
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> localpoints(npoints);
  SOA3D<Precision> directions(npoints);
  NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth());
  NavStatePool statepoolout(npoints, GeoManager::Instance().getMaxDepth());

  std::stringstream pstream;
  std::string geomfilename(argv[1]);
  std::string geomfilename_nopath(geomfilename.substr(1 + geomfilename.find_last_of("\\/")));
  pstream << "points_" << geomfilename_nopath << "_" << argv[2] << ".bin";
  std::stringstream dstream;
  dstream << "directions_" << geomfilename_nopath << "_" << argv[2] << ".bin";
  std::stringstream statestream;
  statestream << "states_" << geomfilename_nopath << "_" << argv[2] << ".bin";
  std::stringstream outstatestream;
  outstatestream << "outstates_" << geomfilename_nopath << "_" << argv[2];
  if (usecached) {
    std::cerr << " loading points from cache \n";
    bool fail = (npoints != points.FromFile(pstream.str()));
    fail |= (npoints != directions.FromFile(dstream.str()));
    fail |= (npoints != statepool.FromFile(statestream.str()));
    if (fail) {
      std::cerr << " loading points from cache failed ... continuing normally \n";
      usecached = false;
    }
  }
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
  }
  if (gBenchWithSafety) {
    benchDifferentNavigators<true>(points, directions, statepool, statepoolout, outstatestream.str());
  } else {
    benchDifferentNavigators<false>(points, directions, statepool, statepoolout, outstatestream.str());
  }
  return 0;
}
