/*
 * SafetyKernelBenchmarker
 *
 *  Created on: 18.11.2015
 *      Author: swenzel
 */

// benchmarking various different safety kernels per logical volume
#include "volumes/utilities/VolumeUtilities.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "base/RNG.h"
#include "navigation/GlobalLocator.h"
#include "navigation/NavStatePool.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedVolume.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "base/Stopwatch.h"
#include "navigation/SimpleSafetyEstimator.h"
#include "navigation/SimpleABBoxSafetyEstimator.h"
#include "navigation/HybridSafetyEstimator.h"
#include "management/HybridManager2.h"

// in case someone has written a special safety estimator for the CAHL logical volume
//#include "navigation/CAHLSafetyEstimator.h"
//#define SPECIALESTIMATOR CAHLSafetyEstimator

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#include <iostream>

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

template <typename T>
__attribute__((noinline)) void benchSafety(SOA3D<Precision> const &points, NavStatePool &pool)
{
  // bench safety
  Precision *safety = new Precision[points.size()];
  Stopwatch timer;
  VSafetyEstimator *se = T::Instance();
  timer.Start();
  for (size_t i = 0; i < points.size(); ++i) {
    safety[i] = se->ComputeSafety(points[i], *(pool[i]));
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (size_t i = 0; i < points.size(); ++i) {
    accum += safety[i];
  }
  delete[] safety;
  std::cerr << "accum  " << T::GetClassName() << " " << accum << "\n";
}

// benchmarks the old vector interface (needing temporary workspace)
template <typename T>
__attribute__((noinline)) void benchVectorSafety(SOA3D<Precision> const &points, NavStatePool &pool)
{
  // bench safety
  SOA3D<Precision> workspace(points.size());
  Precision *safety = (double *)vecCore::AlignedAlloc(64, sizeof(double) * points.size());
  Stopwatch timer;
  VSafetyEstimator *se = T::Instance();
  timer.Start();
  se->ComputeVectorSafety(points, pool, workspace, safety);
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (size_t i = 0; i < points.size(); ++i) {
    accum += safety[i];
  }
  _mm_free(safety);
  std::cerr << "VECTOR accum  " << T::GetClassName() << " " << accum << "\n";
}

// benchmarks the new safety vector interface (which does not need temporary workspace)
template <typename T>
__attribute__((noinline)) void benchVectorSafetyNoWorkspace(SOA3D<Precision> const &points, NavStatePool &pool)
{
  // bench safety
  Precision *safety = (double *)vecCore::AlignedAlloc(64, sizeof(double) * points.size());
  Stopwatch timer;
  VSafetyEstimator *se = T::Instance();
  timer.Start();
  se->ComputeVectorSafety(points, pool, safety);
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (size_t i = 0; i < points.size(); ++i) {
    accum += safety[i];
  }
  _mm_free(safety);
  std::cerr << "VECTOR (NO WORKSP) accum  " << T::GetClassName() << " " << accum << "\n";
}

// benchmarks the ROOT safety interface
#ifdef VECGEOM_ROOT
__attribute__((noinline)) void benchmarkROOTSafety(int nPoints, SOA3D<Precision> const &points, int i)
{

  TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
  TGeoBranchArray *brancharrays[nPoints];
  Precision *safety = new Precision[nPoints];

  for (int i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    rootnav->ResetState();
    rootnav->FindNode(pos.x(), pos.y(), pos.z());
    brancharrays[i] = TGeoBranchArray::MakeInstance(GeoManager::Instance().getMaxDepth());
    brancharrays[i]->InitFromNavigator(rootnav);
  }

#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  Stopwatch timer;
  timer.Start();
  for (int i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    brancharrays[i]->UpdateNavigator(rootnav);
    rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
    safety[i] = rootnav->Safety();
  }
  timer.Stop();
  std::cerr << "ROOT time" << timer.Elapsed() << "\n";
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  double accum(0.);
  for (int i = 0; i < nPoints; ++i) {
    accum += safety[i];
  }
  std::cerr << "ROOT s " << accum << "\n";
  return;
}
#endif

// main routine starting up the individual benchmarks
void benchDifferentSafeties(SOA3D<Precision> const &points, NavStatePool &pool)
{
  std::cerr << "##\n";
  std::cerr << "##\n";
  RUNBENCH(benchSafety<SimpleSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchSafety<SimpleABBoxSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchSafety<HybridSafetyEstimator>(points, pool));

  // std::cerr << "##\n";
  // benchVectorSafety<SPECIALESTIMATOR>(points, pool);

  std::cerr << "##\n";
  RUNBENCH(benchVectorSafety<SimpleSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorSafetyNoWorkspace<SimpleSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchVectorSafety<SimpleABBoxSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchmarkROOTSafety(points.size(), points, 10));
}

// main program
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
  int npoints = 100;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> localpoints(npoints);
  SOA3D<Precision> directions(npoints);
  NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth());

  // setup test points
  TGeoBBox const *rootbbox = dynamic_cast<TGeoBBox const *>(gGeoManager->GetTopVolume()->GetShape());
  Vector3D<Precision> bbox(rootbbox->GetDX(), rootbbox->GetDY(), rootbbox->GetDZ());

  std::string volname(argv[2]);
  volumeUtilities::FillGlobalPointsForLogicalVolume<SOA3D<Precision>>(
      GeoManager::Instance().FindLogicalVolume(volname.c_str()), localpoints, points, npoints);
  std::cerr << "points filled\n";
  for (size_t i = 0; i < points.size(); ++i) {
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
  benchDifferentSafeties(points, statepool);
  return 0;
}
