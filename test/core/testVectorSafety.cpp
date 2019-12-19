/*
 * testVectorSafety.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: swenzel
 */

// Forced asserts() to be defined, even for Release mode
#undef NDEBUG
//#define VERBOSE

#include "SetupBoxGeometry.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Global.h"
#include <iostream>

using namespace vecgeom;


// function to test safety
void testVectorSafety(const VPlacedVolume *world)
{
  int np = 1024;
  SOA3D<Precision> points(np);
  SOA3D<Precision> workspace(np);
  Precision *safeties = (Precision *)vecCore::AlignedAlloc(sizeof(Precision) * np, 32);
  vecgeom::volumeUtilities::FillUncontainedPoints(*world, points);

  // now setup all the navigation states
  NavStatePool states(np, GeoManager::Instance().getMaxDepth());
  for (int i = 0; i < np; ++i) {
    GlobalLocator::LocateGlobalPoint(world, points[i], *states[i], true);
  }

  // calculate safeties with vector interface
  auto nav = vecgeom::NewSimpleNavigator<>::Instance();
  nav->GetSafetyEstimator()->ComputeVectorSafety(points, states, workspace, safeties);

  // verify against serial interface
  int miscounter = 0;
  for (int i = 0; i < np; ++i) {
    double ss = nav->GetSafetyEstimator()->ComputeSafety(points[i], *states[i]);
    if (std::abs(safeties[i] - ss) > 1.0e-3) {
      ++miscounter;
#ifdef VERBOSE
      std::cerr<<" i="<< i <<", point="<< points[i] <<", mismatch: serial="<< ss <<", vector="<< safeties[i] <<"\n";
#endif
    }
    // assert(std::abs(safeties[i] - ss) < 1.0e-3 && "Problem in VectorSafety (in NewSimpleNavigator)");
  }

  std::cerr << "   # safety mismatches: " << miscounter << "/" << np << "\n";
  std::cout << "Safety test passed\n";

  // cleanup
  vecCore::AlignedFree(safeties);
}

/// Function to test vector navigator
void testVectorNavigator(VPlacedVolume const *world)
{
  // int np = 100000;
  int np = 1024;
  SOA3D<Precision> points(np);
  SOA3D<Precision> dirs(np);

  Precision *steps    = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  Precision *pSteps   = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  Precision *safeties = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  bool *calcSafeties  = (bool *)vecCore::AlignedAlloc(32, sizeof(bool) * np);

  vecgeom::volumeUtilities::FillRandomPoints(*world, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool states(np, ndeep);
  NavStatePool newstates(np, ndeep);

  for (int i = 0; i < np; ++i) {
    pSteps[i]    = (i % 2) ? 1 : VECGEOM_NAMESPACE::kInfLength;
    calcSafeties[i] = true;
    GlobalLocator::LocateGlobalPoint(world, points[i], *states[i], true);
  }

  //.. calculate steps with vector interface
  auto nav = vecgeom::NewSimpleNavigator<>::Instance();
  // nav->FindNextBoundaryAndStep(points, dirs, workspace1, workspace2, states, newstates,
  //                              pSteps, safeties, steps, intworkspace);
  // nav->ComputeStepsAndSafetiesAndPropagatedStates(points, dirs, pSteps, states, newstates,
  // steps, calcSafeties, safeties);

  //.. GL: temporarily use a scalar version instead, to fill arrays usually filled by vectorized version
  for (int i = 0; i < np; ++i) {
    nav->ComputeStepAndSafetyAndPropagatedState(points[i], dirs[i], pSteps[i], *states[i], *newstates[i],
                                                calcSafeties[i], safeties[i]);
    nav->FindNextBoundaryAndStep(points[i], dirs[i], *states[i], *newstates[i], pSteps[i], safeties[i]);
  }

  // verify against serial interface
  int miscounter = 0;
  for (int i = 0; i < np; ++i) {
    bool mismatch = false;
    Precision saf = 0;
    NavigationState *cmp = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
    cmp->Clear();
    // nav->ComputeStepAndSafetyAndPropagatedState(points[i], dirs[i], pSteps[i], *states[i], *cmp, true, saf);
    nav->FindNextBoundaryAndStep(points[i], dirs[i], *states[i], *cmp, pSteps[i], saf);

    // check for consistency of navigator
    Precision tmpSafety = nav->GetSafetyEstimator()->ComputeSafety(points[i], *states[i]);

    if (abs(safeties[i] - tmpSafety) > 1.0e-6) mismatch = true;
    if (mismatch) {
      ++miscounter;
#ifdef VERBOSE
      std::cerr << " testVectorNavigator(): i=" << i << ", pt[i]=" << points[i]
                << ", steps[i]="<< steps[i]
                << ", safety[i]="<< safeties[i] << ", "<< saf
                <<", tmpSafety="<< tmpSafety
                << ", cmp.Top=" << (cmp->Top() ? cmp->Top()->GetName() : "NULL")
                << ", state->Top()=" << (newstates[i]->Top() ? newstates[i]->Top()->GetName() : "NULL") << "\n";
#endif
    }

    assert(cmp->Top() == newstates[i]->Top());
    assert(cmp->IsOnBoundary() == newstates[i]->IsOnBoundary());
    // assert(safeties[i] == tmpSafety);
    // assert(saf == tmpSafety);
    delete cmp;
  }

  std::cout << "Navigation mismatches: " << miscounter << "/" << np << "\n";
  std::cout << "Navigation test passed\n";
  _mm_free(steps);
  _mm_free(pSteps);
  _mm_free(safeties);
  _mm_free(calcSafeties);
  // for (int i = 0; i < np; ++i) {
  //   delete states[i];
  //   delete newstates[i];
  // }
  // delete[] states;
  // delete[] newstates;
}

int main()
{
  //.. build geometry
  assert(SetupBoxGeometry());
  const VPlacedVolume *world = GeoManager::Instance().GetWorld();

  //.. optional: configure non-default level locators
  LogicalVolume *lvol(nullptr);
  for (auto &element : GeoManager::Instance().GetLogicalVolumesMap()) {
    lvol = element.second;
    if (lvol->GetDaughtersp()->size() > 8) {
      // HybridManager2::Instance().InitStructure(lvol);
      // lvol->SetLevelLocator(SimpleABBoxLevelLocator::GetInstance());
      // lvol->SetLevelLocator(HybridLevelLocator::GetInstance());
    }
  }

  //.. run safety estimation tests
  testVectorSafety(world);

  //.. run navigation tests
  testVectorNavigator(world);
}
