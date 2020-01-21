/*
 * testVectorSafety.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: swenzel
 */

// Forced asserts() to be defined, even for Release mode
#undef NDEBUG

#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/SimpleNavigator.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Global.h"

using namespace vecgeom;

VPlacedVolume *SetupBoxGeometry()
{
  UnplacedBox *worldUnplaced   = new UnplacedBox(10, 10, 10);
  UnplacedBox *boxUnplaced     = new UnplacedBox(0.5, 0.5, 0.5);
  Transformation3D *placement1 = new Transformation3D(2, 2, 2, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-2, 2, 2, 45, 0, 0);
  Transformation3D *placement3 = new Transformation3D(2, -2, 2, 0, 45, 0);
  Transformation3D *placement4 = new Transformation3D(2, 2, -2, 0, 0, 45);
  Transformation3D *placement5 = new Transformation3D(-2, -2, 2, 45, 45, 0);
  Transformation3D *placement6 = new Transformation3D(-2, 2, -2, 45, 0, 45);
  Transformation3D *placement7 = new Transformation3D(2, -2, -2, 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-2, -2, -2, 45, 45, 45);
  LogicalVolume *world         = new LogicalVolume(worldUnplaced);
  LogicalVolume *box           = new LogicalVolume(boxUnplaced);
  world->PlaceDaughter(box, placement1);
  world->PlaceDaughter(box, placement2);
  world->PlaceDaughter(box, placement3);
  world->PlaceDaughter(box, placement4);
  world->PlaceDaughter(box, placement5);
  world->PlaceDaughter(box, placement6);
  world->PlaceDaughter(box, placement7);
  world->PlaceDaughter(box, placement8);
  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();
  return w;
}

// function to test safety
void testVectorSafety(VPlacedVolume *world)
{
  int np = 1024;
  SOA3D<Precision> points(np);
  SOA3D<Precision> workspace(np);
  Precision *safeties = (Precision *)_vecCore::AlignedAlloc(sizeof(Precision) * np, 32);
  vecgeom::volumeUtilities::FillUncontainedPoints(*world, points);

  // now setup all the navigation states
  NavigationState **states = new NavigationState *[np];
  vecgeom::SimpleNavigator nav;
  for (int i = 0; i < np; ++i) {
    states[i] = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
    nav.LocatePoint(world, points[i], *states[i], true);
  }

  // calculate safeties with vector interface
  nav.GetSafeties(points, states, workspace, safeties);

  // verify against serial interface
  for (int i = 0; i < np; ++i) {
    double ss = nav.GetSafety(points[i], *states[i]);
    assert(std::abs(safeties[i] - ss) < 1E-3 && "Problem in VectorSafety (in SimpleNavigator)");
  }
  std::cout << "Safety test passed\n";
  _mm_free(safeties);
  // free NavigationState instances

  for (int i = 0; i < np; ++i) {
    NavigationState::ReleaseInstance(states[i]);
  }
  delete[] states;
}

// function to test vector navigator
void testVectorNavigator(VPlacedVolume *world)
{
  int np = 100000;
  SOA3D<Precision> points(np);
  SOA3D<Precision> dirs(np);
  SOA3D<Precision> workspace1(np);
  SOA3D<Precision> workspace2(np);

  Precision *steps    = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  Precision *pSteps   = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  Precision *safeties = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);

  int *intworkspace = (int *)vecCore::AlignedAlloc(32, sizeof(int) * np);

  vecgeom::volumeUtilities::FillUncontainedPoints(*world, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  NavigationState **states    = new NavigationState *[np];
  NavigationState **newstates = new NavigationState *[np];

  vecgeom::SimpleNavigator nav;
  for (int i = 0; i < np; ++i) {
    // pSteps[i] = kInfLength;
    pSteps[i]    = (i % 2) ? 1 : VECGEOM_NAMESPACE::kInfLength;
    states[i]    = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
    newstates[i] = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
    nav.LocatePoint(world, points[i], *states[i], true);
  }

  // calculate steps with vector interface
  nav.FindNextBoundaryAndStep(points, dirs, workspace1, workspace2, states, newstates, pSteps, safeties, steps,
                              intworkspace);

  // verify against serial interface
  for (int i = 0; i < np; ++i) {
    Precision s          = 0;
    NavigationState *cmp = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
    cmp->Clear();
    nav.FindNextBoundaryAndStep(points[i], dirs[i], *states[i], *cmp, pSteps[i], s);

    // check for consistency of navigator

    assert(steps[i] == s);
    assert(cmp->Top() == newstates[i]->Top());
    assert(cmp->IsOnBoundary() == newstates[i]->IsOnBoundary());
    assert(safeties[i] == nav.GetSafety(points[i], *states[i]));

    delete cmp;
  }

  std::cout << "Navigation test passed\n";
  _mm_free(steps);
  _mm_free(intworkspace);
  _mm_free(pSteps);
  for (int i = 0; i < np; ++i) {
    delete states[i];
    delete newstates[i];
  }
  delete[] states;
  delete[] newstates;
}

int main()
{
  VPlacedVolume *w;
  testVectorSafety(w = SetupBoxGeometry());
  // fails for the moment testVectorNavigator(w);
}
