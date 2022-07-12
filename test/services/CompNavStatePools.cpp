/*
 * CompNavStatePools.cpp
 *
 *  Created on: 17.01.2016
 *      Author: swenzel
 */

// just a test file / utility to compare 2 navstatepools stored in a binary file

#include "VecGeom/navigation/NavStatePool.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/RootGeoManager.h"

int N = 500000;

using namespace vecgeom;

void test()
{
  Vector3D<Precision> globalpoint(0.5, 0.78, -0.34);
  Transformation3D m1(0.00, -0.00, -13952.20, -1.00, 0.00, -0.00, 0.00, 1.00, 0.00, 0.00, 0.00, -1.00);
  Transformation3D m2(0.00, -0.00, -13951.95, -1.00, 0.00, -0.00, 0.00, 1.00, 0.00, 0.00, 0.00, -1.00);
  auto lp1 = m1.Transform(globalpoint);
  auto lp2 = m2.Transform(globalpoint);

  Transformation3D deltahand(0.00, 0.00, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00);
  auto ld = deltahand.Transform(lp1);
  if (ld == lp2) {
    std::cerr << "ok\n";
  }

  Transformation3D delta;
  NavigationState::DeltaTransformation2(m1, m2, delta);
  deltahand.Print();
  delta.Print();
}

void test_basic()
{
  Transformation3D m1(0, 0, 10, 10, 0, 0);
  Transformation3D m2(0, 0, -10, 10, 0, 0);
  Transformation3D m3(0, 0, -10, 10, 0, 0);
  Transformation3D deltahand(0, 0, -20);

  Transformation3D m1inv;
  m1.Inverse(m1inv);
  m1inv.SetProperties();
  std::cerr << "inverse\n";
  m1inv.Print();
  m3.MultiplyFromRight(m1inv);
  deltahand.Print();
  std::cerr << "relative\n";
  m3.Print();

  Vector3D<Precision> gp(0, 0, 0.75);
  Vector3D<Precision> lp1 = m1.Transform(gp);
  Vector3D<Precision> lp2 = m2.Transform(gp);
  std::cerr << lp1 << "\n";
  std::cerr << lp2 << "\n";
  std::cerr << deltahand.Transform(lp1) << "\n";
  std::cerr << m3.Transform(lp1) << "\n";
}

void test_transformation_stuff(NavigationState const *state1, NavigationState const *state2)
{
  Vector3D<Precision> globalpoint(0.5, 0.78, -0.34);
  Transformation3D g1;
  Transformation3D g2;
  Transformation3D invg1;
  Transformation3D invg2;

  state1->TopMatrix(g1);
  state2->TopMatrix(g2);
  g1.Inverse(invg1);
  g2.Inverse(invg2);

  Vector3D<Precision> g1local = g1.Transform(globalpoint);
  Vector3D<Precision> g2local = g2.Transform(globalpoint);
  Vector3D<Precision> g1back  = invg1.Transform(g1local);

  //    std::cerr << " --- \n";
  //    g1.Print();
  //    g2.Print();
  //    std::cerr << " ---- \n";
  //
  //    invg1.Print();
  //    invg1.SetProperties();
  //    invg2.FixZeroes();
  //    g1.SetProperties();
  //    g1.MultiplyFromRight(invg1);
  //    g1.Print();

  Vector3D<Precision> g2back = invg2.Transform(g2local);
  if (g2back != globalpoint) {
    std::cerr << "scheisse 2\n";
  }
  if (g1back != globalpoint) {
    std::cerr << "scheisse 1\n";
  }

  Transformation3D delta;
  state1->DeltaTransformation(*state2, delta);
  Vector3D<Precision> g2delta = delta.Transform(g1local);
  if (g2delta != g2local) {
    std::cerr << "scheisse 3\n";
    std::cerr << g2delta << " vs " << g2local << "\n";
    std::cerr << g1local << "\n";
    delta.Print();
  }
}

int main(int argc, char *argv[])
{
  //  test(); return 0;
  NavStatePool testpool(0, 0);

  // we need the geometry
  // read in detector passed as argument
  if (argc > 1) {
    RootGeoManager::Instance().set_verbose(3);
    RootGeoManager::Instance().LoadRootGeometry(std::string(argv[1]));
  } else {
    std::cerr << "please give a ROOT geometry file\n";
    return 1;
  }

  int depth;
  int capacity;
  testpool.ReadDepthAndCapacityFromFile("states.bin", capacity, depth);
  std::cerr << " capacity " << capacity << "\n";
  std::cerr << " depth " << depth << "\n";

  NavStatePool inpool(N, depth);
  NavStatePool outpool1(N, depth);
  NavStatePool outpool2(N, depth);

  inpool.FromFile("states.bin");
  outpool1.FromFile("outpool.bin");
  outpool2.FromFile("generatedoutpool.bin");
  // outpool2.FromFile("simplenavoutpool.bin");

  size_t counter = 0;
  for (size_t i = 0; i < N; ++i) {
    auto *instate   = inpool[i];
    auto *outstate1 = outpool1[i];
    auto *outstate2 = outpool2[i];

    test_transformation_stuff(instate, outstate1);
    if (!(outstate1->Distance(*outstate2) == 0)) {
      counter++;
      std::cerr << "state " << i << " " << instate->RelativePath(*outstate1);
      std::cerr << " vs " << instate->RelativePath(*outstate2) << "\n";
    }
  }
  std::cerr << counter << " errors \n";
  return 0;
}
