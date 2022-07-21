// File:    TestVECGEOM-601.cpp
// Purpose: Regression test for Polycone DistanceToOut issue reported by ATLAS
//          in VECGEOM-601. Resolved by MR 878, but we add the test that exposes
//          the problem to confirm we don't reintroduced the bug.
//          Bug in distance calculation caused by different tolerances used
//          between Cone's DistanceToOut and Inside functions. For points very
//          close to surface, can result in Inside returning "on surface" whilst
//          DistanceToOut considers them "outside", returning distance as -1.
//

#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/base/Vector3D.h"
#include "ApproxEqual.h"

#include <iostream>

using namespace vecgeom;

int main()
{
  // Construct ATLAS PixelServMat1 shape
  Precision phiStart = 2.251 * kPi / 180.;
  Precision phiDelta = 25.498 * kPi / 180.;
  constexpr size_t N = 4;
  Precision zVal[N]  = {-0.4525000000001, 0.30834070796467, 0.374182692307775, 0.4525000000001};
  Precision rMin[N]  = {380., 380., 380., 380.};
  Precision rMax[N]  = {452., 452., 416., 380.};

  GenericUnplacedPolycone pixelServMat1{phiStart, phiDelta, N, zVal, rMin, rMax};

  // Test Inputs
  // 1. Point in local coordinates that gives correct DistanceToOut
  Vector3D<Precision> goodPoint{382.70843662077084, 157.84354328757132, -0.45249999999987267};

  // 2. Point in local coordinates that gives incorrect DistanceToOut, exposing the bug
  Vector3D<Precision> badPoint{382.7083744019393, 157.84347166220718, -0.45249943287808492};

  // 3. Direction of track, same between success/fail cases
  Vector3D<Precision> direction{0.87892570216583898, 0.29076414869269152, 0.3780817635212016};

  // 4. Expected value of DistanceToOut given these points and direction, to precision
  //    needed to compare ApproxEqual
  Precision expectedDistance = 2.18652;

  // Check expected good point
  {
    Precision distance = pixelServMat1.DistanceToOut(goodPoint, direction, kInfLength);
    if (!ApproxEqual(distance, expectedDistance)) {
      std::cerr << "FAIL(" << __FILE__ << "," << __LINE__ << "): "
                << " good vector does not yield expected DistanceToOut "
                << "( " << distance << " != " << expectedDistance << ")" << std::endl;
      return 1;
    }
  }

  // Check "Bad" point/direction that results in point on surface
  {
    Precision distance = pixelServMat1.DistanceToOut(badPoint, direction, kInfLength);
    if (!ApproxEqual(distance, expectedDistance)) {
      std::cerr << "FAIL(" << __FILE__ << "," << __LINE__ << "): "
                << " bad vector does not yield expected DistanceToOut "
                << "( " << distance << " != " << expectedDistance << ")" << std::endl;
      return 1;
    }
  }

  return 0;
}
