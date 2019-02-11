//
// File:    TestCoaxialCones.cpp
// Purpose: Unit tests for CoaxialCones
// Author:  Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)
//

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/EllipticUtilities.h"
#include "volumes/CoaxialCones.h"
#include "ApproxEqual.h"

bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;

template <class CoaxialCones_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestCoaxialCones()
{
  /*
   * Add the require unit test
   *
   */
  return true;
}

int main(int argc, char *argv[])
{
  assert(TestCoaxialCones<vecgeom::SimpleCoaxialCones>());
  std::cout << "VecGeomCoaxialCones passed\n";

  return 0;
}
