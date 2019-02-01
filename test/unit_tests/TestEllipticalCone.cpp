//
// File:    TestEllipticalCone.cpp
// Purpose: Unit tests for EllipticalCone
// Author:  Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)
//

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/EllipticUtilities.h"
#include "volumes/EllipticalCone.h"
#include "ApproxEqual.h"

bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;

template <class EllipticalCone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestEllipticalCone()
{
  /*
   * Add the require unit test
   *
   */
  return true;
}

int main(int argc, char *argv[])
{
  assert(TestEllipticalCone<vecgeom::SimpleEllipticalCone>());
  std::cout << "VecGeomEllipticalCone passed\n";

  return 0;
}
