//
// File:    TestVector.cpp
// Purpose: Unit tests for the vecgeom::Vector
//

//-- ensure asserts are compiled in
#undef NDEBUG

#include "VecGeom/base/Vector.h"

int test()
{
  vecgeom::Vector<double> aVector;
  aVector.resize(2, 0.0);
  size_t newSize = aVector.size();

  assert(newSize == 2);

  aVector.reserve(10);
  assert(aVector.capacity() == 10);
  assert(aVector.size() == 2);

  for (int i = 0; i < 12; ++i) {
    aVector.push_back(i);
  }
  assert(aVector.capacity() > 10);
  assert(aVector.size() == 14);

  int i = 0;
  for (auto val : aVector) {
    if (i < 2)
      assert(val == 0);
    else
      assert(val == (i - 2));
    ++i;
  }
  for (i = 0; i < 12; ++i) {
    if (i < 2)
      assert(aVector[i] == 0);
    else
      assert(aVector[i] == (i - 2));
  }

  aVector.clear();
  assert(aVector.capacity() > 10);
  assert(aVector.size() == 0);
  return 0;
}

int main(int, char **)
{
  return test();
}
