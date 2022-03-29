// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit tests for NavStateIndex

// ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/navigation/NavStateIndex.h"
#include <type_traits>

static_assert(std::is_trivially_destructible<vecgeom::NavStateIndex>::value, "");

int main(int argc, char *argv[])
{
  return 0;
}
