#ifndef VECGEOM_BASE_FpeEnable_H_
#define VECGEOM_BASE_FpeEnable_H_

#if defined(__GNUC__) && !defined(__APPLE__)

#include <fenv.h>

static void __attribute__((constructor)) EnableFpeForTests()
{
  // this function is not offered on APPLE MACOS
  feenableexcept(FE_INVALID | FE_DIVBYZERO);
  // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}

#endif

// #ifdef __APPLE__
// #include <xmmintrin.h>
// _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
// #endif

#endif // VECGEOM_BASE_FpeEnable_H_
