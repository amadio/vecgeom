#include <fenv.h>

static void __attribute__((constructor)) EnableFpeForTests()
{
#ifndef __APPLE__
  // this function is not offered on APPLE MACOS
  feenableexcept(FE_INVALID | FE_DIVBYZERO);
#endif
}
