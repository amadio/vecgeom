#ifndef VECGEOM_ASSERT_H
#define VECGEOM_ASSERT_H

#if !defined(__NVCC__)
#include <cassert>
#else

#ifdef assert
#undef assert
#endif

#ifdef NDEBUG
#define assert(x)
#else

#ifndef __CUDA_ARCH__
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%s:%d: Assertion failed: '%s'\n",                       \
        __FILE__, __LINE__, #x);                                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("%s:%d:\n%s: Assertion failed: '%s'\n",                           \
        __FILE__, __LINE__, __PRETTY_FUNCTION__, #x);                          \
      __syncthreads();                                                         \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)
#endif // ifndef __CUDA_ARCH__

#endif // ifdef NDEBUG

#endif // if !defined(__NVCC__)

#endif
