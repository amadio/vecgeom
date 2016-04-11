/// \file Global.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

#if __cplusplus < 201103L
#error VecGeom requires compiler and library support for the ISO C++ 2011 standard.
#endif

#include "base/Assert.h"
#include "base/Cuda.h"
#include "base/Math.h"

#define VECGEOM

#ifdef __INTEL_COMPILER
  // Compiling with icc
  #define VECGEOM_INTEL
  #define VECGEOM_INLINE inline
  #ifndef VECGEOM_NVCC
    #define VECGEOM_ALIGNED __attribute__((aligned(64)))
  #endif
#else
  // Functionality of <mm_malloc.h> is automatically included in icc
  #include <mm_malloc.h>
  #if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) && !defined(__NO_INLINE__) && !defined( VECGEOM_NOINLINE )
    #define VECGEOM_INLINE inline __attribute__((always_inline))
    #ifndef VECGEOM_NVCC
      #define VECGEOM_ALIGNED __attribute__((aligned(64)))
    #endif
  #else
  // Clang or forced inlining is disabled ( by falling back to compiler decision )
    #define VECGEOM_INLINE inline
    #ifndef VECGEOM_NVCC
      #define VECGEOM_ALIGNED
    #endif
  #endif
#endif

// Allow constexpr variables and functions if possible
#define VECGEOM_CONSTEXPR constexpr
#define VECGEOM_CONSTEXPR_RETURN constexpr

// Qualifier(s) of global constants
#ifdef VECGEOM_NVCC_DEVICE
    // constexpr not supported on device in CUDA 6.5
    #define VECGEOM_GLOBAL static __constant__ const
    #define VECGEOM_CLASS_GLOBAL static const
#else
  #define VECGEOM_GLOBAL static constexpr
  #define VECGEOM_CLASS_GLOBAL static constexpr
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE{
 enum EnumInside {
   eInside = 0, /* for USOLID compatibility */
   kInside = eInside,
   eSurface = 1,
   kSurface = eSurface,
   eOutside = 2,
   kOutside = eOutside,
 };
typedef int Inside_t;

#if defined (__MIC__)
  VECGEOM_GLOBAL int kAlignmentBoundary = 64;
#else
  VECGEOM_GLOBAL int kAlignmentBoundary = 32;
#endif

namespace EInside {
VECGEOM_GLOBAL vecgeom::Inside_t kInside = 0;
VECGEOM_GLOBAL vecgeom::Inside_t kSurface = 1;
VECGEOM_GLOBAL vecgeom::Inside_t kOutside = 2;
}

namespace details {
   template <typename DataType, typename Target> struct UseIfSameType {
      VECGEOM_CUDA_HEADER_BOTH
      static Target const *Get(DataType*) { return nullptr; }
   };
   template <typename DataType> struct UseIfSameType<DataType,DataType> {
      VECGEOM_CUDA_HEADER_BOTH
      static DataType const *Get(DataType *ptr) { return ptr; }
   };
}

// some static MACROS
#define VECGEOM_MAXDAUGHTERS 100 // macro mainly used to allocated static (stack) arrays/workspaces

} } // End global namespace

#endif // VECGEOM_BASE_GLOBAL_H_
