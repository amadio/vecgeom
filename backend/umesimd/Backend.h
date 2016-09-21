/// \file umesimd/backend.h
/// \author Przemyslaw Karpinski (przemyslaw.karpinski@cern.ch)

#ifndef VECGEOM_BACKEND_UMESIMDBACKEND_H_
#define VECGEOM_BACKEND_UMESIMDBACKEND_H_

#include "base/Global.h"
#include "backend/scalar/Backend.h"

// Use '()' instead of '[]' brackets for writemask syntax
#define USE_PARENTHESES_IN_MASK_ASSIGNMENT
#include <umesimd/UMESimd.h>

#ifdef kVectorSize
#undef kVectorSize
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/* UME::SIMD allows user to control the length of a vector. Since
   vecgeom is not providing this parameter, it is necessary to
   explicitly select vector lengths to be used for different instruction
   set. */
#ifdef VECGEOM_FLOAT_PRECISION
#if defined(__MIC__)
constexpr int kVectorSize = 16;
#elif defined(__AVX__)
constexpr int kVectorSize = 8;
#elif defined(__SSE__)
constexpr int kVectorSize = 4;
#else // Default fallback to scalar emulation
constexpr int kVectorSize = 1;
#endif
#else
#if defined(__MIC__)
constexpr int kVectorSize = 8;
#elif defined(__AVX__)
constexpr int kVectorSize = 4;
#elif defined(__SSE__)
constexpr int kVectorSize = 2;
#else // Default fallback to scalar emulation
constexpr int kVectorSize = 1;
#endif
#endif

using UMESIMDMask = typename UME::SIMD::SIMDVecMask<kVectorSize>;
#ifdef VECGEOM_FLOAT_PRECISION
using UMESIMDInt_v   = typename UME::SIMD::SIMDVec_i<int, kVectorSize>;
using UMESIMDFloat_v = typename UME::SIMD::SIMDVec_f<float, kVectorSize>;
#else
using UMESIMDInt_v        = typename UME::SIMD::SIMDVec_i<int, kVectorSize>;
using UMESIMDFloat_v      = typename UME::SIMD::SIMDVec_f<double, kVectorSize>;
#endif

struct kUmeSimd {
  using bool_v      = UMESIMDMask;
  using Bool_t      = UMESIMDMask;
  using int_v       = UMESIMDInt_v;
  using inside_v    = UMESIMDInt_v;
  using precision_v = UMESIMDFloat_v;

  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

#define VECGEOM_BACKEND_TYPE kUmeSimd
#define VECGEOM_BACKEND_PRECISION_TYPE UMESIMDFloat_v
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE vecgeom::kVectorSize
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) vecgeom::UMESIMDFloat_v(P)
#define VECGEOM_BACKEND_BOOL UMESIMDMask
#define VECGEOM_BACKEND_INSIDE UMESIMDInt_v

} // End inline namespace
} // End global namespace

#endif // VECGEOM_BACKEND_VCBACKEND_H_
