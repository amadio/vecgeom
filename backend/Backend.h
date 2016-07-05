/// \file vector/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/Backend.h"
#elif defined(VECGEOM_VC)
#include "backend/vc/Backend.h"
#elif defined(VECGEOM_CILK)
#include "backend/cilk/Backend.h"
#elif defined(VECGEOM_MICVEC)
#include "backend/micvec/Backend.h"
#elif defined(VECGEOM_UMESIMD)
#include "backend/umesimd/Backend.h"
#else
#include "backend/scalar/Backend.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
T NonZeroAbs(T const &x)
{
  // additional casting to (T) prevents link error with clang
  // NOTE: why do we need Tiny<T> to be a template? We could just use T(literalconstant) directly?
  return Abs(x) + T(kTiny);
}

template <typename T>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
T NonZero(T const &x)
{
  return x + CopySign(T(kTiny), x);
}

// template specialize these functions for UMESIMD (which does not have CopySign)
#if defined(VECGEOM_UMESIMD)

template <>
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector NonZeroAbs(UmeSimdPrecisionVector const &x)
{
#ifdef VECGEOM_FLOAT_PRECISION
  return (x.abs()).add(std::numeric_limits<float>::lowest());
#else
  return (x.abs()).add(std::numeric_limits<double>::lowest());
#endif
}

template <>
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector NonZero(UmeSimdPrecisionVector const &x)
{
#ifdef VECGEOM_FLOAT_PRECISION
  UmeSimdPrecisionVector t0(std::numeric_limits<float>::lowest());
  UmeSimdMask mask = x < 0.0f;
#else
  UmeSimdPrecisionVector t0(std::numeric_limits<double>::lowest());
  UmeSimdMask mask = x < 0.0;
#endif
  return x.add(t0.neg(mask));
}

template <>
VECGEOM_FORCE_INLINE
UME::SIMD::SIMDVec_f<Precision, kVectorSize> NonZeroAbs(UME::SIMD::SIMDVec_f<Precision, kVectorSize> const &x)
{
#ifdef VECGEOM_FLOAT_PRECISION
  return (x.abs()).add(std::numeric_limits<float>::lowest());
#else
  return (x.abs()).add(std::numeric_limits<double>::lowest());
#endif
}

template <>
VECGEOM_FORCE_INLINE
UME::SIMD::SIMDVec_f<Precision, kVectorSize> NonZero(UME::SIMD::SIMDVec_f<Precision, kVectorSize> const &x)
{
#ifdef VECGEOM_FLOAT_PRECISION
  UmeSimdPrecisionVector t0(std::numeric_limits<float>::lowest());
  UmeSimdMask mask = x < 0.0f;
#else
  UmeSimdPrecisionVector t0(std::numeric_limits<double>::lowest());
  UmeSimdMask mask = x < 0.0;
#endif
  return x.add(t0.neg(mask));
}

#endif
}
}

#endif // VECGEOM_BACKEND_BACKEND_H_
