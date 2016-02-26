/// \file vector/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/Backend.h"
#elif defined(VECGEOM_VC)
#include "backend/vc/Backend.h"
#include "backend/vcfloat/Backend.h"
#elif defined(VECGEOM_CILK)
#include "backend/cilk/Backend.h"
#elif defined(VECGEOM_MICVEC)
#include "backend/micvec/Backend.h"
#else
#include "backend/scalar/Backend.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T NonZeroAbs(T const& x) {
  return Abs(x) + Tiny<T>::kValue;
}

template <typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
T NonZero(T const& x) {
  return x + CopySign( Tiny<T>::kValue, x);
}

} }; // end of vecgeom namespace

#endif // VECGEOM_BACKEND_BACKEND_H_
