/// \file vector/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/Global.h"

#ifdef VECCORE_CUDA
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
VECCORE_ATT_HOST_DEVICE
T NonZeroAbs(T const &x)
{
  return Abs(x) + T(1.0e-30);
}

template <typename T>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
T NonZero(T const &x)
{
  return x + CopySign(T(1.0e-30), x);
}
}
}

#endif // VECGEOM_BACKEND_BACKEND_H_
