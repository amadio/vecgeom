/**
 * @file umesimd/backend.cpp
 * @author Przemyslaw Karpinski (przemyslaw.karpinski@cern.ch)
 */

#include "backend/umesimd/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
  const UmeSimdMask kUmeSimd::kTrue = UmeSimdMask(true);
  const UmeSimdMask kUmeSimd::kFalse = UmeSimdMask(false);
#ifdef VECGEOM_FLOAT_PRECISION
  const UmeSimdPrecisionVector kUmeSimd::kOne = UmeSimdPrecisionVector(1.0f);
  const UmeSimdPrecisionVector kUmeSimd::kZero = UmeSimdPrecisionVector(0.0f);
#else
  const UmeSimdPrecisionVector kUmeSimd::kOne = UmeSimdPrecisionVector(1.0);
  const UmeSimdPrecisionVector kUmeSimd::kZero = UmeSimdPrecisionVector(0.0);
#endif
}
}
