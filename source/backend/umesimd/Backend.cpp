/**
 * @file umesimd/backend.cpp
 * @author Przemyslaw Karpinski (przemyslaw.karpinski@cern.ch)
 */

#include "backend/umesimd/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
const kUmeSimd::bool_v kUmeSimd::kTrue = true;
const kUmeSimd::bool_v kUmeSimd::kFalse = false;
#ifdef VECGEOM_FLOAT_PRECISION
const kUmeSimd::precision_v kUmeSimd::kOne = 1.0f;
const kUmeSimd::precision_v kUmeSimd::kZero = 0.0f;
#else
const kUmeSimd::precision_v kUmeSimd::kOne = 1.0;
const kUmeSimd::precision_v kUmeSimd::kZero = 0.0;
#endif
}
}
