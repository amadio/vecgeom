#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "backend/micvec/Backend.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const MicBool kMic::kTrue      = MicBool(0xffff);
const MicBool kMic::kFalse     = MicBool(0x0);
const MicPrecision kMic::kOne  = MicPrecision(1.0);
const MicPrecision kMic::kZero = MicPrecision(0.0);
}
}

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif
