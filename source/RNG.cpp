/// \file RNG.cpp
/// \author Philippe Canal (pcanal@fnal.gov)

#include "base/RNG.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_NVCC
class RNG;

// Emulating static class member ..
namespace RNGvar {
VECCORE_ATT_DEVICE unsigned long gMaxInstance;
VECCORE_ATT_DEVICE RNG **gInstances;
}
#endif
}
}
