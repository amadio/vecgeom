/// \file RNG.cpp
/// \author Philippe Canal (pcanal@fnal.gov)

#include "VecGeom/base/RNG.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECCORE_CUDA
class RNG;

// Emulating static class member ..
namespace RNGvar {
VECCORE_ATT_DEVICE unsigned long gMaxInstance;
VECCORE_ATT_DEVICE RNG **gInstances;
}
#endif
}
}
