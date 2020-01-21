#include "VecGeom/base/MessageLogger.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

MessageLogger *MessageLogger::gMessageLogger = 0;
#ifndef VECCORE_CUDA
MessageLogger::Map_t MessageLogger::gMessageCount;
#endif
}
}
