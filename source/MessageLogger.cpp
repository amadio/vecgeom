#include "base/MessageLogger.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

MessageLogger *MessageLogger::gMessageLogger = 0;
#ifndef VECGEOM_NVCC
MessageLogger::Map_t MessageLogger::gMessageCount;
#endif
}
}
