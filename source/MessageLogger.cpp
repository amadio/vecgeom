#include "base/MessageLogger.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

MessageLogger* MessageLogger:: gMessageLogger=0;
map<MessageLogger::logging_severity,map<string,map<string,int> > > MessageLogger::gMessageCount;

}
}
