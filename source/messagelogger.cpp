#include "base/messagelogger.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

messagelogger* messagelogger:: gMessageLogger=0;
map<messagelogger::logging_severity,map<string,map<string,int> > > messagelogger:: gMessageCount;

}
}
