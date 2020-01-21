#include "VecGeom/base/Global.h"

namespace vecgeom {
class VPlacedVolume;
// instantiation of global device geometry data
namespace globaldevicegeomdata {
VECCORE_ATT_DEVICE
VPlacedVolume *gCompactPlacedVolBuffer;
}
}
