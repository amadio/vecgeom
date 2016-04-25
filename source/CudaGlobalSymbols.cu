#include "base/Global.h"

namespace vecgeom {
class VPlacedVolume;
// instantiation of global device geometry data
namespace globaldevicegeomdata {
//#ifdef VECGEOM_NVCC_DEVICE
__device__ VPlacedVolume *gCompactPlacedVolBuffer;
//#endif
}

}

