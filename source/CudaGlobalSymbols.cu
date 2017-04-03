#include "base/Global.h"

namespace vecgeom {
class VPlacedVolume;
// instantiation of global device geometry data
namespace globaldevicegeomdata {
//#ifdef VECCORE_CUDA_DEVICE_COMPILATION
__device__ VPlacedVolume *gCompactPlacedVolBuffer;
//#endif
}
}
