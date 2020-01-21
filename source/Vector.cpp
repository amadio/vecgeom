/// \file Vector.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/base/Vector.h"

namespace vecgeom {

#ifdef VECCORE_CUDA

inline namespace cuda {
class VPlacedVolume;
}

namespace cxx {

template size_t DevicePtr<cuda::Vector<Precision>>::SizeOf();
template void DevicePtr<cuda::Vector<Precision>>::Construct(DevicePtr<Precision> const arr, const size_t size) const;
template void DevicePtr<cuda::Vector<Precision>>::Construct(DevicePtr<Precision> const arr, const size_t size,
                                                            const size_t maxsize) const;

template size_t DevicePtr<cuda::Vector<cuda::VPlacedVolume *>>::SizeOf();
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume *>>::Construct(DevicePtr<cuda::VPlacedVolume *> const arr,
                                                                        const size_t size) const;
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume *>>::Construct(DevicePtr<cuda::VPlacedVolume *> const arr,
                                                                        const size_t size, const size_t maxsize) const;

template size_t DevicePtr<cuda::Vector<cuda::VPlacedVolume const *>>::SizeOf();
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume const *>>::Construct(
    DevicePtr<cuda::VPlacedVolume const *> const arr, const size_t size) const;
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume const *>>::Construct(
    DevicePtr<cuda::VPlacedVolume const *> const arr, const size_t size, const size_t maxsize) const;

} // End cxx namespace

#endif

} // End global namespace
