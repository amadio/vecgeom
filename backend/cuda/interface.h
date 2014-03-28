/**
 * @file cuda/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_CUDA_INTERFACE_H_
#define VECGEOM_BACKEND_CUDA_INTERFACE_H_

#include "driver_types.h"

#include "base/global.h"

namespace vecgeom {

class VPlacedVolume;
template <typename Type> class SOA3D;
template <typename Type> class AOS3D;

cudaError_t CudaCheckError(const cudaError_t err);

cudaError_t CudaCheckError();

void CudaAssertError(const cudaError_t err);

void CudaAssertError();

cudaError_t CudaMalloc(void** ptr, unsigned size);

cudaError_t CudaCopyToDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaCopyFromDevice(void* tgt, void const* src, unsigned size);

cudaError_t CudaFree(void* ptr);

template <typename Type>
Type* AllocateOnGpu(const unsigned size) {
  Type *ptr;
  CudaAssertError(CudaMalloc((void**)&ptr, size));
  return ptr;
}

template <typename Type>
Type* AllocateOnGpu() {
  return AllocateOnGpu<Type>(sizeof(Type));
}

template <typename Type>
void FreeFromGpu(Type *const ptr) {
  CudaAssertError(CudaFree(ptr));
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    CudaCopyToDevice(tgt, src, size)
  );
}

template <typename Type>
void CopyToGpu(Type const *const src, Type *const tgt) {
  CopyToGpu<Type>(src, tgt, sizeof(Type));
}

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt, const unsigned size) {
  CudaAssertError(
    CudaCopyFromDevice(tgt, src, size)
  );
}

template <typename Type>
void CopyFromGpu(Type const *const src, Type *const tgt) {
  CopyFromGpu<Type>(src, tgt, sizeof(Type));
}

// Class specific stuf

void CudaManagerPrintGeometry(VPlacedVolume const *const world);

void CudaManagerLocatePoints(VPlacedVolume const *const world,
                             SOA3D<Precision> const *const points,
                             const int n, const int depth, int *const output);

void CudaManagerLocatePoints(VPlacedVolume const *const world,
                             AOS3D<Precision> const *const points,
                             const int n, const int depth, int *const output);

} // End global namespace

#endif // VECGEOM_BACKEND_CUDA_INTERFACE_H_