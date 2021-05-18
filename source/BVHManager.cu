/// \file BVHManager.cu
/// \author Guilherme Amadio

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/backend/cuda/Interface.h"

using vecgeom::cxx::CudaCheckError;

namespace vecgeom {
inline namespace cuda {

static __device__ BVH *dBVH;

void *AllocateDeviceBVHBuffer(size_t n)
{
  BVH *ptr = nullptr;
  CudaCheckError(cudaMalloc((void **)&ptr, n * sizeof(BVH)));
  CudaCheckError(cudaMemcpyToSymbol(dBVH, &ptr, sizeof(ptr)));
  CudaCheckError(cudaDeviceSynchronize());
  return (void*) ptr;
}

void FreeDeviceBVHBuffer()
{
  void *ptr = nullptr;

  CudaCheckError(cudaMemcpyFromSymbol(&ptr, dBVH, sizeof(ptr)));

  if (ptr)
    CudaCheckError(cudaFree(ptr));
}

__device__
BVH *GetDeviceBVH(int id)
{
  return &dBVH[id];
}

} // namespace cuda
} // namespace vecgeom
