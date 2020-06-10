/// \file Raytracer.cu
/// \author Guilherme Amadio

#include "VecGeom/benchmarking/Raytracer.h"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/benchmarking/Raytracer.h>

#include <cassert>
#include <cstdio>

using namespace vecgeom;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d\n", cudaGetErrorString(result), file, line);
    cudaDeviceReset();
    exit(1);
  }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

__global__
void RenderKernel(RaytracerData_t rtdata, char *input_buffer, unsigned char *output_buffer)
{
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  int pixel_index = 4 * (py * rtdata.fSize_px + px);

  Color_t pixel_color = Raytracer::RaytraceOne(px, py, rtdata, input_buffer);

  output_buffer[pixel_index + 0] = pixel_color.fComp.red;
  output_buffer[pixel_index + 1] = pixel_color.fComp.green;
  output_buffer[pixel_index + 2] = pixel_color.fComp.blue;
  output_buffer[pixel_index + 3] = 255;
}

int RaytraceBenchmarkGPU(vecgeom::cxx::RaytracerData_t &rtdata)
int RaytraceBenchmarkGPU(cuda::VPlacedVolume const *const world, int px, int py, int maxdepth)
{
  using Vector3 = cuda::Vector3D<double>;

  // Allocate ray data and output data on the device
  size_t statesize = NavigationState::SizeOfInstance(maxdepth);
  size_t raysize = Ray_t::SizeOfInstance(maxdepth);

  printf("=== Allocating %.3f MB of ray data on the device\n", (float)rtdata.fNrays * raysize / 1048576);
  //char *input_buffer_gpu = nullptr;
  char *input_buffer = new char[statesize + rtdata.fNrays * raysize];
  checkCudaErrors(cudaMallocManaged((void **)&input_buffer, statesize + rtdata.fNrays * raysize));

  unsigned char *output_buffer = nullptr;
  checkCudaErrors(cudaMallocManaged((void **)&output_buffer, 4 * sizeof(unsigned char) * rtdata.fSize_px * rtdata.fSize_py));

  // Load and synchronize the geometry on the GPU
  vecgeom::cxx::CudaManager::Instance().LoadGeometry((vecgeom::cxx::VPlacedVolume*) world);
  vecgeom::cxx::CudaManager::Instance().Synchronize();

  // CudaManager is altering the stack size... setting an appropriate value
  //size_t def_stack_limit = 0, def_heap_limit = 0;
  //cudaDeviceGetLimit( &def_stack_limit, cudaLimitStackSize);
  cudaDeviceGetLimit( &def_heap_limit, cudaLimitMallocHeapSize);
  std::cout << "=== cudaLimitStackSize = " << def_stack_limit << "  cudaLimitMallocHeapSize = " << def_heap_limit << std::endl;
  //auto err = cudaDeviceSetLimit(cudaLimitStackSize, 4096);
  //cudaDeviceGetLimit( &def_stack_limit, cudaLimitStackSize);
  //std::cout << "=== CUDA thread stack size limit set now to: " << def_stack_limit << std::endl;
  
  auto gpu_world = vecgeom::cxx::CudaManager::Instance().world_gpu();
  assert(gpu_world && "GPU world volume is a null pointer");

  // Initialize the navigation state for the view point
  auto vpstate = NavigationState::MakeInstanceAt(rtdata.fMaxDepth, (void *)(input_buffer));
  Raytracer::LocateGlobalPoint(rtdata.fWorld, rtdata.fStart, *vpstate, true);
  rtdata.fVPstate = vpstate;
  rtdata.fWorld   = gpu_world;

  rtdata.Print();

// Construct rays in place
  char *raybuff = input_buffer + statesize;
  for (int iray = 0; iray < rtdata.fNrays; ++iray)
    Ray_t::MakeInstanceAt(raybuff + iray * raysize, rtdata.fMaxDepth);

  dim3 blocks(px / 8 + 1, py / 8 + 1), threads(8, 8);
  RenderKernel<<<blocks, threads>>>(rtdata, input_buffer, output_buffer);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  write_ppm("output.ppm", output_buffer, rtdata.fSize_px, rtdata.fSize_py);

  checkCudaErrors(cudaFree(input_buffer));
  checkCudaErrors(cudaFree(output_buffer));
}
