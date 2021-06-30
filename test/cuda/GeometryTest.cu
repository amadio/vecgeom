// Author: Stephan Hageboeck, CERN, 2021

#include "GeometryTest.h"

#include <cstdio>
#include <err.h>

__managed__ std::size_t g_volumesVisited;
__managed__ bool g_problemDuringVisit;

__device__
void visitVolumes(const vecgeom::cuda::VPlacedVolume * volume, GeometryInfo * geoData, std::size_t & volCounter,
                 const std::size_t nGeoData, unsigned int depth) {
  if (volCounter >= nGeoData) {
    g_problemDuringVisit = true;
    printf("Sorry, hard-coded buffer size exhausted after visiting %lu volumes. Please increase.\n", volCounter);
    return;
  }
  geoData[volCounter++] = GeometryInfo{depth, *volume};

  for (const vecgeom::cuda::VPlacedVolume * daughter : volume->GetDaughters()) {
    visitVolumes(daughter, geoData, volCounter, nGeoData, depth + 1);
    if (g_problemDuringVisit) break;
  }
}

__global__
void kernel_visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume * volume, GeometryInfo * geoData,
                                const std::size_t nGeoData) {
  g_volumesVisited = 0;
  g_problemDuringVisit = false;
  visitVolumes(volume, geoData, g_volumesVisited, nGeoData, 0);
}

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume* volume) {
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Cuda error before visiting device geometry: '%s'", cudaGetErrorString(err));
  }

  constexpr std::size_t maxElem = 100000;

  GeometryInfo * geoDataGPU;
  cudaMalloc(&geoDataGPU, maxElem * sizeof(GeometryInfo));

  kernel_visitDeviceGeometry<<<1,1>>>(volume, geoDataGPU, maxElem);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Visiting device geometry failed with '%s'", cudaGetErrorString(err));
  } else if (g_problemDuringVisit) {
    errx(2, "Visiting device geometry failed.");
  }

  std::vector<GeometryInfo> geoDataCPU(maxElem);
  cudaMemcpy(geoDataCPU.data(), geoDataGPU, maxElem * sizeof(GeometryInfo), cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    errx(2, "Retrieving device geometry data failed with '%s'", cudaGetErrorString(err));
  }

  cudaFree(geoDataGPU);

  geoDataCPU.resize(g_volumesVisited);
  printf(" %lu visited.\n", g_volumesVisited);

  return geoDataCPU;
}


