// Author: Stephan Hageboeck, CERN, 2021

/** \file GeometryTest.cpp
 * This test validates a given geometry on the GPU.
 * - A geometry is read from a GDML file passed as argument.
 * - This geometry is constructed both for CPU and GPU.
 * - It is subsequently visited on the GPU, while information like volume IDs and transformations are recorded in a
 * large array.
 * - This array is copied to the host, and the host geometry is visited, comparing the data coming from the GPU.
 *
 * Which data are recorded and how they are compared is completely controlled by the struct GeometryInfo.
 */

#include "GeometryTest.h"

#include "VecGeom/base/Global.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/CudaManager.h"

#ifdef VECGEOM_GDML
#include "Frontend.h" // VecGeom/gdml/Frontend.h
#endif

#include <err.h>

using namespace vecgeom;

void visitVolumes(const VPlacedVolume *volume, GeometryInfo *geoData, std::size_t &volCounter,
                  const std::size_t nGeoData, unsigned int depth)
{
  assert(volCounter < nGeoData);
  geoData[volCounter++] = GeometryInfo{depth, *volume};

  for (const VPlacedVolume *daughter : volume->GetDaughters()) {
    visitVolumes(daughter, geoData, volCounter, nGeoData, depth + 1);
  }
}

void compareGeometries(const cxx::VPlacedVolume *hostVolume, std::size_t &volumeCounter,
                       const std::vector<GeometryInfo> &deviceGeometry, unsigned int depth)
{
  if (volumeCounter >= deviceGeometry.size()) {
    errx(3, "No device volume corresponds to volume %lu", volumeCounter);
  }

  GeometryInfo info{depth, *hostVolume};
  if (!(info == deviceGeometry[volumeCounter])) {
    printf("CPU transformation:\n");
    info.trans.Print();
    printf("\nGPU transformation\n");
    deviceGeometry[volumeCounter].trans.Print();
    printf("\n");
    printf("CPU amin: (%g, %g, %g)  amax: (%g, %g, %g)\n", info.amin[0], info.amin[1], info.amin[2], info.amax[0],
           info.amax[1], info.amax[2]);
    printf("GPU amin: (%g, %g, %g)  amax: (%g, %g, %g)\n", deviceGeometry[volumeCounter].amin[0],
           deviceGeometry[volumeCounter].amin[1], deviceGeometry[volumeCounter].amin[2],
           deviceGeometry[volumeCounter].amax[0], deviceGeometry[volumeCounter].amax[1],
           deviceGeometry[volumeCounter].amax[2]);
    errx(4, "Volume #%lu (id=%d label=%s logicalId=%d) differs from GPU volume (id=%d logicalId=%d)\n", volumeCounter,
         hostVolume->id(), hostVolume->GetLabel().c_str(), hostVolume->GetLogicalVolume()->id(),
         deviceGeometry[volumeCounter].id, deviceGeometry[volumeCounter].logicalId);
  }
  volumeCounter++;

  for (const cxx::VPlacedVolume *daughter : hostVolume->GetDaughters()) {
    compareGeometries(daughter, volumeCounter, deviceGeometry, depth + 1);
  }
}

int main(int argc, char **argv)
{
#ifdef VECGEOM_GDML
  bool verbose   = true;
  bool validate  = true;
  double mm_unit = 0.1;

  if (argc == 1) errx(ENOENT, "No input GDML file. Usage: ./GeometryTest <gdml>");

  const char *filename = argv[1];

  if (!filename || !vgdml::Frontend::Load(filename, validate, mm_unit, verbose))
    errx(EBADF, "Cannot open file '%s'", filename);

  auto &geoManager  = GeoManager::Instance();
  auto &cudaManager = CudaManager::Instance();

  if (!geoManager.IsClosed()) errx(1, "Geometry not closed");

  cudaManager.LoadGeometry(geoManager.GetWorld());
  cudaManager.Synchronize();

  printf("Visiting device geometry ... ");
  auto deviceGeometry = visitDeviceGeometry(cudaManager.world_gpu());

  printf("Comparing to host geometry ... ");
  std::size_t volumeCounter = 0;
  compareGeometries(geoManager.GetWorld(), volumeCounter, deviceGeometry, 0);
  printf("%lu volumes. Done.\n", volumeCounter);
#endif
  return EXIT_SUCCESS;
}
