#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/NavStatePool.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/management/RootGeoManager.h"
#include "VecGeom/management/GeoManager.h"
#ifdef VECCORE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

#include <iostream>
using namespace vecgeom;

#ifdef VECGEOM_ENABLE_CUDA
// declaration of some external "user-space" kernel
extern void LaunchNavigationKernel(void *gpu_ptr, int depth, int);
#endif

int main()
{
  // Load a geometry
  RootGeoManager::Instance().LoadRootGeometry("ExN03.root");
#ifdef VECGEOM_ENABLE_CUDA
  CudaManager::Instance().set_verbose(3);
  CudaManager::Instance().LoadGeometry();

  // why do I have to do this here??
  CudaManager::Instance().Synchronize();

  CudaManager::Instance().PrintGeometry();

  std::cout << std::flush;
#endif

  // generate some points
  int npoints = 3;
  SOA3D<Precision> testpoints(npoints);
  // testpoints.reserve(npoints)
  // testpoints.resize(npoints);

  // generate some points in the world
  volumeUtilities::FillContainedPoints(*GeoManager::Instance().GetWorld(), testpoints, false);

  NavStatePool pool(npoints, GeoManager::Instance().getMaxDepth());
  pool.Print();
  std::cerr << "#################" << std::endl;

  // fill states
  for (unsigned int i = 0; i < testpoints.size(); ++i) {
    //     std::cerr << testpoints[i] << "\n";
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), testpoints[i], *pool[i], true);
  }
  pool.Print();
  std::cerr << "sizeof navigation state on CPU: " << sizeof(vecgeom::cxx::NavigationState) << " and "
            << sizeof(vecgeom::NavigationState) << " bytes/state\n";

#ifdef VECGEOM_ENABLE_CUDA
  pool.CopyToGpu();
#endif

// launch some kernel on GPU using the
#ifdef VECGEOM_ENABLE_CUDA
  printf("TestNavStatePool: calling LaunchNavigationKernel()...\n");
  LaunchNavigationKernel(pool.GetGPUPointer(), GeoManager::Instance().getMaxDepth(), npoints);
  printf("TestNavStatePool: waiting for CudaDeviceSynchronize()...\n");
  cudaDeviceSynchronize();
  printf("TestNavStatePool: synchronized!\n");
#endif

  // #ifdef VECGEOM_ENABLE_CUDA
  //   pool.CopyFromGpu();
  // #endif

  pool.Print();
}
