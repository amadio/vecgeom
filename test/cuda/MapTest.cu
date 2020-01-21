#include "VecGeom/backend/cuda/Interface.h"
#include "VecGeom/base/Map.h"

__global__ void testNew(vecgeom::map<double, double> *devMap, double *key, int N)
{
  for (int i = 0; i < N; i++) {
    double key1 = (*devMap)[key[i]];
    double key2 = devMap->find(key[i])->second;
    // printf("Key %f, Value from op[] = %f and from find %f\n",key[i],key1, key2);
  }
}

__global__ void rebuildMap(vecgeom::map<double, double> *devMap, double *key, double *value, int N)
{
  //  vecgeom::map<double,double> *myDevMap = new vecgeom::map<double, double>;
  // for (int i=0;i<N;i++)
  // std::cout<<" i "<<value[i]<<std::endl;

  for (int i = 0; i < N; i++) {
    printf(" REBUILDING key %f and value %f\n ", key[i], value[i]);
    (*devMap)[key[i]] = value[i];
  }
}

namespace vecgeom {
namespace cxx {

template size_t DevicePtr<cuda::map<double, double>>::SizeOf();
template void DevicePtr<cuda::map<double, double>>::Construct() const;

} // End cxx namespace
}

void launchTestNew(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double, double>> &devMap,
                   vecgeom::cxx::DevicePtr<double> key, int N, int nBlocks, int nThreads)
{
  int threadsPerBlock = nThreads;
  int blocksPerGrid   = nBlocks;
  testNew<<<blocksPerGrid, threadsPerBlock>>>(devMap, key, N);
}

void launchRebuildMap(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double, double>> &devMap,
                      vecgeom::cxx::DevicePtr<double> key, vecgeom::cxx::DevicePtr<double> value, int N, int nBlocks,
                      int nThreads)
{
  int threadsPerBlock = nThreads;
  int blocksPerGrid   = nBlocks;
  rebuildMap<<<blocksPerGrid, threadsPerBlock>>>(devMap, key, value, N);
}
