#include "VecGeom/backend/cuda/Interface.h"
#include "VecGeom/base/Map.h"
class MyClass {
private:
  float fData;

public:
  VECCORE_ATT_HOST_DEVICE
  MyClass() { fData = 0; }
  VECCORE_ATT_HOST_DEVICE
  MyClass(float data) { fData = data; }
  VECCORE_ATT_HOST_DEVICE
  float getData() const { return fData; }
};

__global__ void testNew(vecgeom::map<double, MyClass> *devMap, double *key, int N)
{
  for (int i = 0; i < N; i++) {
    MyClass key1 = (*devMap)[key[i]];
    MyClass key2 = devMap->find(key[i])->second;
    // printf("Key %f, Value from op[] = %f and from find %f\n",key[i],key1, key2);
  }
}

__global__ void rebuildMap(vecgeom::map<double, MyClass> *devMap, double *key, MyClass *value, int N)
{
  //  vecgeom::map<double,double> *myDevMap = new vecgeom::map<double, double>;
  // for (int i=0;i<N;i++)
  // std::cout<<" i "<<value[i]<<std::endl;

  for (int i = 0; i < N; i++) {
    (*devMap)[key[i]] = value[i];
    printf(" REBUILDING key %f and value %f from op[]\n ", key[i], ((*devMap)[key[i]]).getData());
    auto search = devMap->find(key[i]);
    printf(" REBUILDING key %f and value %f from find\n ", key[i], (search->second).getData());
  }
}

namespace vecgeom {
namespace cxx {

template size_t DevicePtr<MyClass>::SizeOf();
template void DevicePtr<MyClass>::Construct() const;
template size_t DevicePtr<cuda::map<double, MyClass>>::SizeOf();
template void DevicePtr<cuda::map<double, MyClass>>::Construct() const;

} // End cxx namespace
}

void launchTestNew(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double, MyClass>> &devMap,
                   vecgeom::cxx::DevicePtr<double> key, int N, int nBlocks, int nThreads)
{
  int threadsPerBlock = nThreads;
  int blocksPerGrid   = nBlocks;
  testNew<<<blocksPerGrid, threadsPerBlock>>>(devMap, key, N);
}

void launchRebuildMap(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double, MyClass>> &devMap,
                      vecgeom::cxx::DevicePtr<double> key, vecgeom::cxx::DevicePtr<MyClass> value, int N, int nBlocks,
                      int nThreads)
{
  int threadsPerBlock = nThreads;
  int blocksPerGrid   = nBlocks;
  rebuildMap<<<blocksPerGrid, threadsPerBlock>>>(devMap, key, value, N);
}
