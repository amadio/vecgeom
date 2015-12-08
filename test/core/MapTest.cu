#include "backend/cuda/Interface.h"
#include "base/Map.h"


__global__
void test_new(vecgeom::map<double,double>* map, double* key, int N) {
   for (int i=0;i<N;i++)
   {
      double my_1 = (*map)[key[i]];
      double my_2 = map->find(key[i])->second;
      printf("Key %f, Value from op[] = %f and from find %f\n",key[i],my_1, my_2);
   }
}

__global__
void rebuild_map(vecgeom::map<double,double>* devMap, double* key, double* value, int N) {
 //  vecgeom::map<double,double> *myDevMap = new vecgeom::map<double, double>;
   //for (int i=0;i<N;i++)
      //std::cout<<" i "<<value[i]<<std::endl;

   for (int i=0;i<N;i++){
      printf(" REBUILDING key %f and value %f\n ",key[i],value[i]);
      (*devMap)[key[i]] = value[i];
   }
}

namespace vecgeom {
namespace cxx {

template size_t DevicePtr<cuda::map<double,double> >::SizeOf();
template void DevicePtr<cuda::map<double,double> >::Construct() const;

} // End cxx namespace
}

void launch_test_new(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::cxx::DevicePtr<double> key, int N, int nBlocks, int nThreads)
{
   int threadsPerBlock = nThreads;
   int blocksPerGrid   = nBlocks;
   test_new<<< blocksPerGrid, threadsPerBlock >>>(devMap,key, N);
}

void launch_rebuild_map(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::cxx::DevicePtr<double> key, vecgeom::cxx::DevicePtr<double> value, int N, int nBlocks, int nThreads)
{
   int threadsPerBlock = nThreads;
   int blocksPerGrid   = nBlocks;
   rebuild_map<<< blocksPerGrid, threadsPerBlock >>>(devMap,key,value,N);
}

