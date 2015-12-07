#include "backend/cuda/Interface.h"
#include "base/Map.h"


__global__
void test_new(vecgeom::map<double,double>* map, double* key) {
   for (int i=0;i<250;i++)
   {
      double my_1 = (*map)[key[i]];
      double my_2 = map->find(key[i])->second;
      // std::cout<<"From the map= "<<my_1<<" and with find "<<my_2<<std::endl;
   }
}

__global__
void rebuild_map(vecgeom::map<double,double>* devMap, double* key, double* value, int N) {
   vecgeom::map<double,double> *myDevMap = new vecgeom::map<double, double>;
   //for (int i=0;i<N;i++)
      //std::cout<<" i "<<value[i]<<std::endl;

   for (int i=0;i<N;i++){
      //std::cout<<" i "<<key[i]<<" "<<value[i]<<std::endl;
      (*myDevMap)[key[i]] = value[i];
   }
}

namespace vecgeom {
namespace cxx {

template size_t DevicePtr<cuda::map<double,double> >::SizeOf();
template void DevicePtr<cuda::map<double,double> >::Construct() const;

} // End cxx namespace
}

void launch_test_new(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::cxx::DevicePtr<double> key, int nBlocks, int nThreads)
{
   int threadsPerBlock = nThreads;
   int blocksPerGrid   = nBlocks;
   test_new<<< blocksPerGrid, threadsPerBlock >>>(devMap,key);
}

void launch_rebuild_map(vecgeom::cxx::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::cxx::DevicePtr<double> key, vecgeom::cxx::DevicePtr<double> value, int N, int nBlocks, int nThreads)
{
   int threadsPerBlock = nThreads;
   int blocksPerGrid   = nBlocks;
   rebuild_map<<< blocksPerGrid, threadsPerBlock >>>(devMap,key,value,N);
}

