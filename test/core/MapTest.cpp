#include  <map>
#include  <cstdlib>
//#include  <iostream>
#include  <vector>
#include  "base/Map.h"
#include  "base/RNG.h"
using vecgeom::RNG;
#include "backend/cuda/Interface.h"

#if defined(VECGEOM_VTUNE)
#include "ittnotify.h"
__itt_domain* __itt_mymap = __itt_domain_create("myMapTest");
__itt_domain* __itt_stdmap = __itt_domain_create("stdMapTest");
#endif

void launch_test_new(vecgeom::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::DevicePtr<double> key, int N, int nBlocks, int nThreads);
void launch_rebuild_map(vecgeom::DevicePtr<vecgeom::cuda::map<double,double> > &devMap, vecgeom::DevicePtr<double> key, vecgeom::DevicePtr<double> value, int N, int nBlocks, int nThreads);

VECGEOM_CUDA_HEADER_HOST
double get_random()
{
   return RNG::Instance().uniform();
}

VECGEOM_CUDA_HEADER_HOST
void test_std(int size, double* keys,double* values) {
   std::map<double,double> stdMap;
   for (int i=0;i<size;i++)
      {
         stdMap.insert(std::pair<double,double>(keys[i],values[i]));
      }

   for (int i=0;i<size;i++)
      {
         double std_1 = stdMap[keys[i]];
         double std_2 = stdMap.find(keys[i])->second;
         printf("From std map= %f and with find %f\n",std_1,std_2);
      }
}


int main() {
   const int size = 50;

   double* map_keys = new double[size];
   double* map_values =new double[size]   ;
   
   for (int i=0;i<size;i++)
      {
         map_values[i] = get_random();
         map_keys[i] = get_random();
         printf(" vectors %f, %f\n",map_keys[i],map_values[i]);
      }

   // test vecgeom::map
#if defined(VECGEOM_VTUNE)
   __itt_resume();
   __itt_frame_begin_v3(__itt_mymap,NULL);
#endif

   /*
     vecgeom::map<double,double>* hostMap =new vecgeom::map<double,double>();
     for (int i=0;i<size;i++)
     {
     //myMap[map_keys[i]]=map_values[i];
     std::pair<double,double>* p = new std::pair<double,double>(map_keys[i],map_values[i]);
     //myMap.insert(std::pair<int,int>(map_keys[i],map_values[i]));
     hostMap->insert(*p);
     }
   */

   vecgeom::DevicePtr<double> map_keys_dev;
   map_keys_dev.Allocate(size);
   if (cudaGetLastError() != cudaSuccess) {
     printf(" ERROR ALLOC KEYS\n");
     return 0;
   }
   vecgeom::DevicePtr<double> map_values_dev;
   map_values_dev.Allocate(size);
   if (cudaGetLastError() != cudaSuccess) {
     printf(" ERROR ALLOC VALUES\n");
     return 0;
   }
   vecgeom::DevicePtr<vecgeom::cuda::map<double,double> > devMap;
   devMap.Allocate(size);
   if (cudaGetLastError() != cudaSuccess) {
     printf(" ERROR ALLOC MAP\n");
     return 0;
   }
   devMap.Construct();

   map_keys_dev.ToDevice(map_keys,size);
   if (cudaSuccess!=cudaGetLastError()) {
      printf("ERROR MEMCPY keys\n");
      return 0;
   }
   map_values_dev.ToDevice(map_values,size);
   if(cudaSuccess!=cudaGetLastError()) {
      printf("ERROR MEMCPY values\n");
   }
   
   printf(" rebuild map\n");
   launch_rebuild_map(devMap, map_keys_dev,map_values_dev,size,1,1);
   launch_test_new(devMap, map_keys_dev,size,1,1);

#if defined(VECGEOM_VTUNE)
   __itt_frame_end_v3(__itt_mymap,NULL); 
#endif

#ifndef VECGEOM_NVCC
   // test std::map 
#if defined(VECGEOM_VTUNE)
   __itt_frame_begin_v3(__itt_stdmap,NULL); 
#endif

   //test_std(size, map_values,map_keys);


#if defined(VECGEOM_VTUNE)
   __itt_frame_end_v3(__itt_stdmap,NULL); 
   __itt_pause();
#endif
#endif

   delete map_keys;
   delete map_values;
   map_keys_dev.Deallocate();
   map_values_dev.Deallocate();
   return 0;

}
