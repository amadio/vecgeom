#include  <map>
#include  <cassert>
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

VECGEOM_CUDA_HEADER_HOST
double get_random_int()
{
   return RNG::Instance().uniform();
}

VECGEOM_CUDA_HEADER_HOST
void test_std(int size, double* keys,double* values) {
   std::map<double,double> stdMap;
   for (int i=0;i<size;i++)
      {
         //stdMap[map_keys[i]]=map_values[i];
         stdMap.insert(std::pair<double,double>(keys[i],values[i]));
      }

   for (int i=0;i<250;i++)
      {
         double std_1 = stdMap[keys[i]];
         double std_2 = stdMap.find(keys[i])->second;
         std::cout<<"From std map= "<<std_1<<" and with find "<<std_2<<std::endl;
      }
}

VECGEOM_CUDA_HEADER_DEVICE
void test_new(vecgeom::map<double,double>* map, double* key) {
   for (int i=0;i<250;i++)
      {
         double my_1 = (*map)[key[i]];
         double my_2 = map->find(key[i])->second;
         std::cout<<"From the map= "<<my_1<<" and with find "<<my_2<<std::endl;
      }
}


VECGEOM_CUDA_HEADER_DEVICE
void rebuild_map(vecgeom::map<double,double>* &devMap,double* key, double* value, int N) {
   vecgeom::map<double,double> *myDevMap = new vecgeom::map<double, double>;
   for (int i=0;i<N;i++)
      std::cout<<" i "<<value[i]<<std::endl;

   for (int i=0;i<N;i++){
      std::cout<<" i "<<key[i]<<" "<<value[i]<<std::endl;
      (*myDevMap)[key[i]] = value[i];
   }
}


int main() {
   const int size = 50;

   double* map_keys = new double[size];
   double* map_values=new double[size]   ;
   //std::vector<int> retrieve_keys(size);
   vecgeom::map<double,double>* devMap = NULL;
   
   for (int i=0;i<size;i++)
      {
         map_values[i] = get_random_int();
         map_keys[i] = get_random_int();
         //retrieve_keys[i] = get_random_int();
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
   if (cudaGetLastError() != cudaSuccess) std::cout<<" ERROR ALLOC KEYS "<<std::endl;
   vecgeom::DevicePtr<double> map_values_dev;
   map_values_dev.Allocate(size);
   if (cudaGetLastError() != cudaSuccess) std::cout<<" ERROR ALLOC VALUES "<<std::endl;
   
   // vecgeom::cxx::CopyToGpu<double*>(const_cast<double*> (map_keys_dev), map_keys);
   map_keys_dev.ToDevice(map_keys,size);
   if(cudaSuccess!=cudaGetLastError()) std::cout<<"ERROR MEMCPY keys"<<std::endl;
   map_values_dev.ToDevice(map_values,size);
   if(cudaSuccess!=cudaGetLastError()) std::cout<<"ERROR MEMCPY values"<<std::endl;
   std::cout<<" rebuild map "<<std::endl;
   rebuild_map(devMap, map_keys_dev,map_values_dev,size);
   test_new(devMap, map_keys_dev);
  
   //vecgeom::cxx::CopyFromGpu<vecgeom::map<double,double>> (const_cast<vecgeom::map<double,double>*> (hostMap), devMap);
   // cudaMemcpy(hostMap, devMap, sizeof(devMap),cudaMemcpyDeviceToHost);
  
   // std::cout<<"MAP IS BACK"<<std::endl;

   //test_new(hostMap, map_keys);

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
   /*
     ofstream myfile;
     myfile.open("example.txt");
     myfile <<"Sofia Sofia\n";
     myfile.close();
   */

   // free(hostMap);
   delete map_keys;
   delete map_values;
   // cudaFree(devMap);
   map_keys_dev.Deallocate();
   map_values_dev.Deallocate();
   return 0;

}
