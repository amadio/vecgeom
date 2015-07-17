#include  <map>
#include  <cassert>
#include  <cstdlib>
//#include  <iostream>
#include  <vector>
#include  "Map.h"

#if defined(VECGEOM_VTUNE)
#include "ittnotify.h"
 __itt_domain* __itt_mymap = __itt_domain_create("myMapTest");
 __itt_domain* __itt_stdmap = __itt_domain_create("stdMapTest");
#endif

int get_random_int()
{
  static bool initialized = false;
  if (!initialized) {
    srand(time(NULL));
    initialized=true;
  }
  return rand() %100+1;
}


int main() {
  const int size = 5000;

  std::vector<int> map_keys(size);
  std::vector<int> map_values(size);
  std::vector<int> retrieve_keys(size);

   
  for (int i=0;i<size;i++)
  {
    map_values[i] = get_random_int();
    map_keys[i] = get_random_int();
    retrieve_keys[i] = get_random_int();
  }

// test VecCore::map
#if defined(VECGEOM_VTUNE)
 __itt_resume();
 __itt_frame_begin_v3(__itt_mymap,NULL); 
#endif

  VecCore::map<int,int> myMap;
  for (int i=0;i<size;i++)
  {
      //myMap[map_keys[i]]=map_values[i];
      std::pair<int,int>* p = new std::pair<int,int>(map_keys[i],map_values[i]); 
      //myMap.insert(std::pair<int,int>(map_keys[i],map_values[i]));
      myMap.insert(*p);
  }
  for (int i=0;i<250;i++)
  {
     int my_1 = myMap[map_keys[i]];    
     int my_2 = myMap.find(map_keys[i])->second;    
     std::cout<<"from the vector= "<< map_values[i]<<" from the map= "<<my_1<<" and with find "<<my_2<<std::endl;
  }
#if defined(VECGEOM_VTUNE)
  __itt_frame_end_v3(__itt_mymap,NULL); 
#endif
  // test std::map 
#if defined(VECGEOM_VTUNE)
  __itt_frame_begin_v3(__itt_stdmap,NULL); 
#endif
  std::map<int,int> stdMap;
  for (int i=0;i<size;i++)
  {
    //stdMap[map_keys[i]]=map_values[i];
    stdMap.insert(std::pair<char,int>(map_keys[i],map_values[i]));
  }  

  for (int i=0;i<250;i++)
  {
     int std_1 = stdMap[map_keys[i]];
     int std_2 = stdMap.find(map_keys[i])->second;
     std::cout<<"from the vector= "<< map_values[i]<<" from std map= "<<std_1<<" and with find "<<std_2<<std::endl;
  }


#if defined(VECGEOM_VTUNE)
__itt_frame_end_v3(__itt_stdmap,NULL); 
__itt_pause();
#endif
/*
  ofstream myfile;
  myfile.open("example.txt");
  myfile <<"Sofia Sofia\n";
  myfile.close();
*/
  return 0;
}
