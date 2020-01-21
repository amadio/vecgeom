#include <iostream>
#include <string>

// vecgeom::map
#include "VecGeom/base/Map.h"
// stl map
//#include <map>

// to generate random string with a given length
void gen_random(char *s, const int len)
{
  static const char alphanum[] =
      /*"0123456789"*/
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }
  s[len] = 0;
}

int main()
{
  // fill a map with num <string,int> key-value pairs; random string length with max-length of maxL
  int num  = 300;
  int maxL = 35;
  vecgeom::map<std::string, int> aMap;
  // std::map<std::string,int> aMap;

  for (int i = 0; i < num; ++i) {
    char cstr[512];
    int length = rand() % maxL + 1;
    gen_random(cstr, length);
    std::string str(cstr);
    aMap[str] = 1;
  }
  // aMap.clear();

  std::cout << " ==== END ===== " << std::endl;
  return 0;
}
