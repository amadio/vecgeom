#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UGenericTrap.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/GenTrap.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"


typedef vecgeom::SimpleGenTrap GenTrap_t;

int main(int argc,char *argv[]) {
 using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
 using namespace vecgeom;
  std::vector<Vec_t> vertexlist1;
  vertexlist1.push_back( Vec_t(-3,-3, 0 ) );
  vertexlist1.push_back( Vec_t(-3, 3, 0 ) );
  vertexlist1.push_back( Vec_t( 3, 3, 0 ) );
  vertexlist1.push_back( Vec_t( 3,-3, 0 ) );
  vertexlist1.push_back( Vec_t(-2,-2, 0 ) );
  vertexlist1.push_back( Vec_t(-2, 2, 0 ) );
  vertexlist1.push_back( Vec_t( 2, 2, 0 ) );
  vertexlist1.push_back( Vec_t( 2,-2, 0 ) );

  int errCode= 0;
  GenTrap_t* solid=new GenTrap_t("test_VecGeomGenTrap", &vertexlist1[0], 5);    
  
  ShapeTester tester;

  if(argc>1)
  {
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(solid);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(solid);
   }
  std::cout<<"Final Error count for Shape *** "<<solid->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}



