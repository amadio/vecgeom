#include "ShapeTester.h"
#include "VUSolid.hh"
//#include "UTubs.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Hype.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

#define PI 3.14159265358979323846

typedef vecgeom::SimpleHype Hype_t;

int main(int argc,char *argv[]) {
  int errCode= 0;
  VUSolid* hype;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
     hype=new Hype_t("test_VecGeomHype",5.,20,0,PI/4,50); 
    }
    else
    { 
     //hype=new UTubs("test_USolidsHype",1.,6.,2,0,vecgeom::kTwoPi*0.6);
     //hype->StreamInfo(std::cout);
    }
  }
  else
    {
     //hype=new UTubs("test_USolidsHype",1.,6.,2,0,vecgeom::kTwoPi*0.6);
     //hype->StreamInfo(std::cout);
    }
  
  ShapeTester tester;

  if(argc>2)
  {
    if(strcmp(argv[2],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(hype);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(hype);
  }
   std::cout<<"Final Error count for Shape *** "<<hype->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}



