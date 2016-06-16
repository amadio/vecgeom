#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UOrb.hh"

#include "base/Vector3D.h"
#include "volumes/Orb.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimpleOrb Orb_t;

int main(  int argc,char *argv[]) {

  VUSolid* orb=new Orb_t("test_orb",35);
   // VUSolid* orb=new UOrb("test_UOrb",3.);
  ShapeTester tester;
  //tester.EnableDebugger(true);
  if(argc>1)
  {
    /*
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     tester.Run(orb);
     theApp.Run();
     #endif
    }
   */
  tester.Run(orb,argv[1]);
  }
  else
  {
    tester.Run(orb);

   }

  return 0;
}



