/// \file shape_testParaboloid.cpp
/// \author Raman Sehgal (raman.sehgal@cern.ch)
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"
#include "UParaboloid.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Paraboloid.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

// typedef UBox Box_t;
typedef vecgeom::SimpleParaboloid Paraboloid_t;

int main(int argc, char *argv[])
{

  VUSolid *para = new Paraboloid_t("test_para", 6., 10., 10.);
  ShapeTester tester;
  // tester.EnableDebugger(true);
  tester.SetSolidTolerance(1.e-7);
  if (argc > 1) {
    /*
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     tester.Run(para);
     theApp.Run();
     #endif
    }
   */
    tester.Run(para, argv[1]);
  } else {
    tester.Run(para);
  }

  return 0;
}
