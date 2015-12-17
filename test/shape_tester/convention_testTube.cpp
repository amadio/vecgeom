#include "ShapeTester.h"
//#include "VUSolid.hh"

#include "base/Vector3D.h"
#include "volumes/Tube.h"

class VUSolid;
//#ifdef VECGEOM_ROOT
//#include "TApplication.h"
//#endif
//#include "stdlib.h"

#define PI 3.14159265358979323846

typedef vecgeom::SimpleTube Tube_t;

int main(  int argc,char *argv[]) {
  VUSolid* tube = new Tube_t("testTube",0.,20.,25.,0.,2.*PI);
  ShapeTester tester;
  tester.RunConventionChecker(tube);
  return 0;
}
