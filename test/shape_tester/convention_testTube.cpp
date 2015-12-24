
#include "ShapeTester.h"
#include "volumes/Tube.h"

class VUSolid;

#define PI 3.14159265358979323846

typedef vecgeom::SimpleTube Tube_t;

int main(  int argc,char *argv[]) {
  VUSolid* tube = new Tube_t("testTube",30.,50.,50.,0.,2.*PI);
  ShapeTester tester;
  tester.RunConventionChecker(tube);
  return 0;
}
