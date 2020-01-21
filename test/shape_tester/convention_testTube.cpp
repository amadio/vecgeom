#include "ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include "VecGeom/volumes/Tube.h"

#define PI 3.14159265358979323846

typedef vecgeom::SimpleTube Tube_t;

int main(int argc, char *argv[])
{
  Tube_t const *tube = new Tube_t("testTube", 30., 50., 50., 0., 1.67 * PI);
  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.RunConventionChecker(tube);
  tester.SetMaxPoints(300);
  return 0;
}
