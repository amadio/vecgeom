#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"
#include "USphere.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Sphere.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

#define PI 3.14159265358979323846
typedef vecgeom::SimpleSphere Sphere_t;

int main(int argc, char *argv[]) {

  // VUSolid* sphere=new USphere("test_sphere",15. , 20. , 0 ,2*PI/3, 2*PI/3 ,PI/6);
  // VUSolid* sphere=new Sphere_t("test_sphere",15. , 20. ,PI/6, 4.265389, PI/3 ,0.235869);
  // VUSolid* sphere=new Sphere_t("test_sphere",15. , 20. ,0.,  2*PI/3., PI/4., PI/6.);
  VUSolid *sphere = new Sphere_t("test_sphere", 15., 20., 0., 2 * PI / 3., 0., 2 * PI);
  // VUSolid* sphere=new USphere("test_USphere",3.);
  ShapeTester tester;
  tester.EnableDebugger(true);
  if (argc > 1) {

    tester.Run(sphere, argv[1]);
  } else {
    tester.Run(sphere);
  }

  return 0;
}
