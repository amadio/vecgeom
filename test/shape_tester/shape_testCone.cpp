#include "ShapeTester.h"
#include "VUSolid.hh"

#include "UCons.hh"
#include "volumes/Cone.h"

//#include "stdlib.h"

using Cone_t = vecgeom::SimpleCone;

int main(int argc, char *argv[])
{
  VUSolid *solid = 0;

  solid = new Cone_t("VecGeomCone", 5., 6., 5.5, 7., 2, 0, vecgeom::kTwoPi);
  // solid = new UCons("USolidsCone", 5., 6., 5.5, 7., 7., 0, vecgeom::kTwoPi * 0.3);

  // solid->StreamInfo(std::cout);
  vecgeom::VPlacedVolume *vgSolid = dynamic_cast<vecgeom::VPlacedVolume *>(solid);
  if (vgSolid) vgSolid->GetUnplacedVolume()->Print(std::cout);

  ShapeTester tester;
  if (argc > 1) {
    tester.Run(solid, argv[1]);
  } else {
    int errCode = tester.Run(solid);
    // tester.SetMethod("Consistency");
    std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << std::endl;
    std::cout << "=========================================================" << std::endl;
    return errCode;
  }

  return 0;
}
