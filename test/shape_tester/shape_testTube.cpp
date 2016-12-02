#include "ShapeTester.h"
#include "VUSolid.hh"

#include "UTubs.hh"
#include "volumes/Tube.h"

//#include "stdlib.h"

using Tube_t = vecgeom::SimpleTube;

int main(int argc, char *argv[])
{
  VUSolid *solid = 0;

  solid = new Tube_t("VecGeomTube", 3., 6., 2, 0, vecgeom::kTwoPi * 0.6);
  // solid = new UTubs("USolidsTube", 1., 6., 2, 0, vecgeom::kTwoPi * 0.6);

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
