#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "base/Vector3D.h"

#include "UTrap.hh"
#include "volumes/Trapezoid.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif

using VGTrap    = vecgeom::SimpleTrapezoid;
using Precision = vecgeom::Precision;
using Vec_t     = vecgeom::Vector3D<vecgeom::Precision>;

template <typename Trap_t>
VUSolid *buildTrap1()
{
  return new Trap_t("FullTrap", 40, 0, 0, 30, 20, 20, 0, 30, 20, 20, 0);
}

template <typename Trap_t>
VUSolid *buildBoxLikeTrap(double dx, double dy, double dz)
{
  return new Trap_t("BoxLikeTrap", dz, 0, 0, dy, dx, dx, 0, dy, dx, dx, 0);
}

template <typename Trap_t>
VUSolid *buildTrapFromCorners()
{

  // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
  vecgeom::TrapCorners xyz;
  Precision xoffset = 9;
  Precision yoffset = -6;

  // define corner points
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  xyz[0] = Vec_t(-2 + xoffset, -5 + yoffset, -15);
  xyz[1] = Vec_t(2 + xoffset, -5 + yoffset, -15);
  xyz[2] = Vec_t(-3 + xoffset, 5 + yoffset, -15);
  xyz[3] = Vec_t(3 + xoffset, 5 + yoffset, -15);
  xyz[4] = Vec_t(-4 - xoffset, -10 - yoffset, 15);
  xyz[5] = Vec_t(4 - xoffset, -10 - yoffset, 15);
  xyz[6] = Vec_t(-6 - xoffset, 10 - yoffset, 15);
  xyz[7] = Vec_t(6 - xoffset, 10 - yoffset, 15);

  // create trapezoid
  return new Trap_t("slantedTrap", xyz);
}

int main(int argc, char *argv[])
{
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 2);
  OPTION_BOOL(usolids, false);

  VUSolid *solid = 0;
  switch (type) {
  case 0:
    std::cout << "Building default trapezoid\n";
    if (usolids)
      solid = buildTrap1<UTrap>();
    else
      solid = buildTrap1<VGTrap>();
    break;
  case 1:
    std::cout << "Building box-like trapezoid\n";
    if (usolids)
      solid = buildBoxLikeTrap<UTrap>(40, 30, 20);
    else
      solid = buildBoxLikeTrap<VGTrap>(40, 30, 20);
    break;
  case 2:
    std::cout << "Building trapezoid from corners\n";
    if (usolids)
      solid = buildTrapFromCorners<UTrap>();
    else
      solid = buildTrapFromCorners<VGTrap>();
    break;
  default:
    std::cout << "*** No trap type provided.\n";
  }

  // show parameters
  vecgeom::VPlacedVolume *vgSolid = dynamic_cast<vecgeom::VPlacedVolume *>(solid);
  if (vgSolid)
    vgSolid->GetUnplacedVolume()->Print(std::cout);
  else
    solid->StreamInfo(std::cout);

  // setup the shape tester
  ShapeTester tester;
  if (debug) tester.setDebug(true);
  if (stat) tester.setStat(true);

  // run tests
  int errCode = tester.Run(solid);
  // int errCode = tester.SetMethod("Consistency");
  std::cout << "Final Error count for Shape *** " << solid->GetName() << " *** = " << errCode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return errCode;
}
