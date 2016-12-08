#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "VUSolid.hh"
#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"

#include "UTrap.hh"
#include "volumes/Trapezoid.h"
#include <string>

using Precision     = vecgeom::Precision;
using Vec_t         = vecgeom::Vector3D<vecgeom::Precision>;
using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTrap        = vecgeom::SimpleTrapezoid;

template <typename Trap_t>
Trap_t *buildTrap1()
{
  return new Trap_t("FullTrap", 40, 0.12, 0.34, 15, 10, 10, 0, 30, 20, 20, 0);
}

template <typename Trap_t>
Trap_t *buildBoxLikeTrap(double dx, double dy, double dz)
{
  return new Trap_t("BoxLikeTrap", dz, 0, 0, dy, dx, dx, 0, dy, dx, dx, 0);
}

template <typename Trap_t>
Trap_t *buildTrapFromCorners()
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

template <typename Trap_t>
Trap_t *buildATrap(int type)
{

  switch (type) {
  case 0:
    std::cout << "Building default trapezoid\n";
    return buildTrap1<Trap_t>();
    break;
  case 1:
    std::cout << "Building box-like trapezoid\n";
    return buildBoxLikeTrap<Trap_t>(40, 30, 20);
    break;
  case 2:
    std::cout << "Building trapezoid from corners\n";
    return buildTrapFromCorners<Trap_t>();
    break;
  default:
    std::cout << "*** No trap type provided.\n";
  }
  return 0;
}

int main(int argc, char *argv[])
{
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);
  OPTION_INT(type, 2);
  OPTION_INT(npoints, 10000);

  int errcode = 0;
  std::string volname;
  if (usolids) {
    auto trap = buildATrap<UTrap>(type);
    volname   = trap->GetName();
    trap->StreamInfo(std::cout);
    ShapeTester<VUSolid> tester;
    tester.setConventionsMode(usolids);
    if (debug) tester.setDebug(true);
    if (stat) tester.setStat(true);
    tester.SetMaxPoints(npoints);
    errcode = tester.Run(trap);
  }

  else {
    auto trap = buildATrap<VGTrap>(type);
    volname   = trap->GetName();
    trap->Print();
    ShapeTester<VPlacedVolume> tester;
    if (debug) tester.setDebug(true);
    if (stat) tester.setStat(true);
    tester.SetMaxPoints(npoints);
    errcode = tester.Run(trap);
  }

  // int errCode = tester.SetMethod("Consistency");
  std::cout << "Final Error count for Shape *** " << volname << " *** = " << errcode << std::endl;
  std::cout << "=========================================================" << std::endl;
  return errcode;
}
