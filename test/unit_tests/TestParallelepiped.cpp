//
// File: TestParallelepiped.cpp
//
//
//    Ensure asserts are compiled in
//

//.. ensure asserts are compiled in
#undef NDEBUG
#include "base/FpeEnable.h"

#include "base/Vector3D.h"
#include "volumes/Parallelepiped.h"
#include "ApproxEqual.h"
#include <cmath>

template <class Parallelepiped_t>
bool TestParallelepiped()
{
  using namespace vecgeom::VECGEOM_IMPL_NAMESPACE;
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  EnumInside inside;
  const Precision dx    = 20;
  const Precision dy    = 30;
  const Precision dz    = 40;
  const Precision alpha = kDegToRad * 30;
  const Precision theta = kDegToRad * 30;
  const Precision phi   = kDegToRad * 45;

  Parallelepiped_t para("Test Parallelepiped", dx, dy, dz, alpha, theta, phi);

  // Points on faces
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(dx, 0, 0), ponyside(0, dy, 0),
      ponzside(dz * para.GetTanThetaCosPhi(), dz * para.GetTanThetaSinPhi(), dz);
  Vec_t ponmxside(-dx, 0, 0), ponmyside(0, -dy, 0),
      ponmzside(-dz * para.GetTanThetaCosPhi(), -dz * para.GetTanThetaSinPhi(), -dz);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  double Dist;
  Vec_t normal, norm;
  bool valid;

  // check Cubic volume

  double vol      = para.Capacity();
  double volCheck = 8 * dx * dy * dz;
  assert(ApproxEqual(vol, volCheck));

  // Check Surface area

  double surf      = para.SurfaceArea();
  double surfCheck = 8.0 * (dx * dy + dy * dz * sqrt(1. / 0.75 - 1. / 6.) + dx * dz * sqrt(1. / 0.75 - 1. / 6.));
  assert(ApproxEqual(surf, surfCheck));

  // Check Inside

  assert(para.Inside(pzero) == vecgeom::EInside::kInside);
  assert(para.Inside(pbigz) == vecgeom::EInside::kOutside);
  assert(para.Inside(ponxside) == vecgeom::EInside::kSurface);
  assert(para.Inside(ponyside) == vecgeom::EInside::kSurface);
  assert(para.Inside(ponzside) == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponxside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponmxside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponyside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponmyside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponxside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponmxside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponyside);
  assert(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponmyside);
  assert(inside == vecgeom::EInside::kSurface);

  // Check Surface Normal

  valid = para.Normal(ponzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = para.Normal(ponmzside, normal);
  assert(valid && ApproxEqual(normal, Vec_t(0, 0, -1)));

  // SafetyToOut/Outside(P)

  Dist = para.SafetyToOut(pzero);
  assert(Dist < dx);

  Dist = para.SafetyToOut(ponxside);
  assert(Dist == 0.);

  Dist = para.SafetyToOut(ponyside);
  assert(Dist == 0.);

  Dist = para.SafetyToOut(ponzside);
  assert(Dist == 0.);

  Dist = para.SafetyToOut(ponmxside);
  assert(Dist == 0.);

  Dist = para.SafetyToOut(ponmyside);
  assert(Dist == 0.);

  Dist = para.SafetyToOut(ponmzside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(pzero);
  assert(Dist < dx);

  Dist = para.SafetyToIn(ponxside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(ponyside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(ponzside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(ponmxside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(ponmyside);
  assert(Dist == 0.);

  Dist = para.SafetyToIn(ponmzside);
  assert(Dist == 0.);

  assert(ApproxEqual(para.SafetyToIn(pbigx), para.SafetyToIn(pbigmx)));
  assert(ApproxEqual(para.SafetyToIn(pbigy), para.SafetyToIn(pbigmy)));
  assert(ApproxEqual(para.SafetyToIn(pbigz), para.SafetyToIn(pbigmz)));

  // DistanceToOut(P,V)

  Dist = para.DistanceToOut(pzero, vx);
  assert(ApproxEqual(Dist, dx));
  Dist = para.DistanceToOut(pzero, vmx);
  assert(ApproxEqual(Dist, dx));
  Dist = para.DistanceToOut(pzero, vy);
  assert(ApproxEqual(Dist, dy));
  Dist = para.DistanceToOut(pzero, vmy);
  assert(ApproxEqual(Dist, dy));
  Dist = para.DistanceToOut(pzero, vz);
  assert(ApproxEqual(Dist, dz));
  Dist = para.DistanceToOut(pzero, vmz);
  assert(ApproxEqual(Dist, dz));

  Dist = para.DistanceToOut(ponxside, vx);
  assert(ApproxEqual(Dist, 0));
  Dist = para.DistanceToOut(ponmxside, vmx);
  assert(ApproxEqual(Dist, 0));
  Dist = para.DistanceToOut(ponyside, vy);
  assert(ApproxEqual(Dist, 0));
  Dist = para.DistanceToOut(ponmyside, vmy);
  assert(ApproxEqual(Dist, 0));
  Dist = para.DistanceToOut(ponzside, vz);
  assert(ApproxEqual(Dist, 0));
  Dist = para.DistanceToOut(ponmzside, vmz);
  assert(ApproxEqual(Dist, 0));

  // DistanceToIn(P,V)

  Dist = para.DistanceToIn(pbigx, vmx);
  assert(ApproxEqual(Dist, 100 - dx));
  Dist = para.DistanceToIn(pbigmx, vx);
  assert(ApproxEqual(Dist, 100 - dx));
  Dist = para.DistanceToIn(pbigy, vmy);
  assert(ApproxEqual(Dist, 100 - dy));
  Dist = para.DistanceToIn(pbigmy, vy);
  assert(ApproxEqual(Dist, 100 - dy));
  Dist = para.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual(Dist, 100 - dz));
  Dist = para.DistanceToIn(pbigmz, vz);
  assert(ApproxEqual(Dist, 100 - dz));
  Dist = para.DistanceToIn(pbigx, vxy);
  assert(ApproxEqual(Dist, kInfLength));
  Dist = para.DistanceToIn(pbigmx, vxy);
  assert(ApproxEqual(Dist, kInfLength));

  // CalculateExtent
  Vec_t minExtent, maxExtent;
  para.Extent(minExtent, maxExtent);
  std::cout << " min=" << minExtent << " max=" << maxExtent << std::endl;
  assert(ApproxEqual(minExtent,
                     Vec_t(-dx - dy / sqrt(3.) - dz * 0.5 * sqrt(2. / 3.), -dy - dz * 0.5 * sqrt(2. / 3.), -dz)));
  assert(
      ApproxEqual(maxExtent, Vec_t(dx + dy / sqrt(3.) + dz * 0.5 * sqrt(2. / 3.), dy + dz * 0.5 * sqrt(2. / 3.), dz)));

  return true;
}

int main(int argc, char *argv[])
{

  TestParallelepiped<vecgeom::SimpleParallelepiped>();
  std::cout << "VecGeom Parallelepiped passed\n";
  return 0;
}
