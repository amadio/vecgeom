// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for Paralleliped.
/// @file test/unit_tests/TestParallelepiped.cpp
/// @author Created by Mihaela Gheata
/// @author Revised by Evgueni Tcherniaev

// ensure asserts are compiled in
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

  int Npoints = 10000000;
  std::cout << "=== Check Capacity()" << std::endl;
  double vol      = para.Capacity();
  double volCheck = para.GetUnplacedVolume()->EstimateCapacity(Npoints);
  std::cout << " vol = " << vol << "   mc_estimated = " << volCheck << std::endl;
  assert(std::abs(vol - volCheck) < 0.01 * vol);

  // Check Surface area

  std::cout << "=== Check SurfaceArea()" << std::endl;
  double surf      = para.SurfaceArea();
  double surfCheck = para.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " surf = " << surf << "   mc_estimated = " << surfCheck << std::endl;
  assert(std::abs(surf - surfCheck) < 0.01 * surf);

  // Check Extent

  std::cout << "=== Check Extent()" << std::endl;
  Vec_t minExtent, maxExtent;
  Vec_t minCheck(kInfLength, kInfLength, kInfLength);
  Vec_t maxCheck(-kInfLength, -kInfLength, -kInfLength);
  para.Extent(minExtent, maxExtent);
  for (int i = 0; i < Npoints; ++i) {
    Vec_t p = para.GetUnplacedVolume()->SamplePointOnSurface();
    minCheck.Set(std::min(p.x(), minCheck.x()), std::min(p.y(), minCheck.y()), std::min(p.z(), minCheck.z()));
    maxCheck.Set(std::max(p.x(), maxCheck.x()), std::max(p.y(), maxCheck.y()), std::max(p.z(), maxCheck.z()));
  }
  std::cout << " calculated: min = " << minExtent << " max = " << maxExtent << std::endl;
  std::cout << " estimated:  min = " << minCheck << " max = " << maxCheck << std::endl;

  assert(std::abs(minExtent.x() - minCheck.x()) < 0.001 * std::abs(minExtent.x()));
  assert(std::abs(minExtent.y() - minCheck.y()) < 0.001 * std::abs(minExtent.y()));
  assert(minExtent.z() == minCheck.z());
  assert(std::abs(maxExtent.x() - maxCheck.x()) < 0.001 * std::abs(maxExtent.x()));
  assert(std::abs(maxExtent.y() - maxCheck.y()) < 0.001 * std::abs(maxExtent.y()));
  assert(maxExtent.z() == maxCheck.z());

  // Check Inside

  std::cout << "=== Check Inside()" << std::endl;
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

  std::cout << "=== Check Normal()" << std::endl;
  Vec_t pp[8], nn[3], ptest;
  pp[0].Set(-dx, -dy, -dz);
  pp[1].Set(dx, -dy, -dz);
  pp[2].Set(dx, dy, -dz);
  pp[3].Set(-dx, dy, -dz);
  pp[4].Set(-dx, -dy, dz);
  pp[5].Set(dx, -dy, dz);
  pp[6].Set(dx, dy, dz);
  pp[7].Set(-dx, dy, dz);
  for (int i = 0; i < 8; ++i) {
    pp[i].x() += para.GetTanAlpha() * pp[i].y() + para.GetTanThetaCosPhi() * pp[i].z();
    pp[i].y() += para.GetTanThetaSinPhi() * pp[i].z();
  }
  nn[0] = para.GetUnplacedVolume()->GetNormal(0);
  nn[1] = para.GetUnplacedVolume()->GetNormal(1);
  nn[2] = Vec_t(0, 0, 1);

  // check facets
  ptest = Vec_t(dx, 0, 0);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == nn[0]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == nn[0]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == nn[0]);

  ptest = Vec_t(-dx, 0, 0);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == -nn[0]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == -nn[0]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == -nn[0]);

  ptest = Vec_t(0, dy, 0);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == nn[1]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == nn[1]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == nn[1]);

  ptest = Vec_t(0, -dy, 0);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == -nn[1]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == -nn[1]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == -nn[1]);

  ptest = Vec_t(0, 0, dz);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == nn[2]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == nn[2]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == nn[2]);

  ptest = Vec_t(0, 0, -dz);
  valid = para.Normal(ptest, normal);
  assert(valid && normal == -nn[2]);
  valid = para.Normal(1.1 * ptest, normal);
  assert(!valid && normal == -nn[2]);
  valid = para.Normal(0.9 * ptest, normal);
  assert(!valid && normal == -nn[2]);

  // check edges
  valid = para.Normal((pp[0] + pp[1]) / 2, normal);
  assert(valid && normal == (-nn[2] - nn[1]).Unit());
  valid = para.Normal((pp[1] + pp[2]) / 2, normal);
  assert(valid && normal == (-nn[2] + nn[0]).Unit());
  valid = para.Normal((pp[2] + pp[3]) / 2, normal);
  assert(valid && normal == (-nn[2] + nn[1]).Unit());
  valid = para.Normal((pp[3] + pp[0]) / 2, normal);
  assert(valid && normal == (-nn[2] - nn[0]).Unit());

  valid = para.Normal((pp[4] + pp[5]) / 2, normal);
  assert(valid && normal == (nn[2] - nn[1]).Unit());
  valid = para.Normal((pp[5] + pp[6]) / 2, normal);
  assert(valid && normal == (nn[2] + nn[0]).Unit());
  valid = para.Normal((pp[6] + pp[7]) / 2, normal);
  assert(valid && normal == (nn[2] + nn[1]).Unit());
  valid = para.Normal((pp[7] + pp[4]) / 2, normal);
  assert(valid && normal == (nn[2] - nn[0]).Unit());

  valid = para.Normal((pp[0] + pp[4]) / 2, normal);
  assert(valid && normal == (-nn[0] - nn[1]).Unit());
  valid = para.Normal((pp[1] + pp[5]) / 2, normal);
  assert(valid && normal == (nn[0] - nn[1]).Unit());
  valid = para.Normal((pp[2] + pp[6]) / 2, normal);
  assert(valid && normal == (nn[0] + nn[1]).Unit());
  valid = para.Normal((pp[3] + pp[7]) / 2, normal);
  assert(valid && normal == (-nn[0] + nn[1]).Unit());

  // check nodes
  valid = para.Normal(pp[0], normal);
  assert(valid && normal == (-nn[2] - nn[1] - nn[0]).Unit());
  valid = para.Normal(pp[1], normal);
  assert(valid && normal == (-nn[2] - nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[2], normal);
  assert(valid && normal == (-nn[2] + nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[3], normal);
  assert(valid && normal == (-nn[2] + nn[1] - nn[0]).Unit());

  valid = para.Normal(pp[4], normal);
  assert(valid && normal == (nn[2] - nn[1] - nn[0]).Unit());
  valid = para.Normal(pp[5], normal);
  assert(valid && normal == (nn[2] - nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[6], normal);
  assert(valid && normal == (nn[2] + nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[7], normal);
  assert(valid && normal == (nn[2] + nn[1] - nn[0]).Unit());

  // Check SafetyToOut

  std::cout << "=== Check SafetyToOut()" << std::endl;
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

  // Check SafetyToIn

  std::cout << "=== Check SafetyToIn()" << std::endl;
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

  std::cout << "=== Check DistanceToOut()" << std::endl;
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

  std::cout << "=== Check DistanceToIn()" << std::endl;
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

  // Check SamplePointOnSurface()

  std::cout << "=== Check SamplePointOnSurface()" << std::endl;
  Vec_t Vx(1., 0., 0.);
  Vec_t Vy(para.GetTanAlpha(), 1., 0.);
  Vec_t Vz(para.GetTanThetaCosPhi(), para.GetTanThetaSinPhi(), 1.);
  Vec_t Nx = Vy.Cross(Vz);
  Vec_t Ny = Vz.Cross(Vx);
  Vec_t Nz(0., 0., 1.);
  double sx = 4. * para.GetY() * para.GetZ() * Nx.Mag();
  double sy = 4. * para.GetZ() * para.GetX() * Ny.Mag();
  double sz = 4. * para.GetX() * para.GetY();
  Nx.Normalize();
  Ny.Normalize();
  double Dx = -Nx.x() * para.GetX();
  double Dy = -Ny.y() * para.GetY();
  int nxneg = 0, nxpos = 0, nyneg = 0, nypos = 0, nzneg = 0, nzpos = 0;
  int nfactor = 100, ntot = 2. * (sx + sy + sz) * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t p = para.GetUnplacedVolume()->SamplePointOnSurface();
    assert(para.Inside(p) == vecgeom::kSurface);
    if (std::abs(Nx.Dot(p) - Dx) < kHalfTolerance) {
      ++nxneg;
    } else if (std::abs(Nx.Dot(p) + Dx) < kHalfTolerance) {
      ++nxpos;
    } else if (std::abs(Ny.Dot(p) - Dy) < kHalfTolerance) {
      ++nyneg;
    } else if (std::abs(Ny.Dot(p) + Dy) < kHalfTolerance) {
      ++nypos;
    } else if (p.z() == -para.GetZ()) {
      ++nzneg;
    } else {
      ++nzpos;
    }
  }
  std::cout << "facet surface -/+x, -/+y, -/+z: "
            << "\t" << sx << ", \t" << sx << ", \t" << sy << ", \t" << sy << ", \t" << sz << ", \t" << sz << std::endl;
  std::cout << "n. of samples -/+x, -/+y, -/+z: "
            << "\t" << nxneg << ", \t" << nxpos << ", \t" << nyneg << ", \t" << nypos << ", \t" << nzneg << ", \t"
            << nzpos << std::endl;
  assert(std::abs(nxneg - sx * nfactor) < 0.01 * sx * nfactor);
  assert(std::abs(nxpos - sx * nfactor) < 0.01 * sx * nfactor);
  assert(std::abs(nyneg - sy * nfactor) < 0.01 * sy * nfactor);
  assert(std::abs(nypos - sy * nfactor) < 0.01 * sy * nfactor);
  assert(std::abs(nzneg - sz * nfactor) < 0.01 * sz * nfactor);
  assert(std::abs(nzpos - sz * nfactor) < 0.01 * sz * nfactor);

  return true;
}

int main(int argc, char *argv[])
{

  TestParallelepiped<vecgeom::SimpleParallelepiped>();
  std::cout << "\nVecGeom Parallelepiped passed\n" << std::endl;
  return 0;
}
