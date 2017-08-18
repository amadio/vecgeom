#include <VecCore/VecCore>
#include "base/Math.h"
#include "base/Vector3D.h"
#include "volumes/Tessellated.h"

size_t TessellatedOrb(double r, int ngrid, vecgeom::UnplacedTessellated &tsl)
{
  // Create a tessellated orb divided in ngrid*ngrid theta/phi cells
  // Sin/Cos tables
  std::vector<double> sth, cth, sph, cph;
  double dth = vecgeom::kPi / ngrid;
  double dph = vecgeom::kTwoPi / ngrid;

  for (int i = 0; i <= ngrid; ++i) {
    sth.push_back(vecCore::math::Sin(i * dth));
    cth.push_back(vecCore::math::Cos(i * dth));
    sph.push_back(vecCore::math::Sin(i * dph));
    cph.push_back(vecCore::math::Cos(i * dph));
  }

  auto Vtx = [&](int ith, int iph) {
    return vecgeom::Vector3D<double>(r * sth[ith] * cph[iph], r * sth[ith] * sph[iph], r * cth[ith]);
  };

  for (int ith = 0; ith < ngrid; ++ith) {
    for (int iph = 0; iph < ngrid; ++iph) {
      // First/last rows - > triangles
      if (ith == 0) {
        tsl.AddTriangularFacet(vecgeom::Vector3D<double>(0, 0, r), Vtx(ith + 1, iph), Vtx(ith + 1, iph + 1));
      } else if (ith == ngrid - 1) {
        tsl.AddTriangularFacet(Vtx(ith, iph), vecgeom::Vector3D<double>(0, 0, -r), Vtx(ith, iph + 1));
      } else {
        tsl.AddQuadrilateralFacet(Vtx(ith, iph), Vtx(ith + 1, iph), Vtx(ith + 1, iph + 1), Vtx(ith, iph + 1));
      }
    }
  }
  return (tsl.GetNFacets());
}
