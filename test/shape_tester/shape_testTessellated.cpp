#ifndef VECGEOM_ENABLE_CUDA

#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "volumes/Tessellated.h"
typedef vecgeom::SimpleTessellated Tessellated_t;

using namespace vecgeom;

constexpr double r = 10.;
double *sth, *cth, *sph, *cph;

VECGEOM_FORCE_INLINE
Vector3D<double> Vtx(int ith, int iph)
{
  return Vector3D<double>(r * sth[ith] * cph[iph], r * sth[ith] * sph[iph], r * cth[ith]);
}

size_t CreateTessellated(int ngrid, UnplacedTessellated &tsl)
{
  // Create a tessellated sphere divided in ngrid*ngrid theta/phi cells
  // Sin/Cos tables
  double dth = kPi / ngrid;
  double dph = kTwoPi / ngrid;
  sth        = new double[ngrid + 1];
  cth        = new double[ngrid + 1];
  sph        = new double[ngrid + 1];
  cph        = new double[ngrid + 1];

  for (int i = 0; i <= ngrid; ++i) {
    sth[i] = vecCore::math::Sin(i * dth);
    cth[i] = vecCore::math::Cos(i * dth);
    sph[i] = vecCore::math::Sin(i * dph);
    cph[i] = vecCore::math::Cos(i * dph);
  }
  for (int ith = 0; ith < ngrid; ++ith) {
    for (int iph = 0; iph < ngrid; ++iph) {
      // First/last rows - > triangles
      if (ith == 0) {
        tsl.AddTriangularFacet(Vector3D<double>(0, 0, r), Vtx(ith + 1, iph), Vtx(ith + 1, iph + 1));
      } else if (ith == ngrid - 1) {
        tsl.AddTriangularFacet(Vtx(ith, iph), Vector3D<double>(0, 0, -r), Vtx(ith, iph + 1));
      } else {
        tsl.AddQuadrilateralFacet(Vtx(ith, iph), Vtx(ith + 1, iph), Vtx(ith + 1, iph + 1), Vtx(ith, iph + 1));
      }
    }
  }
  delete[] sth;
  delete[] cth;
  delete[] sph;
  delete[] cph;
  return (tsl.GetNFacets());
}
#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  if (argc == 1) {
    std::cout << "Usage: shape_testTessellated <-ngrid N>\n"
                 "       N - number of theta/phi segments for the sphere\n";
  }
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(ngrid, 20);

  Tessellated_t *solid     = 0;
  solid                    = new Tessellated_t("test_VecGeomTessellated");
  UnplacedTessellated *tsl = (UnplacedTessellated *)solid->GetUnplacedVolume();
  size_t nfacets           = CreateTessellated(ngrid, *tsl);
  std::cout << "Testing tessellated sphere with ngrid = " << ngrid << " (nfacets=" << nfacets << ")\n";
  tsl->Close();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setConventionsMode(false);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(1.e-9);
  tester.SetTestBoundaryErrors(false);
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
#endif
  return 0;
}
