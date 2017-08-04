#ifndef VECGEOM_ENABLE_CUDA

#include "volumes/LogicalVolume.h"
#include "volumes/Tessellated.h"
#include "benchmarking/Benchmarker.h"
#include "volumes/kernel/TessellatedImplementation.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Stopwatch.h"
#include <iostream>

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
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(ngrid, 100);

  UnplacedBox worldUnplaced = UnplacedBox(20., 20., 20.);

  UnplacedTessellated tsl = UnplacedTessellated();
  // Create the tessellated solid
  size_t nfacets = CreateTessellated(ngrid, tsl);
  tsl.Close();
  std::cout << "Benchmarking tessellated sphere having " << nfacets << " facets\n";

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume tessellated("tessellated", &tsl);

  Transformation3D placement(0, 0, 0);
  const VPlacedVolume *placedTsl = world.PlaceDaughter("tessellated", &tessellated, &placement);
  (void)placedTsl;

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS) && 0
  vecgeom::Stopwatch timer;
  timer.Start();
  placedTsl->ConvertToUSolids();
  double usolids_btime = timer.Stop();
  std::cout << "+++++++++    nfacets = " << nfacets << "    UTessellatedSolid construction time: " << usolids_btime
            << std::endl;
  return 0;
#endif

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.SetPoolMultiplier(1);

  //  tester.RunToInBenchmark();
  //  tester.RunToOutBenchmark();
  return tester.RunBenchmark();
#else
  return 0;
#endif
}
