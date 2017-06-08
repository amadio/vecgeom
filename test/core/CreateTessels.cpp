#include <VecCore/VecCore>

#include "volumes/TessellatedStruct.h"
#include "test/benchmark/ArgParser.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/Stopwatch.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "TH1.h"
#include "TGraph.h"
#include "TFile.h"
#include "volumes/Box.h"
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

/* Simple test for the scalability of creation of the tessellated structure.
   An orb is split into ngrid theta and phi regions; each cell is represented
   as a quadrilateral. The solid will contain 2*(ngrid-1)*ngrid triangle facets */

using namespace vecgeom;
using Real_v = vecgeom::VectorBackend::Real_v;

double r    = 10.;
double *sth = nullptr;
double *cth = nullptr;
double *sph = nullptr;
double *cph = nullptr;

void RandomDirection(Vector3D<double> &direction)
{
  double phi    = RNG::Instance().uniform(0., 2. * kPi);
  double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
  direction.x() = std::sin(theta) * std::cos(phi);
  direction.y() = std::sin(theta) * std::sin(phi);
  direction.z() = std::cos(theta);
}

VECGEOM_FORCE_INLINE
Vector3D<double> Vtx(int ith, int iph)
{
  return Vector3D<double>(r * sth[ith] * cph[iph], r * sth[ith] * sph[iph], r * cth[ith]);
}

#ifdef VECGEOM_ROOT
void AddFacetToVisualizer(TriangleFacet<double> const *facet, Visualizer &visualizer)
{
  TPolyLine3D pl(3);
  pl.SetLineColor(kBlue);
  for (int i = 0; i < 3; i++)
    pl.SetNextPoint(facet->fVertices[i].x(), facet->fVertices[i].y(), facet->fVertices[i].z());
  visualizer.AddLine(pl);
}

void DrawCluster(TessellatedStruct<double> const &tsl, int icluster, Visualizer &visualizer)
{
  // Draw only segments of the facets which are not shared within the cluster
  TPolyLine3D pl(2);
  pl.SetLineColor(kBlue);
  int nfacets = 0;
  int ifacet  = 0;
  int iother  = 0;
  TriangleFacet<double> *facets[kVecSize];
  while (ifacet < kVecSize) {
    bool add = true;
    for (int i = 0; i < nfacets; ++i) {
      if (tsl.fClusters[icluster]->fFacets[ifacet] == facets[i]) {
        ifacet++;
        add = false;
        break;
      }
    }
    if (add) facets[nfacets++] = tsl.fClusters[icluster]->fFacets[ifacet++];
  }
  // Loop facets
  ifacet = 0;
  int ivert[2];
  while (ifacet < nfacets) {
    // loop segments
    for (int iseg = 0; iseg < 3; iseg++) {
      bool shared = false;
      ivert[0]    = facets[ifacet]->fIndices[iseg];
      ivert[1]    = facets[ifacet]->fIndices[(iseg + 1) % 3];
      // loop remaining facets
      for (iother = 0; iother < nfacets; iother++) {
        if (iother == ifacet) continue;
        // check if the other facet has the 2 vertices
        if (facets[iother]->fIndices[0] != ivert[0] && facets[iother]->fIndices[1] != ivert[0] &&
            facets[iother]->fIndices[2] != ivert[0])
          continue;
        if (facets[iother]->fIndices[0] != ivert[1] && facets[iother]->fIndices[1] != ivert[1] &&
            facets[iother]->fIndices[2] != ivert[1])
          continue;
        // The line is shared
        shared = true;
        break;
      }
      if (shared) continue;
      // Add the line segment to the visualizer
      pl.SetPoint(0, facets[ifacet]->fVertices[iseg].x(), facets[ifacet]->fVertices[iseg].y(),
                  facets[ifacet]->fVertices[iseg].z());
      pl.SetPoint(1, facets[ifacet]->fVertices[(iseg + 1) % 3].x(), facets[ifacet]->fVertices[(iseg + 1) % 3].y(),
                  facets[ifacet]->fVertices[(iseg + 1) % 3].z());
      visualizer.AddLine(pl);
    }
    ifacet++;
  }
}
#endif

int CreateTessellated(int ngrid, TessellatedStruct<double> &tsl)
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
  return int(tsl.fFacets.size());
}

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  //  using Real_v = typename VectorBackend::Real_v;

  OPTION_INT(ngrid, 100);
  OPTION_INT(npoints, 1000);
  OPTION_INT(vis, 0);
  OPTION_INT(scalability, 0);

  int ngrid1         = 10;
  int i              = 0;
  const double sqrt2 = vecCore::math::Sqrt(2.);
  TGraph *gtime      = nullptr;
  if (scalability) {
    gtime = new TGraph(16);
    while (ngrid1 < 2000) {
      TessellatedStruct<double> tsl;
      int nfacets1 = CreateTessellated(ngrid1, tsl);
      // Close the solid
      Stopwatch timer;
      timer.Start();
      tsl.Close();
      double tbuild = timer.Stop();
      gtime->SetPoint(i++, nfacets1, tbuild);
      ngrid1 = sqrt2 * double(ngrid1);
      printf("nfacets=%d  time=%g\n", nfacets1, tbuild);
    }
  }

  TessellatedStruct<double> tsl;
  CreateTessellated(ngrid, tsl);
  tsl.Close();
  std::cout << "=== Tessellated solid statistics: nfacets = " << tsl.fFacets.size()
            << "  nclusters = " << tsl.fClusters.size() << "  kVecSize = " << kVecSize << std::endl;
  std::cout << "    cluster distribution: ";
  for (int i = 1; i <= kVecSize; ++i) {
    std::cout << i << ": " << tsl.fNcldist[i] << " | ";
  }
  std::cout << "\n";

// Visualize the facets
#ifdef VECGEOM_ROOT
  if (vis) {
    Visualizer visualizer;
    // Visualize bounding box
    Vector3D<double> deltas = 0.5 * (tsl.fMaxExtent - tsl.fMinExtent);
    Vector3D<double> origin = 0.5 * (tsl.fMaxExtent + tsl.fMinExtent);
    SimpleBox box("bbox", deltas.x(), deltas.y(), deltas.z());
    visualizer.AddVolume(box, Transformation3D(origin.x(), origin.y(), origin.z()));

    // Visualize facets
    for (auto facet : tsl.fFacets) {
      AddFacetToVisualizer(facet, visualizer);
    }
    visualizer.Show();
  }

  // Test distance to out from origin
  Vector3D<double> point(0, 0, 0);
  Vector3D<double> direction;
  RandomDirection(direction);
  double distance;
  int ifacet;
  tsl.DistanceToSolid<false>(point, direction, InfinityLength<double>(), distance, ifacet);
  printf("Distance = %g\n", distance);

  // Visualize cluster
  int nblobs, nfacets;
  int nfacetstot    = 0;
  TH1F *hdispersion = new TH1F("hdispersion", "Cluster dispersion", 100, 0., 10.);
  TH1I *hblobs      = new TH1I("hblobs", "Blobs in clusters", 8, 0, 8);
  for (int icluster = 0; icluster < tsl.fClusters.size(); ++icluster) {
    // DrawCluster(tsl, icluster, visualizer);
    double dispersion = tsl.fClusters[icluster]->ComputeSparsity(nblobs, nfacets);
    nfacetstot += nfacets;
    hdispersion->Fill(dispersion);
    hblobs->Fill(nblobs);
  }
  printf("Number of facets = %d/%lu\n", nfacetstot, tsl.fFacets.size());
  TFile *file = TFile::Open("dispersion.root", "RECREATE");
  hdispersion->Write();
  hblobs->Write();
  if (gtime) gtime->Write();
  file->Write();
#endif

  return 0;
}
