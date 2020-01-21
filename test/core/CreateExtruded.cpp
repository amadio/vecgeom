#ifndef VECGEOM_ENABLE_CUDA

#include <VecCore/VecCore>

#include "test/benchmark/ArgParser.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/volumes/Extruded.h"
#include "VecGeom/volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
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

void RandomDirection(Vector3D<double> &direction)
{
  double phi    = RNG::Instance().uniform(0., 2. * kPi);
  double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
  direction.x() = std::sin(theta) * std::cos(phi);
  direction.y() = std::sin(theta) * std::sin(phi);
  direction.z() = std::cos(theta);
}

void RandomPointInBBox(Vector3D<double> &point, Vector3D<double> &amin, Vector3D<double> &amax)
{
  Vector3D<double> rnd(RNG::Instance().uniform(0, 1), RNG::Instance().uniform(0, 1), RNG::Instance().uniform(0, 1));
  point = amin + rnd * (amax - amin);
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

void DrawCluster(TessellatedStruct<3, double> const &tsl, int icluster, Visualizer &visualizer, bool boxonly = false)
{
  // Draw only segments of the facets which are not shared within the cluster
  TPolyLine3D pl(2);
  pl.SetLineColor(kBlue);
  if (boxonly) {
    Vector3D<double> minext = tsl.fClusters[icluster]->fMinExtent;
    Vector3D<double> maxext = tsl.fClusters[icluster]->fMaxExtent;
    Vector3D<double> dext   = maxext - minext;
    pl.SetPoint(0, minext.x(), minext.y(), minext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y(), minext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y(), minext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y() + dext.y(), minext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y() + dext.y(), minext.z());
    pl.SetPoint(1, minext.x(), minext.y() + dext.y(), minext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x(), minext.y() + dext.y(), minext.z());
    pl.SetPoint(1, minext.x(), minext.y(), minext.z());
    visualizer.AddLine(pl);

    pl.SetPoint(0, minext.x(), minext.y(), minext.z());
    pl.SetPoint(1, minext.x(), minext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y(), minext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y() + dext.y(), minext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x(), minext.y() + dext.y(), minext.z());
    pl.SetPoint(1, minext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);

    pl.SetPoint(0, minext.x(), minext.y(), minext.z() + dext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y(), minext.z() + dext.z());
    pl.SetPoint(1, minext.x() + dext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x() + dext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    pl.SetPoint(1, minext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    pl.SetPoint(0, minext.x(), minext.y() + dext.y(), minext.z() + dext.z());
    pl.SetPoint(1, minext.x(), minext.y(), minext.z() + dext.z());
    visualizer.AddLine(pl);
    return;
  }

  size_t nfacets = 0;
  size_t ifacet  = 0;
  size_t iother  = 0;
  TriangleFacet<double> *facets[kVecSize];
  while (ifacet < kVecSize) {
    bool add = true;
    for (unsigned i = 0; i < nfacets; ++i) {
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
  size_t ivert[2];
  while (ifacet < nfacets) {
    // loop segments
    for (size_t iseg = 0; iseg < 3; iseg++) {
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
#endif // VECGEOM_ROOT
#endif // VECGEOM_CUDA

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  using namespace vecgeom;
  //  using Real_v = typename VectorBackend::Real_v;

  OPTION_INT(nvert, 8);
  OPTION_INT(nsect, 2);
  OPTION_BOOL(convex, 't');
  OPTION_INT(npoints, 10000);
#ifdef VECGEOM_ROOT
  OPTION_INT(vis, 0);
#endif
  constexpr double rmin = 10.;
  constexpr double rmax = 20.;

  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nvert];

  double phi = 2. * kPi / nvert;
  double r;

  Vector3D<double> start(0, 0, 0);
  Vector3D<double> point;

  Vector3D<double> *dirs = new Vector3D<double>[npoints];
  for (int i = 0; i < npoints; ++i)
    RandomDirection(dirs[i]);

  for (int i = 0; i < nvert; ++i) {
    r                           = rmax;
    if (i % 2 > 0 && !convex) r = rmin;
    vertices[i].x               = r * vecCore::math::Cos(i * phi);
    vertices[i].y               = r * vecCore::math::Sin(i * phi);
  }
  for (int i = 0; i < nsect; ++i) {
    sections[i].fOrigin.Set(0, 0, -20. + i * 40. / (nsect - 1));
    sections[i].fScale = 1;
  }

  std::cout << "Creating extruded polygon having " << nvert << " vertices and " << nsect << " sections\n";
  UnplacedExtruded xtru(nvert, vertices, nsect, sections);

// Visualize the facets
#ifdef VECGEOM_ROOT
  if (vis) {
    Visualizer visualizer;
    // Visualize bounding box
    Vector3D<double> amin, amax;
    xtru.Extent(amin, amax);
    Vector3D<double> deltas = 0.5 * (amax - amin);
    Vector3D<double> origin = 0.5 * (amax + amin);
    SimpleBox box("bbox", deltas.x(), deltas.y(), deltas.z());
    visualizer.AddVolume(box, Transformation3D(origin.x(), origin.y(), origin.z()));

    // Visualize facets

    for (size_t i = 0; i < xtru.GetStruct().fTslHelper.fFacets.size(); ++i)
      AddFacetToVisualizer(xtru.GetStruct().fTslHelper.fFacets[i], visualizer);

    // Visualize clusters
    //    for (unsigned icluster = 0; icluster < tsl.fClusters.size(); ++icluster)
    //      DrawCluster(tsl, icluster, visualizer, false);

    TPolyMarker3D pm(npoints);
    pm.SetMarkerColor(kRed);
    pm.SetMarkerStyle(7);
    // Test contains function

    for (int i = 0; i < npoints; ++i) {
      RandomPointInBBox(point, amin, amax);
      bool contains = xtru.Contains(point);
      if (contains) pm.SetNextPoint(point[0], point[1], point[2]);
    }

    delete[] dirs;

    visualizer.AddPoints(pm);
    visualizer.Show();
    return 0;
  }
#endif // VECGEOM_ROOT
#endif

  return 0;
}
