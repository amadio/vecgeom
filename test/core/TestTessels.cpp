#include <VecCore/VecCore>

#include "volumes/TessellatedCluster.h"
#include "test/benchmark/ArgParser.h"
#include "volumes/utilities/VolumeUtilities.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "volumes/Box.h"
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

//* Simple test demonstrating that we can create/use a structure having SIMD data members
//  to represent the tesselated cluster of triangles. Tested features:
//  - discovery of vector size, Real_v type
//  - filling the Real_v data members from scalar triangle data
//  - Vectorizing scalar queries using multiplexing of the input data to Real_v types, then
//    backend vector operations
using namespace vecgeom;

void RandomDirection(Vector3D<double> &direction)
{
  double phi    = RNG::Instance().uniform(0., 2. * kPi);
  double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
  direction.x() = std::sin(theta) * std::cos(phi);
  direction.y() = std::sin(theta) * std::sin(phi);
  direction.z() = std::cos(theta);
}

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using Real_v = typename VectorBackend::Real_v;

  OPTION_INT(npoints, 10000);

  TessellatedCluster<Real_v> tcl;
  TriangleFacet<double> facet;
  Vector3D<double> p0(0, 0, 10);
  Vector3D<double> p1(-10, 0, 0);
  Vector3D<double> p2(0, 10, 0);
  Vector3D<double> p3(10, 0, 0);
  Vector3D<double> p4(0, -10, 0);
  facet.SetVertices(p0, p2, p1, 0, 2, 1);
  std::cout << facet << std::endl;
  tcl.AddFacet(0, facet);
  facet.SetVertices(p0, p3, p2, 0, 3, 2);
  std::cout << facet << std::endl;
  tcl.AddFacet(1, facet);
  facet.SetVertices(p0, p4, p3, 0, 4, 3);
  std::cout << facet << std::endl;
  tcl.AddFacet(2, facet);
  facet.SetVertices(p0, p1, p4, 0, 1, 4);
  std::cout << facet << std::endl;
  tcl.AddFacet(3, facet);

  std::cout << tcl << std::endl;

// Visualize the facets of the cluster
#ifdef VECGEOM_ROOT
  Visualizer visualizer;
  Vector3D<double> deltas = 0.5 * (tcl.fMaxExtent - tcl.fMinExtent);
  Vector3D<double> origin = 0.5 * (tcl.fMaxExtent + tcl.fMinExtent);
  SimpleBox box("bbox", deltas.x(), deltas.y(), deltas.z());
  visualizer.AddVolume(box, Transformation3D(origin.x(), origin.y(), origin.z()));
  for (size_t ifacet = 0; ifacet < vecgeom::kVecSize; ifacet++) {
    TPolyLine3D pl(3);
    Vector3D<double> vertex;
    pl.SetLineColor(kBlue);
    tcl.GetVertex(ifacet, 0, vertex);
    pl.SetNextPoint(vertex.x(), vertex.y(), vertex.z());
    tcl.GetVertex(ifacet, 1, vertex);
    pl.SetNextPoint(vertex.x(), vertex.y(), vertex.z());
    tcl.GetVertex(ifacet, 2, vertex);
    pl.SetNextPoint(vertex.x(), vertex.y(), vertex.z());
    visualizer.AddLine(pl);
  }
  TPolyMarker3D pm(npoints);
  pm.SetMarkerColor(kRed);
#endif

  Vector3D<double> point;
  Vector3D<double> direction;
  double distance;
  int isurf;
  for (int i = 0; i < npoints; ++i) {
    point.Set(RNG::Instance().uniform(tcl.fMinExtent.x(), tcl.fMaxExtent.x()),
              RNG::Instance().uniform(tcl.fMinExtent.y(), tcl.fMaxExtent.y()),
              RNG::Instance().uniform(tcl.fMinExtent.z(), tcl.fMaxExtent.z()));
    while (1) {
      RandomDirection(direction);
      tcl.DistanceToIn(point, direction, 1e30, distance, isurf);
      if (distance < 1e30) {
        double safety = tcl.SafetyToInSq(point, isurf);
        break;
      }
      tcl.DistanceToOut(point, direction, 1e30, distance, isurf);
      if (distance < 1e30) break;
    }
    point += distance * direction;
#ifdef VECGEOM_ROOT
    pm.SetNextPoint(point[0], point[1], point[2]);
#endif
  }
#ifdef VECGEOM_ROOT
  visualizer.AddPoints(pm);
  visualizer.Show();
#endif

  return 0;
}
