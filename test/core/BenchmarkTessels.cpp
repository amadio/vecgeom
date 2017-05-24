#include <VecCore/VecCore>
#include <sys/time.h>

#include "volumes/TessellatedCluster.h"
#include "test/benchmark/ArgParser.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/Stopwatch.h"

//* Simple test demonstrating that we can create/use a structure having SIMD data members
//  to represent the tesselated cluster of triangles. Tested features:
//  - discovery of vector size, Real_v type
//  - filling the Real_v data members from scalar triangle data
//  - Vectorizing scalar queries using multiplexing of the input data to Real_v types, then
//    backend vector operations
using namespace vecgeom;

//______________________________________________________________________________
void RandomDirection(Vector3D<double> &direction)

{
  double phi    = RNG::Instance().uniform(0., 2. * kPi);
  double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
  direction.x() = std::sin(theta) * std::cos(phi);
  direction.y() = std::sin(theta) * std::sin(phi);
  direction.z() = std::cos(theta);
}

//______________________________________________________________________________
int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using Real_v = typename VectorBackend::Real_v;

  OPTION_INT(npoints, 1000000);
  OPTION_INT(nrep, 10);

  TessellatedCluster<Real_v> tcl;
  TriangleFacet<double> *facet;
  Vector3D<double> p0(0, 0, 10);
  Vector3D<double> p1(-10, 0, 0);
  Vector3D<double> p2(0, 10, 0);
  Vector3D<double> p3(10, 0, 0);
  Vector3D<double> p4(0, -10, 0);
  facet = new TriangleFacet<double>();
  facet->SetVertices(p0, p2, p1, 0, 2, 1);
  //  std::cout << facet << std::endl;
  tcl.AddFacet(0, facet);
  facet = new TriangleFacet<double>();
  facet->SetVertices(p0, p3, p2, 0, 3, 2);
  //  std::cout << facet << std::endl;
  tcl.AddFacet(1, facet);
  facet = new TriangleFacet<double>();
  facet->SetVertices(p0, p4, p3, 0, 4, 3);
  //  std::cout << facet << std::endl;
  tcl.AddFacet(2, facet);
  facet = new TriangleFacet<double>();
  facet->SetVertices(p0, p1, p4, 0, 1, 4);
  //  std::cout << facet << std::endl;
  tcl.AddFacet(3, facet);

  //  std::cout << tcl << std::endl;

  // Generate points and directions
  Vector3D<Precision> *points     = new Vector3D<Precision>[npoints];
  Vector3D<Precision> *directions = new Vector3D<Precision>[npoints];
  Precision *dscalar              = new Precision[npoints];
  Precision *dvector              = new Precision[npoints];
  int *isurfscalar                = new int[npoints];
  int *isurfvector                = new int[npoints];

  for (int i = 0; i < npoints; ++i) {
    points[i].Set(RNG::Instance().uniform(tcl.fMinExtent.x(), tcl.fMaxExtent.x()),
                  RNG::Instance().uniform(tcl.fMinExtent.y(), tcl.fMaxExtent.y()),
                  RNG::Instance().uniform(tcl.fMinExtent.z(), tcl.fMaxExtent.z()));
    RandomDirection(directions[i]);
  }

  Stopwatch timer;
  int nerrors;
  Precision tscalar, tvect;

  // DistanceToIn scalar mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      tcl.DistanceToInScalar(points[i], directions[i], 1e30, dscalar[i], isurfscalar[i]);
    }
  }
  tscalar = timer.Stop();

  // DistanceToIn vector mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      tcl.DistanceToIn(points[i], directions[i], 1e30, dvector[i], isurfvector[i]);
    }
  }
  tvect   = timer.Stop();
  nerrors = 0;
  for (int i = 0; i < npoints; ++i) {
    if (Abs(dvector[i] - dscalar[i]) > vecgeom::kTolerance || isurfvector[i] != isurfscalar[i]) {
      nerrors++;
    }
  }
  printf("DistanceToIn:    scalar %f  vector %f   speedup %f nerrors = %d / %d\n", tscalar, tvect, tscalar / tvect,
         nerrors, npoints);

  // DistanceToOut scalar mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      tcl.DistanceToOutScalar(points[i], directions[i], 1e30, dscalar[i], isurfscalar[i]);
    }
  }
  tscalar = timer.Stop();

  // DistanceToOut vector mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      tcl.DistanceToOut(points[i], directions[i], 1e30, dvector[i], isurfvector[i]);
    }
  }
  tvect = timer.Stop();

  nerrors = 0;
  for (int i = 0; i < npoints; ++i) {
    if (Abs(dvector[i] - dscalar[i]) > vecgeom::kTolerance || isurfvector[i] != isurfscalar[i]) {
      tcl.DistanceToOutScalar(points[i], directions[i], 1e30, dscalar[i], isurfscalar[i]);
      tcl.DistanceToOut(points[i], directions[i], 1e30, dvector[i], isurfvector[i]);
      nerrors++;
    }
  }
  printf("DistanceToOut:   scalar %f  vector %f   speedup %f nerrors = %d / %d\n", tscalar, tvect, tscalar / tvect,
         nerrors, npoints);

  // SafetyToIn scalar mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      dscalar[i] = tcl.SafetySqScalar<true>(points[i], isurfscalar[i]);
    }
  }
  tscalar = timer.Stop();

  // SafetyToIn vector mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      dvector[i] = tcl.SafetySq<true>(points[i], isurfvector[i]);
    }
  }
  tvect   = timer.Stop();
  nerrors = 0;
  for (int i = 0; i < npoints; ++i) {
    if (Abs(dvector[i] - dscalar[i]) > vecgeom::kTolerance || isurfvector[i] != isurfscalar[i]) {
      nerrors++;
    }
  }
  printf("SafetyToIn:      scalar %f  vector %f   speedup %f nerrors = %d / %d\n", tscalar, tvect, tscalar / tvect,
         nerrors, npoints);

  // SafetyToOut scalar mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      dscalar[i] = tcl.SafetySqScalar<false>(points[i], isurfscalar[i]);
    }
  }
  tscalar = timer.Stop();

  // SafetyToOut vector mode
  timer.Start();
  for (int irep = 0; irep < nrep; irep++) {
    for (int i = 0; i < npoints; ++i) {
      dvector[i] = tcl.SafetySq<false>(points[i], isurfvector[i]);
    }
  }
  tvect   = timer.Stop();
  nerrors = 0;
  for (int i = 0; i < npoints; ++i) {
    if (Abs(dvector[i] - dscalar[i]) > vecgeom::kTolerance || isurfvector[i] != isurfscalar[i]) {
      nerrors++;
    }
  }
  printf("SafetyToOut:     scalar %f  vector %f   speedup %f nerrors = %d / %d\n", tscalar, tvect, tscalar / tvect,
         nerrors, npoints);

  return 0;
}
