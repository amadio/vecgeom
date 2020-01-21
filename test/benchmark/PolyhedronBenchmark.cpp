#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

#include <fstream>

using namespace vecgeom;

UnplacedPolyhedron *NoInnerRadii()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -2, 0, 2, 4};
  Precision rInner[nPlanes]  = {0, 0, 0, 0, 0};
  Precision rOuter[nPlanes]  = {2, 3, 2, 3, 2};
  return new UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *WithInnerRadii()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
  Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
  Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return new UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *WithPhiSection()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
  Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
  Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return new UnplacedPolyhedron(15 * kDegToRad, 45 * kDegToRad, 5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *ManySegments()
{
  constexpr int nPlanes      = 17;
  Precision zPlanes[nPlanes] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  Precision rInner[nPlanes]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  Precision rOuter[nPlanes]  = {2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2};
  return new UnplacedPolyhedron(6, nPlanes, zPlanes, rInner, rOuter);
}

int main(int argc, char *argv[])
{

  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  // Polyhedron type:
  //   0=NoInnerRadii
  //   1=WithInnerRadii
  //   2=WithPhiSection
  //   3=ManySegments
  OPTION_INT(type, 2);

  UnplacedBox worldUnplaced = UnplacedBox(10, 10, 10);

  auto RunBenchmark = [&worldUnplaced](UnplacedPolyhedron const *shape, char const *label, int npoints,
                                       int nrep) -> int {
    LogicalVolume logical("pgon", shape);
    // VPlacedVolume *placed = logical.Place();
    LogicalVolume worldLogical(&worldUnplaced);
    //   worldLogical.PlaceDaughter(placed);
    Transformation3D transformation(5, 5, 5);
    worldLogical.PlaceDaughter("pgonplaced", &logical, &transformation);
    GeoManager::Instance().SetWorldAndClose(worldLogical.Place());

    Benchmarker benchmarker(GeoManager::Instance().GetWorld());
    benchmarker.SetVerbosity(2);
    benchmarker.SetPoolMultiplier(1);
    benchmarker.SetRepetitions(nrep);
    benchmarker.SetPointCount(npoints);
    auto errcode                       = benchmarker.RunBenchmark();
    std::list<BenchmarkResult> results = benchmarker.PopResults();
    std::ofstream outStream;
    outStream.open(label, std::fstream::app);
    BenchmarkResult::WriteCsvHeader(outStream);
    for (auto i = results.begin(), iEnd = results.end(); i != iEnd; ++i) {
      i->WriteToCsv(outStream);
    }
    outStream.close();
    return errcode;
  };

  std::cout << "________________________________________________________________________________\n";
  switch (type) {
  case 0:
    std::cout << "Testing NoInnerRadii with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(NoInnerRadii(), "polyhedron_no-inner-radii.csv", npoints, nrep);
  case 1:
    std::cout << "Testing WithInnerRadii with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(WithInnerRadii(), "polyhedron_with-inner-radii.csv", npoints, nrep);
  case 2:
    std::cout << "Testing WithPhiSection with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(ManySegments(), "polyhedron_many-segments.csv", npoints, nrep);
  case 3:
    std::cout << "Testing ManySegments with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(WithPhiSection(), "polyhedron_phi-section.csv", npoints, nrep);
  default:
    std::cout << "Unknown polyhedron type." << std::endl;
  }
  return 1;
}
