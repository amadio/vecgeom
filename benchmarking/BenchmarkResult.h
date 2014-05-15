/// @file BenchmarkResult.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_
#define VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_

#include "base/global.h"

#include <ostream>

namespace vecgeom {

enum EBenchmarkedLibrary {
  kBenchmarkSpecialized = 0,
  kBenchmarkVectorized = 1,
  kBenchmarkUnspecialized = 2,
  kBenchmarkCuda = 3,
  kBenchmarkUSolids = 4,
  kBenchmarkRoot = 5
};

enum EBenchmarkedMethod {
  kBenchmarkInside = 0,
  kBenchmarkDistanceToIn = 1,
  kBenchmarkSafetyToIn = 2,
  kBenchmarkDistanceToOut = 3,
  kBenchmarkSafetyToOut = 4
};

struct BenchmarkResult {
public:
  const Precision elapsed;
  const EBenchmarkedMethod method;
  const EBenchmarkedLibrary library;
  static char const *const fgMethodLabels[];
  static char const *const fgLibraryLabels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const Precision bias;
};

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark);

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_