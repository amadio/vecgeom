add_library(VecGeomBenchmark STATIC
  Benchmarker.h
  Benchmarker.cpp
  BenchmarkResult.h
  BenchmarkResult.cpp
  NavigationBenchmarker.h
  NavigationBenchmarker.cpp
  VolumePointers.h
  VolumePointers.cpp
  $<$<BOOL:${VECGEOM_ENABLE_CUDA}>:
    Benchmarker.cu;
    NavigationBenchmarker.cu>
  )

target_include_directories(VecGeomBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(VecGeomBenchmark PUBLIC vecgeom)