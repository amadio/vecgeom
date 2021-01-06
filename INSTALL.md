# Installation

VecGeom uses [CMake](http://www.cmake.org/) for configuration and GMake for building.

## Requirements

### Compilation

A C++11 compatible compiler is required.
The library has been tested to compile with (but is not necessarily limited to):

- GCC >= 4.7.3
- Clang >= 3.4
- ICC >= 14.0.2

### Dependencies

#### Vc

If Vc is used as the backend for SIMD instructions, CMake must be able to find the package.
The Vc library is an [open source project](https://github.com/VcDevel/Vc).

Tested to compile with Vc >= 1.3.3.

#### CUDA

For CUDA support, an nVIDIA GPU with [compute capability](http://en.wikipedia.org/wiki/CUDA#Supported_GPUs) >= 3.0 must be available,
and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) must be installed.

Tested to compile with CUDA >= 10.0.

Quick start
-----------

Below are instructions to configure and build VecGeom using mostly default settings:

```sh
$ mkdir build && cd build
$ cmake ..
$ cmake --build .
$ cmake --build . --target install
```

The table below shows some options you may want to customise:

|Option|Default|Description|
|------|:-----:|-----------|
|BACKEND|scalar|Vector backend API to be used|
|BENCHMARK|OFF|Enable performance comparisons|
|BUILD_TESTING|ON|Enable build of tests and integration with CTest|
|BUILTIN_VECCORE|OFF|Build VecCore and its dependencies from source|
|COVERAGE_TESTING|OFF|Enable coverage testing flags|
|CUDA|OFF|Enable compilation for CUDA|
|CUDA_ARCH|30|Default CUDA device architecture|
|CUDA_USE_STATIC_CUDA_RUNTIME|OFF|Use the static version of the CUDA runtime library if available|
|CUDA_VOLUME_SPECIALIZATION|OFF|Use specialized volumes for CUDA|
|DATA_DOWNLOAD|OFF|Enable downloading of data for tests|
|DISTANCE_DEBUG|OFF|Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes|
|EMBREE|OFF|Enable Intel Embree|
|FAST_MATH|OFF|Enable the -ffast-math compiler option in Release builds|
|GDML|OFF|Enable GDML persistency. Requres Xerces-C|
|GDMLDEBUG|OFF|Enable additional debug information in GDML module|
|GEANT4|OFF|Build with support for Geant4 (https://geant4.web.cern.ch)|
|INPLACE_TRANSFORMATIONS|ON|Put transformation as members rather than pointers into PlacedVolume objects|
|NO_SPECIALIZATION|ON|Disable specialization of volumes|
|PLANESHELL|ON|Enable the use of PlaneShell class for the trapezoid|
|QUADRILATERAL_ACCELERATION|ON|Enable SIMD vectorization when looping over quadrilaterals|
|ROOT|OFF|Build with support for ROOT (https://root.cern)|
|STATIC_ANALYSIS|OFF|Enable static analysis on VecGeom|
|USE_CACHED_TRANSFORMATIONS|OFF|Use cached transformations in navigation states|
|USE_INDEXEDNAVSTATES|ON|Use indices rather than volume pointers in NavigationState objects|
|USE_NAVINDEX|OFF|Use navigation index table and states|
|VALIDATION|OFF|Enable validation tests from CMS geometry|
|VECGEOM_VECTOR|sse2|Vector instruction set to be used|
