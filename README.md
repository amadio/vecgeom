# Introduction

VecGeom is a geometry modeller library with hit-detection features as needed by particle detector simulation at the LHC and beyond.
It was incubated by a Geant-R&D initiative and the motivation to combine the code of Geant4 and ROOT/TGeo into a single,
better maintainable piece of software within the EU-AIDA program. As such it is close in scope to TGeo and Geant4 geometry modellers.

**Main features** are
   * Build a hierarchic detector geometry out of simple primitives and use it on the CPU or GPU(CUDA)
   * Calculate distances and other geometrical information
   * Collision detection and navigation in complex scenes 
   * SIMD support in various flavours:
       1. True vector interfaces to primitives with SIMD acceleration when benefical
       2. SIMD acceleration of navigation through the use of special voxelization or bounding box hierarchies
   * Runtime specialization of objects to improve execution speed via a factory mechanism and use of C++ templates     
   * VecGeom also compiles under CUDA
   * Few generic kernels serve many instanteations of various simple or vectored interfaces or the CUDA version.

For further information, we provide

   * An installation guide here: tbd
   * A user's guide here: tbd
   * Doxygen reference guide here: tbd
   * Issue tracking system: tbd

## Contact information

Installation
============

VecGeom uses [CMake](http://www.cmake.org/) for configuration and GMake for building.

Quick start
-----------

Configure and build VecGeom using default parameters, installing into the build directory:

    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`
    make install

Requirements
------------

### Configuration
CMake >= 2.8.8.

### Compilation
A C++11 compatible compiler is required.
The library has been tested to compile with (but is not necessarily limited to):

- GCC >= 4.7.3
- Clang >= 3.4
- ICC >= 14.0.2

### Dependencies

#### Vc
If Vc is used as the backend for SIMD instructions, CMake must be able to find the package. The Vc library is an [open source project](http://code.compeng.uni-frankfurt.de/projects/vc/).

Tested to compile with Vc >= 1.0

#### Cilk Plus
When using Intel's Cilk Plus for SIMD instructions, CMake must be configured to compile with ICC.

#### CUDA
For CUDA support, an nVIDIA GPU with [compute capability](http://en.wikipedia.org/wiki/CUDA#Supported_GPUs) >= 2.0 must be available, and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) must be installed.

Tested to compile with CUDA >= 7.0 on GTX series >= 500.

CMake options
-------------

    OPTION [Default]
      Description.

    BACKEND [Vc]
      Specifies the backend used to generate CPU instructions. Available backends are:
      - Vc
      - Cilk
      - Scalar

    CUDA [OFF]
      Enable CUDA support.

    CUDA_ARCH [20]
      nVidia GPU architecture for which CUDA will be compiled. Will be passed directly to NVCC as -arch=<CUDA_ARCH>.

    BENCHMARK [OFF]
      Enables the benchmarking module. Will support a number of targets to benchmark against depending on which optional dependencies are activated. Flags that affect this are:
      - CUDA
      - ROOT
      - GEANT4

    ROOT [OFF]
      Look for a ROOT installation to include ROOT interfacing modules and enable ROOT as a benchmarking case.

    GEANT4 [OFF]
      Enable Geant4 as a benchmarking case.
