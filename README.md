# Introduction

VecGeom is a geometry modeller library with hit-detection features as needed by particle detector simulation at the LHC and beyond.
It was incubated by a Geant-R&D initiative and the motivation to combine the code of Geant4 and ROOT/TGeo into a single,
better maintainable piece of software within the EU-AIDA program. As such it is close in scope to TGeo and Geant4 geometry modellers.

**Main features** are:
   * Build a hierarchic detector geometry out of simple primitives and use it on the CPU or GPU(CUDA)
   * Calculate distances and other geometrical information
   * Collision detection and navigation in complex scenes
   * SIMD support in various flavours:
       1. True vector interfaces to primitives with SIMD acceleration when benefical
       2. SIMD acceleration of navigation through the use of special voxelization or bounding box hierarchies
   * Runtime specialization of objects to improve execution speed via a factory mechanism and use of C++ templates
   * VecGeom also compiles under CUDA
   * Few generic kernels serve many instanteations of various simple or vectored interfaces or the CUDA version.

For further information:

   * [Installation instructions](INSTALL.md)
   * User's guide - tbd
   * [Doxygen reference guide](https://lcgapp-services.cern.ch/spi-jenkins/job/VecGeom-Doxygen/doxygen/)
   * [Issue tracking system](http://sft.its.cern.ch/jira/projects/VECGEOM)
