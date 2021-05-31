# - Setup the project
cmake_minimum_required(VERSION 3.8...3.19)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

include(cmake/modules/RecordCmdLine.cmake)
include(ExternalProject)

project(VecGeom VERSION 1.1.15)

################################################################################
# - Core CMake settings
# - Core/Custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)
include(MacroUtilities)
include(CMakeDependentOption)
include(IntelCompileFeatures)

# - Though we check for some absolute paths, ensure there are no others
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

# - Never export to or search in user/system package registry
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

# - Force project directories to appear first in any list of includes
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

# - Only relink shared libs when interface changes
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

# - Only report newly installed files
set(CMAKE_INSTALL_MESSAGE LAZY)

################################################################################
# - Core build settings
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)

# - Setting the C++ compiler to use
set("${CMAKE_CXX_COMPILER_ID}" TRUE CACHE STRING "C++ Compiler")
if(NOT GNU AND NOT Clang AND NOT AppleClang AND NOT Intel)
  message(WARNING "Unsupported compiler. Build will likely fail.")
endif()

# - C++ standard and extensions
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard")
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Use ccache if available ( to avoid recompilation on branch switches )
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  message(STATUS "found ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

# - Basic library settings
# Build statics with Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Review this as this is strictly incorrect for macOS (likely ROOT-ism)
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Install paths
include(GNUInstallDirs)
# Add a path for CMake config files (immutable)
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake")
set(CMAKE_INSTALL_FULL_CMAKEDIR "${CMAKE_INSTALL_FULL_LIBDIR}/cmake")

# Add uninstall target if required
if(NOT TARGET uninstall)
  configure_file(cmake/cmake_uninstall.cmake.in
    "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
    @ONLY)

  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

################################################################################
# Configuration options
# - ISA
set(VECGEOM_ISAS empty)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(i686|x86_64)")
  set(VECGEOM_ISAS sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 native empty)
endif()

enum_option(VECGEOM_VECTOR DOC "Vector instruction set to be used"
  TYPE STRING
  VALUES ${VECGEOM_ISAS}
  CASE_INSENSITIVE)

# - ISA Backend
set(VECGEOM_BACKENDS scalar umesimd vc)
enum_option(BACKEND DOC "Vector backend API to be used"
  TYPE STRING
  VALUES ${VECGEOM_BACKENDS}
  CASE_INSENSITIVE)

# Case adjustment because pass to VecCore components is
if(BACKEND STREQUAL "scalar")
  set(BACKEND "Scalar")
elseif(BACKEND STREQUAL "umesimd")
  set(BACKEND "UMESIMD")
elseif(BACKEND STREQUAL "vc")
  set(BACKEND "Vc")
endif()

# Configure backend for export to VecGeom's Config.h and setting up VecCore
string(TOUPPER "${BACKEND}" _BACKEND_UP)
set(VECGEOM_${_BACKEND_UP} True)
message(STATUS "Configuring with ${BACKEND} vector backend")

if("${BACKEND}" MATCHES "Vc|UMESIMD")
  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} ${BACKEND})
endif()

# - Core
option(BUILTIN_VECCORE "Build VecCore and its dependencies from source" OFF)

# - Precision
option(SINGLE_PRECISION "Use single precision throughout the package" OFF)
if (SINGLE_PRECISION)
  message(STATUS "VecGeom compiles in single precision mode.")
  set(VECGEOM_FLOAT_PRECISION ON)
endif()

option(CUDA "Enable compilation for CUDA." OFF)
cmake_dependent_option(CUDA_VOLUME_SPECIALIZATION "Use specialized volumes for CUDA." OFF "CUDA" OFF)
cmake_dependent_option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled." ON "CUDA" OFF)
cmake_dependent_option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF "CUDA" OFF)

if(CUDA)
  set(VECGEOM_ENABLE_CUDA ON)
  if(CUDA_VOLUME_SPECIALIZATION)
    set(VECGEOM_CUDA_VOLUME_SPECIALIZATION ON)
  endif()

  set(CUDA_ARCH 30 CACHE STRING "CUDA device architecture.")
  set(CUDA_ARCH "-arch=sm_${CUDA_ARCH}")

  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} "CUDA")
endif()

option(NO_SPECIALIZATION "Disable specialization of volumes." ON)
if(NO_SPECIALIZATION)
  set(VECGEOM_NO_SPECIALIZATION ON)
endif()

option(PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)
if(NOT PLANESHELL)
  set(VECGEOM_PLANESHELL_DISABLE TRUE)
endif()

option(QUADRILATERAL_ACCELERATION "Enable SIMD vectorization when looping over quadrilaterals (in Polyhedron)." ON)
if(QUADRILATERAL_ACCELERATION)
  set(VECGEOM_QUADRILATERALS_VC ON)
endif()

option(DISTANCE_DEBUG "Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes" OFF)
if(DISTANCE_DEBUG)
  set(VECGEOM_DISTANCE_DEBUG ON)
endif()

option(INPLACE_TRANSFORMATIONS "Put transformation as members rather than pointers into PlacedVolume objects" ON)
if(INPLACE_TRANSFORMATIONS)
  set(VECGEOM_INPLACE_TRANSFORMATIONS ON)
endif()

option(USE_INDEXEDNAVSTATES "Use indices rather than volume pointers in NavigationState objects" ON)
if(USE_INDEXEDNAVSTATES)
  set(VECGEOM_USE_INDEXEDNAVSTATES ON)
endif()

option(USE_NAVINDEX "Use navigation index table and states" OFF)
if(CUDA)
  ### CUDA=ON requires NAVINDEX
  message(STATUS CUDA=ON implies USE_NAVINDEX=ON)
  set(USE_NAVINDEX ON)
endif()
if(USE_NAVINDEX)
  set(VECGEOM_USE_NAVINDEX ON)
endif()

option(GDML "Enable GDML persistency. Requires Xerces-C" OFF)
if(GDML)
  set(VECGEOM_GDML ON)
endif()
option(GDMLDEBUG "Enable additional debug information in GDML module" OFF)

option(EMBREE "Enable Intel Embree" OFF)
if(EMBREE)
  set(VECGEOM_EMBREE ON)
endif()

option(USE_CACHED_TRANSFORMATIONS "Use cached transformations in navigation states" OFF)
if(USE_CACHED_TRANSFORMATIONS)
  set(VECGEOM_CACHED_TRANS ON)
endif()

option(FAST_MATH "Enable the -ffast-math compiler option in Release builds" OFF)

################################################################################
# - Testing, unit, coverage, benchmarking, validation etc
include(CTest)

# TODO: not completely set up. Will build VG+tests with correct flags/links, but no report gen.
cmake_dependent_option(COVERAGE_TESTING "Enable coverage testing flags" OFF "BUILD_TESTING" OFF)

# TODO: Check use of this option, not totally clear on use
cmake_dependent_option(VALIDATION "Enable validation tests from CMS geometry" OFF "BUILD_TESTING" OFF)

# Static tests build/use clang-tidy, so require compile commands for VecGeom to be exported
cmake_dependent_option(STATIC_ANALYSIS "Enable static analysis/testing on VecGeom" OFF "BUILD_TESTING" OFF)
if(STATIC_ANALYSIS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Following options result in changes to VecGeom API/ABI, but are strictly only for testing (AFAK)
cmake_dependent_option(VTUNE "Enable the use of profiler Vtune" OFF "BUILD_TESTING" OFF)
if(VTUNE)
  set(VECGEOM_VTUNE ON)
endif()

# These are also testing only
cmake_dependent_option(ROOT "Enable testing using ROOT" OFF "BUILD_TESTING" OFF)
if(ROOT)
  set(VECGEOM_ROOT ON)
endif()

cmake_dependent_option(GEANT4 "Enable testing using Geant4" OFF "BUILD_TESTING" OFF)
if(GEANT4)
  set(VECGEOM_GEANT4 ON)
endif()

cmake_dependent_option(BENCHMARK "Enable benchmark/performance tests" OFF "BUILD_TESTING" OFF)
if(BENCHMARK)
  set(VECGEOM_BENCHMARK ON)
endif()


################################################################################
# Minimum version of VecCore we need.
set(VecCore_VERSION "0.5.2")

# Let's see if we can find VecCore
if(NOT BUILTIN_VECCORE)
  # Find VecCore with selected components turned on (CUDA and backend)
  find_package(VecCore ${VecCore_VERSION} QUIET COMPONENTS ${VecCore_COMPONENTS})
  if(NOT VecCore_FOUND)
    # This is forced because otherwise re-cmake will setup non-builtin VecCore
    set(BUILTIN_VECCORE ON CACHE BOOL "Build VecCore and its dependencies from source" FORCE)
  endif()
endif()

# Have to recheck because the above may re-set the variable
if(BUILTIN_VECCORE)
  include(BuiltinVecCore)
endif()

# Find VecCore with selected components turned on (CUDA and backend)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_COMPONENTS})

# Should no longer need this as VecCore provides usage requirements
#include_directories(${VecCore_INCLUDE_DIRS})
set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} VecCore::VecCore)

################################################################################
# - Compiler Flags/Settings for each build type
# We do this here as flags are dependent on chosen ISA
# Flags will also be forwarded by CUDA when compiling C++.
set(VECGEOM_ERROR_LIMIT 20 CACHE STRING "Limit number of errors output by compiler")
mark_as_advanced(VECGEOM_ERROR_LIMIT)

if(GNU)
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fmax-errors=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
     set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ggdb -O0")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ftree-vectorize -finline-limit=10000000")
    if(FAST_MATH)
      set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ffast-math")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -faligned-new")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "4.9")
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()

elseif(Intel)
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fmax-errors=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ggdb -O0")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fno-alias")
  endif()

elseif(Clang OR AppleClang)
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -ferror-limit=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS}  -ggdb -O0")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ftree-vectorize")
    if(FAST_MATH)
      set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ffast-math")
    endif()
  endif()

  # Whilst we don't test clang < 4, it is used to build the custom static
  # analysis tools, and this causes it to barf...
  if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "4")
   set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -faligned-new")
  endif()
endif()

# - Additional flags for coverage testing support
if(COVERAGE_TESTING)
  if(GNU)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} gcov)
  endif()

  if(Clang OR AppleClang)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} --coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} profile_rt)
  endif()

  if(Intel)
    message(FATAL_ERROR "Coverage testing not supported for icc.")
  endif()
endif()

# - ISA-specific flags
string(TOLOWER ${VECGEOM_VECTOR} _arch_lo)
string(TOUPPER ${VECGEOM_VECTOR} _arch_up)

if(${_arch_lo} MATCHES native)
  if(Intel)
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -xHost")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64le")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -mcpu=${_arch_lo} -mtune=${_arch_lo}")
  else()
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -march=${_arch_lo}")
  endif()
elseif(NOT ${_arch_lo} MATCHES empty)
  if(Intel)
    if(${_arch_lo} MATCHES avx2)
      set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -xCORE-${_arch_up}")
    else()
      set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -x${_arch_up}")
    endif()
  else()
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -m${_arch_lo}")
  endif()
endif()
message(STATUS "Compiling for ${_arch_up} SIMD architecture")

################################################################################
# - CUDA compiler flags
if(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -g -G ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()
if(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -use_fast_math ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()

if(VECGEOM_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
    -std=c++${CMAKE_CUDA_STANDARD}
    -Xcompiler -Wno-unused-function
    -Xcudafe "--diag_suppress=code_is_unreachable"
    -Xcudafe "--diag_suppress=initialization_not_reachable")

  if(NOT NO_SPECIALIZATION)
    set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
      -Xptxas --disable-optimizer-constants)
  endif()

  set(VECGEOM_NVCC_CONFIGS
    DEBUG -g -G -O0
    RELEASE  -O3 -use_fast_math
    RELWITHDEBINFO -g -G -O3 -use_fast_math
    MINSIZEREL -use_fast_math)

  # FindCUDA.mk generate cmake file for each .o file using
  # the cmake command:
  #   file(GENERATE
  #      OUTPUT "${custom_target_script}"
  #      INPUT "${custom_target_script_pregen}"
  #      )
  # However this fails to register the output as a 'BYPRODUCTS'
  # and thus, since the .o file depends on it, trigger the CMP0058
  # warning (when using Ninja).
  # Using a add_custom_target/command does not work there as
  # neither the OUTPUT not the BYPRODUCTS parameter supports
  # generator expression (and ${custom_target_script} is a
  # generator expression depending on the configuration type).
  # Consequently there is no (known to the author) way of
  # correctly adding the correct information at the moment.
  # So let's just use the old policy (even-though it is
  # deprecated ... )
  # Note: this is a feature introduced in CMake 3.6 to
  # support  generator expressions in CUDA_NVCC_FLAGS
  # and since we imported that version's FindCUDA.cmake
  # it applies to all cmake that support CMP0058.
  if(NOT (CMAKE_VERSION LESS 3.3))
    cmake_policy(SET CMP0058 OLD)
  endif()

  # Even-though VecCore (if enabled) has already enabled CUDA, if we do not do it a second time,
  # the compilation flags are incorrectly set.  The symptoms is that cuda_add_library for vecgeomcuda
  # misinterpret the ${VECGEOM_NVCC_CONFIGS} flag and rather than splitting in configurations, pass it as is
  # to the NVCC command line ...
  find_package(CUDA REQUIRED)

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()

### Set CXXFLAGS early, so that VECGEOM_GEANT4 and VECGEOM_ROOT are already defined when compiling benchmarks, or it breaks
# Pass flags to compilers
# We may have addition in compiler flags from the above included packages
# We don't want to have trailing CMAKE_CXX_FLAGS_"BUILD_TYPE" options on the command line
# but rather have it as first set of argument.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type}} ${VECGEOM_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS}")

# At this point, all changes to CXXFLAGS have been made.
# The line below is necessary to allow passing extra options
# via -DCMAKE_CXX_FLAGS or CXXFLAGS environment variable.
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")
if(VECGEOM_ENABLE_CUDA)
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
endif()

if(VECGEOM_ROOT)
  # ROOT install may be relying on ROOTSYS
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

  find_package(ROOT REQUIRED COMPONENTS Geom Graf3d)

  #---setup ROOT include + lib dirs
  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIRS})
  link_directories(${ROOT_LIBRARY_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${ROOT_LIBRARIES})
  set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${ROOT_INCLUDE_DIRS}")
endif()

# Intel Embree
if(VECGEOM_EMBREE)
  find_package(embree 3.1 REQUIRED)
  include_directories(${EMBREE_INCLUDE_DIRS})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${EMBREE_LIBRARIES})
endif()

if(VECGEOM_GEANT4)
  find_package(Geant4 REQUIRED)
  # need to define G4MULTITHREADED for Geant4-MT
  add_definitions(${Geant4_DEFINITIONS})
  include_directories(AFTER SYSTEM ${Geant4_INCLUDE_DIRS})
  set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${Geant4_INCLUDE_DIRS}")
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${Geant4_LIBRARIES})

  if(VECGEOM_ROOT)
    # We use VGM to convert (test) geometries from ROOT input to G4
    # without needing an intermediate gdml file
    # for user friendliness .. added as external project for now
    set(VGM_INSTALL "${PROJECT_BINARY_DIR}/vgminstall/")
    ExternalProject_Add(VGM
      GIT_REPOSITORY "https://github.com/alisw/VGM.git"
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${VGM_INSTALL}
        -DWITH_EXAMPLES=OFF
        -DGeant4_DIR=${Geant4_DIR}
        -DROOT_DIR=${ROOT_DIR}
        -DCMAKE_INSTALL_LIBDIR=${VGM_INSTALL}/lib)

    # now modify includes and libs for later linking
    include_directories(${VGM_INSTALL}/include)
    link_directories(${VGM_INSTALL}/lib)
    set(VECGEOM_LIBRARIES_EXTERNAL "-L${VGM_INSTALL}/lib" BaseVGM;ClhepVGM;XmlVGM;Geant4GM;RootGM ${VECGEOM_LIBRARIES_EXTERNAL})
    set(VGM ON)
  endif()
endif()

if(VECGEOM_VTUNE)
  find_package(VTUNE REQUIRED)
  include_directories(AFTER SYSTEM ${VTUNE_INCLUDE_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${VTUNE_LIBRARIES} -lpthread -ldl)
endif()

################################################################################
# - Configure source lists for library(ies)
# VECGEOM_COMMON_SRCS: C++ files compiled by both CXX and CUDA
# VECGEOM_CPPONLY_SRCS: C++ files only compiled for CXX
# VECGEOM_CUDA_SRCS: CUDA files (both from copies of common cpp->cu, and .cu)
file(GLOB _createnuclei RELATIVE ${PROJECT_SOURCE_DIR} source/generated/CreateNuclei*.cpp)

# Sources common to C++ and CUDA
set(VECGEOM_COMMON_SRCS
  #.. moved up for faster compilation
  ${_createnuclei}
  source/UnplacedPolycone.cpp
  source/UnplacedPolyhedron.cpp
  source/UnplacedTet.cpp
  source/UnplacedTorus2.cpp
  source/UnplacedTube.cpp
  source/UnplacedEllipticalTube.cpp
  source/UnplacedEllipticalCone.cpp
  source/UnplacedEllipsoid.cpp
  source/UnplacedCoaxialCones.cpp
  source/UnplacedGenericPolycone.cpp
  source/UnplacedCone.cpp
  source/UnplacedCutTube.cpp
  source/UnplacedGenTrap.cpp

  source/LogicalVolume.cpp
  source/PlacedPolyhedron.cpp
  source/PlacedPolycone.cpp
  source/PlacedCone.cpp
  source/PlacedAssembly.cpp
  source/PlacedBox.cpp
  source/PlacedSExtru.cpp
  source/PlacedTet.cpp
  source/PlacedHype.cpp
  source/PlacedTube.cpp
  source/PlacedEllipticalTube.cpp
  source/PlacedEllipticalCone.cpp
  source/PlacedEllipsoid.cpp
  source/PlacedCoaxialCones.cpp
  source/PlacedGenericPolycone.cpp
  source/PlacedCutTube.cpp
  source/PlacedTorus2.cpp
  source/PlacedTrd.cpp
  source/PlacedGenTrap.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedScaledShape.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedTessellated.cpp
  source/PlacedMultiUnion.cpp
  source/PlacedExtruded.cpp
  source/PlacedVolume.cpp
  source/Planes.cpp
  source/Plane.cpp
  source/CutPlanes.cpp
  source/Quadrilaterals.cpp
  source/Rectangles.cpp
  source/TessellatedHelpers.cpp
  source/Tile.cpp
  source/Scale3D.cpp
  source/Transformation3D.cpp
  source/UnplacedAssembly.cpp
  source/UnplacedBox.cpp
  source/UnplacedSExtruVolume.cpp
  source/UnplacedHype.cpp
  source/UnplacedTrd.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedScaledShape.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedTessellated.cpp
  source/UnplacedMultiUnion.cpp
  source/UnplacedExtruded.cpp
  source/UnplacedVolume.cpp
  source/NavStateIndex.cpp
  source/NavStatePath.cpp
  source/NavIndexTable.cpp

  source/UnplacedOrb.cpp
  source/PlacedOrb.cpp
  source/UnplacedSphere.cpp
  source/PlacedSphere.cpp
  source/UnplacedBooleanVolume.cpp
  source/PlacedBooleanVolume.cpp
  source/Wedge.cpp
  source/Wedge_Evolution.cpp
  source/ABBoxManager.cpp
  source/HybridManager2.cpp
  source/FlatVoxelManager.cpp
  source/BVH.cpp
  source/BVHManager.cpp

  source/MessageLogger.cpp

  services/NavigationSpecializer.cpp

  source/ResultComparator.cpp
  source/ReducedPolycone.cpp
  source/Utils3D.cpp
  source/SolidMesh.cpp)

set(VECGEOM_CPPONLY_SRCS
  test/shape_tester/ShapeTester.cpp
  test/shape_tester/ConventionChecker.cpp
  source/GeoManager.cpp
  source/CppExporter.cpp)

if(VECGEOM_EMBREE)
  list(APPEND VECGEOM_COMMON_SRCS source/EmbreeManager.cpp)
endif()

if(VECGEOM_GEANT4)
  list(APPEND VECGEOM_COMMON_SRCS source/G4GeoManager.cpp)
endif()

if(VECGEOM_ROOT)
  list(APPEND VECGEOM_COMMON_SRCS source/RootGeoManager.cpp)
  list(APPEND VECGEOM_CPPONLY_SRCS
    source/PlacedRootVolume.cpp
    source/UnplacedRootVolume.cpp
    source/ShapeDebugger.cpp
    source/Visualizer.cpp)
endif()

# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.
if(VECGEOM_ENABLE_CUDA)
  list(APPEND VECGEOM_COMMON_SRCS
    source/RNG.cpp
    source/AOS3D.cpp
    source/SOA3D.cpp
    source/Vector.cpp)

  list(APPEND VECGEOM_CPPONLY_SRCS
    source/CudaManager.cpp
    source/backend/cuda/Interface.cpp)

  # some CUDA kernel code in "userspace" and test
  set(VECGEOM_CUDA_SRCS
    source/BVHManager.cu
    source/CudaManager.cu
    source/CudaGlobalSymbols.cu
    # ?? Test code get compiled into vecgeomcuda ??
    test/cuda/BVHTest.cu
    test/cuda/MapTest.cu
    test/cuda/MapTestClass.cu)

  # Filter file prefixes that won't be compiled for CUDA
  set(NOT_FOR_CUDA
    ABBoxManager
    HybridManager2
    FlatVoxelManager
    Medium
    G4GeoManager
    RootGeoManager
    NavigationSpecializer
    ResultComparator
    UnplacedTessellated
    UnplacedMultiUnion
    PlacedTessellated
    PlacedMultiUnion
    TessellatedCluster
    TessellatedHelpers
    Tile
    UnplacedExtruded
    PlacedExtruded
    SolidMesh)

  foreach(SRC_FILE ${VECGEOM_COMMON_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)
    list(FIND NOT_FOR_CUDA ${SRC_FILENAME} _index)

    if(${_index} EQUAL -1)
      set(SRC_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE})
      set(SRC_FILE_CU ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu)

      add_custom_command(
        OUTPUT ${SRC_FILE_CU}
        COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE_CPP} ${SRC_FILE_CU}
        DEPENDS ${SRC_FILE_CPP})

      list(APPEND VECGEOM_CUDA_SRCS ${SRC_FILE_CU})
    endif()
  endforeach()
endif()

# - Additional sources for benchmarking
if(VECGEOM_BENCHMARK)
  list(APPEND VECGEOM_CPPONLY_SRCS
    source/benchmarking/NavigationBenchmarker.cpp
    source/benchmarking/BenchmarkResult.cpp
    source/benchmarking/Benchmarker.cpp
    source/benchmarking/VolumePointers.cpp)

  list(APPEND VECGEOM_CUDA_SRCS
    source/benchmarking/Benchmarker.cu
    source/benchmarking/NavigationBenchmarker.cu
    test/benchmark/TestNavIndex.cu)
endif()

################################################################################
# Build libraries
if(NOT APPLE)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} rt pthread)
endif()

# - Configure settings header
configure_file(VecGeom/base/Config.h.in ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h @ONLY)

add_library(vecgeom
  ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h
  ${VECGEOM_COMMON_SRCS}
  ${VECGEOM_CPPONLY_SRCS})
target_compile_features(vecgeom PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(vecgeom PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vecgeom
  PUBLIC ${VECGEOM_LIBRARIES} ${VECGEOM_LIBRARIES_EXTERNAL}
  PRIVATE ${CMAKE_DL_LIBS})

if(VGM)
  add_dependencies(vecgeom VGM)
endif()

set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeom)

if(GDML)
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vgdml)
endif()

# build the CUDA version of the library
if(VECGEOM_ENABLE_CUDA)
  # This is a hack to force the project include dirs as the first in
  # the list passed to nvcc. Due to FindCUDA not respecting CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE
  # Will disappear once migration to native CMake CUDA support is done.
  cuda_include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

  cuda_add_library(vecgeomcuda ${VECGEOM_CUDA_SRCS} SHARED
    OPTIONS ${CUDA_ARCH}
    # For debug information we might need
    #    -O0 -g -G -lineinfo
    ${VECGEOM_NVCC_CONFIGS})

  target_link_libraries(vecgeomcuda ${VECGEOM_LIBRARIES})
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeomcuda)

  # build the CUDA user-code library
  cuda_add_library(cudauserlib userexamples/src/TestNavigationStatePool.cu SHARED
    OPTIONS ${CUDA_ARCH})

  # Note: the repeat below is due the author of  cuda_add_library_depend
  # not knowing how to automatically go from the target to the real
  # file in add_custom_command
  cuda_add_library_depend(cudauserlib vecgeomcuda_static libvecgeomcuda_static.a)
  target_link_libraries(cudauserlib ${VECGEOM_LIBRARIES})
  set(USERKERNELLIB cudauserlib)
endif()

# Make sure we build VecCore before building VecGeom
if(BUILTIN_VECCORE)
  add_dependencies(vecgeom VecCore)
  if(VECGEOM_ENABLE_CUDA)
    add_dependencies(vecgeomcuda VecCore)
  endif()
endif()

# build the libraries for GDML persistency
if(GDML)
  add_subdirectory(persistency/gdml)
endif()

################################################################################
# TESTING
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

################################################################################
# Installation
# Install headers and libraries
install(DIRECTORY VecGeom DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN "*.h.in" EXCLUDE)

install(FILES ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/VecGeom/base")

install(TARGETS vecgeom EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")

if(VECGEOM_ENABLE_CUDA)
  install(TARGETS vecgeomcuda EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS vecgeomcuda_static EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# CMake support files
include(CMakePackageConfigHelpers)

file(RELATIVE_PATH INSTALL_INCLUDE_DIR_RELATIVE
  "${CMAKE_INSTALL_FULL_CMAKEDIR}/VecGeom" "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
file(RELATIVE_PATH INSTALL_LIB_DIR_RELATIVE
  "${CMAKE_INSTALL_FULL_CMAKEDIR}/VecGeom" "${CMAKE_INSTALL_FULL_LIBDIR}")

# Common
write_basic_package_version_file(VecGeomConfigVersion.cmake COMPATIBILITY AnyNewerVersion)

# Build
set(CONF_INCLUDE_DIR "${PROJECT_SOURCE_DIR}")

if(BUILTIN_VECCORE)
  set(VECCORE_PREFIX "${VecCore_ROOTDIR}")
  set(PATH_VARS_ARG PATH_VARS VECCORE_PREFIX)
endif()

configure_package_config_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}/VecGeomConfig.cmake"
  ${PATH_VARS_ARG}
  INSTALL_PREFIX "${PROJECT_BINARY_DIR}"
  INSTALL_DESTINATION "${PROJECT_BINARY_DIR}")
export(EXPORT VecGeomTargets NAMESPACE VecGeom::)

# Installation
set(CONF_INCLUDE_DIR "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}")

if(BUILTIN_VECCORE)
  set(VECCORE_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(PATH_VARS_ARG PATH_VARS VECCORE_PREFIX)
endif()

configure_package_config_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  ${PATH_VARS_ARG}
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/VecGeom")

# Install the VecGeom{Config,Version,Targets}.cmake
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  "${PROJECT_BINARY_DIR}/VecGeomConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/VecGeom" COMPONENT dev)
install(EXPORT VecGeomTargets
  NAMESPACE VecGeom::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VecGeom")


################################################################################
# Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYFILE_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/doxygen)
  foreach(d doc/doxygen vecgeom persistency scripts source services)
    set(DOXYFILE_SOURCE_DIRS "${DOXYFILE_SOURCE_DIRS} ${PROJECT_SOURCE_DIR}/${d}")
  endforeach()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMENT "Writing documentation to ${DOXYFILE_OUTPUT_DIR}..."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  add_custom_target(doxydir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYFILE_OUTPUT_DIR}
    COMMENT "Creating doc directory")
  add_dependencies(doxygen doxydir)
endif()
