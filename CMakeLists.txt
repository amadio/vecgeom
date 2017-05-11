#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.1.0)

include(cmake/modules/RecordCmdLine.cmake)
include(ExternalProject)

project(VecGeom)

set(VECGEOM ON)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard")
set(CMAKE_CXX_STANDARD_REQUIRED True)

################################################################################
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)

# Setting the C++ compiler to use
set("${CMAKE_CXX_COMPILER_ID}" TRUE CACHE STRING "C++ Compiler")
if (NOT GNU AND NOT Clang AND NOT AppleClang AND NOT Intel)
  message(WARNING "Unsupported compiler. Build will likely fail.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

include(MacroUtilities)
################################################################################

# Configuration options
option(USOLIDS_VECGEOM "Forward USOLIDS volumes to their VecGeom implementation" OFF)

option(CUDA "Enable compilation for CUDA." OFF)

option(CUDA_VOLUME_SPECIALIZATION "Use specialized volumes for CUDA." OFF)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(GEANT4 "Include Geant4." OFF)

option(BENCHMARK "Enable performance comparisons." OFF)

option(NO_SPECIALIZATION "Disable specialization of volumes." ON)

option(COVERAGE_TESTING "Enable coverage testing flags." OFF)

option(CTEST "Enable CTest when building." ON)

option(VALIDATION "Enable validation tests from CMS geometry." OFF)

option(PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)

option(QUADRILATERAL_ACCELERATION "Enable SIMD vectorization when
looping over quadrilaterals (in Polyhedron)." ON)

option(DISTANCE_DEBUG "Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes" OFF)

option(INPLACE_TRANSFORMATIONS "Put transformation as members rather than pointers into PlacedVolume objects" ON)

option(USE_INDEXEDNAVSTATES "Use indices rather than volume pointers in NavigationState objects" ON)

option(DATA_DOWNLOAD "Enable downloading of data for tests" OFF)

option(GENERATE_MATERIALS_DB "Include materials and particle declarations in library" OFF)

option(STATIC_ANALYSIS "enable static analysis on VecGeom" OFF)

set(VECGEOM_ARCH native sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 mic mic_avx512)

enum_option(VECGEOM_VECTOR
      DOC "Vector instruction set to be used"
      TYPE STRING
      VALUES ${VECGEOM_ARCH}
      CASE_INSENSITIVE
    )

set(VECGEOM_BACKEND default Scalar Vc Cilk MicVec umesimd)

enum_option(BACKEND
            DOC "Backend infrastructure to be used"
            TYPE STRING
            VALUES ${VECGEOM_BACKEND}
            CASE_INSENSITIVE
           )

if (CTEST)
  message(STATUS "Testing with CTest enabled.")
  enable_testing()
endif()

option(VTUNE "Enable the use of profiler Vtune" OFF)

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries.")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files.")
set(INSTALL_CMAKE_DIR lib/CMake/VecGeom CACHE PATH
    "Installation directory for CMake files.")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

if (USOLIDS)
  set(USolids ON)
endif()

if (APPLE)
   set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if (GEANT4)
   set(Geant4 ON)
endif()

if (CTEST)
  set (DATA_DOWNLOAD ON)
endif()

################################################################################

# VecCore will initialize the CUDA support so we must have those over-ride placed here as they
# need to be done before loading FindCUDA.cmake
if (CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Compile CUDA objects with separable compilation enabled.")
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available")
endif()

# Determine backend

string(TOUPPER ${BACKEND} _backend_up)

if (_backend_up MATCHES VC)
  set(Vc ON)
  set(VC ON CACHE BOOL "" FORCE) # enable Vc in VecCore
  message(STATUS "Configuring with \"Vc\" vector backend")
elseif (_backend_up MATCHES SCALAR)
  set(Scalar ON)
  message(STATUS "Configuring with \"Scalar\" backend.")
elseif (_backend_up MATCHES CILK)
  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icpc) for Cilk+ backend.")
  endif()
  set(Cilk ON)
  message(STATUS "Configuring with \"Cilk+\" vector backend.")
elseif (_backend_up MATCHES MICVEC)
  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icpc) with MIC SIMD for MICVEC backend.")
  endif()
  set(MICVEC ON)
  message(STATUS "Configuring with \"Micvec\" vector backend.")
elseif (_backend_up MATCHES UMESIMD)
  set(UMESIMD ON)
  find_package(UMESIMD)
  if(UMESIMD_FOUND)
    set(UMESIMD ON CACHE BOOL "" FORCE) # enable UMESIMD in VecCore
    include_directories(SYSTEM ${UMESIMD_INCLUDE_DIR})
    set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${UMESIMD_INCLUDE_DIR}")
  endif()
  message(STATUS "Configuring with \"UME::SIMD\" vector backend.")
elseif (_backend_up MATCHES DEFAULT)
  set(Vc ON)
  set(VC ON CACHE BOOL "" FORCE) # enable Vc in VecCore
  set(BACKEND VC CACHE STRING "Set by default to VC" FORCE)
  message(STATUS "Backend not specified. Defaulting to Vc.")
else()
  message(FATAL_ERROR "Value '${BACKEND}' for variable BACKEND is not allowed. ")
endif()

add_subdirectory(VecCore)
# A hack for more user friendliness
# If Vc wanted but was not found .. we fall back to scalar
if (Vc)
  if (NOT Vc_FOUND)
    message(STATUS "****************************************************************")
    message(STATUS "***** Tried to configure using Vc backend but Vc was not found *")
    message(STATUS "***** Falling back to Scalar backend                           *")
    message(STATUS "***** If you want to use Vc; please install it                 *")
    message(STATUS "******  ...   and/or fix your cmake command options            *")
    message(STATUS "****************************************************************")
    set(Vc OFF)
    set(VC OFF)
    set(Scalar ON)
    set(BACKEND Scalar CACHE STRING "Fallback by default to Scalar" FORCE)
  endif()
endif()


# CMake does not propagate these up, must be added by hand
include_directories(${PROJECT_SOURCE_DIR}/VecCore/include)
include_directories(${PROJECT_BINARY_DIR}/VecCore/include)

################################################################################

# Set Default compiler flags for each build type

if (GNU)
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fPIC")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ggdb -O0")

  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -finline-limit=10000000 -ffast-math -ftree-vectorize")
  endif()
  if (Vc)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fabi-version=6")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -fabi-version=6")
  endif()
elseif(Intel)
  # CMake fails to add -std option for icc, so add it manually
  if("${CMAKE_VERSION}" VERSION_LESS "3.6")
    if(CMAKE_CXX_STANDARD STREQUAL 11)
      list(APPEND CMAKE_CXX_FLAGS -std=c++11)
    elseif(CMAKE_CXX_STANDARD STREQUAL 14)
      list(APPEND CMAKE_CXX_FLAGS -std=c++14)
    else()
      message(FATAL_ERROR "Unsupported C++ standard requested")
    endif()
  endif()
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fPIC -diag-disable 3438")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -g -O0")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fno-alias")
  endif()
elseif(Clang OR AppleClang)
  if (APPLE)
    # prefer LLVM's standard C++ library on Mac OS X
    set (VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -stdlib=libc++")
  endif()
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fPIC")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS}  -ggdb -O0")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ffast-math -ftree-vectorize")
  endif()
else()
   message(WARNING "No compiler type matched, the compiler flags for VecGeom have *not* been set.")
endif()

if (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -g -G ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()
if (CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -use_fast_math ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()

string(TOLOWER ${VECGEOM_VECTOR} _arch_lo)
string(TOUPPER ${VECGEOM_VECTOR} _arch_up)


if (${_arch_lo} MATCHES mic_avx512)
  if (Intel)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -xMIC_AVX512")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -xMIC_AVX512")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -march=knl")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -march=knl")
  endif()
elseif (${_arch_lo} MATCHES mic)
  if (Intel)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -m${_arch_lo}")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -m${_arch_lo}")
    set(MIC TRUE)
  else()
   message(FATAL_ERROR "MIC native or offload compilation requires Intel Compiler.")
  endif()
elseif (${_arch_lo} MATCHES native)
  if (Intel)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -xHost")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -xHost")
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -march=${_arch_lo}")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -march=${_arch_lo}")
  endif()
else()
  if (Intel)
    if (${_arch_lo} MATCHES avx2)
      set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -xCORE-${_arch_up}")
      set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -xCORE-${_arch_up}")
    else()
      set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -x${_arch_up}")
      set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -x${_arch_up}")
    endif()
  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -m${_arch_lo}")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -m${_arch_lo}")
  endif()
endif()
message(STATUS "Compiling for ${_arch_up} SIMD architecture")

################################################################################

# Set C++ compiler flags. Will also be forwarded by CUDA when compiling C++.

if ((NOT PLANESHELL))
  message(STATUS "Disabling PlaneShell class")
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_PLANESHELL_DISABLE")
endif()

# Backend specific
if (Scalar)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_SCALAR")
endif()
if (Vc)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_VC")
endif()
if (Cilk)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_CILK")
endif()
if (MICVEC)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_MICVEC")
endif()
if (UMESIMD)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_UMESIMD")
endif()

if (CUDA)
  set(CUDA_ARCH 30 CACHE STRING "CUDA device architecture.")
  set(CUDA_ARCH "-arch=sm_${CUDA_ARCH}")
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_ENABLE_CUDA")
  if (CUDA_VOLUME_SPECIALIZATION)
    set(VECGEOM_DEFINITIONS
        "${VECGEOM_DEFINITIONS} -DVECGEOM_CUDA_VOLUME_SPECIALIZATION")
  else()
    set(VECGEOM_DEFINITIONS
        "${VECGEOM_DEFINITIONS} -DVECGEOM_CUDA_NO_VOLUME_SPECIALIZATION")
  endif()
  set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
      -std=c++11
      -Xcompiler -Wno-unused-function
      -Xcudafe "--diag_suppress=code_is_unreachable"
      -Xcudafe "--diag_suppress=initialization_not_reachable")
  if (NOT NO_SPECIALIZATION)
     set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
        -Xptxas --disable-optimizer-constants )
  endif()
  set(VECGEOM_NVCC_CONFIGS
        DEBUG -g -G
        RELEASE  -O3 -use_fast_math
        RELWITHDEBINFO -g -G -O3 -use_fast_math
        MINSIZEREL -use_fast_math )
endif()

if (Clang OR AppleClang)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} dl)
endif()

# Enable pretty diagnostics coloring if gcc 4.9+ is being used
if (GNU)
  # gcc needs -ldl to be added explicitly
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} dl)
  set(VECGEOM_ERROR_LIMIT 20 CACHE STRING
      "Limit number of errors output by diagnostics.")
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fmax-errors=${VECGEOM_ERROR_LIMIT}")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()
endif()

# Coverage testing support
if (COVERAGE_TESTING)
  if (GNU)
    set(VECGEOM_CXX_FLAGS
        "${VECGEOM_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} gcov)
  endif()
  if (Clang OR AppleClang)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} --coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} profile_rt)
  endif()
  if (Intel)
    message(FATAL_ERROR "Coverage testing not supported for icc.")
  endif()
endif()

# Forward CMake options
if (USOLIDS_VECGEOM)
  set(VECGEOM_DEFINITIONS
      "${VECGEOM_DEFINITIONS} -DVECGEOM_REPLACE_USOLIDS")
endif()
if (QUADRILATERAL_ACCELERATION)
  set(VECGEOM_DEFINITIONS
      "${VECGEOM_DEFINITIONS} -DVECGEOM_QUADRILATERALS_VC")
endif()
if (BENCHMARK)
  set(VECGEOM_DEFINITIONS
      "${VECGEOM_DEFINITIONS} -DVECGEOM_BENCHMARK")
endif()
if (NO_SPECIALIZATION)
  set(VECGEOM_DEFINITIONS
      "${VECGEOM_DEFINITIONS} -DVECGEOM_NO_SPECIALIZATION")
endif()
if (ROOT)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_ROOT")
endif()
if (USolids)
  set(VECGEOM_DEFINITIONS
      "${VECGEOM_DEFINITIONS} -DVECGEOM_USOLIDS")
endif()
if (Geant4)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_GEANT4")
endif()
if (DISTANCE_DEBUG)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_DISTANCE_DEBUG")
endif()
if (VTUNE)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_VTUNE")
endif()
if (INPLACE_TRANSFORMATIONS)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_INPLACE_TRANSFORMATIONS")
endif()
if (USE_INDEXEDNAVSTATES)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_USE_INDEXEDNAVSTATES")
endif()
if (GENERATE_MATERIALS_DB)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_GENERATE_MATERIALS_DB")
endif()

################################################################################

# see if we have CCACHE ( to avoid recompilation on branch switches )
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  message(STATUS "found ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)


# Backend configuration

if (Vc)
  if (MIC)
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${Vc_MIC_LIBRARIES})
  else()
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${Vc_LIBRARIES})
  endif()
  include_directories(AFTER SYSTEM ${Vc_INCLUDE_DIR})
  set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${Vc_INCLUDE_DIR}")

  if (Vc)
    set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/vc/Backend.cpp)
  endif()
endif()

if (Cilk)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/cilk/Backend.cpp)
endif()

if (MICVEC)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/micvec/Backend.cpp)
endif()

if (UMESIMD)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/umesimd/Backend.cpp)
endif()

if (CUDA)
  # See the setting of CUDA_SEPARABLE_COMPILATION and CUDA_USE_STATIC_CUDA_RUNTIME
  # ealier.

  # Even-though VecCore (if enabled) has already enabled CUDA, if we do not do it a second time,
  # the compilation flags are incorrectly set.  The symptoms is that cuda_add_library for vecgeomcuda
  # misinterpret the ${VECGEOM_NVCC_CONFIGS} flag and rather than splitting in configurations, pass it as is
  # to the NVCC command line ...

  find_package(CUDA REQUIRED)

  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/CudaManager.cpp source/backend/cuda/Interface.cpp)
  set(SRC_CUDA ${SRC_CUDA}
      source/CudaManager.cu source/CudaGlobalSymbols.cu)

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()

### CXXFLAGS needs to be set early, so that VECGEOM_GEANT4, VECGEOM_ROOT,
### and VECGEOM_USOLIDS are already defined when compiling USolids, or it breaks

# Pass flags to compilers
# We may have addition in compiler flags from the above included packages
# We don't want to have trailing CMAKE_CXX_FLAGS_"BUILD_TYPE" options on the command line
# but rather have it as first set of argument.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type}} ${VECGEOM_CXX_FLAGS} ${VECGEOM_DEFINITIONS} ${VECGEOM_COMPILATION_FLAGS}")
#set(VECGEOM_CXX_FLAGS ${CMAKE_CXX_FLAGS_${_build_type}})
#unset(CMAKE_CXX_FLAGS_${_build_type} CACHE)

# At this point, all changes to CXXFLAGS have been made.
# The line below is necessary to allow passing extra options
# via -DCMAKE_CXX_FLAGS or CXXFLAGS environment variable.
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")
if (CUDA)
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
endif()

if (ROOT)

# You need to tell CMake where to find the ROOT installation. This can be done in a number of ways:
#   - ROOT built with classic configure/make use the provided $ROOTSYS/etc/cmake/FindROOT.cmake
#   - ROOT built with CMake. Add in CMAKE_PREFIX_PATH the installation prefix for ROOT
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

#---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
  find_package(ROOT REQUIRED COMPONENTS Geom Graf3d)

  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
#---setup ROOT include + lib dirs
  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIRS})
  link_directories(${ROOT_LIBRARY_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${ROOT_LIBRARIES})
  set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${ROOT_INCLUDE_DIRS}")

  set(SRC_EXTERNAL ${SRC_EXTERNAL}
      source/PlacedRootVolume.cpp
      source/UnplacedRootVolume.cpp
      source/ShapeDebugger.cpp
      source/Visualizer.cpp)

endif(ROOT)

if (USolids)

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIRS OR NOT USOLIDS_LIBRARIES_DIR))
    message(STATUS "Building with shipped USolids.")
    add_subdirectory(USolids)
    include_directories(${CMAKE_SOURCE_DIR}/USolids/include)
    link_directories(${CMAKE_BINARY_DIR}/USolids)
  endif()
  if (USOLIDS_INCLUDE_DIRS)
    include_directories(${USOLIDS_INCLUDE_DIRS})
  elseif (USOLIDS_DIR)
    include_directories(${USOLIDS_DIR}/include)
  endif()
  if (USOLIDS_LIBRARIES_DIR)
    link_directories(${USOLIDS_LIBRARIES_DIR})
  elseif (USOLIDS_DIR)
    link_directories(${USOLIDS_DIR}/lib)
  endif()

  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} usolids)

endif()

if (Geant4)

  find_package(Geant4 REQUIRED)
  # need to define G4MULTITHREADED for Geant4-MT
  add_definitions(${Geant4_DEFINITIONS})
  include_directories(AFTER SYSTEM ${Geant4_INCLUDE_DIRS})
  set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${Geant4_INCLUDE_DIRS}")
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL}
      ${Geant4_LIBRARIES})
  message(STATUS "Geant4 found in: ${Geant4_INCLUDE_DIRS}")

if(ROOT)
  # We use VGM to convert (test) geometries from ROOT input to G4
  # without needing an intermediate gdml file
  # for user friendliness .. added as external project for now
  set(VGM_INSTALL "${CMAKE_BINARY_DIR}/vgminstall/")
  ExternalProject_Add(VGM
                      GIT_REPOSITORY "https://github.com/alisw/VGM.git"
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${VGM_INSTALL} -DWITH_EXAMPLES=OFF -DGeant4_DIR=${Geant4_DIR} -DROOT_DIR=${ROOT_DIR}
                                 -DCMAKE_INSTALL_LIBDIR=${VGM_INSTALL}/lib)

  # now modify includes and libs for later linking
  include_directories(${VGM_INSTALL}/include)
  link_directories(${VGM_INSTALL}/lib)
  set(VECGEOM_LIBRARIES_EXTERNAL "-L${VGM_INSTALL}/lib" BaseVGM;ClhepVGM;XmlVGM;Geant4GM;RootGM ${VECGEOM_LIBRARIES_EXTERNAL})
  set(VGM ON)
endif(ROOT)
endif(Geant4)

if (VTUNE)
  find_package(VTUNE REQUIRED)
  include_directories(AFTER SYSTEM ${VTUNE_INCLUDE_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${VTUNE_LIBRARIES} -lpthread -ldl)
endif()

################################################################################

include_directories(BEFORE ${CMAKE_SOURCE_DIR})

file(GLOB _createnuclei RELATIVE ${CMAKE_SOURCE_DIR} source/generated/CreateNuclei*.cpp)
set(VECGEOM_SRCS
  #.. moved up for faster compilation
  ${_createnuclei}
  source/UnplacedPolycone.cpp
  source/UnplacedPolyhedron.cpp
  source/PlacedPolycone.cpp
  source/PlacedPolyhedron.cpp

  source/LogicalVolume.cpp
  source/PlacedCone.cpp
  source/PlacedAssembly.cpp
  source/PlacedBox.cpp
  source/PlacedSExtru.cpp
  source/PlacedHype.cpp
  source/PlacedTube.cpp
  source/PlacedCutTube.cpp
  source/PlacedTorus2.cpp
  source/PlacedTrd.cpp
  source/PlacedGenTrap.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedScaledShape.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedVolume.cpp
  source/Planes.cpp
  source/Plane.cpp
  source/CutPlanes.cpp
  source/Quadrilaterals.cpp
  source/Rectangles.cpp
  source/Scale3D.cpp
  source/Transformation3D.cpp
  source/UnplacedAssembly.cpp
  source/UnplacedCone.cpp
  source/UnplacedBox.cpp
  source/UnplacedSExtruVolume.cpp
  source/UnplacedTube.cpp
  source/UnplacedCutTube.cpp
  source/UnplacedHype.cpp
  source/UnplacedTorus2.cpp
  source/UnplacedTrd.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedScaledShape.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedGenTrap.cpp
  source/UnplacedVolume.cpp
  source/NavigationState.cpp
  source/SimpleNavigator.cpp

  source/UnplacedOrb.cpp
  source/PlacedOrb.cpp
  source/UnplacedSphere.cpp
  source/PlacedSphere.cpp
  source/UnplacedBooleanVolume.cpp
  source/PlacedBooleanVolume.cpp
  source/Wedge.cpp
  source/ABBoxManager.cpp
  source/HybridManager2.cpp
  source/ABBoxNavigator.cpp

  source/Material.cpp
  source/Medium.cpp
  source/Nucleus.cpp
  source/MessageLogger.cpp

  services/NavigationSpecializer.cpp

  source/ResultComparator.cpp
 )

if (CUDA)
  set(VECGEOM_CUDA_SRCS
    source/RNG.cpp
    source/AOS3D.cpp
    source/SOA3D.cpp
    source/Vector.cpp
  )
endif()

if (Geant4)
  set(VECGEOM_GEANT4_SRCS
    source/G4GeoManager.cpp
  )
endif()

if (ROOT)
  set(VECGEOM_ROOT_SRCS
    source/RootGeoManager.cpp
  )
endif()

if (USolids)
  set(VECGEOM_USOLIDS_SRCS
    source/USolidsInterfaceHelper.cpp
  )
endif()

set(VECGEOM_SRCS
  ${VECGEOM_SRCS}
  ${VECGEOM_CUDA_SRCS}
  ${VECGEOM_GEANT4_SRCS}
  ${VECGEOM_ROOT_SRCS}
  ${VECGEOM_USOLIDS_SRCS}
)

foreach(SRC ${VECGEOM_SRCS})
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/${SRC})
endforeach()

# file prefixes that won't be compiled for CUDA
set(NOT_FOR_CUDA
   ABBoxManager
   ABBoxNavigator
   HybridManager2
   Medium
 # Material
   G4GeoManager
   RootGeoManager
   NavigationSpecializer
   ResultComparator
)


# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.

if (CUDA)

  foreach(SRC_FILE ${SRC_CPP})

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

    list(FIND NOT_FOR_CUDA ${SRC_FILENAME} _index)
    if(${_index} EQUAL -1)
      ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
        COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE}
            ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
        DEPENDS ${SRC_FILE}
      )

      set(
        SRC_CUDA ${SRC_CUDA}
        ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      )
    else()
      message(STATUS "EXCLUDING ${SRC_FILENAME} FROM CUDA")
    endif()
  endforeach()

endif()

# Add files that won't be compiled for CUDA

#if (USOLIDS)
  set(SRC_CPP ${SRC_CPP}
    test/shape_tester/ShapeTester.cpp
    test/shape_tester/ConventionChecker.cpp
  )
#endif()
if (BENCHMARK)
  set(SRC_CPP ${SRC_CPP}
    source/benchmarking/BenchmarkResult.cpp
    source/benchmarking/Benchmarker.cpp
    source/benchmarking/VolumePointers.cpp
  )
  set(SRC_CUDA ${SRC_CUDA}
    source/benchmarking/Benchmarker.cu
    source/benchmarking/NavigationBenchmarker.cu
  )
  add_subdirectory(test/benchmark)
endif()
set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL}
  source/GeoManager.cpp
  source/CppExporter.cpp
  source/benchmarking/NavigationBenchmarker.cpp
)

# some CUDA kernel code in "userspace" and test
if (CUDA)
   set(SRC_USER_CUDA_CPP
       userexamples/src/TestNavigationStatePool.cu
   )
  set(SRC_CUDA ${SRC_CUDA}
        test/cuda/MapTest.cu
        test/cuda/MapTestClass.cu
   )
 endif()


# if static analysis
if (STATIC_ANALYSIS)
  # enable export compile commands ... to be used by clang-tidy
  SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  message(STATUS "including VecGeom static analysis code")
  add_subdirectory(test/static_analysis)
endif()
#

################################################################################

# Build libraries

add_library(vecgeom ${SRC_CPP})
target_link_libraries(vecgeom ${VECGEOM_LIBRARIES}
                      ${VECGEOM_LIBRARIES_EXTERNAL})
if(VGM)
add_dependencies(vecgeom VGM)
endif(VGM)
set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeom)

if (BUILD_SHARED_LIBS)
    set(VECGEOM_LIBNAME "libvecgeom${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    set(VECGEOM_LIBNAME "libvecgeom${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

if (NOT APPLE)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} rt)
endif()

# build the CUDA version of the library
if (CUDA)
  cuda_add_library(
    vecgeomcuda
    ${SRC_CUDA}
    SHARED
    OPTIONS ${CUDA_ARCH}
    # For debug information we might need
    #    -O0 -g -G -lineinfo
    ${VECGEOM_NVCC_CONFIGS}
  )
  target_link_libraries(vecgeomcuda ${VECGEOM_LIBRARIES}
                        ${VECGEOM_LIBRARIES_EXTERNAL})
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeomcuda)

endif()


# build the CUDA user-code library
if (CUDA)
  cuda_add_library(
    cudauserlib
    ${SRC_USER_CUDA_CPP}
    SHARED
    OPTIONS ${CUDA_ARCH}
  )
  # Note: the repeat below is due the author of  cuda_add_library_depend
  # not knowing how to automatically go from the target to the real
  # file in add_custom_command
  cuda_add_library_depend(cudauserlib vecgeomcuda_static libvecgeomcuda_static.a)
  target_link_libraries(cudauserlib ${VECGEOM_LIBRARIES} ${VECGEOM_LIBRARIES_EXTERNAL})
  set(USERKERNELLIB cudauserlib)
endif()
################################################################################

# Define executables

set(TEST_EXECUTABLES_CORE
  ${CMAKE_SOURCE_DIR}/test/core/SafetyEstimatorTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/ContainerTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/create_geometry.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/testVectorSafety.cpp
  ${CMAKE_SOURCE_DIR}/test/core/PlanesTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/QuadrilateralTest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/SOATest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/Transformation3DTest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/boolminustest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/boolminustest2.cpp
  ${CMAKE_SOURCE_DIR}/test/core/PhiWedgeTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/ThetaConeTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestConvexity.cpp
  ${CMAKE_SOURCE_DIR}/test/core/BooleanConvexityTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestVecGeomPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestSExtru.cpp
# to be enabled when running cleanly
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBooleans.cpp
  ${CMAKE_SOURCE_DIR}/test/core/AssemblyTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestNavigationState.cpp
#  ${CMAKE_SOURCE_DIR}/services/CompNavStatePools.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestMaskedAssign.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestVector.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestMap.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestHybridBVH.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestEarlyReturns.cpp
)


set(TEST_EXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/core/TestNavigationStatePool.cpp
  ${CMAKE_SOURCE_DIR}/test/root/root_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/E03Test.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportFromRootFileTest.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportTGeoPgon.cpp
  ${CMAKE_SOURCE_DIR}/test/root/TestExportToROOT.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ExitingOrEntering.cpp
  ${CMAKE_SOURCE_DIR}/test/core/BoxBoxIntersectionTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/SplittedABBox.cpp
  ${CMAKE_SOURCE_DIR}/test/root/AssemblyExample.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestRegions.cpp
)

set(TEST_EXECUTABLES_CORE
  ${TEST_EXECUTABLES_CORE}
  ${CMAKE_SOURCE_DIR}/test/core/BitSetTest.cpp
)
# higher level benchmarks or executables
set(TEST_EXECUTABLES_ROOT
  ${TEST_EXECUTABLES_ROOT}
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/LocatePointsBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/SafetyKernelBenchmarker.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/NavigationKernelBenchmarker.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/NavigationBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/TraceTrack.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/XRayBenchmarkFromROOTFile.cpp
  ${CMAKE_SOURCE_DIR}/services/NavigationSpecializerTest.cpp
  ${CMAKE_SOURCE_DIR}/services/LibraryGenerator.cpp
)

# separate list for ROOT UNIT tests
set(TEST_UNITTESTEXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/E03Test.cpp
  ${CMAKE_SOURCE_DIR}/test/root/TestExportToROOT.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportTGeoPgon.cpp
)

if(BENCHMARK)
  set(TEST_EXECUTABLES_ROOT
  ${TEST_EXECUTABLES_ROOT}
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile.cpp
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile_WithVisualization.cpp
  ${CMAKE_SOURCE_DIR}/test/root/CompareDistances.cpp
  ${CMAKE_SOURCE_DIR}/test/root/GenerateSurfacePoints.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shapeDebug.cpp
)
endif()

set(TEST_EXECUTABLES_VISUALIZATION
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizePolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizePolyhedron.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeParboloid.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeCone.cpp
 # ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTorus.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTube.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeScaled.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/DebugTube.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/DebugPolyhedron.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeSphere.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeGenTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeHype.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeParallelepiped.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeCutTube.cpp
)

set(TEST_EXECUTABLES_USOLIDS
  ${CMAKE_SOURCE_DIR}/test/usolids/CompatibilityTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBox.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestCons.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestGenTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTube.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestHype.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestParallelepiped.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolyhedra.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTet.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestSphere.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBoxlikeTrapezoid.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTorus2.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/usolidTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestParaboloid.cpp
)

set(TEST_EXECUTABLES_CUDA
  # Files go here
  ${CMAKE_SOURCE_DIR}/test/cuda/MapTest.cpp
  ${CMAKE_SOURCE_DIR}/test/cuda/MapTestClass.cpp
)
set(TEST_EXECUTABLES_SHAPETESTER
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testBox.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testSExtru.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testSphere.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testCone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testHype.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrapezoid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParaboloid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testGenTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParallelepiped.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/convention_testTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testPolyhedron.cpp 
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTorus.cpp 
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testCutTube.cpp 
)

if(ROOT) # add shape testers that depend on ROOT
  set(TEST_EXECUTABLES_SHAPETESTER
  ${TEST_EXECUTABLES_SHAPETESTER}
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testFromROOTFile.cpp )
endif()


# Build executables


set(VECGEOM_EXECUTABLES)

macro(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_link_libraries(${TARGET_NAME} ${VECGEOM_LIBRARIES}
                          ${VECGEOM_LIBRARIES_EXTERNAL}
                          ${USERKERNELLIB})
    set(VECGEOM_EXECUTABLES ${VECGEOM_EXECUTABLES} ${TARGET_NAME})
    if (MIC)
      get_target_property(current_property ${TARGET_NAME} LINK_FLAGS)
      if(NOT current_property)
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-B/usr/linux-k1om-4.7/linux-k1om/usr/lib64/k1om-mpss-linux/4.7.4 -B/usr/linux-k1om-4.7/linux-k1om/lib64/4.7.4")
      else()
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${current_property} -B/usr/linux-k1om-4.7/linux-k1om/usr/lib64/k1om-mpss-linux/4.7.4 -B/usr/linux-k1om-4.7/linux-k1om/lib64/4.7.4")
      endif()
    endif()
  endforeach()
endmacro()

function(add_to_ctest EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    if (CTEST)
      get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
      add_test(${TARGET_NAME} ${TARGET_NAME})
    endif()
  endforeach()
endfunction()

if (CTEST)
  build_executables("${TEST_EXECUTABLES_CORE}")
  add_to_ctest("${TEST_EXECUTABLES_CORE}")
  set_tests_properties(TestSExtru PROPERTIES TIMEOUT 30)
endif()

if (BENCHMARK)
  build_executables("${TEST_EXECUTABLES_BENCHMARK}")
  add_to_ctest("${CTESTS_BENCHMARK}")
  build_executables("${TEST_EXECUTABLES_SHAPETESTER}")
endif()

if (ROOT)
  build_executables("${TEST_EXECUTABLES_ROOT}")
  build_executables("${TEST_EXECUTABLES_VISUALIZATION}")
  add_to_ctest("${TEST_UNITTESTEXECUTABLES_ROOT}")

  if (VALIDATION)
    macro(add_cmsshapevalidation_test TESTNAME SHAPEFILE)
      add_test(NAME ${TESTNAME} COMMAND bash -c "${CMAKE_SOURCE_DIR}/test/scripts/RunRandomValidation ${CMAKE_BINARY_DIR}/BenchmarkShapeFromROOTFile  ${CMAKE_SOURCE_DIR}/test/cmstestdata/cms2015.root  ${CMAKE_SOURCE_DIR}/test/cmstestdata/${SHAPEFILE}" )
    endmacro()

    # Adding various shapes tests for nightlies
    add_cmsshapevalidation_test( tubevalidation cmstubes.txt )
    add_cmsshapevalidation_test( trapvalidation cmstraps.txt )
    add_cmsshapevalidation_test( polyconevalidation cmspolycones.txt )
    add_cmsshapevalidation_test( polyhedravalidation cmspolyhedra.txt )
    add_cmsshapevalidation_test( conevalidation cmscones.txt )
    add_cmsshapevalidation_test( boxvalidation cmsboxes.txt )
    #taken out due to problems: add_cmsshapevalidation_test( booleanvalidation cmsbooleans.txt )
    #taken out due to Issue-133: add_cmsshapevalidation_test( torusvalidation cmstori.txt )
  endif()
else()
  if (VALIDATION)
    message(WARNING "Validation tests depend on ROOT being enabled")
  endif()
endif()

if (USolids)
  build_executables("${TEST_EXECUTABLES_USOLIDS}")
  #add_to_ctest("${TEST_EXECUTABLES_USOLIDS}")
  #add_to_ctest("${TEST_EXECUTABLES_SHAPETESTER}")

  # add unit tests
  add_test(NAME USolidsBox COMMAND TestBox --usolids)
  add_test(NAME VecGeomBox COMMAND TestBox --vecgeom)
  add_test(NAME USolidsTrap COMMAND TestTrap --usolids)
  add_test(NAME VecGeomTrap COMMAND TestTrap --vecgeom)
  add_test(NAME USolidsTube COMMAND TestTube --usolids)
  add_test(NAME VecGeomTube COMMAND TestTube --vecgeom)
  add_test(NAME USolidsCone COMMAND TestCons --usolids)
  add_test(NAME VecGeomCone COMMAND TestCons --vecgeom)
  add_test(NAME USolidsTrd COMMAND TestTrd --usolids)
  add_test(NAME VecGeomTrd COMMAND TestTrd --vecgeom)
  add_test(NAME USolidsOrb COMMAND TestOrb --usolids)
  add_test(NAME VecGeomOrb COMMAND TestOrb --vecgeom)
  add_test(NAME VecGeomPolyhedra COMMAND TestPolyhedra --vecgeom)
#  add_test(NAME USolidsPolyhedra COMMAND TestPolyhedra --usolids)
#  add_test(NAME USolidsPolycone COMMAND TestPolycone --usolids)
  add_test(NAME USolidsParallelepiped COMMAND TestParallelepiped --usolids)
  add_test(NAME VecGeomParallelepiped COMMAND TestParallelepiped --vecgeom)
  add_test(NAME USolidsHype COMMAND TestHype --usolids)
#  add_test(NAME VecGeomHype COMMAND TestHype --vecgeom)
  add_test(NAME USolidsGenTrap COMMAND TestGenTrap --usolids)
  add_test(NAME VecGeomGenTrap COMMAND TestGenTrap --vecgeom)
endif()

if (CUDA)
  build_executables("${TEST_EXECUTABLES_CUDA}")
  #  add_to_ctest("${TEST_EXECUTABLES_CUDA}")
endif()

#---------------------------------------------------------------------------
# Include creation and submission to dashboard cdash.cern.ch

include(CTest)

#---------------------------------------------------------------------------
# Set data to be dowloaded

if(DATA_DOWNLOAD)
  message(STATUS "Downloading data files")

  # define a function downloading from a URL into a local file LOCALFILE
  function(FILE_DOWNLOAD FILE_URL LOCALFILE )
   if(APPLE)
       execute_process(COMMAND curl -o ${LOCALFILE} ${FILE_URL})
   else()
      execute_process(COMMAND wget -q ${FILE_URL} -O ${LOCALFILE})
   endif()
  endfunction(FILE_DOWNLOAD)
  # end of function FILE DOWNLOAD

  # define a function checking md5 hashes
  # result is stored in MD5MATCHES ( 1 == true, 0 == false )
  function(CHECKMD5 FILETOCHECK EXPECTEDMD5HASH MD5MATCHES)
      if(APPLE)
          execute_process(COMMAND md5 ${FILETOCHECK} OUTPUT_VARIABLE MD5SUM)
          string(LENGTH ${MD5SUM} MD5LENGTH)
          MATH(EXPR START "${MD5LENGTH} - 33")
          string(SUBSTRING ${MD5SUM} ${START} 32 MD5SUM)
      else()
          execute_process(COMMAND md5sum ${FILETOCHECK} OUTPUT_VARIABLE MD5SUM)
          string(SUBSTRING ${MD5SUM} 0 32 MD5SUM)
      endif()
      if(MD5SUM STREQUAL EXPECTEDMD5HASH)
        set(${MD5MATCHES} 1 PARENT_SCOPE)
      else()
        set(${MD5MATCHES} 0 PARENT_SCOPE)
      endif()
  endfunction(CHECKMD5)

  # actual function for managing the download
  function(DOWNLOAD_IF_NOT_INSTALLED FILE_URL LOCALFILE TARGETPATH MD5HASH )
    find_file(FOUNDFILE ${LOCALFILE} ${TARGETPATH} )
    if(FOUNDFILE STREQUAL "FOUNDFILE-NOTFOUND")
        # set need download
        message(STATUS "need download of ${LOCALFILE} since not found")
        set( NEEDTODOWNLOAD 1 )
    else()
        # check md5
        message(STATUS "found existing file ${LOCALFILE}")
        CHECKMD5( ${FOUNDFILE} ${MD5HASH} MD5CORRECT )
        if( ${MD5CORRECT} STREQUAL "1" )
            # do not set download flag
            set( NEEDTODOWNLOAD 0 )
        else( )
            # set need download
            message(STATUS "hash ${MD5HASH} not correct for file ${FOUNDFILE} ${MD5CORRECT}" )
            set( NEEDTODOWNLOAD 1 )
        endif( )
    endif()

    if( ${NEEDTODOWNLOAD} STREQUAL 1 )
        message(STATUS " downloading ... ")
        set(DOWNLOADLOCATION "${TARGETPATH}/${LOCALFILE}")
        FILE_DOWNLOAD( ${FILE_URL} ${DOWNLOADLOCATION} )
    else()
        message(STATUS " doing nothing ... ")
    endif()
    # in principle have to check now if download succeeded and has right MD5
    # TOBEDONE

    # this is annoying but we have to clear FOUNDFILE SINCE THIS IS TREATED LIKE A STATIC VARIABLE
    unset(FOUNDFILE CACHE)
  endfunction(DOWNLOAD_IF_NOT_INSTALLED)

  DOWNLOAD_IF_NOT_INSTALLED( "http://mnovak.web.cern.ch/mnovak/data/cms2015.root" "cms2015.root"
                             "${CMAKE_SOURCE_DIR}/test/cmstestdata/" "fb3ede867532b372c5e6f7138d00c07e" )
  DOWNLOAD_IF_NOT_INSTALLED( "http://mnovak.web.cern.ch/mnovak/data/ExN03.root" "ExN03.root"
                             "${CMAKE_BINARY_DIR}"                   "b6b0cfdd5035117171bfe1b7f1f40c3f" )
  # add more data to download here
  #
endif()


################################################################################

# Installation
file(RELATIVE_PATH INSTALL_INCLUDE_DIR_RELATIVE
     "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}/.")
file(RELATIVE_PATH INSTALL_LIB_DIR_RELATIVE
     "${INSTALL_CMAKE_DIR}" "${INSTALL_LIB_DIR}/.")
# Build
set(CONF_TYPE "build")
set(CONF_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")
set(CONF_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL}
    "${CONF_LIBRARY_DIR}/${VECGEOM_LIBNAME}")
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES}
      "${CONF_LIBRARY_DIR}/USolids/libusolids.a")
endif()
if (CUDA)
   set(CONF_LIBRARIES ${CONF_LIBRARIES}  "${CONF_LIBRARY_DIR}/libvecgeomcuda.so")
   set(CONF_CUDA_STATIC_LIBRARY "${CONF_LIBRARY_DIR}/libvecgeomcuda_static.a")
endif()
configure_file(VecGeomConfig.cmake.in
               "${PROJECT_BINARY_DIR}/VecGeomConfig.cmake" @ONLY)
# Installation
set(CONF_TYPE "install")
set(CONF_INCLUDE_DIR "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}")
set(CONF_LIBRARY_DIR "\${THIS_DIR}/${INSTALL_LIB_DIR_RELATIVE}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL} ${CONF_LIBRARY_DIR}/${VECGEOM_LIBNAME})
if (CUDA)
   set(CONF_LIBRARIES ${CONF_LIBRARIES}  "${CONF_LIBRARY_DIR}/libvecgeomcuda.so")
   set(CONF_CUDA_STATIC_LIBRARY "${CONF_LIBRARY_DIR}/libvecgeomcuda_static.a")
endif()
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES} "${CONF_LIBRARY_DIR}/libusolids.a")
  configure_file(VecGeomConfig.cmake.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/USolidsConfig.cmake" @ONLY)
endif()
configure_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake" @ONLY)

# Install the VecGeomConfig.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
if (USolids)
  install(FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/USolidsConfig.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}/../USolids" COMPONENT dev)
endif()

# Install headers and libraries
foreach(FOLDER base benchmarking management navigation volumes materials backend)
  install(DIRECTORY ${FOLDER} DESTINATION ${INSTALL_INCLUDE_DIR})
endforeach()

string(REPLACE "." ";" SYSTEM_VERSION_LIST ${CMAKE_SYSTEM_VERSION})
list(GET SYSTEM_VERSION_LIST 0 SYSTEM_VERSION_MAJOR)
list(GET SYSTEM_VERSION_LIST 1 SYSTEM_VERSION_MINOR)
list(GET SYSTEM_VERSION_LIST 2 SYSTEM_VERSION_PATCH)
if(APPLE AND ${SYSTEM_VERSION_MAJOR} LESS 14)
   install(FILES ${PROJECT_BINARY_DIR}/${VECGEOM_LIBNAME} DESTINATION ${INSTALL_LIB_DIR})
else ()
   install(TARGETS vecgeom DESTINATION ${INSTALL_LIB_DIR})
endif ()
if (CUDA)
  install(TARGETS vecgeomcuda DESTINATION ${INSTALL_LIB_DIR})
  install(TARGETS vecgeomcuda_static DESTINATION ${INSTALL_LIB_DIR})
endif()
