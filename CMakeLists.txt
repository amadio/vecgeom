#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.1.0)

include(cmake/modules/RecordCmdLine.cmake)
include(ExternalProject)

project(VecGeom)

set(VECGEOM_VERSION 1.1.4)
set(VECGEOM ON)

enable_language(CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
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

option(BUILTIN_VECCORE "Build VecCore and its dependencies from source" OFF)
option(CUDA "Enable compilation for CUDA." OFF)
option(CUDA_VOLUME_SPECIALIZATION "Use specialized volumes for CUDA." OFF)
option(ROOT "Include ROOT." OFF)
option(GEANT4 "Include Geant4." OFF)
option(BENCHMARK "Enable performance comparisons." OFF)
option(NO_SPECIALIZATION "Disable specialization of volumes." ON)
option(COVERAGE_TESTING "Enable coverage testing flags." OFF)
option(CTEST "Enable CTest when building." ON)
option(VALIDATION "Enable validation tests from CMS geometry." OFF)
option(PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)
option(QUADRILATERAL_ACCELERATION "Enable SIMD vectorization when looping over quadrilaterals (in Polyhedron)." ON)
option(DISTANCE_DEBUG "Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes" OFF)
option(INPLACE_TRANSFORMATIONS "Put transformation as members rather than pointers into PlacedVolume objects" ON)
option(USE_INDEXEDNAVSTATES "Use indices rather than volume pointers in NavigationState objects" ON)
option(DATA_DOWNLOAD "Enable downloading of data for tests" OFF)
option(STATIC_ANALYSIS "enable static analysis on VecGeom" OFF)
option(GDML "Enable GDML persistency. Requres Xerces-C" OFF)
option(EMBREE "Enable Intel Embree" OFF)
option(USE_CACHED_TRANSFORMATIONS "Use cached transformations in navigation states" OFF)

option(FAST_MATH "Enable the -ffast-math compiler option in Release builds" OFF)

set(VECGEOM_ARCH native sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 mic mic_avx512)

enum_option(VECGEOM_VECTOR
      DOC "Vector instruction set to be used"
      TYPE STRING
      VALUES ${VECGEOM_ARCH}
      CASE_INSENSITIVE
    )

if ("${BACKEND}" MATCHES "vc|VC")
  set(BACKEND "Vc")
endif()
if ("${BACKEND}" MATCHES "scalar|SCALAR")
  set(BACKEND "Scalar")
endif()

set(VECGEOM_BACKEND Scalar UMESIMD Vc)

enum_option(BACKEND
            DOC "Backend infrastructure to be used"
            TYPE STRING
            VALUES ${VECGEOM_BACKEND}
            CASE_SENSITIVE
           )

if (CTEST)
  message(STATUS "Testing with CTest enabled.")
  enable_testing()
endif()

option(VTUNE "Enable the use of profiler Vtune" OFF)

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries.")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files.")
set(INSTALL_CMAKE_DIR lib/cmake/VecGeom CACHE PATH
    "Installation directory for CMake files.")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

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

if (CUDA)
  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} "CUDA")
  set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Compile CUDA objects with separable compilation enabled.")
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available")
endif()

# Determine backend

set(${BACKEND} True)
string(TOUPPER "${BACKEND}" _BACKEND_UP)
string(TOLOWER "${BACKEND}" _BACKEND_LO)

if("${BACKEND}" MATCHES "Scalar")
  set(VECGEOM_SCALAR True)
  message(STATUS "Configuring with Scalar backend")
elseif ("${BACKEND}" MATCHES "Vc|UMESIMD")
  set(VECGEOM_${_BACKEND_UP} True)
  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} ${BACKEND})
  message(STATUS "Configuring with ${BACKEND} backend")
else()
  message(FATAL_ERROR "Unknown backend: '${BACKEND}'. Known values are: Scalar, UMESIMD, Vc.")
endif()


# Which version of VecCore do we need.
set(VecCore_VERSION "0.5.2")

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Let's see if we can find VecCore
if (NOT BUILTIN_VECCORE)
  # Find VecCore with selected components turned on (CUDA and backend)
  find_package(VecCore ${VecCore_VERSION} QUIET COMPONENTS ${VecCore_COMPONENTS})
  if (NOT VecCore_FOUND)
    set(BUILTIN_VECCORE ON)
  else()
    message("-- Found VecCore at: ${VecCore_ROOT_DIR}")
  endif()
endif()
if (BUILTIN_VECCORE)
    #  This function is used to force a build on a dependant project at cmake configuration phase.
    #
    function (build_external_project target globpattern ) #FOLLOWING ARGUMENTS are the ARGS of ExternalProject_Add

      set(trigger_build_topdir ${CMAKE_BINARY_DIR}/buildExternals/${target})
      set(trigger_src_dir ${CMAKE_BINARY_DIR}/buildExternals/${target}/trigger_src)
      set(trigger_build_dir ${CMAKE_BINARY_DIR}/buildExternals/${target}/build)

      #mktemp dir in build tree
      file(MAKE_DIRECTORY ${trigger_build_dir} ${trigger_src_dir} ${trigger_build_dir})

      #generate false dependency project
      set(CMAKE_LIST_CONTENT "
          cmake_minimum_required(VERSION 2.8)

          include(ExternalProject)
          ExternalProject_add(${target}
                  ${ARGN}
                  )

          add_custom_target(trigger_${target})
          add_dependencies(trigger_${target} ${target})
      ")

      file(WRITE ${trigger_src_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")

      execute_process(COMMAND ${CMAKE_COMMAND} ${trigger_src_dir} -G ${CMAKE_GENERATOR}
        WORKING_DIRECTORY ${trigger_build_dir}
      )
      execute_process(COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${trigger_build_dir}
      )

      FILE(GLOB VecCoreHeaders string(${globpattern}))

      add_custom_target(${target})
      add_custom_command(
              OUTPUT ${VecCoreHeaders}
              COMMAND ${CMAKE_COMMAND} --build .
              WORKING_DIRECTORY ${trigger_build_dir}
      )
    endfunction()


    set(VecCore_PROJECT "VecCore-${VecCore_VERSION}")
    set(VecCore_SRC_URI "http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources")
    set(VecCore_SRC_MD5 "caca009e8ad655fc449ec593ce9764c1")
    set(VecCore_DESTDIR "${CMAKE_BINARY_DIR}/installExternals/${VecCore_PROJECT}")
    set(VecCore_ROOTDIR "${VecCore_DESTDIR}/${CMAKE_INSTALL_PREFIX}")
    set(VecCore_SRC_TAG "v${VecCore_VERSION}")

    if (Vc)
      if (MIC)
        set(KNC_SUFFIX "_MIC")
      endif()
      set(Vc_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}Vc${KNC_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(Vc_LIBRARY ${VecCore_ROOTDIR}/lib${LIB_SUFFIX}/${Vc_LIBNAME})
    endif()


    if (DEFINED UMESIMD_DIR AND NOT DEFINED UMESIMD_ROOT)
      set(UMESIMD_ROOT ${UMESIMD_DIR})
    endif()

    build_external_project(${VecCore_PROJECT}
      "${VecCore_ROOTDIR}/include/VecCore/*.h;${VecCore_ROOTDIR}/include/VecCore/VecCore;${VecCore_ROOTDIR}/include/VecCore/Backend/*.h"
      "URL \"${VecCore_SRC_URI}/VecCore-${VecCore_VERSION}.tar.gz\"
      URL_MD5 ${VecCore_SRC_MD5}
      #GIT_REPOSITORY \"${VecCore_SRC_URI}\"
      #GIT_TAG \"${VecCore_SRC_TAG}\"

      PREFIX external
      STAMP_DIR external/stamp
      BINARY_DIR external/build
      BUILD_IN_SOURCE 0
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      CMAKE_ARGS -G \"${CMAKE_GENERATOR}\"
                 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_TESTING=OFF
                 -DCUDA=${CUDA} -DVC=${Vc} -DUMESIMD=${UMESIMD}
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               \"-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}\"
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               \"-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}\"
                 -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               \"-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\"
               \"-DVc_DIR=${Vc_DIR}\"
               \"-DUMESIMD_ROOT=${UMESIMD_ROOT}\"
      INSTALL_COMMAND env DESTDIR=${VecCore_DESTDIR} ${CMAKE_COMMAND} --build . --target install"
    )

    add_custom_target(VecCore)
    add_dependencies(VecCore ${VecCore_PROJECT})

    install(DIRECTORY ${VecCore_ROOTDIR}/ DESTINATION "." )

    # Find VecCore with selected components turned on (CUDA and backend)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${VecCore_DESTDIR}/${CMAKE_INSTALL_PREFIX})

    # Make sure to look again for a version of VecCore so that we can make sure to find
    # the one we just build.
    unset(VecCore_DIR CACHE)

endif()

# Find VecCore with selected components turned on (CUDA and backend)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_COMPONENTS})

message("-- Found VecCore at: ${VecCore_ROOT_DIR}")

include_directories(${VecCore_INCLUDE_DIRS})
set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} ${VecCore_DEFINITIONS}")
set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_${_BACKEND_UP}")
set(VECGEOM_EXTERNAL_INCLUDES "${VECGEOM_EXTERNAL_INCLUDES};${VecCore_INCLUDE_DIRS}")
set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${VecCore_LIBRARIES})

################################################################################

# Set Default compiler flags for each build type

string(TOLOWER ${VECGEOM_VECTOR} _arch_lo)
string(TOUPPER ${VECGEOM_VECTOR} _arch_up)

if (GNU)
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -Wall -fPIC")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ggdb -O0")

  else()
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -finline-limit=10000000 -ftree-vectorize")
    if (FAST_MATH)
        set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ffast-math")
    endif()
  endif()
  set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -fabi-version=0")
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -fabi-version=0")
  if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7")
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -faligned-new")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -faligned-new")
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
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ftree-vectorize")
    if (FAST_MATH)
      set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -ffast-math")
    endif()
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
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64le")
    set(VECGEOM_CXX_FLAGS "${VECGEOM_CXX_FLAGS} -mcpu=${_arch_lo} -mtune=${_arch_lo}")
    set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -mcpu=${_arch_lo} -mtune=${_arch_lo}")
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
  if (NOT (CMAKE_VERSION LESS 3.3))
     cmake_policy(SET CMP0058 OLD)
  endif()
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
if (EMBREE)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_EMBREE")
endif()
if (USE_CACHED_TRANSFORMATIONS)
  set(VECGEOM_DEFINITIONS "${VECGEOM_DEFINITIONS} -DVECGEOM_CACHED_TRANS")
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

################################################################################

# see if we have CCACHE ( to avoid recompilation on branch switches )
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  message(STATUS "found ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

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

### Set CXXFLAGS early, so that VECGEOM_GEANT4 and VECGEOM_ROOT are already defined when compiling benchmarks, or it breaks

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

# Intel Embree
if (EMBREE)
  find_package(embree 3.1 REQUIRED)
  include_directories(${EMBREE_INCLUDE_DIRS})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${EMBREE_LIBRARIES})
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
  source/NavigationState.cpp

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
 
  source/MessageLogger.cpp

  services/NavigationSpecializer.cpp

  source/ResultComparator.cpp
  source/ReducedPolycone.cpp
  source/Utils3D.cpp
  source/SolidMesh.cpp
 )
 
if (EMBREE)
set(VECGEOM_SRCS
    ${VECGEOM_SRCS}
    source/EmbreeManager.cpp  
)
endif()


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

set(VECGEOM_SRCS
  ${VECGEOM_SRCS}
  ${VECGEOM_CUDA_SRCS}
  ${VECGEOM_GEANT4_SRCS}
  ${VECGEOM_ROOT_SRCS}
)

foreach(SRC ${VECGEOM_SRCS})
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/${SRC})
endforeach()

# file prefixes that won't be compiled for CUDA
set(NOT_FOR_CUDA
   ABBoxManager
   HybridManager2
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
   SolidMesh
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

set(SRC_CPP ${SRC_CPP}
  test/shape_tester/ShapeTester.cpp
  test/shape_tester/ConventionChecker.cpp
)

if (BENCHMARK)
  set(SRC_CPP ${SRC_CPP}
    source/benchmarking/BenchmarkResult.cpp
    source/benchmarking/Benchmarker.cpp
    source/benchmarking/VolumePointers.cpp
  )
  set(SRC_CUDA ${SRC_CUDA}
    source/benchmarking/Benchmarker.cu
   # source/benchmarking/NavigationBenchmarker.cu
  )
  add_subdirectory(test/benchmark)
endif()
set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL}
  source/GeoManager.cpp
  source/CppExporter.cpp
 # source/benchmarking/NavigationBenchmarker.cpp
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

if (NOT APPLE)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} rt)
endif()

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
  target_link_libraries(vecgeomcuda ${VECGEOM_LIBRARIES})
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeomcuda)
endif()

# Make sure we build VecCore before building VecGeom
if (BUILTIN_VECCORE)
  add_dependencies(vecgeom VecCore)
  if (CUDA)
    add_dependencies(vecgeomcuda VecCore)
  endif()
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
  target_link_libraries(cudauserlib ${VECGEOM_LIBRARIES})
  set(USERKERNELLIB cudauserlib)
endif()

if(GDML AND CUDA)
  message(STATUS "GDML is not supported for CUDA builds. Turning GDML OFF")
  set(GDML OFF)
endif()
# build the libraries for GDML persistency
if(GDML)
  option(GDMLDEBUG "Enable additional debug information in GDML module" OFF)
  add_subdirectory(persistency/gdml)
endif() # GDML

################################################################################

# Define executables

set(TEST_EXECUTABLES_CORE
  ${CMAKE_SOURCE_DIR}/test/core/SafetyEstimatorTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/ContainerTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/create_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/core/BitSetTest.cpp
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
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestEstimateSurfaceArea.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestVecGeomPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestSExtru.cpp
# to be enabled when running cleanly
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBooleans.cpp
  ${CMAKE_SOURCE_DIR}/test/core/AssemblyTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestMakeInstance.cpp
#  ${CMAKE_SOURCE_DIR}/services/CompNavStatePools.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestMaskedAssign.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestVector.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestMap.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestHybridBVH.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestEarlyReturns.cpp
  ${CMAKE_SOURCE_DIR}/test/core/CreateTessels.cpp
  ${CMAKE_SOURCE_DIR}/test/core/CreateExtruded.cpp
)
if (USE_INDEXEDNAVSTATES)
   set(TEST_EXECUTABLES_CORE
      ${TEST_EXECUTABLES_CORE}
      ${CMAKE_SOURCE_DIR}/test/core/TestNavigationState.cpp
   )
endif()


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
if (EMBREE)
 set(TEST_EXECUTABLES_ROOT 
      ${TEST_EXECUTABLES_ROOT}
      ${CMAKE_SOURCE_DIR}/test/core/EmbreeManagerTest.cpp
    )
endif()

# higher level benchmarks or executables
set(TEST_EXECUTABLES_ROOT
  ${TEST_EXECUTABLES_ROOT}
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/LocatePointsBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/SafetyKernelBenchmarker.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/NavigationKernelBenchmarker.cpp
  # ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/NavigationBenchmark.cpp
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
  #${CMAKE_SOURCE_DIR}/test/visualization/VisualizeExtruded.cpp
)

if(NOT CUDA)
set(TEST_EXECUTABLES_VISUALIZATION
  ${TEST_EXECUTABLES_VISUALIZATION}
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeMultiUnion.cpp)
endif()

set(TEST_EXECUTABLES_SHAPES
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBox.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestCons.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestGenTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTube.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestEllipticalTube.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestEllipticalCone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestEllipsoid.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestCoaxialCones.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestGenericPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestHype.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestParallelepiped.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolyhedra.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTet.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestSphere.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTorus2.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestParaboloid.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestReducedPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestUtils3D.cpp
)

if(NOT CUDA)
set(TEST_EXECUTABLES_SHAPES
  ${TEST_EXECUTABLES_SHAPES}
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTessellated.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestMesh.cpp)
endif()

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
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testEllipticalCone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testEllipticalTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testEllipsoid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testHype.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrapezoid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTet.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParaboloid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testGenericPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testGenTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParallelepiped.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/convention_testTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testPolyhedron.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTorus2.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testCutTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testExtruded.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testMultiUnion.cpp
)

if(ROOT) # add shape testers that depend on ROOT
  set(TEST_EXECUTABLES_SHAPETESTER
    ${TEST_EXECUTABLES_SHAPETESTER}
    ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testFromROOTFile.cpp
    ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_debugFromROOTFile.cpp
  )
endif()

if(NOT CUDA)
  set(TEST_EXECUTABLES_SHAPETESTER
    ${TEST_EXECUTABLES_SHAPETESTER}
    ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTessellated.cpp
  )
endif()


### Build executables

set(VECGEOM_EXECUTABLES)

macro(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_link_libraries(${TARGET_NAME} ${VECGEOM_LIBRARIES}
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

if (CTEST OR VALIDATION)
  build_executables("${TEST_EXECUTABLES_SHAPES}")
endif()

if (CTEST)
  build_executables("${TEST_EXECUTABLES_CORE}")
  add_to_ctest("${TEST_EXECUTABLES_CORE}")
  # set_tests_properties(TestSExtru PROPERTIES TIMEOUT 30)
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

if(CTEST OR VALIDATION)
  # add unit tests
  add_test(NAME TestBox COMMAND TestBox)
  add_test(NAME TestTrap COMMAND TestTrap)
  add_test(NAME TestTube COMMAND TestTube)
  add_test(NAME TestCone COMMAND TestCons)
  add_test(NAME TestTrd COMMAND TestTrd)
  add_test(NAME TestOrb COMMAND TestOrb)
  add_test(NAME TestPolycone COMMAND TestPolycone)
  add_test(NAME TestPolyhedra COMMAND TestPolyhedra)
  add_test(NAME TestParallelepiped COMMAND TestParallelepiped)
  add_test(NAME TestHype COMMAND TestHype)
  add_test(NAME TestGenTrap COMMAND TestGenTrap)
  if(NOT CUDA)
    add_test(NAME TestTessellated COMMAND TestTessellated)
  endif(NOT CUDA)
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
configure_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake" @ONLY)

# Install the VecGeomConfig.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install headers and libraries
foreach(FOLDER base benchmarking management navigation volumes backend)
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

################################################################################

# Doxygen documentation

find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYFILE_OUTPUT_DIR  ${CMAKE_BINARY_DIR}/doxygen)
  foreach(d doc/doxygen backend base benchmarking management navigation scripts source volumes)
    set(DOXYFILE_SOURCE_DIRS "${DOXYFILE_SOURCE_DIRS} ${CMAKE_SOURCE_DIR}/${d}")
  endforeach()
  
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in
                  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen
                    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                    COMMENT "Writing documentation to ${DOXYFILE_OUTPUT_DIR}..."
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_custom_target(doxydir
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYFILE_OUTPUT_DIR}
                    COMMENT "Creating doc directory")
  add_dependencies(doxygen doxydir)
endif()
