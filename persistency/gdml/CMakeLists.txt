#    \file CMakeLists.txt
#    \brief builds and tests a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

cmake_minimum_required (VERSION 3.3)
project (vgdml)

message(STATUS "vgdml CMAKE_GENERATOR is ${CMAKE_GENERATOR}")
message(STATUS "vgdml CMAKE_MAKE_PROGRAM is ${CMAKE_MAKE_PROGRAM}")

if (NOT BUILTIN_XercesC)
  # Find XercesC with selected components turned on (CUDA and backend)
  find_package(XercesC)
  if (NOT XercesC_FOUND)
    message(WARNING "Could not find XercesC, will try to build from source")
    set(BUILTIN_XercesC ON)
  else()
    message("-- Found XercesC at: ${XercesC_DIR}")
  endif()
endif()

if(BUILTIN_XercesC)
  add_subdirectory(external/XercesC)
  set(XercesC_ROOT_DIR ${CMAKE_BINARY_DIR}/persistency/gdml/external/XercesC/Install/XercesC-3.2.1)
  set(XercesC_DIR ${XercesC_ROOT_DIR}/lib64/cmake/XercesC/)
  set(XercesC_INCLUDE_DIR ${XercesC_ROOT_DIR}/include/)
  set(XercesC_LIBRARY ${XercesC_ROOT_DIR}/lib64/libxerces-c.so)
  find_package(XercesC REQUIRED)
endif()

if(VECGEOM)
  set(VECGEOM_LIBRARIES vecgeom)
  get_target_property(VECGEOM_INCLUDE_DIR vecgeom INCLUDE_DIRECTORIES)
else()
  find_package(VecGeom REQUIRED)
endif()

add_subdirectory(source)

if(NOT VECGEOM)
  enable_testing()
endif()

add_subdirectory(test)
