#    \file CMakeLists.txt
#    \brief builds and tests a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

cmake_minimum_required(VERSION 3.8...3.19)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(vgdml)

# Find XercesC with selected components turned on (CUDA and backend)
find_package(XercesC REQUIRED)

# We are only a part of VecGeom, so only ever link to vecgeom target
# Easy to resotore this later if vgdml is re-split out
set(VECGEOM_LIBRARIES vecgeom)

if(GDMLDEBUG)
  # Only used in Middleware.cpp, and to set a constant
  add_definitions(-DGDMLDEBUG)
endif()

add_subdirectory(source)

option(GDMLTESTING "Enable testing of GDML persitency" ON)
if(GDMLTESTING)
  # enable_testing should be safe even if called at a lower level
  enable_testing()
  add_subdirectory(test)
endif()
