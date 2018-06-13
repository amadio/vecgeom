#    \file CMakeLists.txt
#    \brief downloads and builds xerces-c as an external project
#
#    \authors Author:  Dmitry Savin
#

cmake_minimum_required(VERSION 3.3)
project(XercesC-internal)


message(STATUS "Trying to build XercesC from source")
include(ExternalProject)
set(XercesCEP ${CMAKE_BINARY_DIR})

message(STATUS "XercesC-internal CMAKE_GENERATOR is ${CMAKE_GENERATOR}")
message(STATUS "XercesC-internal CMAKE_MAKE_PROGRAM is ${CMAKE_MAKE_PROGRAM}")

ExternalProject_Add(XercesC-3.2.1
  URL https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.1.tar.gz
  URL_HASH MD5=fe951ca5d93713db31b026fab2d042d7
  BUILD_IN_SOURCE 0
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${XercesCEP}/Install/XercesC-3.2.1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -G ${CMAKE_GENERATOR} -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  )

add_custom_target(XercesC-internal)
add_dependencies(XercesC-internal XercesC-3.2.1)


execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}/persistency/gdml/external/XercesC -G ${CMAKE_GENERATOR}
  WORKING_DIRECTORY ${XercesCEP}/persistency/gdml/external/XercesC
)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${XercesCEP}/persistency/gdml/external/XercesC
)

