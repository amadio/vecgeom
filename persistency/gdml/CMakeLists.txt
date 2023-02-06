#    \file CMakeLists.txt
#    \brief builds and tests a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

# Find XercesC 
find_package(XercesC REQUIRED)

add_subdirectory(source)
