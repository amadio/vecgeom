#    \file CMakeLists.txt
#    \brief builds a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

add_library(vgdml
  include/Backend.h
  include/Frontend.h
  include/Helper.h
  include/MaterialInfo.h
  include/RegionInfo.h
  include/Middleware.h
  src/Backend.cpp
  src/Frontend.cpp
  src/Helper.cpp
  src/Middleware.cpp
)

# Don't force this yet
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

target_include_directories(vgdml PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(vgdml PUBLIC ${VECGEOM_LIBRARIES} XercesC::XercesC)

install(TARGETS vgdml EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/VecGeom/gdml"
  FILES_MATCHING PATTERN "*.h"
)