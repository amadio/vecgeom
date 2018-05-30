#    \file CMakeLists.txt
#    \brief builds a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

add_subdirectory(backend)
add_subdirectory(middleware)

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

add_library(vgdml ${sources} ${headers})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
install(TARGETS vgdml DESTINATION ${CMAKE_INSTALL_PREFIX})
target_link_libraries(vgdml vgdmlMiddleware)
target_include_directories(vgdml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
get_target_property(MIDDLEWARE_INCLUDES vgdmlMiddleware INCLUDE_DIRECTORIES)
target_include_directories(vgdml PUBLIC ${MIDDLEWARE_INCLUDES})
set_property(TARGET vgdml PROPERTY CXX_STANDARD 14)
