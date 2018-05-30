#    \file CMakeLists.txt
#    \brief builds a library for reading and writing GDML files 
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
add_library(vgdmlMiddleware ${sources} ${headers})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
install(TARGETS vgdmlMiddleware DESTINATION ${CMAKE_INSTALL_PREFIX})
target_link_libraries(vgdmlMiddleware vgdmlBackend ${VECGEOM_LIBRARIES})
target_include_directories(vgdmlMiddleware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_include_directories(vgdmlMiddleware PUBLIC ${VECGEOM_INCLUDE_DIR})
get_target_property(BACKEND_INCLUDES vgdmlBackend INCLUDE_DIRECTORIES)
target_include_directories(vgdmlMiddleware PUBLIC ${BACKEND_INCLUDES})
set_property(TARGET vgdmlMiddleware PROPERTY CXX_STANDARD 14)

