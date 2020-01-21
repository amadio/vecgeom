#    \file CMakeLists.txt
#    \brief builds a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

#add_subdirectory(backend)
#add_subdirectory(middleware)

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

message(STATUS "XercesC_INCLUDE_DIR is ${XercesC_INCLUDE_DIR}")
if(BUILTIN_XercesC)
  add_dependencies(vgdmlBackend XercesC-internal)
endif()

add_library(vgdml ${sources} ${backend_sources} ${middleware_sources} ${headers} ${backend_headers} ${middleware_headers})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
install(TARGETS vgdml DESTINATION lib)
install(FILES ${headers} DESTINATION include/VecGeom/gdml)
target_link_libraries(vgdml ${VECGEOM_LIBRARIES} ${XercesC_LIBRARY})
target_include_directories(vgdml PUBLIC 
  ${XercesC_INCLUDE_DIR} ${VECGEOM_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/backend/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/middleware/include/
)
