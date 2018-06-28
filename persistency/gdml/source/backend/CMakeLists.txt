#    \file CMakeLists.txt
#    \brief builds a library for reading and writing GDML files 
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0



file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
message(STATUS "XercesC_INCLUDE_DIR is ${XercesC_INCLUDE_DIR}")
add_library(vgdmlBackend ${sources} ${headers})
if(BUILTIN_XercesC)
  add_dependencies(vgdmlBackend XercesC-internal)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
install(TARGETS vgdmlBackend DESTINATION ${CMAKE_INSTALL_PREFIX})
target_link_libraries(vgdmlBackend ${XercesC_LIBRARY})
target_include_directories(vgdmlBackend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_include_directories(vgdmlBackend PUBLIC ${XercesC_INCLUDE_DIR})
