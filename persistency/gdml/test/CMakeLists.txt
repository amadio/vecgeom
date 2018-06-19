#    \file CMakeLists.txt
#    \brief tests the library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

option(GDMLTESTING "Enable testing of GDML persitency" ON)

if(NOT GDMLTESTING)
  return()
endif()

# find_package(XercesC REQUIRED)
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

foreach(testname ${SUBDIRS})
  file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/${testname}/*.cpp)
  file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/${testname}/*.hpp)
  add_executable(${testname} ${sources} ${headers})
  set_property(TARGET ${testname} PROPERTY CXX_STANDARD 14)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
  install(TARGETS ${testname} DESTINATION ${CMAKE_INSTALL_PREFIX})
  target_link_libraries(${testname} vgdml)
  get_target_property(VGDML_INCLUDES vgdml INCLUDE_DIRECTORIES)
  target_include_directories(${testname} PUBLIC ${VGDML_INCLUDES})
endforeach()

file(GLOB GDMLS ${CMAKE_CURRENT_SOURCE_DIR}/../gdmls/*.gdml)
foreach(GDML ${GDMLS})
  get_filename_component(BASENAMEGDML ${GDML} NAME)
  add_test(NAME TestXercesBackend-${BASENAMEGDML} COMMAND TestXercesBackend ${GDML})
endforeach()

foreach(BASENAMEGDML oneOrb oneSphere oneTorus oneBox oneTube oneConeSegment oneParallelepiped oneTrapezoid oneParaboloid oneGeneralTrapezoid oneHyperbolicTube oneCutTube)
  set(GDMLFILE ${CMAKE_CURRENT_SOURCE_DIR}/../gdmls/${BASENAMEGDML}.gdml)
  add_test(NAME TestXercesMiddleware-${BASENAMEGDML} COMMAND TestXercesMiddleware ${GDMLFILE})
endforeach()


