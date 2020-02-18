#    \file CMakeLists.txt
#    \brief tests the library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0
# One executable for Backend/Frontend/Middleware
foreach(_vgdml_test TestXercesBackend TestXercesFrontend TestXercesMiddleware)
  add_executable(${_vgdml_test} ${_vgdml_test}/${_vgdml_test}.cpp)
  # Not 100% clear that USERKERNELLIB is needed
  target_link_libraries(${_vgdml_test} vgdml ${USERKERNELLIB})
endforeach()

# Add tests for each type with needed input GDML file
# - Backend
file(GLOB GDMLS ${CMAKE_CURRENT_SOURCE_DIR}/../gdmls/*.gdml)
foreach(GDML ${GDMLS})
  get_filename_component(BASENAMEGDML ${GDML} NAME)
  add_test(NAME TestXercesBackend-${BASENAMEGDML} COMMAND TestXercesBackend ${GDML})
endforeach()

# - Middleware
foreach(BASENAMEGDML oneOrb oneSphere oneTorus oneBox oneTube oneConeSegment oneParallelepiped oneTrapezoid oneParaboloid oneGeneralTrapezoid oneHyperbolicTube oneCutTube aConstant twoBoxes aPosition aRotation solidIntersection solidSubtraction solidUnion onePolycone onePolyhedron cms2018)
  set(GDMLFILE ${CMAKE_CURRENT_SOURCE_DIR}/../gdmls/${BASENAMEGDML}.gdml)
  add_test(NAME TestXercesMiddleware-${BASENAMEGDML} COMMAND TestXercesMiddleware ${GDMLFILE})
endforeach()


