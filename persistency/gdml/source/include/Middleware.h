//!    \file Middleware.h
//!    \brief Declares the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once

#ifndef VGDMLMiddleware_h
#define VGDMLMiddleware_h

#include <string>
#include <vector>
#include <map>

#include "xercesc/util/XercesDefs.hpp"

#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/BooleanStruct.h"

#include "MaterialInfo.h"

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMLSParser;
class DOMNode;
class DOMNamedNodeMap;
XERCES_CPP_NAMESPACE_END

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
class VUnplacedVolume;
class UnplacedTessellated;
// class UnplacedOrb;
// class UnplacedBox;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

namespace vgdml {
class Middleware {
public:
  Middleware() {}
  bool Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument);
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Save(void const *);

private:
  bool processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume *const processPhysicalVolume(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processConstant(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processPosition(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processScale(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processRotation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  bool processIsotope(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processElement(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processMaterial(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  bool processFacet(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                    vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTessellated &storage);
  template <vecgeom::BooleanOperation Op>
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processBoolean(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processMultiUnion(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *processMultiUnionNode(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processOrb(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processBox(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCutTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processPolycone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processPolyhedron(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTorus(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processSphere(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParallelepiped(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrd(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrapezoid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParaboloid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processHype(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTesselated(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processExtruded(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTet(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processScaledShape(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  double GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  double GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

private:
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *> unplacedVolumeMap;
  std::map<std::string, double> constantMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> positionMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> scaleMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> rotationMap;
  std::map<std::string, vgdml::Isotope> isotopeMap;
  std::map<std::string, vgdml::Element> elementMap;
  std::map<std::string, vgdml::Material> materialMap;

  double GetDoubleAttribute(std::string const &attrName,
                            XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
};
} // namespace vgdml
#endif
