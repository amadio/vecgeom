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

#include "xercesc/util/XercesDefs.hpp"

#include "volumes/UnplacedVolume.h"
#include "volumes/BooleanStruct.h"

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
  void *Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument);
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Save(void const *);

private:
  static bool processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *processPhysicalVolume(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processConstant(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processPosition(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processRotation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processFacet(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                           vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTessellated &storage);
  template <vecgeom::BooleanOperation Op>
  static vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processBoolean(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processMultiUnion(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *processMultiUnionNode(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  static vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processOrb(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processBox(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCutTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTorus(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processSphere(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParallelepiped(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrd(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrapezoid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParaboloid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processHype(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTesselated(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  static double GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static double GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
};
} // namespace vgdml
#endif
