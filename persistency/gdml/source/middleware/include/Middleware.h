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

//#include "volumes/Orb.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMLSParser;
class DOMNode;
class DOMNamedNodeMap;
XERCES_CPP_NAMESPACE_END

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
class VUnplacedVolume;
// class UnplacedOrb;
// class UnplacedBox;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

namespace vgdml {
class Middleware {
public:
  Middleware();
  void *Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument);
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Save(void const *);

private:
  static bool processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static bool processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedOrb const *processOrb(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBox *processBox(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  static const vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTube *processTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  static vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processSimpleVolume(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode, std::string volumeName);
  static double GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *attributes);
  static double GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *attributes);
};
} // namespace vgdml
#endif
