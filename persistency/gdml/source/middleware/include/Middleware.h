//!    \file Middleware.h
//!    \brief Declares the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#pragma once

#ifndef VGDMLMiddleware_h
#define VGDMLMiddleware_h

#include <string>

#include "xercesc/util/XercesDefs.hpp"

#include "volumes/PlacedVolume.h"
#include "volumes/Orb.h"

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMLSParser;
class DOMNode;
XERCES_CPP_NAMESPACE_END

namespace vgdml {
class Middleware {
public:
  Middleware();
  void *Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument);
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Save(void const *);

private:
  static void processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  // TODO other precision
  static vecgeom::cxx::OrbStruct<double> processOrb(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
};
} // namespace vgdml
#endif
