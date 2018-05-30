//!    \file Middleware.cpp
//!    \brief Defines the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#include "Middleware.h"
#include "Helper.h"
#include "xercesc/dom/DOMNode.hpp"
#include "xercesc/dom/DOMDocument.hpp"
#include "xercesc/dom/DOMNamedNodeMap.hpp"
#include "xercesc/util/XMLString.hpp"

#include <iostream>

namespace {
constexpr bool debug = true;
}

namespace vgdml {
extern template std::string Transcode(const XMLCh *const anXMLstring);
extern template double Transcode(const XMLCh *const anXMLstring);
extern template int Transcode(const XMLCh *const anXMLstring);

Middleware::Middleware()
{
}

void *Middleware::Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument)
{
  auto *rootDocElement = aDOMDocument->getDocumentElement();
  processNode((XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *)rootDocElement);
  exit(-1);
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Middleware::Save(void const *)
{
  exit(-1);
}

void Middleware::processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto *theXMLNodeName = aDOMNode->getNodeName();
  auto const name      = Transcode(theXMLNodeName);
  if (debug) {
    std::cout << "Middleware::processNode: processing: " << name << std::endl;
  }

  if (name == "orb") {
    if (debug) {
      std::cout << "Middleware::processNode: found orb" << std::endl;
    }
    auto *attributes   = aDOMNode->getAttributes();
    auto const orbName = GetAttribute("name", attributes);
    auto const unit    = GetAttribute("lunit", attributes); // TODO set to default in empty
                                                            //    TODO use variant
    auto anOrb = processOrb(aDOMNode);
    std::cout << "Middleware::processNode: read orb \"" << orbName << "\""
              << " with radius " << anOrb.fR << " " << unit << std::endl;
    return;
  }
  //  if do not know what to do, process the children
  for (auto *child = aDOMNode->getFirstChild(); child != nullptr; child = child->getNextSibling()) {
    processNode(child);
  }
}

vecgeom::VECGEOM_IMPL_NAMESPACE::OrbStruct<double> Middleware::processOrb(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processOrb was called" << std::endl;
  }
  auto *attributes = aDOMNode->getAttributes();

  auto radiusStr = GetAttribute("r", attributes);
  if (!radiusStr.size()) {
    std::cout << "Middleware::processOrb: error: did not found expected attribute" << std::endl;
    return vecgeom::VECGEOM_IMPL_NAMESPACE::OrbStruct<double>();
  } else {
    if (debug) {
      std::cout << "Middleware::processOrb: found expected attribute" << std::endl;
    }
    auto radius = std::stod(radiusStr);
    // TODO units precision cleanup
    auto const anOrbStruct = vecgeom::VECGEOM_IMPL_NAMESPACE::OrbStruct<double>(radius);
    return anOrbStruct;
  }
}

} // namespace vgdml
