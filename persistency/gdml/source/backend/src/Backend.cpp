//!    \file Backend.cpp
//!    \brief Defines the class for loading files to DOM and writing back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#include "Backend.h"

// TODO leave only the needed
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMLSParser.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
namespace vgdml {

Backend::Backend()
{
  // TODO catch errors, do once per program
  xercesc::XMLPlatformUtils::Initialize();

  static const XMLCh gLS[]         = {xercesc::chLatin_L, xercesc::chLatin_S, xercesc::chNull};
  xercesc::DOMImplementation *impl = xercesc::DOMImplementationRegistry::getDOMImplementation(gLS);
  fDOMLSParser                     = (static_cast<xercesc::DOMImplementationLS *>(impl))
                     ->createLSParser(xercesc::DOMImplementationLS::MODE_SYNCHRONOUS, nullptr);
  xercesc::DOMConfiguration *config = fDOMLSParser->getDomConfig();

  config->setParameter(xercesc::XMLUni::fgDOMNamespaces, true);
  config->setParameter(xercesc::XMLUni::fgXercesSchema, true);
  config->setParameter(xercesc::XMLUni::fgXercesHandleMultipleImports, true);
  config->setParameter(xercesc::XMLUni::fgXercesSchemaFullChecking, true);
  config->setParameter(xercesc::XMLUni::fgDOMValidateIfSchema, true);
  config->setParameter(xercesc::XMLUni::fgDOMDatatypeNormalization, true);
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Backend::Load(std::string const &aFilename)
{
  fDOMLSParser->resetDocumentPool();
  auto doc = fDOMLSParser->parseURI(aFilename.c_str());
  return doc;
}

void Backend::Save(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDoc, std::string const &aFilename)
{
  // TODO set parameters, maybe take form Geant4
  XMLCh tempStr[3]                        = {xercesc::chLatin_L, xercesc::chLatin_S, xercesc::chNull};
  xercesc::DOMImplementation *impl        = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
  xercesc::DOMLSSerializer *theSerializer = (static_cast<xercesc::DOMImplementationLS *>(impl))->createLSSerializer();
  xercesc::DOMLSOutput *theOutputDesc     = (static_cast<xercesc::DOMImplementationLS *>(impl))->createLSOutput();
  xercesc::XMLFormatTarget *myFormTarget  = new xercesc::LocalFileFormatTarget(aFilename.c_str());
  theOutputDesc->setByteStream(myFormTarget);
  theSerializer->write(aDOMDoc, theOutputDesc);

  theOutputDesc->release();
  theSerializer->release();
  return;
}
} // namespace vgdml
