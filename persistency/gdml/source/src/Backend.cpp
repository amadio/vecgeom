//!    \file Backend.cpp
//!    \brief Defines the class for loading files to DOM and writing back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Backend.h"

// TODO leave only the needed
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSParser.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/framework/psvi/PSVIHandler.hpp>

#include <cstdlib>

namespace vgdml {

Backend::Backend()
{
  // TODO catch errors, do once per program
  xercesc::XMLPlatformUtils::Initialize();

  fDOMParser           = new xercesc::XercesDOMParser;
  auto const schemaDir = std::getenv("GDMLDIR"); // get the alternative schema location for offline use
  if (schemaDir) {
    auto const shemaFile = (schemaDir + std::string("gdml.xsd")).c_str();
    fDOMParser->setExternalNoNamespaceSchemaLocation(shemaFile);
    fDOMParser->loadGrammar(shemaFile, xercesc::Grammar::SchemaGrammarType, true);
  }
  fDOMParser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
  fDOMParser->setDoNamespaces(true);
  fDOMParser->setDoSchema(true);
  fDOMParser->setDoXInclude(true);
  fDOMParser->setCreateSchemaInfo(true);
  fDOMParser->setIncludeIgnorableWhitespace(false);
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Backend::Load(std::string const &aFilename)
{
  fDOMParser->resetDocumentPool();
  fDOMParser->parse(aFilename.c_str());
  auto doc = fDOMParser->getDocument();
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
