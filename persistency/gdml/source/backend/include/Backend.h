//!    \file Backend.h
//!    \brief Declares class for loading files to DOM and writing back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#pragma once

#ifndef VGDMLBackend_h
#define VGDMLBackend_h

#include <string>

#include "xercesc/util/XercesDefs.hpp"

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMLSParser;
XERCES_CPP_NAMESPACE_END

namespace vgdml {
class Backend {
  xercesc::DOMLSParser *fDOMLSParser;

public:
  Backend();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Load(std::string const &aFilename);
  void Save(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDoc, std::string const &aFilename);
};
} // namespace vgdml
#endif
