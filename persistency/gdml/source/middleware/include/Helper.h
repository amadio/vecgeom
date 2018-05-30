//!    \file Helper.h
//!    \brief Declares helper functions to manipulate data from XML
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0
#pragma once

#ifndef Helper_h
#define Helper_h

#include <string>

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/util/XMLChar.hpp"
#include "xercesc/util/XMLString.hpp"

XERCES_CPP_NAMESPACE_BEGIN
class DOMNamedNodeMap;
XERCES_CPP_NAMESPACE_END

namespace vgdml {
// has overhead for object creation, but cleaner than char*, TODO profile if critical
template <typename T = std::string>
T Transcode(const XMLCh *const anXMLstring);

template <typename F, typename T, typename R>
R TranscodeAndApply(F aFunction, T anArgument);

template <typename T = std::string>
T GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap *theAttributes);
} // namespace vgdml

namespace vgdml {
template <typename F, typename R>
R TranscodeAndApply(F aFunction, std::string anArgument)
{
  auto anXMLargument = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(anArgument.c_str());
  auto result        = aFunction(anXMLargument);
  XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&anXMLargument);
  return std::forward<R>(result);
}

} // namespace vgdml

#endif
