//!    \file Helper.h
//!    \brief Declares helper functions to manipulate data from XML
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once

#ifndef Helper_h
#define Helper_h

#include <string>
#include <map>

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/util/XMLChar.hpp"
#include "xercesc/util/XMLString.hpp"

XERCES_CPP_NAMESPACE_BEGIN
class DOMNamedNodeMap;
class DOMNode;
XERCES_CPP_NAMESPACE_END

namespace vgdml {

namespace Helper {
// has overhead for object creation, but cleaner than char*, TODO profile if critical
template <typename T = std::string>
T Transcode(const XMLCh *const anXMLstring);

template <typename F, typename T, typename R>
R TranscodeAndApply(F aFunction, T anArgument);

template <typename T = std::string>
T GetAttribute(std::string const &attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

std::map<std::string const, std::string const> GetAttributes(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
std::map<std::string const, std::string const> GetAttributes(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

std::string GetNodeInformation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

} // namespace Helper
} // namespace vgdml

namespace vgdml {
namespace Helper {

template <typename F, typename R>
R TranscodeAndApply(F aFunction, std::string anArgument)
{
  auto anXMLargument = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(anArgument.c_str());
  auto result        = aFunction(anXMLargument);
  XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&anXMLargument);
  return std::forward<R>(result);
}

bool IsWhitespace(std::string const &aString);

} // namespace Helper
} // namespace vgdml

#endif
