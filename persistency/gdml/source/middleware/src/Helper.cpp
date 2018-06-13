//!    \file Helper.cpp
//!    \brief Defines helper functions to manipulate data from XML
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#include "Helper.h"

#include <string>

#include "xercesc/dom/DOMNamedNodeMap.hpp"
#include "xercesc/dom/DOMNode.hpp"

namespace vgdml {
template <>
std::string Transcode(const XMLCh *const anXMLstring);
template <>
double Transcode(const XMLCh *const anXMLstring);
template <>
int Transcode(const XMLCh *const anXMLstring);

template <>
std::string GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
template <>
double GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
template <>
int GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

template <>
std::string GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto *attrXMLName = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(attrName.c_str());
  auto *attribute   = theAttributes->getNamedItem(attrXMLName);
  XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&attrXMLName);
  return attribute ? vgdml::Transcode(attribute->getNodeValue()) : "";
};

template <>
double GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto const strAttribure = GetAttribute(attrName, theAttributes);
  return std::stod(strAttribure);
};

template <>
int GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto const strAttribure = GetAttribute(attrName, theAttributes);
  return std::stoi(strAttribure);
};

template <>
std::string Transcode(const XMLCh *const anXMLstring)
{
  // TODO use u16string and then c++ standard codecvt
  auto *aCstring     = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(anXMLstring);
  auto const aString = std::string(aCstring);
  XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&aCstring);
  return aString;
}

// TODO other types, use constexpr_if or SFINAE
template <>
double Transcode(const XMLCh *const anXMLstring)
{
  auto const aString = Transcode(anXMLstring);
  auto const aDouble = std::stod(aString);
  return aDouble;
}

template <>
int Transcode(const XMLCh *const anXMLstring)
{
  auto const aString = Transcode(anXMLstring);
  auto const anInt   = std::stoi(aString);
  return anInt;
}

} // namespace vgdml
