//!    \file Helper.cpp
//!    \brief Defines helper functions to manipulate data from XML
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!
//!    \license Distributed under the Apache license 2.0

#include "Helper.h"

#include <string>
#include <stdexcept>
#include <limits>

#include "xercesc/dom/DOMNamedNodeMap.hpp"
#include "xercesc/dom/DOMNode.hpp"
#include "xercesc/dom/DOMElement.hpp"

namespace vgdml {
namespace Helper {
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
  return attribute ? vgdml::Helper::Transcode(attribute->getNodeValue()) : "";
};

template <>
double GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto const strAttribure = GetAttribute(attrName, theAttributes);
  try {
    return std::stod(strAttribure);

  } catch (std::invalid_argument) {
    return std::numeric_limits<double>::quiet_NaN();
  }
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

std::stringstream GetNodeInformation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto aStream                   = std::stringstream{};
  auto const theNodeName         = Transcode(aDOMNode->getNodeName());
  auto const theNodeText         = Transcode(aDOMNode->getTextContent());
  auto const *const asDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement const *>(aDOMNode);
  aStream << "node name is \"" << theNodeName << "\"";
  aStream << ", node text content is \"" << theNodeText << "\"";
  if (asDOMElement) {
    auto const theNodeLocalName = Transcode(aDOMNode->getLocalName());
    auto const theChildTagName  = Transcode(asDOMElement->getTagName());
    aStream << ", local node name is \"" << theNodeLocalName << "\"";
    aStream << ", node tag name is \"" << theChildTagName << "\"";
    auto const *const attributes = aDOMNode->getAttributes();
    auto const nAttributes       = attributes->getLength();
    aStream << ", it has " << nAttributes << " attributes ( ";
    for (auto ind = 0u; ind < nAttributes; ++ind) {
      auto const *const anAttribute = attributes->item(ind);
      auto const attributeName      = Helper::Transcode(anAttribute->getNodeName());
      auto const attributeValue     = GetAttribute(attributeName, attributes);
      aStream << "\"" << attributeName << "\":\"" << attributeValue << "\" ";
    }
    aStream << ")";
  }
  return aStream;
}

} // namespace Helper
} // namespace vgdml
