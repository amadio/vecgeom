//!    \file Middleware.cpp
//!    \brief Defines the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Middleware.h"
#include "Helper.h"
#include "xercesc/dom/DOMNode.hpp"
#include "xercesc/dom/DOMNodeList.hpp"
#include "xercesc/dom/DOMDocument.hpp"
#include "xercesc/dom/DOMElement.hpp"
#include "xercesc/dom/DOMNamedNodeMap.hpp"
#include "xercesc/util/XMLString.hpp"

#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedBox.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/VolumeFactory.h"
#include "management/GeoManager.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr bool debug = true;
// a container with simple solids description (fixed number of attributes, no loops), generated from the schema
auto const simpleSolids = std::map<std::string, std::vector<std::string>>{
    {"reflectedSolid", {"solid", "sx", "sy", "sz", "rx", "ry", "rz", "dx", "dy", "dz"}},
    {"box", {"x", "y", "z"}},
    {"twistedbox", {"x", "y", "z", "PhiTwist"}},
    {"twistedtrap", {"PhiTwist", "z", "Theta", "Phi", "y1", "x1", "y2", "x2", "x3", "x4", "Alph"}},
    {"twistedtrd", {"PhiTwist", "z", "y1", "x1", "y2", "x2"}},
    {"paraboloid", {"rlo", "rhi", "dz"}},
    {"sphere", {"rmin", "rmax", "startphi", "deltaphi", "starttheta", "deltatheta"}},
    {"ellipsoid", {"ax", "by", "cz", "zcut1", "zcut2"}},
    {"tube", {"z", "rmin", "rmax", "startphi", "deltaphi"}},
    {"twistedtubs",
     {"twistedangle", "endinnerrad", "endouterrad", "midinnerrad", "midouterrad", "negativeEndz", "positiveEndz",
      "zlen", "nseg", "totphi", "phi"}},
    {"cutTube", {"z", "rmin", "rmax", "startphi", "deltaphi", "lowX", "lowY", "lowZ", "highX", "highY", "highZ"}},
    {"cone", {"z", "rmin1", "rmin2", "rmax1", "rmax2", "startphi", "deltaphi"}},
    {"elcone", {"dx", "dy", "zmax", "zcut"}},
    {"polycone", {"deltaphi", "startphi"}},
    {"genericPolycone", {"deltaphi", "startphi"}},
    {"para", {"x", "y", "z", "alpha", "theta", "phi"}},
    {"trd", {"x1", "x2", "y1", "y2", "z"}},
    {"trap", {"z", "theta", "phi", "y1", "x1", "x2", "alpha1", "y2", "x3", "x4", "alpha2"}},
    {"torus", {"rmin", "rmax", "rtor", "startphi", "deltaphi"}},
    {"orb", {"r"}},
    {"hype", {"rmin", "rmax", "inst", "outst", "z"}},
    {"eltube", {"dx", "dy", "dz"}},
    {"tet", {"vertex1", "vertex2", "vertex3", "vertex4"}},
    {"arb8",
     {"v1x", "v1y", "v2x", "v2y", "v3x", "v3y", "v4x", "v4y", "v5x", "v5y", "v6x", "v6y", "v7x", "v7y", "v8x", "v8y",
      "dz"}}};
// for getting referenced unplaced volumes
auto unplacedVolumeMap = std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *>{};
} // namespace

namespace vgdml {
extern template std::string Transcode(const XMLCh *const anXMLstring);
extern template double Transcode(const XMLCh *const anXMLstring);
extern template int Transcode(const XMLCh *const anXMLstring);
extern template std::string GetAttribute(std::string attrName,
                                         XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
extern template double GetAttribute(std::string attrName,
                                    XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

Middleware::Middleware()
{
}

void *Middleware::Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument)
{
  auto *rootDocElement = aDOMDocument->getDocumentElement();
  auto *rootDocNode    = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *>(rootDocElement);

  //  auto *attributes = rootDocNode->getAttributes();
  //  auto const rootDocNodeName    = Transcode(rootDocNode->getNodeName());
  //  if(debug) std::cout << "rootDocNodeName is " << rootDocNodeName << std::endl;

  // rootDocNode->getChildNodes();
  // auto theAttributes = rootDocNode->getAttributes();
  // auto structXMLname = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode("structure");
  // auto structNode = theAttributes->getNamedItem(structXMLname);
  //  auto structureNodes = aDOMDocument->getElementsByTagName(XERCES_CPP_NAMESPACE_QUALIFIER
  //  XMLString::transcode("gdml/structure"));
  processNode(rootDocNode);
  return nullptr;
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Middleware::Save(void const *)
{
  exit(-1);
}

bool Middleware::processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto *theXMLNodeName = aDOMNode->getNodeName();
  auto const name      = Transcode(theXMLNodeName);
  if (debug) {
    std::cout << "Middleware::processNode: processing: " << name << std::endl;
  }

  if (simpleSolids.count(name)) {
    if (debug) processSimpleVolume(aDOMNode, name);
    auto const success = processSolid(aDOMNode);
    return success;
  } else if (name == "volume") {
    auto const success = processLogicVolume(aDOMNode);
    return success;
  } else if (name == "world") {
    auto const success = processWorld(aDOMNode);
    return success;
  }

  //  if do not know what to do, process the children
  for (auto *child = aDOMNode->getFirstChild(); child != nullptr; child = child->getNextSibling()) {
    processNode(child);
  }
  return false;
}

bool Middleware::processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto *theXMLNodeName = aDOMNode->getNodeName();
  auto const name      = Transcode(theXMLNodeName);
  if (debug) {
    std::cout << "Middleware::processNode: processing: " << name << std::endl;
  }

  auto *attributes     = aDOMNode->getAttributes();
  auto const solidName = GetAttribute("name", attributes);
  //    TODO use variant
  vecgeom::VUnplacedVolume const *anUnplacedSolid = nullptr;
  if (name == "orb") {
    if (debug) {
      std::cout << "Middleware::processNode: found orb" << std::endl;
    }
    auto anUnplacedOrb = processOrb(aDOMNode);
    if (debug) {
      std::cout << "Middleware::processNode: read orb \"" << solidName << "\""
                << " with radius " << anUnplacedOrb->GetRadius() << " mm" << std::endl;
    }
    anUnplacedSolid = anUnplacedOrb;
  } else if (name == "box") {
    anUnplacedSolid = processBox(aDOMNode);
  } else if (name == "tube") {
    anUnplacedSolid = processTube(aDOMNode);
  } // TODO more volumes

  if (!anUnplacedSolid) {
    std::cout << "Middleware::processNode: an unknown solid " << name << " with name " << solidName << std::endl;
    return false;
  }
  auto success = unplacedVolumeMap.insert(std::make_pair(solidName, anUnplacedSolid)).second;
  if (!success) {
    std::cout << "Middleware::processNode: failed to insert volume with name " << solidName << std::endl;
  }
  return success;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedOrb const *Middleware::processOrb(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processOrb was called" << std::endl;
  }
  auto *attributes            = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(attributes);
  auto const radius           = lengthMultiplier * GetAttribute<double>("r", attributes);
  auto const anUnplacedOrbPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedOrb>(radius);
  return anUnplacedOrbPtr;
  // TODO precision
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processSimpleVolume(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode, std::string volumeName)
{

  if (debug) {
    std::cout << "Middleware::processSimpleVolume was called" << std::endl;
  }
  auto *attributes = aDOMNode->getAttributes();
  //  std::vector<std::string> paramNames = {std::string("r")}; // FIXME temporary
  std::vector<std::string> paramNames = simpleSolids.at(volumeName);
  std::vector<double> params;
  std::transform(paramNames.begin(), paramNames.end(), std::back_inserter(params),
                 [attributes](std::string &name) { return GetAttribute<double>(name, attributes); });
  if (debug) {
    std::cout << "Middleware::processSimpleVolume: " << volumeName << " with:\n";
    for (auto ind = 0u; ind < paramNames.size(); ++ind) {
      std::cout << "\t" << paramNames.at(ind) << " = " << params.at(ind) << "\n";
    }
  }

  return nullptr;
  //  return params;
  // TODO units precision cleanup references
}

double Middleware::GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *attributes)
{
  auto const unit = GetAttribute("lunit", attributes);
  auto const lengthMultiplier =
      (unit == "cm")
          ? 1e1
          : (unit == "m")
                ? 1e3
                : (unit == "km") ? 1e6 : (unit == "um") ? 1e-3 : (unit == "nm") ? 1e-6 : 1.; // TODO more units
  return lengthMultiplier;
}

double Middleware::GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *attributes)
{
  auto const unit            = GetAttribute("aunit", attributes);
  auto const angleMultiplier = (unit == "deg") ? vecgeom::VECGEOM_IMPL_NAMESPACE::kPi / 180. : 1.;
  return angleMultiplier;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBox const *Middleware::processBox(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto *attributes            = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(attributes);
  auto const x                = lengthMultiplier * GetAttribute<double>("x", attributes);
  auto const y                = lengthMultiplier * GetAttribute<double>("y", attributes);
  auto const z                = lengthMultiplier * GetAttribute<double>("z", attributes);
  auto const anUnplacedBoxPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBox>(x, y, z);
  return anUnplacedBoxPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTube *Middleware::processTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto *attributes            = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(attributes);
  auto const angleMultiplier  = GetAngleMultiplier(attributes);
  auto const z                = lengthMultiplier * GetAttribute<double>("z", attributes);
  auto const rmin             = lengthMultiplier * GetAttribute<double>("rmin", attributes);
  auto const rmax             = lengthMultiplier * GetAttribute<double>("rmax", attributes);
  auto const startphi         = angleMultiplier * GetAttribute<double>("startphi", attributes);
  // FIXME the default value is not 0
  auto const deltaphi = angleMultiplier * GetAttribute<double>("deltaphi", attributes);
  auto const anUnplacedTubbePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTube>(
          rmin, rmax, z, startphi, deltaphi);
  return anUnplacedTubbePtr;
}

bool Middleware::processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{

  auto *attributes      = aDOMNode->getAttributes();
  auto const volumeName = GetAttribute("name", attributes);
  if (debug) {
    std::cout << "Middleware::processLogicVolume: processing volume named " << volumeName << std::endl;
  }

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement            = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    auto const theChildNodeName = Transcode(it->getNodeName());
    auto const theChildText     = Transcode(it->getTextContent());
    if (debug) {
      std::cout << "\tchild node name is " << theChildNodeName << std::endl;
      std::cout << "\tthe child text content is " << theChildText << std::endl;
    }
    if (aDOMElement) {
      auto const theChildTagName = Transcode(aDOMElement->getTagName());
      if (debug) std::cout << "\tthe child tag name is " << theChildTagName << std::endl;
    }
    if (theChildNodeName == "solidref") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());
      if (debug) std::cout << "volume " << volumeName << " refernces solid " << solidName << std::endl;

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        std::cout << "Could not find solid " << solidName << std::endl;
        return false;
      } else {
        if (debug) std::cout << "Found solid " << solidName << std::endl;
        auto *logicVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(volumeName.c_str(), foundSolid->second);
        vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().RegisterLogicalVolume(logicVolume);
        return true;
      }
    }
  }
  //    auto const solidRef = GetAttribute("materialref", attributes);
  //    vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(volumeName, solidRef);

  return false;
}

bool Middleware::processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto const logicalVolumeName = GetAttribute("ref", aDOMNode->getAttributes());
  auto logicalVolume =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().FindLogicalVolume(logicalVolumeName.c_str());

  if (!logicalVolume) {
    std::cout << "Middleware::processWorld: could not find world volume " << logicalVolumeName << std::endl;
    return false;
  } else {
    if (debug) std::cout << "Middleware::processWorld: found world volume " << logicalVolumeName << std::endl;
    auto placedWorld = logicalVolume->Place(); // TODO use the setup name
    vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().RegisterPlacedVolume(placedWorld); // FIXME is it needed?
    vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().SetWorldAndClose(placedWorld);
  }
  return false;
}

} // namespace vgdml
