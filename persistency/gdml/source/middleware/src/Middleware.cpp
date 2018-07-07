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
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedTorus2.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedHype.h"
#include "volumes/UnplacedCutTube.h"
#include "volumes/UnplacedMultiUnion.h"

#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "management/VolumeFactory.h"
#include "management/GeoManager.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <array>

// requires lengthMultiplier, angleMultiplier and attributes already declared
#define DECLAREANDGETLENGTVAR(x) auto const x = lengthMultiplier * GetAttribute<double>(#x, attributes);
#define DECLAREANDGETANGLEVAR(x) auto const x = angleMultiplier * GetAttribute<double>(#x, attributes);
#define DECLAREANDGETPLAINVAR(x) auto const x = GetAttribute<double>(#x, attributes);

#define DECLAREHALF(x) auto const half##x = x / 2.;

namespace {
constexpr bool debug = true;
// a container with simple solids description (fixed number of attributes, no loops), generated from the schema
auto const gdmlSolids = std::map<std::string, std::vector<std::string>>{
    {"multiUnion", {}},
    {"reflectedSolid", {"solid", "sx", "sy", "sz", "rx", "ry", "rz", "dx", "dy", "dz"}},
    {"scaledSolid", {}},
    {"union", {}},
    {"subtraction", {}},
    {"intersection", {}},
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
    {"polyhedra", {"startphi", "deltaphi", "numsides"}},
    {"genericPolyhedra", {"startphi", "deltaphi", "numsides"}},
    {"xtru", {}},
    {"hype", {"rmin", "rmax", "inst", "outst", "z"}},
    {"eltube", {"dx", "dy", "dz"}},
    {"tet", {"vertex1", "vertex2", "vertex3", "vertex4"}},
    {"arb8",
     {"v1x", "v1y", "v2x", "v2y", "v3x", "v3y", "v4x", "v4y", "v5x", "v5y", "v6x", "v6y", "v7x", "v7y", "v8x", "v8y",
      "dz"}},
    {"tessellated", {}}};
// for getting referenced unplaced volumes
auto unplacedVolumeMap = std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *>{};
auto constantMap       = std::map<std::string, double>{};
auto positionMap       = std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>>{};
auto rotationMap       = std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>>{};

template <typename T = std::string>
T GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  return vgdml::Helper::GetAttribute<T>(attrName, theAttributes);
}

template <>
double GetAttribute(std::string attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto const simpleDouble = vgdml::Helper::GetAttribute<double>(attrName, theAttributes);
  if (std::isnan(simpleDouble)) {
    auto const referencedConstant = constantMap[vgdml::Helper::GetAttribute(attrName, theAttributes)];
    return referencedConstant;
  }
  return simpleDouble;
}

std::array<double, 9> makeRotationMatrixFromCartesianAngles(double x, double y, double z)
{
  auto const s1 = -std::sin(x);
  auto const c1 = std::cos(x);
  auto const s2 = -std::sin(y);
  auto const c2 = std::cos(y);
  auto const s3 = -std::sin(z);
  auto const c3 = std::cos(z);
  auto const xx = c2 * c3;
  auto const xy = -c2 * s3;
  auto const xz = s2;
  auto const yx = c1 * s3 + c3 * s1 * s2;
  auto const yy = c1 * c3 - s1 * s2 * s3;
  auto const yz = -c2 * s1;
  auto const zx = s1 * s3 - c1 * c3 * s2;
  auto const zy = c3 * s1 + c1 * s2 * s3;
  auto const zz = c1 * c2;
  return {{xx, xy, xz, yx, yy, yz, zx, zy, zz}};
}

} // namespace

namespace vgdml {
extern template std::string Helper::Transcode(const XMLCh *const anXMLstring);
extern template double Helper::Transcode(const XMLCh *const anXMLstring);
extern template int Helper::Transcode(const XMLCh *const anXMLstring);
extern template std::string Helper::GetAttribute(std::string attrName,
                                                 XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
extern template double Helper::GetAttribute(std::string attrName,
                                            XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

void *Middleware::Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument)
{
  auto *rootDocElement = aDOMDocument->getDocumentElement();
  auto *rootDocNode    = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *>(rootDocElement);

  //  auto const* const attributes = rootDocNode->getAttributes();
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
  if (debug) {
    std::cout << "Middleware::processNode: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const name = Helper::Transcode(aDOMNode->getNodeName());

  if (gdmlSolids.count(name)) {
    auto const success = processSolid(aDOMNode);
    return success;
  } else if (name == "constant") {
    auto const success = processConstant(aDOMNode);
    return success;
  } else if (name == "position") {
    auto const success = processPosition(aDOMNode);
    return success;
  } else if (name == "rotation") {
    auto const success = processRotation(aDOMNode);
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

bool Middleware::processConstant(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processConstant: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const constantName      = GetAttribute("name", attributes);
  auto const constantValue     = GetAttribute<double>("value", attributes);
  auto const success           = constantMap.insert(std::make_pair(constantName, constantValue)).second;
  if (!success) {
    std::cout << "Middleware::processNode: failed to insert constant with name " << constantName << " and value "
              << constantValue << std::endl;
  }
  return success;
}

bool Middleware::processPosition(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processPosition: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const positionName      = GetAttribute("name", attributes);
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  auto const positionValue = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
  auto const success       = positionMap.insert(std::make_pair(positionName, positionValue)).second;
  if (!success) {
    std::cout << "Middleware::processNode: failed to insert position with name " << positionName << " and value "
              << positionValue << std::endl;
  }
  return success;
}

bool Middleware::processRotation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processPosition: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const rotationName      = GetAttribute("name", attributes);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETANGLEVAR(x)
  DECLAREANDGETANGLEVAR(y)
  DECLAREANDGETANGLEVAR(z)
  auto const rotationValue = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
  auto const success       = rotationMap.insert(std::make_pair(rotationName, rotationValue)).second;
  if (!success) {
    std::cout << "Middleware::processNode: failed to insert rotation with name " << rotationName << " and value "
              << rotationValue << std::endl;
  }
  return success;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processUnion(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processUnion: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *firstSolid  = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *secondSolid = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement            = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      std::cout << "Child: " << Helper::GetNodeInformation(it) << std::endl;
    }
    if (theChildNodeName == "first") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        std::cout << "Could not find solid " << solidName << std::endl;
        return nullptr;
      } else {
        if (debug) std::cout << "Found solid " << solidName << std::endl;
        firstSolid = foundSolid->second;
      }
    } else if (theChildNodeName == "second") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());
      auto foundSolid      = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        std::cout << "Could not find solid " << solidName << std::endl;
        return nullptr;
      } else {
        if (debug) std::cout << "Found solid " << solidName << std::endl;
        secondSolid = foundSolid->second;
      }
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    }
  }
  if (!secondSolid || !firstSolid) {
    std::cout << "Middleware::processUnion: one of the requested soilds not found" << std::endl;
    return nullptr;
  }
  auto const r              = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  auto const transformation = vecgeom::VECGEOM_IMPL_NAMESPACE::Transformation3D(
      position.x(), position.y(), position.z(), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
  auto *multiUnionPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedMultiUnion>();
  multiUnionPtr->AddNode(firstSolid, vecgeom::VECGEOM_IMPL_NAMESPACE::Transformation3D());
  multiUnionPtr->AddNode(secondSolid, transformation);
  return multiUnionPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processMultiUnion(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processMultiUnion: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  std::vector<vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *> placedNodes;

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      std::cout << "Child: " << Helper::GetNodeInformation(it) << std::endl;
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      placedNodes.emplace_back(processMultiUnionNode(it));
    }
  }
  auto *multiUnionPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedMultiUnion>();

  for (auto const *const node : placedNodes) {
    multiUnionPtr->AddNode(node);
  }
  return multiUnionPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *Middleware::processMultiUnionNode(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processMultiUnionNode: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *solid = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;
  auto const name = Helper::GetAttribute("name", aDOMNode->getAttributes());
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement            = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      std::cout << "Child: " << Helper::GetNodeInformation(it) << std::endl;
    }
    if (theChildNodeName == "solid") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        std::cout << "Could not find solid " << solidName << std::endl;
        return nullptr;
      } else {
        if (debug) std::cout << "Found solid " << solidName << std::endl;
        solid = foundSolid->second;
      }
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    }
  }
  if (!solid) {
    std::cout << "Middleware::processUnion: one of the requested soilds not found" << std::endl;
    return nullptr;
  }
  auto const r              = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  auto const transformation = vecgeom::VECGEOM_IMPL_NAMESPACE::Transformation3D(
      position.x(), position.y(), position.z(), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
  auto const logicVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(name.c_str(), solid);
  auto *placedSolidPtr   = logicVolume->Place(name.c_str(), &transformation);
  return placedSolidPtr;
}

bool Middleware::processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processSolid: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const name              = Helper::Transcode(aDOMNode->getNodeName());
  auto const *const attributes = aDOMNode->getAttributes();
  auto const solidName         = GetAttribute("name", attributes);

  auto const *const anUnplacedSolid = [name, aDOMNode]() {
    if (name == "orb") {
      return processOrb(aDOMNode);
    } else if (name == "box") {
      return processBox(aDOMNode);
    } else if (name == "tube") {
      return processTube(aDOMNode);
    } else if (name == "cutTube") {
      return processCutTube(aDOMNode);
    } else if (name == "cone") {
      return processCone(aDOMNode);
    } else if (name == "torus") {
      return processTorus(aDOMNode);
    } else if (name == "sphere") {
      return processSphere(aDOMNode);
    } else if (name == "para") {
      return processParallelepiped(aDOMNode);
    } else if (name == "trd") {
      return processTrd(aDOMNode);
    } else if (name == "trap") {
      return processTrapezoid(aDOMNode);
    } else if (name == "paraboloid") {
      return processParaboloid(aDOMNode);
    } else if (name == "union") {
      return processUnion(aDOMNode);
    } else if (name == "multiUnion") {
      return processMultiUnion(aDOMNode);
    } else if (name == "hype") {
      return processHype(aDOMNode);
    } else
      return static_cast<vecgeom::VUnplacedVolume const *>(nullptr); // TODO more volumes
  }();
  if (!anUnplacedSolid) {
    std::cout << "Middleware::processNode: an unknown solid " << name << " with name " << solidName << std::endl;
    return false;
  }
  auto const success = unplacedVolumeMap.insert(std::make_pair(solidName, anUnplacedSolid)).second;
  if (!success) {
    std::cout << "Middleware::processNode: failed to insert volume with name " << solidName << std::endl;
  }
  return success;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processOrb(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processOrb: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(r)
  auto const anUnplacedOrbPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedOrb>(r);
  return anUnplacedOrbPtr;
  // TODO precision
}

double Middleware::GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nodeName          = Helper::Transcode(aDOMNode->getNodeName());
  auto const unitTag           = (nodeName == "position") ? "unit" : "lunit";
  auto const unit              = GetAttribute(unitTag, attributes);
  auto const lengthMultiplier =
      (unit == "mm")
          ? 1e-1
          : (unit == "m")
                ? 1e2
                : (unit == "km") ? 1e5 : (unit == "um") ? 1e-4 : (unit == "nm") ? 1e-7 : 1.; // TODO more units
  return lengthMultiplier;
}

double Middleware::GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nodeName          = Helper::Transcode(aDOMNode->getNodeName());
  auto const unitTag           = (nodeName == "rotation") ? "unit" : "aunit";
  auto const unit              = GetAttribute(unitTag, attributes);
  auto const angleMultiplier   = (unit == "deg") ? vecgeom::VECGEOM_IMPL_NAMESPACE::kPi / 180. : 1.;
  return angleMultiplier;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processBox(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processBox: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(x)
  DECLAREHALF(y)
  DECLAREHALF(z)
  auto const anUnplacedBoxPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBox>(
          halfx, halfy, halfz);
  return anUnplacedBoxPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processTube: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREHALF(z)
  auto const anUnplacedTubePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTube>(
          rmin, rmax, halfz, startphi, deltaphi);
  return anUnplacedTubePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processCutTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processCutTube: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREANDGETLENGTVAR(lowX)
  DECLAREANDGETLENGTVAR(lowY)
  DECLAREANDGETLENGTVAR(lowZ)
  DECLAREANDGETLENGTVAR(highX)
  DECLAREANDGETLENGTVAR(highY)
  DECLAREANDGETLENGTVAR(highZ)
  DECLAREHALF(z)
  if (highZ < 0 || lowZ > 0) {
    std::cout << "Middleware::processCutTube: for compatibility, the normal must point outwards, expected to fail"
              << std::endl;
  }
  auto const bottomNormal = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{lowX, lowY, lowZ};
  auto const topNormal    = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{highX, highY, highZ};
  auto const anUnplacedCutTubePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedCutTube>(
          rmin, rmax, halfz, startphi, deltaphi, bottomNormal, topNormal);
  return anUnplacedCutTubePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processCone(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processCone: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin1)
  DECLAREANDGETLENGTVAR(rmin2)
  DECLAREANDGETLENGTVAR(rmax1)
  DECLAREANDGETLENGTVAR(rmax2)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREHALF(z)
  auto const anUnplacedConePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedCone>(
          rmin1, rmax1, rmin2, rmax2, halfz, startphi, deltaphi);
  return anUnplacedConePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTorus(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processTorus: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETLENGTVAR(rtor)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  auto const anUnplacedTorusPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTorus2>(
          rmin, rmax, rtor, startphi, deltaphi);
  return anUnplacedTorusPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processSphere(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processSphere: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREANDGETANGLEVAR(starttheta)
  DECLAREANDGETANGLEVAR(deltatheta) // FIXME the default value is not 0
  auto const anUnplacedSpherePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedSphere>(
          rmin, rmax, startphi, deltaphi, starttheta, deltatheta);
  return anUnplacedSpherePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processParallelepiped(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processParallelepiped: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETANGLEVAR(alpha)
  DECLAREANDGETANGLEVAR(theta)
  DECLAREANDGETANGLEVAR(phi)
  DECLAREHALF(x)
  DECLAREHALF(y)
  DECLAREHALF(z)
  auto const anUnplacedParallelepipedPtr = vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<
      vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedParallelepiped>(halfx, halfy, halfz, alpha, theta, phi);
  return anUnplacedParallelepipedPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTrd(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processTrd: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x1)
  DECLAREANDGETLENGTVAR(x2)
  DECLAREANDGETLENGTVAR(y1)
  DECLAREANDGETLENGTVAR(y2)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(x1)
  DECLAREHALF(x2)
  DECLAREHALF(y1)
  DECLAREHALF(y2)
  DECLAREHALF(z)
  auto const anUnplacedTrdPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTrd>(
          halfx1, halfx2, halfy1, halfy2, halfz);
  return anUnplacedTrdPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTrapezoid(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processTrapezoid: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(y1)
  DECLAREANDGETLENGTVAR(x1)
  DECLAREANDGETLENGTVAR(x2)
  DECLAREANDGETLENGTVAR(y2)
  DECLAREANDGETLENGTVAR(x3)
  DECLAREANDGETLENGTVAR(x4)
  DECLAREANDGETANGLEVAR(theta)
  DECLAREANDGETANGLEVAR(phi)
  DECLAREANDGETANGLEVAR(alpha1)
  DECLAREANDGETANGLEVAR(alpha2)
  DECLAREHALF(z)
  DECLAREHALF(x1)
  DECLAREHALF(y1)
  DECLAREHALF(x2)
  DECLAREHALF(y2)
  DECLAREHALF(x3)
  DECLAREHALF(x4)
  auto const anUnplacedTrapezoidPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTrapezoid>(
          halfz, theta, phi, halfy1, halfx1, halfx2, alpha1, halfy2, halfx3, halfx4, alpha2);
  return anUnplacedTrapezoidPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processParaboloid(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processParaboloid: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rlo)
  DECLAREANDGETLENGTVAR(rhi)
  DECLAREANDGETLENGTVAR(dz)
  auto const anUnplacedParaboloidPtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedParaboloid>(
          rlo, rhi, dz);
  return anUnplacedParaboloidPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processHype(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processHype: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETPLAINVAR(inst)
  DECLAREANDGETPLAINVAR(outst)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(z)
  auto const anUnplacedHypePtr =
      vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedHype>(
          rmin, rmax, inst, outst, halfz);
  return anUnplacedHypePtr;
}

bool Middleware::processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processLogicVolume: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume *logicVolume = nullptr;
  auto const *const attributes                                = aDOMNode->getAttributes();
  auto const volumeName                                       = GetAttribute("name", attributes);

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement            = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      std::cout << "Child: " << Helper::GetNodeInformation(it) << std::endl;
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
        logicVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(volumeName.c_str(), foundSolid->second);
        vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().RegisterLogicalVolume(logicVolume);
      }
    } else if (theChildNodeName == "physvol") {
      auto const daughterVolume = processPhysicalVolume(aDOMElement);
      logicVolume->PlaceDaughter(daughterVolume);
    }
  }
  return logicVolume;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *Middleware::processPhysicalVolume(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processPhysicalVolume: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume *logicalVolume = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement            = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      std::cout << "Child: " << Helper::GetNodeInformation(it) << std::endl;
    }
    if (theChildNodeName == "volumeref") {
      auto const logicalVolumeName = GetAttribute("ref", aDOMElement->getAttributes());
      logicalVolume =
          vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().FindLogicalVolume(logicalVolumeName.c_str());
      if (!logicalVolume) {
        std::cout << "Middleware::processPhysicalVolume: could not find volume " << logicalVolumeName << std::endl;
        return nullptr;
      } else {
        if (debug) std::cout << "Middleware::processPhysicalVolume: found volume " << logicalVolumeName << std::endl;
      }
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    }
  }
  if (!logicalVolume) return nullptr;
  auto const r              = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  auto const transformation = vecgeom::VECGEOM_IMPL_NAMESPACE::Transformation3D(
      position.x(), position.y(), position.z(), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
  auto const placedVolume = logicalVolume->Place(&transformation); // TODO position, rotation, label
  vecgeom::VECGEOM_IMPL_NAMESPACE::GeoManager::Instance().RegisterPlacedVolume(placedVolume);
  return placedVolume;
}

bool Middleware::processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    std::cout << "Middleware::processWorld: processing: " << Helper::GetNodeInformation(aDOMNode) << std::endl;
  }
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
