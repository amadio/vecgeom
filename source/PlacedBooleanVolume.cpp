/*
 * PlacedBooleanVolume.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: swenzel
 */

#include "volumes/PlacedBooleanVolume.h"
#include "volumes/SpecializedBooleanVolume.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/LogicalVolume.h"
#include "base/Vector3D.h"
#include "base/RNG.h"
#include <map>

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#endif

#include <iostream>

namespace vecgeom {

template <>
Vector3D<Precision> PlacedBooleanVolume<kUnion>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  double arearatio(0.5);
  double leftarea, rightarea;

  auto unplaced = GetUnplacedVolume();

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, double> idtoareamap;
  auto leftid = unplaced->GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(unplaced->GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(leftid, leftarea));
  }

  auto rightid = unplaced->GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(unplaced->GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? unplaced->GetLeft() : unplaced->GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

template <>
Vector3D<Precision> PlacedBooleanVolume<kIntersection>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  double arearatio(0.5);
  double leftarea, rightarea;

  auto unplaced = GetUnplacedVolume();

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, double> idtoareamap;
  auto leftid = unplaced->GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(unplaced->GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(leftid, leftarea));
  }

  auto rightid = unplaced->GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(unplaced->GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? unplaced->GetLeft() : unplaced->GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

template <>
Vector3D<Precision> PlacedBooleanVolume<kSubtraction>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  double arearatio(0.5);
  double leftarea, rightarea;

  auto unplaced = GetUnplacedVolume();

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, double> idtoareamap;
  auto leftid = unplaced->GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(unplaced->GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(leftid, leftarea));
  }

  auto rightid = unplaced->GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(unplaced->GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? unplaced->GetLeft() : unplaced->GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

#ifdef VECGEOM_ROOT
template <>
TGeoShape const *PlacedBooleanVolume<kUnion>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();

  TGeoUnion *node =
      new TGeoUnion(const_cast<TGeoShape *>(left->ConvertToRoot()), const_cast<TGeoShape *>(right->ConvertToRoot()),
                    leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
  return new TGeoCompositeShape("RootComposite", node);
}

template <>
TGeoShape const *PlacedBooleanVolume<kIntersection>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();
  TGeoIntersection *node         = new TGeoIntersection(const_cast<TGeoShape *>(left->ConvertToRoot()),
                                                const_cast<TGeoShape *>(right->ConvertToRoot()),
                                                leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
  return new TGeoCompositeShape("RootComposite", node);
}

template <>
TGeoShape const *PlacedBooleanVolume<kSubtraction>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();
  TGeoSubtraction *node          = new TGeoSubtraction(const_cast<TGeoShape *>(left->ConvertToRoot()),
                                              const_cast<TGeoShape *>(right->ConvertToRoot()),
                                              leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
  return new TGeoCompositeShape("RootComposite", node);
}
#endif

#ifdef VECGEOM_GEANT4
template <>
G4VSolid const *PlacedBooleanVolume<kUnion>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    std::cerr << "WARNING : For the moment left transformations are not implemented\n";
  }

  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  auto rot                       = rightm->Rotation();
  // HepRep3x3 seems? to be column major order:
  g4rot->set(CLHEP::HepRep3x3(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]));
  return new G4UnionSolid(GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()),
                          const_cast<G4VSolid *>(right->ConvertToGeant4()), g4rot,
                          G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
template <>
G4VSolid const *PlacedBooleanVolume<kIntersection>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    std::cerr << "WARNING : For the moment left transformations are not implemented\n";
  }

  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  auto rot                       = rightm->Rotation();
  // HepRep3x3 seems? to be column major order:
  g4rot->set(CLHEP::HepRep3x3(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]));
  return new G4IntersectionSolid(GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()),
                                 const_cast<G4VSolid *>(right->ConvertToGeant4()), g4rot,
                                 G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
template <>
G4VSolid const *PlacedBooleanVolume<kSubtraction>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    std::cerr << "WARNING : For the moment left transformations are not implemented\n";
  }
  return new G4SubtractionSolid(GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()),
                                const_cast<G4VSolid *>(right->ConvertToGeant4()), g4rot,
                                G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
#endif

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kUnion)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kIntersection)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kSubtraction)

#endif // VECCORE_CUDA

} // End namespace vecgeom
