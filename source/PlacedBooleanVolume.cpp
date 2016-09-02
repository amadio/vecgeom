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

Vector3D<Precision> PlacedBooleanVolume::GetPointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  double arearatio(0.5);
  double leftarea, rightarea;

  UnplacedBooleanVolume *unplaced = (UnplacedBooleanVolume *)GetUnplacedVolume();

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, double> idtoareamap;
  auto leftid = unplaced->fLeftVolume->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(unplaced->fLeftVolume)->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, double>(leftid, rightarea));
  }

  auto rightid = unplaced->fRightVolume->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(unplaced->fRightVolume)->SurfaceArea();
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

    auto *selected((RNG::Instance().uniform() < arearatio) ? unplaced->fLeftVolume : unplaced->fRightVolume);
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->GetPointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedBooleanVolume::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->fLeftVolume;
  VPlacedVolume const *right     = GetUnplacedVolume()->fRightVolume;
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();

  TGeoShape *shape = NULL;
  if (GetUnplacedVolume()->GetOp() == kSubtraction) {
    TGeoSubtraction *node = new TGeoSubtraction(const_cast<TGeoShape *>(left->ConvertToRoot()),
                                                const_cast<TGeoShape *>(right->ConvertToRoot()),
                                                leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
    shape = new TGeoCompositeShape("RootComposite", node);
  }
  if (GetUnplacedVolume()->GetOp() == kUnion) {
    TGeoUnion *node =
        new TGeoUnion(const_cast<TGeoShape *>(left->ConvertToRoot()), const_cast<TGeoShape *>(right->ConvertToRoot()),
                      leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
    shape = new TGeoCompositeShape("RootComposite", node);
  }
  if (GetUnplacedVolume()->GetOp() == kIntersection) {
    TGeoIntersection *node = new TGeoIntersection(const_cast<TGeoShape *>(left->ConvertToRoot()),
                                                  const_cast<TGeoShape *>(right->ConvertToRoot()),
                                                  leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
    shape = new TGeoCompositeShape("RootComposite", node);
  }
  return shape;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedBooleanVolume::ConvertToGeant4() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->fLeftVolume;
  VPlacedVolume const *right     = GetUnplacedVolume()->fRightVolume;
  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  g4rot->set(CLHEP::HepRep3x3(rightm->Rotation()));
  if (GetUnplacedVolume()->GetOp() == kSubtraction) {
    return new G4SubtractionSolid(
        GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()), const_cast<G4VSolid *>(right->ConvertToGeant4()),
        g4rot, G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
  }
  if (GetUnplacedVolume()->GetOp() == kUnion) {
    return new G4UnionSolid(GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()),
                            const_cast<G4VSolid *>(right->ConvertToGeant4()), g4rot,
                            G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
  }
  if (GetUnplacedVolume()->GetOp() == kIntersection) {
    return new G4IntersectionSolid(
        GetLabel(), const_cast<G4VSolid *>(left->ConvertToGeant4()), const_cast<G4VSolid *>(right->ConvertToGeant4()),
        g4rot, G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
  }
  return NULL;
}
#endif

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kUnion)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kIntersection)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kSubtraction)

#endif // VECGEOM_NVCC

} // End namespace vecgeom
