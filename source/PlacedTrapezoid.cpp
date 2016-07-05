/// @file PlacedTrapezoid.cpp
/// @author Guilherme Lima (lima at fnal dot gov)

#include "volumes/PlacedTrapezoid.h"
#include "volumes/Trapezoid.h"
#include "volumes/PlacedBox.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTrap.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Trap.hh"
#endif

#endif // VECGEOM_NVCC

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

PlacedTrapezoid::~PlacedTrapezoid()
{
}

#ifndef VECGEOM_NVCC

VPlacedVolume const *PlacedTrapezoid::ConvertToUnspecialized() const
{
  return new SimpleTrapezoid(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTrapezoid::ConvertToRoot() const
{
  return new TGeoTrap(GetLabel().c_str(), GetZHalfLength(), GetTheta() * kRadToDeg, GetPhi() * kRadToDeg,
                      GetYHalfLength1(), GetXHalfLength1(), GetXHalfLength2(), GetAlpha1() * kRadToDeg,
                      GetYHalfLength2(), GetXHalfLength3(), GetXHalfLength4(), GetAlpha2() * kRadToDeg);
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedTrapezoid::ConvertToUSolids() const
{
  const UnplacedTrapezoid &unp = *(GetUnplacedVolume());
  return new ::UTrap(GetLabel().c_str(), unp.GetDz(), unp.GetTheta(), unp.GetPhi(), unp.GetDy1(), unp.GetDx1(),
                     unp.GetDx2(), unp.GetAlpha1(), unp.GetDy2(), unp.GetDx3(), unp.GetDx4(), unp.GetAlpha2());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTrapezoid::ConvertToGeant4() const
{
  const UnplacedTrapezoid &unp = *(GetUnplacedVolume());
  return new G4Trap(GetLabel().c_str(), unp.GetDz(), unp.GetTheta(), unp.GetPhi(), unp.GetDy1(), unp.GetDx1(),
                    unp.GetDx2(), unp.GetAlpha1(), unp.GetDy2(), unp.GetDx3(), unp.GetDx4(), unp.GetAlpha2());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTrapezoid)

#endif // VECGEOM_NVCC

// function provided for compatibility with USolids
void PlacedTrapezoid::SetAllParameters(double dz, double theta, double phi, double dy1, double dx1, double dx2,
                                       double alp1, double dy2, double dx3, double dx4, double alp2)
{
  // mark unused
  (void)theta;
  (void)phi;
  (void)alp1;
  (void)alp2;

  UnplacedTrapezoid &utrap = *const_cast<UnplacedTrapezoid *>(GetUnplacedVolume());
  double mm                = 0.1; // to cm
  utrap.fDz                = dz * mm;
  utrap.fDy1               = dy1 * mm;
  utrap.fDy2               = dy2 * mm;
  utrap.fDx1               = dx1 * mm;
  utrap.fDx2               = dx2 * mm;
  utrap.fDx3               = dx3 * mm;
  utrap.fDx4               = dx4 * mm;
  utrap.fTanAlpha1         = dz * mm;
  utrap.fTanAlpha2         = dz * mm;
}

/*
void PlacedTrapezoid::ComputeBoundingBox() {
  Vector3D<Precision> aMin, aMax;
  GetUnplacedVolume()->Extent(aMin, aMax) ;

  // try a box with no rotation
  Vector3D<Precision> bbdims1 = 0.5*(aMax-aMin);
  Vector3D<Precision> center1 = 0.5*(aMax+aMin);
  UnplacedBox *box1 = new UnplacedBox(bbdims1);
  Precision vol1 = box1->volume();

  // try a box with a rotation by theta,phi
  Transformation3D* matrix2 =
    new Transformation3D(center1.x(), center1.y(), center1.z(),
                         this->GetTheta(), this->GetPhi(), 0);
  Vector3D<Precision> newMin, newMax;
  matrix2->Transform(aMin, newMin);
  matrix2->Transform(aMax, newMax);
  UnplacedBox *box2 = new UnplacedBox(0.5*(newMax-newMin));
  Precision vol2 = box2->volume();

  if(vol2>0.5*vol1) {
    // use box1
    bounding_box_ =
      new PlacedBox(new LogicalVolume(box1),
                    new Transformation3D(center1.x(), center1.y(), center1.z()),
                    SimpleBox(box1));
    delete box2, matrix2;
  }
  else {
    // use box2
    bounding_box_ = new PlacedBox(new LogicalVolume(box2), matrix2, 0);
    delete box1;
  }
*/

} // End global namespace
