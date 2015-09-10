/*
 * @file   volumes/PlacedTrapezoid.h
 * @author Guilherme Lima (lima 'at' fnal 'dot' gov)
 *
 * 2014-05-01 - Created, based on the Parallelepiped draft
 */

#ifndef VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedTrapezoid.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedTrapezoid; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedTrapezoid )

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrapezoid : public VPlacedVolume {

public:
  typedef UnplacedTrapezoid UnplacedShape_t;
  using TrapSidePlane = vecgeom::UnplacedTrapezoid::TrapSidePlane;

#ifndef VECGEOM_NVCC

  PlacedTrapezoid(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  // PlacedTrapezoid(LogicalVolume const *const logical_volume,
  //                 Transformation3D const *const transformation,
  //                 PlacedBox const *const boundingBox)
  //   : PlacedTrapezoid("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTrapezoid(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTrapezoid();

  /// Accessors
  /// @{

  /* Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced trapezoid.
   */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid const* GetUnplacedVolume() const {
    return static_cast<UnplacedTrapezoid const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }


  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZHalfLength() const { return GetUnplacedVolume()->GetDz(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetYHalfLength1() const { return GetUnplacedVolume()->GetDy1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetXHalfLength1() const { return GetUnplacedVolume()->GetDx1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetXHalfLength2() const { return GetUnplacedVolume()->GetDx2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha1() const { return GetUnplacedVolume()->GetTanAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetYHalfLength2() const { return GetUnplacedVolume()->GetDy2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetXHalfLength3() const { return GetUnplacedVolume()->GetDx3(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetXHalfLength4() const { return GetUnplacedVolume()->GetDx4(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha2() const { return GetUnplacedVolume()->GetTanAlpha2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha1() const { return GetUnplacedVolume()->GetAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha2() const { return GetUnplacedVolume()->GetAlpha2(); }

  VECGEOM_CUDA_HEADER_BOTH
  double GetThetaCphi() const {
      return (double)GetUnplacedVolume()->GetTanThetaCosPhi();
  }

  VECGEOM_CUDA_HEADER_BOTH
  double GetThetaSphi() const {
      return (double)GetUnplacedVolume()->GetTanThetaSinPhi();
  }


  // This is specifically designed for Geant4 - input parameters in mm
  VECGEOM_CUDA_HEADER_BOTH
  void SetAllParameters(double dz, double theta, double phi,
                        double dy1, double dx1, double dx2, double alp1,
                        double dy2, double dx3, double dx4, double alp2);

  VECGEOM_CUDA_HEADER_BOTH
  PlacedTrapezoid(double dx, double dy, double dz, double);

  VECGEOM_CUDA_HEADER_BOTH
  void SetPlanes(const UVector3 upt[8]) {
    vecgeom::UnplacedTrapezoid* utrap = const_cast<vecgeom::UnplacedTrapezoid*>(GetUnplacedVolume());
    if(sizeof(upt)==8*sizeof(Vector3D<Precision>)) {
      utrap->fromCornersToParameters(upt);
    }
    else {
      // just in case Precision is float
      Vector3D<Precision> pt[8];
      for(unsigned i=0; i<8; ++i) {
        pt[i] = Vector3D<Precision>(upt[i].x(), upt[i].y(), upt[i].z());
      }
      utrap->fromCornersToParameters(pt);
    }
  }

  VECGEOM_CUDA_HEADER_BOTH
  TrapSidePlane GetSidePlane(int n) const {

    TrapSidePlane sidePlane;

#ifndef VECGEOM_PLANESHELL_DISABLE
      using TrapPlanes = vecgeom::UnplacedTrapezoid::Planes;
      TrapPlanes const* planes = GetUnplacedVolume()->GetPlanes();
      sidePlane.fA = (double)planes->fA[n];
      sidePlane.fB = (double)planes->fA[n];
      sidePlane.fC = (double)planes->fC[n];
      sidePlane.fD = (double)planes->fD[n];
#else
      using TrapSidePlane = vecgeom::UnplacedTrapezoid::TrapSidePlane;
      TrapSidePlane const* planes = GetUnplacedVolume()->GetPlanes();
      sidePlane.fA = (double)(planes[n].fA);
      sidePlane.fB = (double)(planes[n].fB);
      sidePlane.fC = (double)(planes[n].fC);
      sidePlane.fD = (double)(planes[n].fD);
#endif
      return sidePlane;
  }

  UVector3 GetSymAxis() const {
    vecgeom::UnplacedTrapezoid const* utrap = GetUnplacedVolume();
    double tanThetaSphi = utrap->GetTanThetaSinPhi();
    double tanThetaCphi = utrap->GetTanThetaCosPhi();
    double tan2Theta = tanThetaSphi*tanThetaSphi + tanThetaCphi*tanThetaCphi;
    double cosTheta = 1.0 / std::sqrt(1 + tan2Theta);
    return UVector3( tanThetaCphi*cosTheta, tanThetaSphi*cosTheta, cosTheta );
  }

#ifndef VECGEOM_NVCC
  virtual
  Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual
  Precision SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea();}

  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const override {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  Vector3D<Precision>  GetPointOnSurface() const override {
    return GetUnplacedVolume()->GetPointOnSurface();
  }

#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType() ;}
#endif
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBoundingBox();

  VECGEOM_CUDA_HEADER_BOTH
  void GetParameterList() const { return GetUnplacedVolume()->GetParameterList() ;}

#if defined(VECGEOM_USOLIDS)
//  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& StreamInfo(std::ostream &os) const override {
    return GetUnplacedVolume()->StreamInfo(os);
  }
#endif

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const* ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override;
#endif
#endif // VECGEOM_NVCC

protected:

  // static PlacedBox* make_bounding_box(LogicalVolume const *const logical_volume,
  //                                     Transformation3D const *const transformation) {

  //   UnplacedTrapezoid const *const utrap = static_cast<UnplacedTrapezoid const *const>(logical_volume->GetUnplacedVolume());
  //   UnplacedBox const *const unplaced_bbox = new UnplacedBox(
  //     std::max(std::max(utrap->GetDx1(),utrap->GetDx2()),std::max(utrap->GetDx3(),utrap->GetDx4())),
  //     std::max(utrap->GetDy1(),utrap->GetDy2()), utrap->GetDz());
  //   LogicalVolume const *const box_volume =  new LogicalVolume(unplaced_bbox);
  //   return new PlacedBox(box_volume, transformation);
  // }

}; // end of class PlacedTrapezoid

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
