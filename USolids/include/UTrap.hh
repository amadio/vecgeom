//
// ********************************************************************
// * This Software is part of the AIDA Unified Solids Library package *
// * See: https://aidasoft.web.cern.ch/USolids                        *
// ********************************************************************
//
// $Id:$
//
// --------------------------------------------------------------------
//
// UTrap
//
// Class description:
//
// A UTrap is a general trapezoid: The faces perpendicular to the
// z planes are trapezia, and their centres are not necessarily on
// a line parallel to the z axis.
//
// Note that of the 11 parameters described below, only 9 are really
// independent - a check for planarity is made in the calculation of the
// equation for each plane. If the planes are not parallel, a call to
// UException is made.
//
//      pDz    Half-length along the z-axis
//      pTheta  Polar angle of the line joining the centres of the faces
//              at -/+pDz
//      pPhi    Azimuthal angle of the line joing the centre of the face at
//              -pDz to the centre of the face at +pDz
//      pDy1    Half-length along y of the face at -pDz
//      pDx1    Half-length along x of the side at y=-pDy1 of the face at -pDz
//      pDx2    Half-length along x of the side at y=+pDy1 of the face at -pDz
//      pAlp1  Angle with respect to the y axis from the centre of the side
//              at y=-pDy1 to the centre at y=+pDy1 of the face at -pDz
//
//      pDy2    Half-length along y of the face at +pDz
//      pDx3    Half-length along x of the side at y=-pDy2 of the face at +pDz
//      pDx4    Half-length along x of the side at y=+pDy2 of the face at +pDz
//      pAlp2  Angle with respect to the y axis from the centre of the side
//              at y=-pDy2 to the centre at y=+pDy2 of the face at +pDz
//
//   Member Data:
//
//      fDz    Half-length along the z axis
//      fTthetaCphi = std::tan(pTheta)*std::cos(pPhi)
//      fTthetaSphi = std::tan(pTheta)*std::sin(pPhi)
//      These combinations are suitable for creation of the trapezoid corners
//
//      fDy1    Half-length along y of the face at -fDz
//      fDx1    Half-length along x of the side at y=-fDy1 of the face at -fDz
//      fDx2    Half-length along x of the side at y=+fDy1 of the face at -fDz
//      fTalpha1   Tan of Angle with respect to the y axis from the centre of
//                 the side at y=-fDy1 to the centre at y=+fDy1 of the face
//                 at -fDz
//
//      fDy2    Half-length along y of the face at +fDz
//      fDx3    Half-length along x of the side at y=-fDy2 of the face at +fDz
//      fDx4    Half-length along x of the side at y=+fDy2 of the face at +fDz
//      fTalpha2   Tan of Angle with respect to the y axis from the centre of
//                 the side at y=-fDy2 to the centre at y=+fDy2 of the face
//                 at +fDz
//
//      UTrapSidePlane fPlanes[4]  Plane equations of the faces not at +/-fDz
//                                 NOTE: order is important !!!
//
// 12.02.13 Marek Gayer
//          Created from original implementation in Geant4
// 19.07.15 Guilherme Lima - Add VecGeom implementation as option for underlying implementation
//
// --------------------------------------------------------------------

#ifndef UTrap_HH
#define UTrap_HH

#ifdef VECGEOM_REPLACE_USOLIDS

//============== here for VecGeom-based implementation

#include "volumes/SpecializedTrapezoid.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedTrapezoid.h"
#include "base/Transformation3D.h"
#include "VecCore/VecMath.h"

struct UTrapSidePlane {
  double a, b, c, d; // Normal Unit vector (a,b,c) and offset (d)
                     // => ax + by + cz + d = 0
  // #ifdef VECGEOM_PLANESHELL_DISABLE
  // UTrapSidePlane(const TrapezoidStruct<double>::TrapSidePlane &oth)
  // {
  //   this->a = oth.fA;
  //   this->b = oth.fB;
  //   this->c = oth.fC;
  //   this->d = oth.fD;
  // }
  // #endif
};

#include "volumes/UnplacedTrapezoid.h"
#include "volumes/USolidsAdapter.h"

class UTrap : public vecgeom::USolidsAdapter<vecgeom::UnplacedTrapezoid> {

  // just forwards UTrap to vecgeom trapezoid
  using Shape_t = vecgeom::UnplacedTrapezoid;
  using Base_t  = vecgeom::USolidsAdapter<vecgeom::UnplacedTrapezoid>;

  // inherit all constructors
  using Base_t::Base_t;

public:
  using Base_t::GetTanAlpha1;
  using Base_t::GetTanAlpha2;

  // add default constructor for tests
  UTrap() : Base_t("", 0., 0., 0) {}
  virtual ~UTrap() {}

  VECGEOM_FORCE_INLINE
  double GetZHalfLength() const { return static_cast<Shape_t const *>(this)->GetDz(); }

  VECGEOM_FORCE_INLINE
  double GetYHalfLength1() const { return static_cast<Shape_t const *>(this)->GetDy1(); }

  VECGEOM_FORCE_INLINE
  double GetXHalfLength1() const { return static_cast<Shape_t const *>(this)->GetDx1(); }

  VECGEOM_FORCE_INLINE
  double GetXHalfLength2() const { return static_cast<Shape_t const *>(this)->GetDx2(); }

  VECGEOM_FORCE_INLINE
  double GetYHalfLength2() const { return static_cast<Shape_t const *>(this)->GetDy2(); }

  VECGEOM_FORCE_INLINE
  double GetXHalfLength3() const { return static_cast<Shape_t const *>(this)->GetDx3(); }

  VECGEOM_FORCE_INLINE
  double GetXHalfLength4() const { return static_cast<Shape_t const *>(this)->GetDx4(); }

  VECGEOM_FORCE_INLINE
  double GetThetaCphi() const { return static_cast<Shape_t const *>(this)->GetTanThetaCosPhi(); }

  VECGEOM_FORCE_INLINE
  double GetThetaSphi() const { return static_cast<Shape_t const *>(this)->GetTanThetaSinPhi(); }

  UTrapSidePlane GetSidePlane(int i)
  {
    vecgeom::UnplacedTrapezoid const *vgtrap = (Shape_t *)this;
    UTrapSidePlane plane;
    plane.a = vgtrap->GetStruct().GetPlane(i).fA;
    plane.b = vgtrap->GetStruct().GetPlane(i).fB;
    plane.c = vgtrap->GetStruct().GetPlane(i).fC;
    plane.d = vgtrap->GetStruct().GetPlane(i).fD;
    return plane;
  }

  UVector3 GetSymAxis() const
  {
    vecgeom::UnplacedTrapezoid const *vgtrap = (Shape_t *)this;
    double tanThetaSphi                      = vgtrap->GetTanThetaSinPhi();
    double tanThetaCphi                      = vgtrap->GetTanThetaCosPhi();
    double tan2Theta                         = tanThetaSphi * tanThetaSphi + tanThetaCphi * tanThetaCphi;
    double cosTheta                          = 1.0 / vecCore::math::Sqrt(1 + tan2Theta);
    return UVector3(tanThetaCphi * cosTheta, tanThetaSphi * cosTheta, cosTheta);
  }

  void SetPlanes(const UVector3 pt[8])
  {
    vecgeom::UnplacedTrapezoid *vgtrap = static_cast<Shape_t *>(this);
    // std::cout<<"PlacedTrap.h: sizeof's for: pt="<< sizeof(pt)
    //          <<", opt[8]="<< sizeof(pt[8])
    //          <<", Vec3D: "<< sizeof(Vector3D<vecgeom::Precision>) <<"\n";
    if (sizeof(pt[8]) == 8 * sizeof(vecgeom::Vector3D<vecgeom::Precision>)) {
      vgtrap->fromCornersToParameters(pt);
    } else {
      // just in case Precision is float
      vecgeom::Vector3D<vecgeom::Precision> vgpt[8];
      for (unsigned i = 0; i < 8; ++i) {
        vgpt[i].Set(pt[i].x(), pt[i].y(), pt[i].z());
      }
      vgtrap->fromCornersToParameters(vgpt);
    }
  }

  // provide a new object which is a clone of the original solid
  VUSolid *Clone() const override
  {
    return new UTrap(GetName().c_str(), dz(), theta(), phi(), dy1(), dx1(), dx2(), tanAlpha1(), dy2(), dx3(), dx4(),
                     tanAlpha2());
  }

  void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) override {}

  UGeometryType GetEntityType() const override { return "UTrap"; }

  // Visualisation
  void GetParametersList(int, double *aArray) const override
  {
    aArray[0]  = dz();
    aArray[1]  = theta();
    aArray[2]  = phi();
    aArray[3]  = dy1();
    aArray[4]  = dx1();
    aArray[5]  = dx2();
    aArray[6]  = tanAlpha1();
    aArray[7]  = dy2();
    aArray[8]  = dx3();
    aArray[9]  = dx4();
    aArray[10] = tanAlpha2();
  }

  void SetAllParameters(double pDz, double pTheta, double pPhi, double pDy1, double pDx1, double pDx2, double pAlp1,
                        double pDy2, double pDx3, double pDx4, double pAlp2)
  {
    // const double _mm                   = 0.1; // conversion factor from Geant4 default (mm) to VecGeom's (cm)
    vecgeom::UnplacedTrapezoid *vgtrap = (Shape_t *)this;
    vgtrap->SetDz(pDz);
    vgtrap->SetDy1(pDy1);
    vgtrap->SetDy2(pDy2);
    vgtrap->SetDx1(pDx1);
    vgtrap->SetDx2(pDx2);
    vgtrap->SetDx3(pDx3);
    vgtrap->SetDx4(pDx4);
    vgtrap->SetTanAlpha1(vecCore::math::Tan(pAlp1));
    vgtrap->SetTanAlpha1(vecCore::math::Tan(pAlp2));
    // last two will also reset cached variables
    vgtrap->SetTheta(pTheta);
    vgtrap->SetPhi(pPhi);
  }

  std::ostream &StreamInfo(std::ostream &os) const override
  {
    using vecgeom::cxx::kRadToDeg;
    int oldprc = os.precision(16);
    os << "-----------------------------------------------------------\n"
       << "     *** Dump for solid - " << GetEntityType() << " ***\n"
       << "     ===================================================\n"
       << " Solid type: Trapezoid\n"
       << " Parameters: \n"
       << "     half lengths X1-X4: " << dx1() << "mm, " << dx2() << "mm, " << dx3() << "mm, " << dx4() << "mm\n"
       << "     half lengths Y1,Y2: " << dy1() << "mm, " << dy2() << "mm\n"
       << "     half length Z: " << dz() << "mm\n"
       << "     Solid axis angles: Theta=" << theta() * kRadToDeg << "deg, "
       << " Phi=" << phi() * kRadToDeg << "deg\n"
       << "     Face axis angles: TanAlpha1=" << tanAlpha1() * kRadToDeg << "deg, "
       << " TanAlpha2=" << tanAlpha2() * kRadToDeg << "deg\n";
    os << "-----------------------------------------------------------\n";
    os.precision(oldprc);
    return os;
  }
};
//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation
#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

struct UTrapSidePlane {
  double a, b, c, d; // Normal Unit vector (a,b,c) and offset (d)
};

class UTrap : public VUSolid {

public: // with description
  UTrap(const std::string &pName, double pDz, double pTheta, double pPhi, double pDy1, double pDx1, double pDx2,
        double pAlp1, double pDy2, double pDx3, double pDx4, double pAlp2);
  //
  // The most general constructor for UTrap which prepares plane
  // equations and corner coordinates from parameters

  UTrap(const std::string &pName, const UVector3 pt[8]);
  //
  // Prepares plane equations and parameters from corner coordinates

  UTrap(const std::string &pName, double pZ, double pY, double pX, double pLTX);
  //
  // Constructor for Right Angular Wedge from STEP (assumes pLTX<=pX)

  UTrap(const std::string &pName, double pDx1, double pDx2, double pDy1, double pDy2, double pDz);
  //
  // Constructor for UTrd

  UTrap(const std::string &pName, double pDx, double pDy, double pDz, double pAlpha, double pTheta, double pPhi);
  //
  // Constructor for UPara

  UTrap(const std::string &pName);
  //
  // Constructor for "nominal" UTrap whose parameters are to be Set
  // by a UVPVParamaterisation later

  virtual ~UTrap();
  //
  // Destructor

  // Accessors

  inline double GetZHalfLength() const;
  inline double GetYHalfLength1() const;
  inline double GetXHalfLength1() const;
  inline double GetXHalfLength2() const;
  inline double GetTanAlpha1() const;
  inline double GetYHalfLength2() const;
  inline double GetXHalfLength3() const;
  inline double GetXHalfLength4() const;
  inline double GetTanAlpha2() const;
  //
  // Returns coordinates of Unit vector along straight
  // line joining centers of -/+fDz planes

  inline UTrapSidePlane GetSidePlane(int n) const;
  inline UVector3 GetSymAxis() const;

  // Modifiers

  void SetAllParameters(double pDz, double pTheta, double pPhi, double pDy1, double pDx1, double pDx2, double pAlp1,
                        double pDy2, double pDx3, double pDx4, double pAlp2);

  void SetPlanes(const UVector3 pt[8]);

  // Methods for solid

  inline double Capacity();
  inline double SurfaceArea();

  VUSolid::EnumInside Inside(const UVector3 &p) const;

  UVector3 SurfaceNormal(const UVector3 &p) const;

  bool Normal(const UVector3 &aPoint, UVector3 &aNormal) const;

  double DistanceToIn(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;

  double SafetyFromOutside(const UVector3 &p, bool precise = false) const;

  double DistanceToOut(const UVector3 &p, const UVector3 &v, UVector3 &aNormalVector, bool &aConvex,
                       double aPstep = UUtils::kInfinity) const;

  double SafetyFromInside(const UVector3 &p, bool precise = false) const;

  UGeometryType GetEntityType() const;

  UVector3 GetPointOnSurface() const;

  VUSolid *Clone() const;

  virtual void Extent(UVector3 &aMin, UVector3 &aMax) const;

  std::ostream &StreamInfo(std::ostream &os) const;

  // Visualisation functions

public: // without description
  UTrap(const UTrap &rhs);
  UTrap &operator=(const UTrap &rhs);
  // Copy constructor and assignment operator.

  inline double GetThetaCphi() const;
  inline double GetThetaSphi() const;

protected: // with description
  bool MakePlanes();
  bool MakePlane(const UVector3 &p1, const UVector3 &p2, const UVector3 &p3, const UVector3 &p4, UTrapSidePlane &plane);

private:
  UVector3 ApproxSurfaceNormal(const UVector3 &p) const;
  // Algorithm for SurfaceNormal() following the original
  // specification for points not on the surface

  inline double GetFaceArea(const UVector3 &p1, const UVector3 &p2, const UVector3 &p3, const UVector3 &p4);
  //
  // Provided four corners of plane in clockwise fashion,
  // it returns the area of finite face

  UVector3 GetPointOnPlane(UVector3 p0, UVector3 p1, UVector3 p2, UVector3 p3, double &area) const;
  //
  // Returns a random point on the surface of one of the faces

  void GetParametersList(int /*aNumber*/, double * /*aArray*/) const {}

  void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

private:
  double fDz, fTthetaCphi, fTthetaSphi;
  double fDy1, fDx1, fDx2, fTalpha1;
  double fDy2, fDx3, fDx4, fTalpha2;
  UTrapSidePlane fPlanes[4];

  double fCubicVolume;
  double fSurfaceArea;
};

#include "UTrap.icc"

//============== end of USolids-based implementation

#endif // VECGEOM_REPLACE_USOLIDS
#endif // UTrap_HH
