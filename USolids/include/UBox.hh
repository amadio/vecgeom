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
// UBox
//
// Class description:
//
//  A simple box defined by half-lengths on the three axis.
//  The center of the box matches the origin of the local reference frame.
//
// 10.06.11 J.Apostolakis, G.Cosmo, A.Gheata
//          Created from original implementation in Geant4 and ROOT
// 30.06.15 Guilherme Lima - Add VecGeom implementation as option for underlying implementation
//
// --------------------------------------------------------------------

#ifndef USOLIDS_UBox
#define USOLIDS_UBox

#ifdef VECGEOM_REPLACE_USOLIDS

//============== here for VecGeom-based implementation

#include "volumes/SpecializedBox.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedBox.h"
#include "base/Transformation3D.h"

#ifndef VECGEOM_MASTER
#include "volumes/UnplacedBox.h"
#include "volumes/USolidsAdapter.h"

class UBox: public vecgeom::USolidsAdapter<vecgeom::UnplacedBox> {

  // just forwards UBox to vecgeom box
  using Shape_t = vecgeom::UnplacedBox;
  using Base_t = vecgeom::USolidsAdapter<vecgeom::UnplacedBox>;

  // inherit all constructors
  using Base_t::Base_t;
public:
  // add default constructor for tests
  UBox() : Base_t("", 0.,0.,0) {}
  virtual ~UBox() {}

  inline double GetXHalfLength() const { return x(); }

  inline double GetYHalfLength() const { return y(); }

  inline double GetZHalfLength() const { return z(); }

  inline void SetXHalfLength(double dx) { SetX(dx); }
  inline void SetYHalfLength(double dy) { SetY(dy); }

  inline void SetZHalfLength(double dz) { SetZ(dz); }

  inline void Set(double xx, double yy, double zz) {
    SetX(xx);
    SetY(yy);
    SetZ(zz);
  }

  // o provide a new object which is a clone of the solid
  VUSolid* Clone() const override { return new UBox(*this); }

  void ComputeBBox(UBBox* /*aBox*/, bool /*aStore = false*/) override {}

  UGeometryType GetEntityType() const override { return "UBox"; }

  // Visualisation
  void GetParametersList(int, double* aArray) const override
  {
    aArray[0] = GetXHalfLength();
    aArray[1] = GetYHalfLength();
    aArray[2] = GetZHalfLength();
  }

  std::ostream& StreamInfo(std::ostream &os) const override {
    int oldprc = os.precision(16);
    os << "-----------------------------------------------------------\n"
       << "     *** Dump for solid - " << GetEntityType() << " ***\n"
       << "     ===================================================\n"
       << " Solid type: Box\n"
       << " Parameters: \n"
       << "     half-dimensions in mm: x,y,z: " << dimensions() <<"\n"
       << "-----------------------------------------------------------\n";
    os.precision(oldprc);
    return os;
  }
};

#else

class UBox: public vecgeom::SpecializedBox<vecgeom::translation::kIdentity, vecgeom::rotation::kIdentity> {
  // just forwards UBox to vecgeom box
  typedef typename vecgeom::SpecializedBox<vecgeom::translation::kIdentity, vecgeom::rotation::kIdentity> Shape_t;
  // inherit all constructors
  using Shape_t::Shape_t;

public:
  // add default constructor for tests
  UBox() : Shape_t(new vecgeom::LogicalVolume(new vecgeom::UnplacedBox(0.,0.,0.)),
                   &vecgeom::Transformation3D::kIdentity,
                   this) {}
};
#endif
//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation
#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

class UBox : public VUSolid
{

  public:
  UBox() : VUSolid(), fDx(0), fDy(0), fDz(0),fCubicVolume(0.), fSurfaceArea(0.) {}
    UBox(const std::string& name, double dx, double dy, double dz);
    virtual ~UBox();

    UBox(const UBox& rhs);
    UBox& operator=(const UBox& rhs);

    // Copy constructor and assignment operator

    void Set(double dx, double dy, double dz);
    void Set(const UVector3& vec);

    // Accessors and modifiers

    inline double GetXHalfLength() const;
    inline double GetYHalfLength() const;
    inline double GetZHalfLength() const;

    void SetXHalfLength(double dx);
    void SetYHalfLength(double dy);
    void SetZHalfLength(double dz);


    // Navigation methods
    EnumInside     Inside(const UVector3& aPoint) const;

    double  SafetyFromInside(const UVector3& aPoint,
                             bool aAccurate = false) const;
    double  SafetyFromOutside(const UVector3& aPoint,
                              bool aAccurate = false) const;
    double  DistanceToIn(const UVector3& aPoint,
                         const UVector3& aDirection,
                         // UVector3       &aNormalVector,

                         double aPstep = UUtils::kInfinity) const;

    double DistanceToOut(const UVector3& aPoint,
                         const UVector3& aDirection,
                         UVector3&       aNormalVector,
                         bool&           aConvex,
                         double aPstep = UUtils::kInfinity) const;
    bool Normal(const UVector3& aPoint, UVector3& aNormal) const;
//  void Extent ( EAxisType aAxis, double &aMin, double &aMax ) const;
    void Extent(UVector3& aMin, UVector3& aMax) const;
    inline double Capacity();
    inline double SurfaceArea();
    VUSolid* Clone() const;
    UGeometryType GetEntityType() const;
   
    void    ComputeBBox(UBBox* /*aBox*/, bool /*aStore = false*/) {}

    // Visualisation
    void GetParametersList(int, double* aArray) const
    {
      aArray[0] = GetXHalfLength();
      aArray[1] = GetYHalfLength();
      aArray[2] = GetZHalfLength();
    }

    UVector3 GetPointOnSurface() const;
    UVector3 GetPointOnEdge() const;

    std::ostream& StreamInfo(std::ostream& os) const;


  private:
    double                fDx;   // Half-length on X
    double                fDy;   // Half-length on Y
    double                fDz;   // Half-length on Z
    double       fCubicVolume;   // Cubic Volume
    double       fSurfaceArea;   // Surface Area

};


inline double UBox::GetXHalfLength() const
{
  return fDx;
}
inline double UBox::GetYHalfLength() const
{
  return fDy;
}
inline double UBox::GetZHalfLength() const
{
  return fDz;
}

inline double UBox::Capacity()
{
  if (fCubicVolume != 0.)
  {
    ;
  }
  else
  {
    fCubicVolume = 8 * fDx * fDy * fDz;
  }
  return fCubicVolume;
}

inline double UBox::SurfaceArea()
{
  if (fSurfaceArea != 0.)
  {
    ;
  }
  else
  {
    fSurfaceArea = 8 * (fDx * fDy + fDx * fDz + fDy * fDz);
  }
  return fSurfaceArea;
}
//============== end of USolids-based implementation

#endif  // VECGEOM_REPLACE_USOLIDS
#endif  // USOLIDS_UBox
