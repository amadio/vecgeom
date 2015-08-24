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
// UTorus - Inherits from the VecGeom implementation.
//   Please note that there is an incomplete/incorrect implementation
//   of the USolids class UTorus, used only to fix compilation errors in
//   case of USolids usage in the context of Geant4 integration.
//
// Class description:
//
// 2015-08-19 Guilherme Lima - Including VecGeom torus implementation
//         Note: quick and dirty implementation (incomplete) for integration tests only.
//
// --------------------------------------------------------------------

#ifndef USOLIDS_UTorus
#define USOLIDS_UTorus

#ifdef VECGEOM_REPLACE_USOLIDS

//============== here for VecGeom-based implementation

#include "volumes/SpecializedTorus.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedTorus.h"
#include "base/Transformation3D.h"

class UTorus: public vecgeom::SimpleTorus {
  // just forwards UTorus to vecgeom::SimpleTorus
  using vecgeom::SimpleTorus::SimpleTorus;
};
//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation
#ifndef USOLIDS_VUSolid
#include "UUtils.hh"
#include "VUSolid.hh"
#endif

/// GL NOTE: this is not meant to be a COMPLETE implementation of UTorus!!!
class UTorus : public VUSolid {
public:
  UTorus() : VUSolid(), fRmin(0), fRmax(0), fRtor(0), fSphi(0), fDphi(0) { }
  UTorus(const std::string& name, double val1, double val2, double val3, double val4, double val5)
    : VUSolid(name)
    , fRmin(val1)
    , fRmax(val2)
    , fRtor(val3)
    , fSphi(val4)
    , fDphi(val5)
    { }

  virtual ~UTorus() {};

  // Copy constructor and assignment operator

  UTorus(const UTorus& rhs)
    : VUSolid(rhs.GetName() )
    , fRmin( rhs.GetRmin() )
    , fRmax( rhs.GetRmax() )
    , fRtor( rhs.GetRtor() )
    , fSphi( rhs.GetSPhi() )
    , fDphi( rhs.GetDPhi() )
    {  }

  UTorus& operator=(const UTorus& rhs) = delete;

  // Accessors and modifiers

  inline double GetRmin() const;
  inline double GetRmax() const;
  inline double GetRtor() const;
  inline double GetSPhi() const;
  inline double GetDPhi() const;

  void SetRmin(double arg);
  void SetRmax(double arg);
  void SetRtor(double arg);
  void SetSPhi(double arg);
  void SetDPhi(double arg);

  // Navigation methods
  EnumInside     Inside(const UVector3& aPoint) const;

  double  SafetyFromInside(const UVector3& /*aPoint*/,
                           bool /*aAccurate*/ = false) const {
    assert(false && "Not implemented.");
    return 0.;
  }

  double  SafetyFromOutside(const UVector3& /*aPoint*/,
                            bool /*aAccurate*/ = false) const {
    assert(false && "Not implemented.");
    return 0.;
  }

  double  DistanceToIn(const UVector3& /*aPoint*/,
                       const UVector3& /*aDirection*/,
                       // UVector3       &aNormalVector,
                       double /*aPstep*/ = UUtils::kInfinity) const {
    assert(false && "Not implemented.");
    return 0.;
  }

  double DistanceToOut(const UVector3& /*aPoint*/,
                       const UVector3& /*aDirection*/,
                       UVector3&       /*aNormalVector*/,
                       bool&           /*aConvex*/,
                       double /*aPstep*/ = UUtils::kInfinity) const {
    assert(false && "Not implemented.");
    return 0.;
  }

  bool Normal(const UVector3& /*aPoint*/, UVector3& /*aNormal*/) const {
    assert(false && "Not implemented.");
    return 0.;
  }

  void Extent(UVector3& aMin, UVector3& aMax) const {
    // Returns the full 3D cartesian extent of the solid.
    aMax.x() = fRtor+fRmax;
    aMax.y() = fRtor+fRmax;
    aMax.z() = fRmax;
    aMin = -aMax;
  }

  inline double Capacity();
  inline double SurfaceArea();

  VUSolid* Clone() const { return new UTorus(*this); }

  UGeometryType GetEntityType() const { return "Torus"; }

  std::ostream& StreamInfo( std::ostream& os ) const;

  void    ComputeBBox(UBBox* /*aBox*/, bool /*aStore = false*/) {}

  // Visualisation
  void GetParametersList(int, double* aArray) const {
    aArray[0] = GetRmin();
    aArray[1] = GetRmax();
    aArray[2] = GetRtor();
    aArray[3] = GetSPhi();
    aArray[4] = GetDPhi();
  }

  UVector3 GetPointOnSurface() const;
  UVector3 GetPointOnEdge() const;

private:
  double fRmin;  // Rmin of torus tube
  double fRmax;  // Rmax of torus tube
  double fRtor;  // Main radius of torus to center of the tube
  double fSphi;  // start phi angle
  double fDphi;  // delta phi angle
};

inline double UTorus::GetRmin() const {
  return fRmin;
}

inline double UTorus::GetRmax() const {
  return fRmax;
}

inline double UTorus::GetRtor() const {
  return fRtor;
}

inline double UTorus::GetSPhi() const {
  return fSphi;
}

inline double UTorus::GetDPhi() const {
  return fDphi;
}

inline void UTorus::SetRmin(double val) { fRmin = val; }
inline void UTorus::SetRmax(double val) { fRmax = val; }
inline void UTorus::SetRtor(double val) { fRtor = val; }
inline void UTorus::SetSPhi(double val) { fSphi = val; }
inline void UTorus::SetDPhi(double val) { fDphi = val; }

//============== end of USolids-based implementation

#endif  // VECGEOM_REPLACE_USOLIDS
#endif  // USOLIDS_UTorus
