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
// UParaboloid - Inherits from the VecGeom implementation.
//
//   Please note that there is an incomplete/incorrect implementation
//   of the USolids class UParaboloid, used only to fix compilation errors in
//   case of USolids usage in the context of Geant4 integration.
//
// Class description:
//
// 22.09.14 J. de Fine Licht
//          Including VecGeom paraboloid implementation by Marilena
//          Bandieramonte.
// 2015-08-20 Guilherme Lima - Integration with Geant4
//         Note: quick and dirty implementation (incomplete) for integration tests only.
//
// --------------------------------------------------------------------

#ifndef USOLIDS_UParaboloid
#define USOLIDS_UParaboloid

#ifdef VECGEOM_REPLACE_USOLIDS

//============== here for VecGeom-based implementation

#include "volumes/SpecializedParaboloid.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedParaboloid.h"
#include "base/Transformation3D.h"

class UParaboloid: public vecgeom::SimpleParaboloid {
  // just forwards UParaboloid to vecgeom::SimpleParaboloid
  using vecgeom::SimpleParaboloid::SimpleParaboloid;
};
//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation
#ifndef USOLIDS_VUSolid
#include "UUtils.hh"
#include "VUSolid.hh"
#endif

/// GL NOTE: this is not meant to be a COMPLETE implementation of UParaboloid!!!
class UParaboloid : public VUSolid {
public:
  UParaboloid() : VUSolid(), fDz(0), fRhi(0), fRlo(0) { }
  UParaboloid(const std::string& name, double val1, double val2, double val3)
    : VUSolid(name)
    , fDz(val1)
    , fRhi(val2)
    , fRlo(val3)
    { }

  ~UParaboloid() {}

  // Copy constructor and assignment operator

  UParaboloid(const UParaboloid& rhs)
    : VUSolid(rhs.GetName() )
    , fDz( rhs.GetDz() )
    , fRhi( rhs.GetRhi() )
    , fRlo( rhs.GetRlo() )
    {  }

  UParaboloid& operator=(const UParaboloid& rhs) = delete;

  // Accessors and modifiers

  inline double GetDz() const;
  inline double GetRhi() const;
  inline double GetRlo() const;

  void SetDz(double arg);
  void SetRhi(double arg);
  void SetRlo(double arg);

  // Navigation methods
  EnumInside     Inside(const UVector3& /*aPoint*/) const {
    assert(false && "Not implemented.");
    return EnumInside::eInside;
  }

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
    return false;
  }

  void Extent(UVector3& aMin, UVector3& aMax) const {
    // Returns the full 3D cartesian extent of the solid.
    aMax.x() = fRhi;
    aMax.y() = fRhi;
    aMax.z() = fDz;
    aMin = -aMax;
  }

    // Computes capacity of the shape in [length^3]
  double Capacity() {
    return UUtils::kPi*fDz*(fRlo*fRlo+fRhi*fRhi);
  }

  inline double SurfaceArea();

  VUSolid* Clone() const { return new UParaboloid(*this); }

  UGeometryType GetEntityType() const { return "Paraboloid"; }

  std::ostream& StreamInfo( std::ostream& os ) const;

  void    ComputeBBox(UBBox* /*aBox*/, bool /*aStore = false*/) {}

  // Visualisation
  void GetParametersList(int, double* aArray) const {
    aArray[0] = GetDz();
    aArray[1] = GetRhi();
    aArray[2] = GetRlo();
  }

  UVector3 GetPointOnSurface() const;
  UVector3 GetPointOnEdge() const;

private:
  double fDz;   // Half-length
  double fRhi;  // external radius
  double fRlo;  // internal radius
  double fA, fB;
};

inline double UParaboloid::GetDz() const {
  return fDz;
}

inline double UParaboloid::GetRhi() const {
  return fRhi;
}

inline double UParaboloid::GetRlo() const {
  return fRlo;
}

inline void UParaboloid::SetDz(double val)  { fDz = val; }
inline void UParaboloid::SetRhi(double val) { fRhi = val; }
inline void UParaboloid::SetRlo(double val) { fRlo = val; }

//============== end of USolids-based implementation

#endif  // VECGEOM_REPLACE_USOLIDS
#endif  // USOLIDS_UParaboloid
