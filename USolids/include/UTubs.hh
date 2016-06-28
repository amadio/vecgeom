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
// UTubs
//
// Class description:
//
//   A tube or tube segment with curved sides parallel to
//   the z-axis. The tube has a specified half-length along
//   the z-axis, about which it is centered, and a given
//   minimum and maximum radius. A minimum radius of 0
//   corresponds to filled tube /cylinder. The tube segment is
//   specified by starting and delta angles for phi, with 0
//   being the +x axis, PI/2 the +y axis.
//   A delta angle of 2PI signifies a complete, unsegmented
//   tube/cylinder.
//
//   Member Data:
//
//   fRMin  Inner radius
//   fRMax  Outer radius
//   fDz  half length in z
//
//   fSPhi  The starting phi angle in radians,
//          adjusted such that fSPhi+fDPhi<=2PI, fSPhi>-2PI
//
//   fDPhi  Delta angle of the segment.
//
//   fPhiFullTube  Boolean variable used for indicate the Phi Section
//
// 19.10.12 Marek Gayer
//          Created from original implementation in Geant4
// 29.07.15 Guilherme Lima - Add VecGeom implementation as option for underlying implementation
// --------------------------------------------------------------------

#ifndef UTUBS_HH
#define UTUBS_HH

#ifdef VECGEOM_REPLACE_USOLIDS

//============== here for VecGeom-based implementation
#include "base/Transformation3D.h"
#include "volumes/LogicalVolume.h"
#include "volumes/SpecializedTube.h"
#include "volumes/kernel/shapetypes/TubeTypes.h"
#include "volumes/UnplacedTube.h"

#ifndef VECGEOM_MASTER

#include "volumes/UnplacedTube.h"
#include "volumes/USolidsAdapter.h"

class UTubs : public vecgeom::USolidsAdapter<vecgeom::GenericUnplacedTube> {

  // just forwards UBox to vecgeom box
  using Shape_t = vecgeom::GenericUnplacedTube;
  using Base_t  = vecgeom::USolidsAdapter<vecgeom::GenericUnplacedTube>;

  // inherit all constructors
  using Base_t::Base_t;

public:
  // Accessors

  double GetInnerRadius() const { return rmin(); }
  double GetOuterRadius() const { return rmax(); }
  double GetZHalfLength() const { return z(); }
  double GetStartPhiAngle() const { return sphi(); }
  double GetDeltaPhiAngle() const { return dphi(); }

  // Modifiers

  void SetInnerRadius(double newRMin) { SetRMin(newRMin); }
  void SetOuterRadius(double newRMax) { SetRMax(newRMax); }
  void SetZHalfLength(double newDz) { SetDz(newDz); }
  void SetStartPhiAngle(double newSPhi, bool /* trig */ = true) { SetSPhi(newSPhi); }
  void SetDeltaPhiAngle(double newDPhi) { SetDPhi(newDPhi); }

  // o provide a new object which is a clone of the solid
  VUSolid *Clone() const override { return new UTubs(GetName().c_str(), rmin(), rmax(), z(), sphi(), dphi()); }

  void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) override {}

  std::string GetEntityType() const override { return "Tube"; }

  // Visualisation
  void GetParametersList(int, double *aArray) const override
  {
    aArray[0] = GetInnerRadius();
    aArray[1] = GetOuterRadius();
    aArray[2] = GetZHalfLength();
    aArray[3] = GetStartPhiAngle();
    aArray[4] = GetDeltaPhiAngle();
  }

  std::ostream &StreamInfo(std::ostream &os) const override
  {
    int oldprc = os.precision(16);
    os << "-----------------------------------------------------------\n"
       << "     *** Dump for solid - tube ***\n"
       << "     ===================================================\n"
       << " Solid type: " << GetEntityType() << "\n"
       << " Parameters: \n"
       << "     Tube Radii Rmin, Rmax: " << rmin() << "mm, " << rmax() << "mm \n"
       << "     Half-length Z = " << z() << "mm\n";
    if (dphi() < vecgeom::kTwoPi) {
      os << "     Wedge starting angles: fSPhi=" << sphi() * vecgeom::kRadToDeg << "deg, "
         << ", fDphi=" << dphi() * vecgeom::kRadToDeg << "deg\n";
    }
    os << "-----------------------------------------------------------\n";
    os.precision(oldprc);
    return os;
  }
};

#else

class UTubs : public vecgeom::SpecializedTube<vecgeom::translation::kIdentity, vecgeom::rotation::kIdentity,
                                              vecgeom::TubeTypes::UniversalTube> {
  // just forwards UTubs to vecgeom tube
  typedef typename vecgeom::SpecializedTube<vecgeom::translation::kIdentity, vecgeom::rotation::kIdentity,
                                            vecgeom::TubeTypes::UniversalTube>
      Shape_t;
  // inherits all constructors
  using Shape_t::Shape_t;
};

#endif

//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation
#include "VUSolid.hh"
#include <sstream>

class UTubs : public VUSolid {
public:
  UTubs(const std::string &pName, double pRMin, double pRMax, double pDz, double pSPhi, double pDPhi);
  //
  // Constructs a tubs with the given name and dimensions

  virtual ~UTubs();
  //
  // Destructor

  // Accessors

  inline double GetInnerRadius() const;
  inline double GetOuterRadius() const;
  inline double GetZHalfLength() const;
  inline double GetStartPhiAngle() const;
  inline double GetDeltaPhiAngle() const;

  // Modifiers

  inline void SetInnerRadius(double newRMin);
  inline void SetOuterRadius(double newRMax);
  inline void SetZHalfLength(double newDz);
  inline void SetStartPhiAngle(double newSPhi, bool trig = true);
  inline void SetDeltaPhiAngle(double newDPhi);

  // Methods for solid

  inline double Capacity();
  inline double SurfaceArea();

  VUSolid::EnumInside Inside(const UVector3 &p) const;

  bool Normal(const UVector3 &p, UVector3 &normal) const;

  double DistanceToIn(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;
  double SafetyFromInside(const UVector3 &p, bool precise = false) const;
  double DistanceToOut(const UVector3 &p, const UVector3 &v, UVector3 &n, bool &validNorm,
                       double aPstep = UUtils::kInfinity) const;
  double SafetyFromOutside(const UVector3 &p, bool precise = false) const;

  inline double SafetyFromInsideR(const UVector3 &p, const double rho, bool precise = false) const;
  inline double SafetyFromOutsideR(const UVector3 &p, const double rho, bool precise = false) const;
  UGeometryType GetEntityType() const;

  UVector3 GetPointOnSurface() const;

  VUSolid *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;

  void Extent(UVector3 &aMin, UVector3 &aMax) const;

  virtual void GetParametersList(int /*aNumber*/, double * /*aArray*/) const;
  virtual void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

public:
  UTubs();
  //
  // Fake default constructor for usage restricted to direct object
  // persistency for clients requiring preallocation of memory for
  // persistifiable objects.

  UTubs(const UTubs &rhs);
  UTubs &operator=(const UTubs &rhs);
  // Copy constructor and assignment operator.

  inline double GetRMin() const;
  inline double GetRMax() const;
  inline double GetDz() const;
  inline double GetSPhi() const;
  inline double GetDPhi() const;

protected:
  //    UVector3List*
  //    CreateRotatedVertices( const UAffineTransform& pTransform ) const;
  //
  // Creates the List of transformed vertices in the format required
  // for VUSolid:: ClipCrossSection and ClipBetweenSections

  inline void Initialize();
  //
  // Reset relevant values to zero

  inline void CheckSPhiAngle(double sPhi);
  inline void CheckDPhiAngle(double dPhi);
  inline void CheckPhiAngles(double sPhi, double dPhi);
  //
  // Reset relevant flags and angle values

  inline void InitializeTrigonometry();
  //
  // Recompute relevant trigonometric values and cache them

  virtual UVector3 ApproxSurfaceNormal(const UVector3 &p) const;
  //
  // Algorithm for SurfaceNormal() following the original
  // specification for points not on the surface

  inline double SafetyToPhi(const UVector3 &p, const double rho, bool &outside) const;

protected:
  double fCubicVolume, fSurfaceArea;
  // Used by distanceToOut
  //
  enum ESide { kNull, kRMin, kRMax, kSPhi, kEPhi, kPZ, kMZ };

  // Used by normal
  //
  enum ENorm { kNRMin, kNRMax, kNSPhi, kNEPhi, kNZ };

  double kRadTolerance, kAngTolerance;
  //
  // Radial and angular tolerances

  double fRMin, fRMax, fDz, fSPhi, fDPhi;
  //
  // Radial and angular dimensions

  double fSinCPhi, fCosCPhi, fCosHDPhiOT, fCosHDPhiIT, fSinSPhi, fCosSPhi, fSinEPhi, fCosEPhi, fSinSPhiDPhi,
      fCosSPhiDPhi;
  //
  // Cached trigonometric values

  bool fPhiFullTube;
  //
  // Flag for identification of section or full tube
};

#include "UTubs.icc"

//============== end of USolids-based implementation

#endif // VECGEOM_REPLACE_USOLIDS
#endif // UTubs_HH
