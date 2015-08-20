//
// File: UTorus.cc
// Purpose: tests (not a complete/correct implementation)
//
// 2015-08-20 Guilherme Lima - Note: quick and dirty implementation for integration tests only
//

#include "UTorus.hh"

double UTorus::Capacity() {
  assert(false && "Not implemented!");
  return -1.0;
}

double UTorus::SurfaceArea() {
  using UUtils::kTwoPi;
  double surfaceArea = fDphi*kTwoPi*fRtor*(fRmax+fRmin);
  if(fDphi < kTwoPi) {
    surfaceArea = surfaceArea + kTwoPi*(fRmax*fRmax-fRmin*fRmin);
  }
  return surfaceArea;
}

std::ostream& UTorus::StreamInfo(std::ostream& os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "       *** Dump for solid - " << GetName() << " ***\n"
     << "       ===================================================\n"
     << " Solid type: UBox\n"
     << " Parameters: \n"
     << "       tube Rmin: " << fRmin << " mm \n"
     << "       tube Rmax: " << fRmax << " mm \n"
     << "       torus Rad: " << fRtor << " mm \n"
     << "       start Phi: " << fSphi << " rad \n"
     << "       delta Phi: " << fDphi << " rad \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

VUSolid::EnumInside UTorus::Inside(const UVector3& /*aPoint*/) const
{
  // Classify point location with respect to solid:
  //  o eInside       - inside the solid
  //  o eSurface      - close to surface within tolerance
  //  o eOutside      - outside the solid

  assert(false && "Not implemented.");
  return EnumInside::eInside;
}


UVector3 UTorus::GetPointOnSurface() const {
  // start with a random point along the axis of the torus tube
  double radtor = GetRtor();
  double radmax = GetRmax();
  double phi = UUtils::Random( fSphi, fDphi );
  double theta = UUtils::Random( 0, UUtils::kTwoPi );

  double x = (radtor + radmax*cos(theta))*cos(phi);  // using only outer surface (incomplete)
  double y = (radtor + radmax*cos(theta))*cos(phi);
  double z = radmax*sin(theta);
  return UVector3(x,y,z);
}
