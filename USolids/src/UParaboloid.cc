//
// File: UParaboloid.cc
// Purpose: tests (not a complete/correct implementation)
//
// 2015-08-20 Guilherme Lima - Note: quick and dirty implementation for integration tests only
//

#include "base/Global.h"
#include "UParaboloid.hh"

constexpr double kPi = 3.14159265358979323846;

double UParaboloid::SurfaceArea() {
  //G4 implementation
  double h1, h2, A1, A2;
  double Rhi2 = fRhi*fRhi;
  double Rlo2 = fRlo*fRlo;
  double dd = 1./(Rhi2 - Rlo2);
  // double A = 2.*fDz*dd;
  double B = - fDz * (Rlo2 + Rhi2) * dd;
  h1 = -B + fDz;
  h2 = -B - fDz;

  // Calculate surface area for the paraboloid full paraboloid
  // cutoff at z = dz (not the cutoff area though).
  A1 = fRhi*fRhi + 4 * h1*h1;
  A1 *= (A1*A1); // Sets A1 = A1^3
  A1 = kPi * fRhi /6 / (h1*h1) * ( sqrt(A1) - fRhi*fRhi*fRhi);

  // Calculate surface area for the paraboloid full paraboloid
  // cutoff at z = -dz (not the cutoff area though).
  A2 = 0.;
  if(h2 != 0) {
    A2 = fRlo*fRlo + 4 * (h2*h2);
    A2 *= (A2*A2); // Sets A2 = A2^3
    A2 = kPi * fRlo /6 / (h2*h2) * (sqrt(A2) - fRlo*fRlo*fRlo);
  }

  return (A1 - A2 + (fRlo*fRlo + fRhi*fRhi)*kPi);
}

std::ostream& UParaboloid::StreamInfo(std::ostream& os) const {
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "       *** Dump for solid - " << GetName() << " ***\n"
     << "       ===================================================\n"
     << " Solid type: UParaboloid\n"
     << " Parameters: \n"
     << "       Rhi: " << fRhi << " mm \n"
     << "       Rlo: " << fRlo << " mm \n"
     << "       Dz: " << fDz << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

UVector3 UParaboloid::GetPointOnSurface() const {
  //G4 implementation
  double area = const_cast<UParaboloid*>(this)->SurfaceArea();
  double z = UUtils::Random( 0., 1. );
  double phi = UUtils::Random(0.,2*kPi);
  double Rhi2 = fRhi*fRhi;
  double Rlo2 = fRlo*fRlo;
  double dd = 1./(Rhi2 - Rlo2);
  double A = 2.*fDz*dd;
  double B = - fDz * (Rlo2 + Rhi2) * dd;
  if(kPi*(Rlo2 + Rhi2)/area >= z) {
    double rho = sqrt(UUtils::Random(0.,1.));
    if(kPi * Rlo2/area > z) {
      //points on the cutting circle surface at -dZ
      rho *= fRlo;
      return UVector3(rho * cos(phi), rho * sin(phi), -fDz);
    }
    else {
      //points on the cutting circle surface at dZ
      rho = fRhi * sqrt(UUtils::Random(0.,1.));
      return UVector3(rho * cos(phi), rho * sin(phi), fDz);
    }
  }
  else {
    //points on the paraboloid surface
    z = fDz * UUtils::Random(-1.,1.);
    double rho = sqrt((z-B)/A);
    return UVector3(rho*cos(phi), rho*sin(phi), z);
  }
}
