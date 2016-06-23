#ifndef Nucleus_H
#define Nucleus_H

#include "base/Global.h"

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <math.h>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Nucleus;)

inline namespace VECGEOM_IMPL_NAMESPACE {

class Nucleus {
public:
  class Decay;
  Nucleus(std::string name, int n, int z, int iso, double a, double dm, double life, double natab, double toxa,
          double toxb, int ind1, int ind2);

  double A() const { return fA; }
  double Z() const { return fZ; }
  double Iso() const { return fIso; }
  double Life() const { return fLife; }

  double ToxA() const { return fToxa; }
  double ToxB() const { return fToxb; }
  int Indx1() const { return fInd1; }
  int Indx2() const { return fInd2; }

  static void CreateNuclei();

  const std::vector<Decay> &DecayList() const { return fDecayList; }

  std::string Name() const
  {
    std::stringstream ss;
    ss << fZ << "-" << fName << "-" << fN << "-" << fIso;
    return ss.str();
  }

  static void ReadFile(std::string infilename, std::string outfilename = "");

  void NormDecay();

  friend std::ostream &operator<<(std::ostream &os, const Nucleus &nuc);

  void AddDecay(int da, int dz, int diso, double qval, double br);

  static const std::map<int, Nucleus *> &Nuclei() { return fNuclei; }
  static const std::map<int, std::vector<Nucleus *>> &IsoList() { return fIsoList; }
  static const std::map<int, std::vector<Nucleus *>> &NatIsoList() { return fNatIsoList; }

  class Decay {
  public:
    Decay(int da, int dz, int diso, double qval, double br) : fDa(da), fDz(dz), fDiso(diso), fQval(qval), fBr(br) {}

    bool operator==(const Decay &d1) const
    {
      return (this->fDa == d1.fDa) && (this->fDz == d1.fDz) && (this->fDiso == d1.fDiso) &&
             (fabs(this->fQval - d1.fQval) / (this->fQval + d1.fQval) > 0 ? (this->fQval + d1.fQval) : 1 > 5e-7) &&
             (fabs(this->fBr - d1.fBr) / (this->fBr + d1.fBr) > 0 ? (this->fBr + d1.fBr) : 1 > 5e-7);
    }

    int Dz() const { return fDz; }
    int Da() const { return fDa; }
    int Diso() const { return fDiso; }
    double Qval() const { return fQval; }
    double Br() const { return fBr; }
    void Br(double br) { fBr = br; }
    const std::string Name() const;

  private:
    int fDa;
    int fDz;
    int fDiso;
    double fQval;
    double fBr;
  };

private:
  static void Getmat(std::string line, int &n, int &z, int &iso, std::string &name, double &a, double &dm, double &life,
                     int &da, int &dz, int &diso, double &br, double &qval, double &natab, double &toxa, double &toxb,
                     int &ind1, int &ind2);

  std::string fName;             // Name
  int fN;                        // Nucleon number
  int fZ;                        // Atomic number
  int fIso;                      // Isomer level
  double fA;                     // Atomic weight
  int fIsolevel;                 // Isomeric mass excess
  double fLife;                  // Lifetime
  double fNatab;                 // Natural abundance
  double fToxa;                  // Radiotoxicity
  double fToxb;                  // Radiotoxicity
  int fInd1;                     // Misterious index 1
  int fInd2;                     // Misterious index 2
  std::vector<Decay> fDecayList; // Decay channels

  static std::map<int, Nucleus *> fNuclei;                  // Nuclei list indexed by 10,000*z + 10*n + iso
  static std::map<int, std::vector<Nucleus *>> fIsoList;    // List of isotopes for a given z
  static std::map<int, std::vector<Nucleus *>> fNatIsoList; // List of stable isotopes for a given z
};
}
}
#endif
