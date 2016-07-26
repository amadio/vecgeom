#ifndef Nucleus_H
#define Nucleus_H

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "base/Map.h"
#include "base/Vector.h"
#include <string.h>
template <class T> using vector = vecgeom::Vector<T>;
template <class T, class V> using map = vecgeom::map<T, V>;
#else
#include <map>
#include <vector>
#include <string>
using std::vector;
using std::map;
#endif
#include <fstream>
#include <sstream>
#include <math.h>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Nucleus;);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_NVCC
class Nucleus;
extern VECGEOM_CUDA_HEADER_DEVICE vecgeom::map<int, Nucleus *> *
    fNucleiDev; // Nuclei list indexed by 10,000*z + 10*n + iso
extern vecgeom::map<int, Nucleus *> *
    fNucleiHost; // Nuclei list indexed by 10,000*z + 10*n + iso
extern VECGEOM_CUDA_HEADER_DEVICE vecgeom::map<int, vector<Nucleus *>> *
    fIsoListDev; // List of isotopes for a given z
extern vecgeom::map<int, vector<Nucleus *>> *
    fIsoListHost; // List of isotopes for a given z
extern VECGEOM_CUDA_HEADER_DEVICE vecgeom::map<int, vector<Nucleus *>> *
    fNatIsoListDev; // List of stable isotopes for a given z
extern vecgeom::map<int, vector<Nucleus *>> *
    fNatIsoListHost; // List of stable isotopes for a given z
#endif

class Nucleus {
public:
  class Decay;
  VECGEOM_CUDA_HEADER_BOTH
  Nucleus(const char *name, int n, int z, int iso, double a, double dm,
          double life, double natab, double toxa, double toxb, int ind1,
          int ind2);

  VECGEOM_CUDA_HEADER_BOTH
  double A() const { return fA; }
  VECGEOM_CUDA_HEADER_BOTH
  double Z() const { return fZ; }
  VECGEOM_CUDA_HEADER_BOTH
  double Iso() const { return fIso; }
  VECGEOM_CUDA_HEADER_BOTH
  double Life() const { return fLife; }

  VECGEOM_CUDA_HEADER_BOTH
  double ToxA() const { return fToxa; }
  VECGEOM_CUDA_HEADER_BOTH
  double ToxB() const { return fToxb; }
  VECGEOM_CUDA_HEADER_BOTH
  int Indx1() const { return fInd1; }
  VECGEOM_CUDA_HEADER_BOTH
  int Indx2() const { return fInd2; }

  VECGEOM_CUDA_HEADER_BOTH
  static void CreateNuclei();

  VECGEOM_CUDA_HEADER_BOTH
  const vector<Decay> &DecayList() const { return fDecayList; }

  VECGEOM_CUDA_HEADER_BOTH
  std::string Name() const {
    char name[15];
    snprintf(name,14,"%d-%s-%d-%d",fZ,fName,fN,fIso);
    name[14]='\0';
    return std::string(name);
  }

#ifndef VECGEOM_NVCC
  static void ReadFile(std::string infilename, bool outfile = false);
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void NormDecay();

#ifndef VECGEOM_NVCC
  friend std::ostream &operator<<(std::ostream &os, const Nucleus &nuc);
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void AddDecay(int da, int dz, int diso, double qval, double br);

  VECGEOM_CUDA_HEADER_BOTH
  static const map<int, Nucleus *> &Nuclei() {
#ifndef VECGEOM_NVCC
    if (!fNuclei)
      fNuclei = new map<int, Nucleus *>;
    return *fNuclei;
#else
#ifndef VECGEOM_NVCC_DEVICE
    if (!fNucleiHost)
      fNucleiHost = new map<int, Nucleus *>;
    return *fNucleiHost;
#else
    if (!fNucleiDev)
      fNucleiDev = new map<int, Nucleus *>;
    return *fNucleiDev;
#endif
#endif
  }

  VECGEOM_CUDA_HEADER_BOTH
  static const map<int, vector<Nucleus *>> &IsoList() {
#ifndef VECGEOM_NVCC
    if (!fIsoList)
      fIsoList = new map<int, vector<Nucleus *>>;
    return *fIsoList;
#else
#ifndef VECGEOM_NVCC_DEVICE
    if (!fIsoListHost)
      fIsoListHost = new map<int, vector<Nucleus *>>;
    return *fIsoListHost;
#else
    if (!fIsoListDev)
      fIsoListDev = new map<int, vector<Nucleus *>>;
    return *fIsoListDev;
#endif
#endif
  }

  VECGEOM_CUDA_HEADER_BOTH
  static const map<int, vector<Nucleus *>> &NatIsoList() {
#ifndef VECGEOM_NVCC
    if (!fNatIsoList)
      fNatIsoList = new map<int, vector<Nucleus *>>;
    return *fNatIsoList;
#else
#ifndef VECGEOM_NVCC_DEVICE
    return *fNatIsoListHost;
#else
    return *fNatIsoListDev;
#endif
#endif
  }

  class Decay {
  public:
    VECGEOM_CUDA_HEADER_BOTH
    Decay() : fDa(0), fDz(0), fDiso(0), fQval(0), fBr(0) {}

    VECGEOM_CUDA_HEADER_BOTH
    Decay(int da, int dz, int diso, double qval, double br)
        : fDa(da), fDz(dz), fDiso(diso), fQval(qval), fBr(br) {}

    VECGEOM_CUDA_HEADER_BOTH
    bool operator==(const Decay &d1) const {
      return (fDa == d1.fDa) && (fDz == d1.fDz) && (fDiso == d1.fDiso) &&
             (fabs(fQval - d1.fQval) / (fQval + d1.fQval) > 0
                  ? (fQval + d1.fQval)
                  : 1 > 5e-7) &&
             (fabs(fBr - d1.fBr) / (fBr + d1.fBr) > 0 ? (fBr + d1.fBr)
                                                      : 1 > 5e-7);
    }

    VECGEOM_CUDA_HEADER_BOTH
    int Dz() const { return fDz; }
    VECGEOM_CUDA_HEADER_BOTH
    int Da() const { return fDa; }
    VECGEOM_CUDA_HEADER_BOTH
    int Diso() const { return fDiso; }
    VECGEOM_CUDA_HEADER_BOTH
    double Qval() const { return fQval; }
    VECGEOM_CUDA_HEADER_BOTH
    double Br() const { return fBr; }
    VECGEOM_CUDA_HEADER_BOTH
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
#ifndef VECGEOM_NVCC
  static void Getmat(std::string line, int &n, int &z, int &iso,
                     std::string &name, double &a, double &dm, double &life,
                     int &da, int &dz, int &diso, double &br, double &qval,
                     double &natab, double &toxa, double &toxb, int &ind1,
                     int &ind2);
#endif

  char fName[50];           // Name
  int fN;                   // Nucleon number
  int fZ;                   // Atomic number
  int fIso;                 // Isomer level
  double fA;                // Atomic weight
  int fIsolevel;            // Isomeric mass excess
  double fLife;             // Lifetime
  double fNatab;            // Natural abundance
  double fToxa;             // Radiotoxicity
  double fToxb;             // Radiotoxicity
  int fInd1;                // Misterious index 1
  int fInd2;                // Misterious index 2
  vector<Decay> fDecayList; // Decay channels

#ifndef VECGEOM_NVCC
  static std::map<int, Nucleus *> *
      fNuclei; // Nuclei list indexed by 10,000*z + 10*n + iso
  static std::map<int, vector<Nucleus *>> *
      fIsoList; // List of isotopes for a given z
  static std::map<int, vector<Nucleus *>> *
      fNatIsoList; // List of stable isotopes for a given z
#endif
};
}
}
#endif
