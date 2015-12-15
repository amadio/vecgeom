#ifndef Particle_H
#define Particle_H

#include "base/Global.h"

#include <string.h>
#ifdef VECGEOM_NVCC
#include "base/Map.h"
#include "base/Vector.h"
#include <string.h>
#else
#include <map>
#include <vector>
#include <string>
#endif
#include <fstream>
#include <math.h>

#include <iostream>

#ifdef VECGEOM_NVCC
using vecgeom::map;
using vecgeom::Vector;
#else
using std::map;
using std::string;
using std::vector;
#endif
using std::ostream;

namespace vecgeom {
  
  VECGEOM_DEVICE_FORWARD_DECLARE( class Material; )
  
    inline namespace VECGEOM_IMPL_NAMESPACE {
 
#ifdef VECGEOM_NVCC_DEVICE
class Particle; 
extern VECGEOM_CUDA_HEADER_DEVICE map<int,Particle> *fParticles;              // Particle list indexed by PDG code
#endif

class Particle {
public:
   class Decay;
   VECGEOM_CUDA_HEADER_BOTH
   Particle();
   VECGEOM_CUDA_HEADER_BOTH
   Particle(const char* name, int pdg, bool matter, const char* pclass, int pcode, double charge, double mass,
	    double width, int isospin, int iso3, int strange, int flavor, int track, int code=-1);

   VECGEOM_CUDA_HEADER_BOTH
   //Particle(const Particle & other):fName(other.fName), fPDG(other.fPDG), fMatter(other.fMatter), fClass(other.fClass), fPcode(other.fPcode), fCharge(other.fCharge), fMass(other.fMass),fWidth(other.fWidth),fIsospin(other.fIsospin),fStrange(other.fStrange),fFlavor(other.fFlavor),fTrack(other.fTrack),fCode(other.fCode){}

   VECGEOM_CUDA_HEADER_BOTH
   static void CreateParticles();
#ifdef VECGEOM_NVCC
   VECGEOM_CUDA_HEADER_BOTH
Particle operator=(const Particle &part) {
   return part;
}
#endif

   const char* Name() const {return fName;}
   int PDG() const {return fPDG;}
   bool Matter() const {return fMatter;}
   double Mass() const {return fMass;}
   const char* Class() const {return fClass;}
   int Pcode() const {return fPcode;}
   double Charge() const {return fCharge;}
   double Width() const {return fWidth;}
   int Isospin() const {return fIsospin;}
   int Iso3() const {return fIso3;}
   int Strange() const {return fStrange;}
   int Flavor() const {return fFlavor;}
   int Track() const {return fTrack;}
   int Ndecay() const {return fNdecay;}
   int Code() const  {return fCode;}

   void SetCode(int code) {fCode = code;}
   
#ifndef VECGEOM_NVCC
   const vector<Decay> & DecayList() const {return fDecayList;}
#else
   const Vector<Decay> & DecayList() const {return fDecayList;}
#endif
#ifndef VECGEOM_NVCC
   static void ReadFile(string infilename, string outfilename="");
#endif
VECGEOM_CUDA_HEADER_BOTH
   static void CreateParticle();

#ifndef VECGEOM_NVCC_DEVICE   
   static const Particle& GetParticle(int pdg) {
      if(fParticles->find(pdg)!=fParticles->end()) return (*fParticles)[pdg];
      static Particle p;
      std::cout << __func__ << "::pdg:" << pdg << " does not exist" << std::endl; return p;}

   void NormDecay();

   friend ostream& operator<<(ostream& os, const Particle& part);
   
#endif
   void AddDecay(const Decay &decay) {fDecayList.push_back(decay); fNdecay = fDecayList.size();}

   static const map<int,Particle> & Particles() {return *fParticles;}
   
   class Decay {
   public:
VECGEOM_CUDA_HEADER_BOTH
      Decay(): fType(0), fBr(0) {}
VECGEOM_CUDA_HEADER_BOTH
      Decay(const Decay & other): fType(other.fType), fBr(other.fBr), fDaughters(other.fDaughters) {}
#ifndef VECGEOM_NVCC
      Decay(int type, double br, const vector<int>& daughters): fType(type), fBr(br), fDaughters(daughters) {}
#else
VECGEOM_CUDA_HEADER_BOTH
      Decay(int type, double br, const Vector<int>& daughters): fType(type), fBr(br), fDaughters(daughters) {}
#endif
      void Clear() {fType = 0; fBr = 0; fDaughters.clear();}

      int Type() const {return fType;}
      double Br() const {return fBr;}
#ifndef VECGEOM_NVCC
      const vector<int> &Daughters() const {return fDaughters;}
#else
      const Vector<int> &Daughters() const {return fDaughters;}
#endif
      int NDaughters() const {return fDaughters.size();}
      int Daughter(int i) const {return fDaughters[i];}
      
      void SetType(int type) {fType = type;}
      void SetBr(double br) {fBr = br;}
      void AddDaughter(int daughter) {fDaughters.push_back(daughter);}

#ifndef VECGEOM_NVCC
      friend ostream& operator<<(ostream& os, const Decay& dec);
#else
   VECGEOM_CUDA_HEADER_BOTH
      Decay operator=(const Decay &dec) {
         return dec;
      }
#endif
   private:
      char fType;
      float fBr;
#ifndef VECGEOM_NVCC
      vector<int> fDaughters;
#else
      Vector<int> fDaughters;
#endif
   };

private:

#ifndef VECGEOM_NVCC  
   static void GetPart(const string &line, int &count, string &name, int &pdg, bool &matter, int &pcode, 
		       string &pclass, int &charge, double &mass, double &width, int &isospin, int &iso3, 
		       int &strange, int &flavor, int &track, int &ndecay, int &ipart, int &acode);

   static void GetDecay(const string &line, int &dcount, Decay &decay);
#endif
   char fName[256];  // Name
   int fPDG;      // PDG code
   bool fMatter;  // False if antiparticle
   char fClass[256]; // Particle class
   int fPcode;   // Particle code
   float fCharge;// Charge
   float fMass;  // Mass in GeV
   float fWidth; // Width in GeV
   float fLife;  // Lifetime in seconds
   char  fIsospin;// Isospin
   char  fIso3;   // Isospin 3
   char  fStrange; // Strangeness
   char  fFlavor;   // Flavor code (?)
   char  fTrack;   // Track code (?)
   unsigned char fNdecay;  // Number of decay channels
   short fCode;    // Particle code for a given MC
#ifndef VECGEOM_NVCC
   vector<Decay>  fDecayList; // Decay channels
#else
   Vector<Decay>  fDecayList; // Decay channels
#endif
#ifndef VECGEOM_NVCC_DEVICE
   static map<int,Particle> *fParticles;              // Particle list indexed by PDG code
#endif

};

}
}
#endif
