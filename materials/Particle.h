#ifndef Particle_H
#define Particle_H

#include "base/Global.h"

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


namespace vecgeom {


  VECGEOM_DEVICE_FORWARD_DECLARE( class Material; )
  VECGEOM_DEVICE_FORWARD_DECLARE( class Particle; )

    inline  namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_NVCC
class Particle;
extern VECGEOM_CUDA_HEADER_DEVICE vecgeom::map<int,Particle> *fParticlesDev;              // Particle list indexed by PDG code
extern vecgeom::map<int,Particle> *fParticlesHost;              // Particle list indexed by PDG code
#endif

class Particle {
public:

   class Decay;
#ifdef VECGEOM_NVCC
   using Map_t = vecgeom::map<int,Particle>;
   using VectorDecay_t = vecgeom::Vector<Decay>;
   using VectorInt_t = vecgeom::Vector<int>;
#else
   using Map_t = std::map<int,Particle>;
   using VectorDecay_t = std::vector<Decay>;
   using VectorInt_t = std::vector<int>;
#endif


   VECGEOM_CUDA_HEADER_BOTH
   Particle();
   VECGEOM_CUDA_HEADER_BOTH
   Particle(const char* name, int pdg, bool matter, const char* pclass, int pcode, double charge, double mass,
	    double width, int isospin, int iso3, int strange, int flavor, int track, int code=-1);

  VECGEOM_CUDA_HEADER_BOTH
   Particle(const Particle & other);

   VECGEOM_CUDA_HEADER_BOTH
   Particle &operator=(const Particle &part);


   VECGEOM_CUDA_HEADER_BOTH
   static void CreateParticles();
   VECGEOM_CUDA_HEADER_BOTH
   const char* Name() const {return fName;}
   VECGEOM_CUDA_HEADER_BOTH
   int PDG() const {return fPDG;}
   VECGEOM_CUDA_HEADER_BOTH
   bool Matter() const {return fMatter;}
   VECGEOM_CUDA_HEADER_BOTH
   double Mass() const {return fMass;}
   const char* Class() const {return fClass;}
   VECGEOM_CUDA_HEADER_BOTH
   int Pcode() const {return fPcode;}
   VECGEOM_CUDA_HEADER_BOTH
   double Charge() const {return fCharge;}
   VECGEOM_CUDA_HEADER_BOTH
   double Width() const {return fWidth;}
   VECGEOM_CUDA_HEADER_BOTH
   int Isospin() const {return fIsospin;}
   VECGEOM_CUDA_HEADER_BOTH
   int Iso3() const {return fIso3;}
   VECGEOM_CUDA_HEADER_BOTH
   int Strange() const {return fStrange;}
   VECGEOM_CUDA_HEADER_BOTH
   int Flavor() const {return fFlavor;}
   VECGEOM_CUDA_HEADER_BOTH
   int Track() const {return fTrack;}
   VECGEOM_CUDA_HEADER_BOTH
   int Ndecay() const {return fNdecay;}
   VECGEOM_CUDA_HEADER_BOTH
   int Code() const  {return fCode;}

VECGEOM_CUDA_HEADER_BOTH
   void SetCode(int code) {fCode = code;}

   const VectorDecay_t & DecayList() const {return fDecayList;}
#ifndef VECGEOM_NVCC
   static void ReadFile(std::string infilename, std::string outfilename="");
#endif
VECGEOM_CUDA_HEADER_BOTH
  static void CreateParticle();

#ifndef VECGEOM_NVCC
   static const Particle& GetParticle(int pdg) {
      if(fParticles->find(pdg)!=fParticles->end()) return (*fParticles)[pdg];
      static Particle p;
      std::cout << __func__ << "::pdg:" << pdg << " does not exist" << std::endl; return p;
 }
#else
   static const Particle& GetParticle(int pdg) {
      if(fParticlesHost->find(pdg)!=fParticlesHost->end()) return (*fParticlesHost)[pdg];
      static Particle p;
      printf(" pdg %d does not exist\n",pdg);
      return p;
 }
VECGEOM_CUDA_HEADER_DEVICE
   static const Particle& GetParticleDev(int pdg) {
      if(fParticlesDev->find(pdg)!=fParticlesDev->end()) return (*fParticlesDev)[pdg];
      //Particle p;
      printf(" pdg %d does not exist\n",pdg);
      return (*fParticlesDev)[1];
 }
#endif

#ifndef VECGEOM_NVCC
   void NormDecay();

   friend std::ostream& operator<<(std::ostream& os, const Particle& part);

#endif
VECGEOM_CUDA_HEADER_BOTH
   void AddDecay(const Decay &decay) {fDecayList.push_back(decay); fNdecay = fDecayList.size();}
VECGEOM_CUDA_HEADER_BOTH
   static const Map_t& Particles() {
   #ifndef VECGEOM_NVCC
     return *fParticles;
   #else
   #ifndef VECGEOM_NVCC_DEVICE
     return *fParticlesHost;
   #else
     return *fParticlesDev;
   #endif
   #endif
   }

   class Decay {
   public:
VECGEOM_CUDA_HEADER_BOTH
      Decay(): fType(0), fBr(0) {}
VECGEOM_CUDA_HEADER_BOTH
      Decay(const Decay & other): fType(other.fType), fBr(other.fBr), fDaughters(other.fDaughters) {}
VECGEOM_CUDA_HEADER_BOTH
      Decay(int type, double br, const VectorInt_t& daughters): fType(type), fBr(br), fDaughters(daughters) {}
VECGEOM_CUDA_HEADER_BOTH
      void Clear() {fType = 0; fBr = 0; fDaughters.clear();}

VECGEOM_CUDA_HEADER_BOTH
      int Type() const {return fType;}
VECGEOM_CUDA_HEADER_BOTH
      double Br() const {return fBr;}
VECGEOM_CUDA_HEADER_BOTH
      const VectorInt_t &Daughters() const {return fDaughters;}
VECGEOM_CUDA_HEADER_BOTH
      int NDaughters() const {return fDaughters.size();}
VECGEOM_CUDA_HEADER_BOTH
      int Daughter(int i) const {return fDaughters[i];}

VECGEOM_CUDA_HEADER_BOTH
      void SetType(int type) {fType = type;}
VECGEOM_CUDA_HEADER_BOTH
      void SetBr(double br) {fBr = br;}
VECGEOM_CUDA_HEADER_BOTH
      void AddDaughter(int daughter) {fDaughters.push_back(daughter);}

#ifndef VECGEOM_NVCC
   friend std::ostream& operator<<(std::ostream& os, const Decay& dec);
#else
   VECGEOM_CUDA_HEADER_BOTH
      Decay operator=(const Decay &dec) {
         return dec;
      }
#endif
   private:
      char fType;
      float fBr;
      VectorInt_t fDaughters;
   };

private:

#ifndef VECGEOM_NVCC
   static void GetPart(const std::string &line, int &count, std::string &name, int &pdg, bool &matter, int &pcode,
		       std::string &pclass, int &charge, double &mass, double &width, int &isospin, int &iso3,
		       int &strange, int &flavor, int &track, int &ndecay, int &ipart, int &acode);

   static void GetDecay(const std::string &line, int &dcount, Decay &decay);
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
   VectorDecay_t  fDecayList; // Decay channels
#ifndef VECGEOM_NVCC
   static Map_t *fParticles;              // Particle list indexed by PDG code
#endif

};

}
}
#endif
