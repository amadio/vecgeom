#ifndef Particle_H
#define Particle_H

#include "base/Global.h"

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <math.h>

#include <iostream>

using std::string;
using std::vector;
using std::map;
using std::ostream;
using std::to_string;

namespace vecgeom {
  
  VECGEOM_DEVICE_FORWARD_DECLARE( class Material; )
  
    inline namespace VECGEOM_IMPL_NAMESPACE {
  
class Particle {
public:
   class Decay;
   Particle();
   Particle(const char* name, int pdg, bool matter, const char* pclass, int pcode, double charge, double mass,
	    double width, int isospin, int iso3, int strange, int flavor, int track);

   static void CreateParticles();

   const char* Name() const {return fName.c_str();}
   int PDG() const {return fPDG;}
   bool Matter() const {return fMatter;}
   double Mass() const {return fMass;}
   const char* Class() const {return fClass.c_str();}
   int Pcode() const {return fPcode;}
   double Charge() const {return fCharge;}
   double Width() const {return fWidth;}
   int Isospin() const {return fIsospin;}
   int Iso3() const {return fIso3;}
   int Strange() const {return fStrange;}
   int Flavor() const {return fFlavor;}
   int Track() const {return fTrack;}
   int Ndecay() const {return fNdecay;}
   

   const vector<Decay> & DecayList() const {return fDecayList;}
   
   static void ReadFile(string infilename, string outfilename="");
   static void CreateParticle();

   const Particle& GetParticle(int pdg) {return fParticles[pdg];}

   void NormDecay();

   friend ostream& operator<<(ostream& os, const Particle& part);

   void AddDecay(const Decay &decay) {fDecayList.push_back(decay); fNdecay = fDecayList.size();}

   static const map<int,Particle> & Particles() {return fParticles;}
   
   class Decay {
   public:
      Decay(): fType(0), fBr(0) {}
      Decay(int type, double br, const vector<int>& daughters): fType(type), fBr(br), fDaughters(daughters) {}
      void Clear() {fType = 0; fBr = 0; fDaughters.clear();}

      int Type() const {return fType;}
      double Br() const {return fBr;}
      const vector<int> &Daughters() const {return fDaughters;}
      int NDaughters() const {return fDaughters.size();}
      int Daughter(int i) const {return fDaughters[i];}
      
      void SetType(int type) {fType = type;}
      void SetBr(double br) {fBr = br;}
      void AddDaughter(int daughter) {fDaughters.push_back(daughter);}

      friend ostream& operator<<(ostream& os, const Decay& dec);

   private:
      char fType;
      float fBr;
      vector<int> fDaughters;
   };

private:

   static void GetPart(const string &line, int &count, string &name, int &pdg, bool &matter, int &pcode, 
		       string &pclass, int &charge, double &mass, double &width, int &isospin, int &iso3, 
		       int &strange, int &flavor, int &track, int &ndecay, int &ipart, int &acode);

   static void GetDecay(const string &line, int &dcount, Decay &decay);

   string fName;  // Name
   int fPDG;      // PDG code
   bool fMatter;  // False if antiparticle
   string fClass; // Particle class
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
   char  fNdecay;  // Number of decay channels
   vector<Decay>  fDecayList; // Decay channels

   static map<int,Particle> fParticles;              // Particle list indexed by PDG code


};

}
}
#endif
