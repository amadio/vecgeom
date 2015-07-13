#include "volumes/Particle.h"
#include <iostream>
#include <fstream>
#include <sstream>


using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::stringstream;

static const double kPlankBar = 6.5821192815e-25; // GeV s

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
  
map<int,Particle> Particle::fParticles;

ostream& operator<<(ostream& os, const Particle& part)
{
   os << part.fName << "(" << part.fPDG << ") Class:" << part.fClass << " Q:" << part.fCharge << " m:" << part.fMass
      << " lt:" << part.fLife << " I:" << (int) part.fIsospin << " I3:" << (int) part.fIso3 << " S:" << (int) part.fStrange 
      << " F:" << (int) part.fFlavor << " #:" << (int) part.fNdecay;
   return os;
}

//________________________________________________________________________________________________
Particle::Particle(): fName(""), fPDG(0), fMatter(true), fClass(""), fPcode(0), fCharge(0), fMass(0),
   fWidth(0), fIsospin(0), fIso3(0), fStrange(0), fFlavor(0), fTrack(0), fNdecay(0) { }

//________________________________________________________________________________________________
Particle::Particle(const char* name, int pdg, bool matter, const char* pclass, int pcode, double charge, 
		   double mass, double width, int isospin, int iso3, int strange, int flavor, int track):
   fName(name), fPDG(pdg), fMatter(matter), fClass(pclass), fPcode(pcode), fCharge(charge), fMass(mass),
   fWidth(width), fIsospin(isospin), fIso3(iso3), fStrange(strange), fFlavor(flavor), fTrack(track), fNdecay(0) {

   if(fParticles.count(fPDG) != 0) {
      cout << "Particle " << fPDG << " already there" << endl;
      return;
   }

   if(fWidth > 0) fLife = kPlankBar/fWidth;
   else fLife = 0;
   fParticles[fPDG] = *this;
}

//________________________________________________________________________________________________
void Particle::ReadFile(string infilename, string outfilename) {
   int count;
   string name;
   int pdg;
   bool matter;
   int pcode;
   string pclass;
   int charge;
   double mass, width;
   int isospin, iso3, strange, flavor, track, ndecay;
   int ipart, acode;
   
   bool output=!outfilename.empty();
   ofstream outfile;
   if(output) outfile.open(outfilename);
   ifstream infile(infilename);
   string line;

   int np=0;
   while(getline(infile,line)) {
      if(np == 0) for(int i=0; i<3; ++i) {
	    if(line.substr(0,1) != "#") {
	       cout << "There should be three lines of comments at the beginning " << line << endl;
	       exit(1);
	    }
	    getline(infile, line);
	 }
      if(line.substr(0,1) == "#") {
	 cout << "Disaster embedded comment!!!" << endl;
	 exit(1);
      }

      ++np;
      //      cout << "line: " << line << endl;
      GetPart(line, count, name, pdg, matter, pcode, pclass, charge, mass, width, isospin, iso3, 
	      strange, flavor, track, ndecay, ipart, acode);
      if(np != count) {
	 cout << "Disaster count np(" <<np << ") != count(" << count << ")"<< endl;
	 exit(1);
      }
      /*
      cout << " count:" << count << " name:" << name << " pdg:" << pdg << " matter:" << matter << " pcode:" << pcode 
	   <<" pclass:" << pclass << " charge:" << charge << " mass:" << mass << " width:" << width << " isospin:" 
	   << isospin << " iso3:" << iso3 << " strange:" << strange << " flavor:" << flavor << " track:" << track 
	   << " ndecay:" << ndecay << " ipart:" << ipart << " acode:" << acode << endl;
      */
      if(pdg>=0) {
	 if(isospin != -100) isospin = (isospin-1)/2; 
	 if(strange != -100) strange = (strange-1)/2;
	 new Particle(name.c_str(), pdg, matter, pclass.c_str(), pcode, charge/3., mass, width, isospin, iso3, 
			     strange, flavor, track);
	 Particle &part = fParticles.at(pdg);
	 if(ndecay > 0) {
	    for(int i=0; i<3; ++i) {
	       getline(infile, line);
	       if(line.substr(0,1) != "#") {
		  cout << "Disaster comment!!!" << endl;
		  exit(1);
	       }
	    }
	    vector<Decay> decaylist;
	    decaylist.clear();
	    for(int i=0; i< ndecay; ++i) {
	       getline(infile,line);
	       if(line.substr(0,1) == "#") {
		  cout << "Disaster embedded comment!!!" << endl;
		  exit(1);
	       }
	       Decay decay;
	       int dcount;
	       GetDecay(line,dcount,decay);
	       if(dcount != i+1) {
		  cout << "dcount (" << dcount << ") != i+1 (" << i+1 << ")" << endl;
		  exit(1);
	       }
	       cout << "         " << dcount << " " << decay << endl;
	       part.AddDecay(decay);
	    }
	    part.NormDecay();
	 }
      } else {
	 // Create antiparticle from particle
	 if(fParticles.find(-pdg) == fParticles.end()) {
	    cout << "Cannot build the antiparticle because the partiche " << -pdg << " is not there!" << endl;
	    exit(1);
	 }
	 Particle p = fParticles.at(-pdg);
	 new Particle(name.c_str(), pdg, matter, p.Class(), p.Pcode(), -p.Charge(), p.Mass(), p.Width(), -p.Isospin(), 
			     -p.Iso3(), -p.Strange(), -p.Flavor(), p.Track());
	 Particle &part = fParticles.at(pdg);
	 Decay decay;
	 vector<Particle::Decay>  dl = p.DecayList();
	 for(int i=0; i<p.Ndecay(); ++i) {
	    decay.Clear();
	    decay.SetType(dl.at(i).Type());
	    decay.SetBr(dl.at(i).Br());
	    for(int j=0; j<dl.at(i).NDaughters(); ++j) 
	       decay.AddDaughter(-dl.at(i).Daughter(j));
	    cout << "         " << i << " " << decay << endl;
	    part.AddDecay(decay);
	 }
      }
   }

   if(output) {
      outfile << "#include \"volumes/Particle.h\"" << endl;
      outfile << "namespace vecgeom {" << endl;
      outfile << "   inline namespace VECGEOM_IMPL_NAMESPACE {" << endl << endl;
      outfile << "void Particle::CreateParticles() {" << endl;
      outfile << "   Particle *part=0;" << endl;

      for(map<int,Particle>::const_iterator p=Particle::Particles().begin(); p != Particle::Particles().end(); ++p) {
	 const Particle &part = p->second;
	 string name(part.Name());
	 outfile << endl << "   // Creating " << name << endl;
	 size_t quote = name.find_first_of("\\\"");
	 if(quote != string::npos) 
	    name = name.substr(0,quote) + "\\\"" + name.substr(quote+1,name.size()-quote);
	 outfile << "   new Particle(" 
	      << "\"" << name << "\", "
	      << part.PDG() << ", "
	      << part.Matter() << ", "
	      << "\"" << part.Class() << "\", "
	      << part.Pcode() << ", "
	      << part.Charge() << ", "
	      << part.Mass() << ", "
	      << part.Width() << ", "
	      << part.Isospin() << ", "
	      << part.Iso3() << ", "
	      << part.Strange() << ", "
	      << part.Flavor() << ", "
	      << part.Track() << ");" << endl;
	 cout << part.Ndecay() << endl;
	 if(part.Ndecay() > 0) {
	    outfile << "   part = &fParticles.at(" << part.PDG() << ");" << endl;
	    for(vector<Decay>::const_iterator d=part.DecayList().begin(); d!=part.DecayList().end(); ++d) {
	       outfile << "      part->AddDecay(Decay(" << d->Type() << ", " << d->Br() << ", "
		       << " vector<int>{";
	       for(int i=0; i<d->NDaughters(); ++i) {
		  outfile << d->Daughter(i);
		  if(i<d->NDaughters()-1) outfile << ",";
	       }
	       outfile << "}));" << endl;
	    }
	 }
      }
   
      outfile << "}" << endl;      
      outfile << " } // End of inline namespace" << endl;
      outfile << " } // End of vecgeom namespace" << endl;
   }
}

//________________________________________________________________________________________________
void Particle::GetDecay(const string &line, int &dcount, Decay &decay) {
   int dtype;
   double br;
   int ndec;
   int daughter;

   stringstream ss(line);
   decay.Clear();
   ss >> dcount;
   ss >> dtype;
   ss >> br;
   decay.SetType(dtype);
   decay.SetBr(br);
   ss >> ndec;
   for(int i=0; i<ndec; ++i) {
      ss >> daughter;
      decay.AddDaughter(daughter);
   }
}

//________________________________________________________________________________________________
void Particle::NormDecay() {
   double brt=0;

   int ndec = fDecayList.size();
   if(ndec) {
      for(vector<Decay>::iterator idec=fDecayList.begin(); idec!=fDecayList.end(); ++idec) 
	 brt += idec->Br();
      if(brt) {
	 brt = 1/brt;
	 for(vector<Decay>::iterator idec=fDecayList.begin(); idec!=fDecayList.end(); ++idec) 
	    idec->SetBr(idec->Br()*brt);
	 for(int i=0; i<fDecayList.size()-1; ++i)
	    for(int j=i+1; j<fDecayList.size(); ++j)
	       if(fDecayList.at(i).Br() < fDecayList.at(j).Br()) {
		  Decay dec = fDecayList.at(i);
		  fDecayList.at(i) = fDecayList.at(j);
		  fDecayList.at(j) = dec;
	       }
      }
   } else
      if(fLife == 0) fLife = 1e38;
}

//________________________________________________________________________________________________
void Particle::GetPart(const string &line, int &count, string &name, int &pdg, bool &matter, int &pcode, 
		       string &pclass, int &charge, double &mass, double &width, int &isospin, int &iso3, 
		       int &strange, int &flavor, int &track, int &ndecay, int &ipart, int &acode)
{
   
   count = 0;
   name="";
   pdg=0;
   matter=false;
   pclass="";
   charge=0;
   mass=0;
   width=0;
   isospin=0;
   iso3=0;
   strange=0;
   flavor=0;
   track=0;
   ndecay=0;
   ipart=0;
   acode=0;
   pcode=0;

   stringstream ss(line);

   ss >> count;
   ss >> name;
   ss >> pdg;

   if(pdg<0) {
      matter = false;
      ss >> ipart;
      ss >> acode;
   } else {
   
      int imatter = 0;
      ss >> imatter;
      matter = (imatter == 1);
      
      ss >> pcode;
      ss >> pclass;
      ss >> charge;
      ss >> mass;
      ss >> width;
      ss >> isospin;
      ss >> iso3;
      ss >> strange;
      ss >>  flavor;
      ss >> track;
      ss >> ndecay;
   }
}

ostream& operator<<(ostream& os, const Particle::Decay& dec)
{
   os << "Type " << static_cast<int>(dec.fType) << " br " << dec.fBr << " products: ";
   for(int i=0; i<dec.fDaughters.size(); ++i)
      os << " " << dec.fDaughters[i];
   
   /*   for(int i=0; i<dec.fDaughters.size(); ++i)
	os << " " << Particle::Particles().at(dec.fDaughters[i]).Name(); */
   
   return os;
}


}
}
