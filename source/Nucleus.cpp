#include "materials/Nucleus.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>


using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::stringstream;
using std::setw;
using std::setfill;

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
  
map<int,Nucleus*> Nucleus::fNuclei;
map<int,vector<Nucleus*> > Nucleus::fIsoList;
map<int,vector<Nucleus*> > Nucleus::fNatIsoList;

ostream& operator<<(ostream& os, const Nucleus& nuc)
{
   os << nuc.Name() << ": A=" << nuc.fA;
   if(nuc.fNatab > 0) os << " Ab=" << nuc.fNatab;
   if(nuc.fIso != 0) os << " Level=" << nuc.fIsolevel << "MeV";
   if(nuc.fLife > 0) os << " Lifetime=" << nuc.fLife << "s";
   return os;
}

//________________________________________________________________________________________________
Nucleus::Nucleus(string name, int n, int z, int iso, double a, double dm, double life, 
		 double natab, double toxa, double toxb, int ind1, int ind2): 
   fName(name), fN(n), fZ(z), fIso(iso), fA(a), fIsolevel(dm), fLife(life),
   fNatab(natab), fToxa(toxa), fToxb(toxb), fInd1(ind1), fInd2(ind2) {
   
   int zniso = 10000 * fZ + 10 * fN + fIso;
   if(fNuclei.count(zniso) != 0) {
      cout << "Nucleus " << zniso << " already there" << endl;
      return;
   }
   
   fNuclei[zniso] = this;
   
   fIsoList[fZ].push_back(this);
   if(natab>0) fNatIsoList[fZ].push_back(this);
}

//________________________________________________________________________________________________
void Nucleus::ReadFile(string infilename, string outfilename) {
   string name;
   int n,z,iso;
   double a, dm, life;
   int da, dz, diso;
   double br, qval, natab, toxa, toxb;
   int ind1, ind2;
   int kcount=0;
   const int ksplit=50;
   int kfunc=0;
   
   bool output=!outfilename.empty();
   ofstream outfile;
   if(output) outfile.open(outfilename);
   ifstream infile(infilename);
   string line;
   getline(infile,line);  // Get title
   Nucleus *nuc=0;
   if(output) {
      outfile << "#ifdef __clang__" << endl;
      outfile << "#pragma clang optimize off" << endl;
      outfile << "#endif" << endl;
      outfile << "#include \"materials/Nucleus.h\"" << endl;
      outfile << "namespace vecgeom {" << endl;
      outfile << "   inline namespace VECGEOM_IMPL_NAMESPACE {" << endl << endl;
   }
   while(getline(infile,line)) {
      Getmat(line, n, z, iso, name, a, dm, life, da, dz, diso, br, qval, 
	     natab, toxa, toxb, ind1, ind2);
      if(z==0) continue ; // skip neutron
      
      int zniso = 10000*z+10*n+iso;
      if(Nucleus::Nuclei().count(zniso) == 0) {
	 nuc = new Nucleus(name, n, z, iso, a, dm, life, natab, toxa, toxb, ind1, ind2);
	 if(output) {
	    if(kcount%ksplit ==0) {
	       if(kcount > 0) {
		  outfile << "}" << endl << endl;
	       }
	       outfile << endl << "//" << setw(80) << setfill('_') << "_" << endl << setfill(' ') << setw(0);
	       outfile << "static void CreateNuclei" << setfill('0') << setw(4) << kfunc << "() {" << endl << setfill(' ') << setw(0);
	       outfile << "   Nucleus *nuc=0;" << endl << endl;
	       ++kfunc;
	    }
	    outfile << endl << "   // Adding " << nuc->Name() << endl;
	    outfile << "   nuc = new Nucleus(" << "\"" << name << "\"" << "," << n<< "," << z<< "," << iso<< "," << a<< "," 
		    << dm<< "," << life<< "," << natab<< "," << toxa<< "," << toxb<< "," << ind1<< "," << ind2 <<");" << endl;
	    ++kcount;
	 }
      }
      if(da != 0 || dz !=0 || diso != 0) {
	 nuc->AddDecay(da, dz, diso, qval, br);
	 if(output) 
	    outfile << "   nuc->AddDecay(" << da<< "," << dz<< "," << diso<< "," << qval<< "," << br << ");" << endl;
      }
   }
   
   for(map<int,Nucleus*>::const_iterator inuc=Nucleus::Nuclei().begin(); inuc != Nucleus::Nuclei().end(); ++inuc) 
      inuc->second->NormDecay();
   
   if(output) {
      outfile << "}" << endl;

      outfile << endl << "//" << setw(80) << setfill('_') << "_" << endl << setfill(' ') << setw(0);
      outfile << "void Nucleus::CreateNuclei() {" << endl;
      outfile << "   static bool initDone=false;" << endl;
      outfile << "   if(initDone) return;" << endl;
      outfile << "   initDone = true;" << endl;
      for(int i=0; i<kfunc; ++i) 
	 outfile << "  CreateNuclei" << setfill('0') << setw(4) << i << "();" << endl;
      outfile << endl;
      outfile << "   for(map<int,Nucleus*>::const_iterator inuc=Nucleus::Nuclei().begin(); inuc != Nucleus::Nuclei().end(); ++inuc)" << endl;
      outfile << "      inuc->second->NormDecay();" << endl;
      outfile << "}" << endl;      
      outfile << " } // End of inline namespace" << endl;
      outfile << " } // End of vecgeom namespace" << endl;
      outfile << "#ifdef __clang__" << endl;
      outfile << "#pragma clang optimize on" << endl;
      outfile << "#endif" << endl;
   }
}

//________________________________________________________________________________________________
void Nucleus::NormDecay() {
   double brt=0;
   for(vector<Decay>::iterator idec=fDecayList.begin(); idec!=fDecayList.end(); ++idec) 
      brt += idec->Br();
   brt = 1/brt;
   for(vector<Decay>::iterator idec=fDecayList.begin(); idec!=fDecayList.end(); ++idec) 
      idec->Br(100*idec->Br()*brt);
}


//________________________________________________________________________________________________
void Nucleus::AddDecay(int da, int dz, int diso, double qval, double br) {
   Decay dec(da,dz,diso,qval,br);
   bool found = false;
   if(std::find(fDecayList.begin(),fDecayList.end(),dec) != fDecayList.end()) {
      cout << "Decay already there!" << endl;
      found = true;
   }
   if(!found) fDecayList.push_back(dec);
}

void Nucleus::Getmat(string line, int &n, int &z, int &iso, string &name, double &a, double &dm, double &life, int &da,
                     int &dz, int &diso, double &br, double &qval, double &natab, double &toxa, double &toxb, int &ind1,
                     int &ind2) {
   int beg=5;
   int ic=0;
   int len[17]={5,5,5,5,15,15,15,5,5,5,15,15,15,15,15,5,5};
   
   beg = 5;
   ic=0;
   stringstream(line.substr(beg,len[ic])) >> n;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> z;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> iso;
   
   beg +=len[ic];
   ++ic;
   name = line.substr(beg,len[ic]);
   name = name.substr(name.find_first_not_of(" "),name.find_last_not_of(" ")-name.find_first_not_of(" ")+1);
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> a;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> dm;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> life;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> da;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> dz;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> diso;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> br;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> qval;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> natab;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> toxa;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> toxb;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> ind1;
   
   beg +=len[ic];
   ++ic;
   stringstream(line.substr(beg,len[ic])) >> ind2;
   
}

//________________________________________________________________________________________________
const string Nucleus::Decay::Name() const {
   stringstream name;
   name << "(" <<fBr<<"%) ";
   if(fDz == -2 && fDa == -4) {
      name<< "Alpha";
      if(fDiso != 0) name<< " iso"; }
   else if(fDz == 1 && fDa == 0) {
      name<< "Beta-";
      if(fDiso != 0) name << " iso"; }
   else if(fDz == -1 && fDa == 0) {
      name << "Beta+";
      if(fDiso != 0) name << " iso"; }
   else if(fDz == 0 && fDa == 0 && fDiso == -1) name << "IC";
   else if(fDz == 1000) name << "Fission";
   else name <<fDa<<":"<<fDz<<":"<<fDiso;
   return name.str();
}

}
}
