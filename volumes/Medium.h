#ifndef Medium_H
#define Medium_H


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Medium                                                                 //
//                                                                      //
// Material for GV per material                                         //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// The following is here for the ROOT I/O
//#include "TStorage.h"

// The following only to provide a 1-1 replacement, to be changed

#include <iostream>
#include <vector>

using std::string;
using std::vector;
using std::cout;
using std::endl;

#include "base/Global.h"

namespace vecgeom {
  
  VECGEOM_DEVICE_FORWARD_DECLARE( class Medium; )
  
  inline namespace VECGEOM_IMPL_NAMESPACE {

class Material;

class Medium {

public:
   Medium();
   Medium(const char *name, const char *title, Material *mat);
   virtual ~Medium();
 
   // Getters and setters
   Material* GetMaterial() const {return fMat;}
   bool IsUsed() const {return fUsed;}
   void Used(bool used=true) {fUsed=used;}
   const char* GetName() const {return fName.c_str();}
   const char* GetTitle() const {return fTitle.c_str();}
   void Dump() const {cout << "To be implemented" << endl;}
   static vector<Medium*>& GetMedia() {return fMedDB;}

private:
   Medium(const Medium&);      // Not implemented
   Medium& operator=(const Medium&);      // Not implemented

   static vector<Medium*> fMedDB;

   string fName; // name of the material
   string fTitle; // title of the material
   bool fUsed; // whether the material is used or not
   Material *fMat; //Material

//   ClassDef(Medium,1)  //Medium

};
    
}
} // End global namespace


#endif
