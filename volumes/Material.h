#ifndef Material_H
#define Material_H


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Material                                                             //
//                                                                      //
// Material for VecGeom                                                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// The following is here for the ROOT I/O
//#include "TStorage.h"

// The following only to provide a 1-1 replacement, to be changed
class TGeoRCExtension;

#include <iostream> 
#include <vector>

#include "base/Global.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace vecgeom {
  
  VECGEOM_DEVICE_FORWARD_DECLARE( class Material; )
  
    inline namespace VECGEOM_IMPL_NAMESPACE {
  
class Material {

public:
   Material();
   Material(const char *name, const char *title, double a, double z, double dens, double radlen=-1,
	double intlen=-1);
   template <typename T, typename U, typename V> 
      Material(const char *name, const char *title, const T a[], const U z[], const V w[], 
	   int nelements, double dens, double radlen=-1,
	   double intlen=-1);
   virtual ~Material();
 
   // Getters and setters
   double GetZ() const {return fZ;}
   bool IsUsed() const {return fUsed;}
   void Used(bool used=true) {fUsed=used;}
   int GetNelements() const {return fNelem;}
   const char* GetName() const {return fName.c_str();}
   const char* GetTitle() const {return fTitle.c_str();}
   void GetElementProp(double &ad, double &zd, double &wd, int iel) const;
   double GetDensity() const {return fDensity;}
   void Dump() const {cout << "To be implemented" << endl;}
   int GetIndex() const {return fIndex;}
   static vector<Material*>& GeMaterials() {return fMatDB;}

   // remove 
   void SetFWExtension(TGeoRCExtension *ext) {fGeoRCExtension = ext;}
   TGeoRCExtension* GetFWExtension() const {return fGeoRCExtension;}


   struct Element {
      Element(): fA(0), fZ(0), fW(0) {}
      Element(double a, double z, double w): fA(a), fZ(z), fW(w) {}
      double fA;
      double fZ;
      double fW;
   };
      
private:
   Material(const Material&);      // Not implemented
   Material& operator=(const Material&);      // Not implemented

   static vector<Material*> fMatDB;

   string fName; // name of the material
   string fTitle; // title of the material
   bool fUsed; // whether the material is used or not
   double fDensity; // density in g/cm3
   double fZ; // z of the material
   int fNelem; // number of element
   int fIndex; // index of the material in the vector
   vector<Element> fElements;

   //remove
   TGeoRCExtension *fGeoRCExtension;

//   ClassDef(Material,1)  //Material X-secs

};

}
} // End global namespace

#endif
