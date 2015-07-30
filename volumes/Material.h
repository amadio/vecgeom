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

/**
 * @brief Material class
 * @details Class to define materials for GeantV
 */

// The following is here for the ROOT I/O
//#include "TStorage.h"

#include <iostream> 
#include <vector>

#include "base/Global.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::ostream;

namespace vecgeom {
  
  VECGEOM_DEVICE_FORWARD_DECLARE( class Material; )
  
    inline namespace VECGEOM_IMPL_NAMESPACE {
  
class Material {

public:
   Material();
   Material(const char *name, double a, double z, double dens, double radlen=-1,
	double intlen=-1);
   template <typename T, typename U, typename V> 
      Material(const char *name, const T a[], const U z[], const V w[], 
	   int nelements, double dens, double radlen=-1,
	   double intlen=-1);
   virtual ~Material();
 
   // Getters and setters
   double GetZ() const {return fZ;}
   double GetA() const {return fA;}
   bool IsUsed() const {return fUsed;}
   void Used(bool used=true) {fUsed=used;}
   int GetNelements() const {return fNelem;}
   const char* GetName() const {return fName.c_str();}
   void GetElementProp(double &ad, double &zd, double &wd, int iel) const;
   double GetDensity() const {return fDensity;}
   void Dump() const {cout << "To be implemented" << endl;}
   int GetIndex() const {return fIndex;}
   template <typename T> int GetRatioBW(vector<T> &rbw) const;
   static vector<Material*>& GetMaterials() {return fMatDB;}

   // remove 
   void SetXsecPtr(void *ptr) {xsecPtr = ptr;}
   void *GetXsecPtr() const {return xsecPtr;}

   friend ostream& operator<<(ostream& os, const Material& mat);

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
   bool fUsed; // whether the material is used or not
   double fDensity; // density in g/cm3
   double fZ; // z of the material
   double fA; // A of the material
   int fNelem; // number of element
   int fIndex; // index of the material in the vector
   vector<Element> fElements;

   //remove
   void *xsecPtr;

//   ClassDef(Material,1)  //Material X-secs

};


//____________________________________________________________________________

/**
 * Material class constructor
 * @param name [in]  name of the material. It must be unique
 * @param a    [in]  array of atomic masses
 * @param z    [in]  array of atomic numbers
 * @param w    [in]  if nelements > 0 array of weight
                     if nelements < 0 array of molecular abundance
 * @param nelements [in] number of elements
 * @param dens      [in] density in g/cm3
 * @param radlen    [in] radiation length
 * @param intlen    [in] interaction length
 */

template <typename T, typename U, typename V> 
Material::Material(const char *name, const T a[], const U z[], const V w[], 
	   int nelements, double dens, double radlen,
		   double intlen): fName(name), fDensity(dens), fZ(0), fA(0), fNelem(abs(nelements))
{
   Element elem;
   double totw=0;
   fA = 0;
   fZ = 0;
   for(int iel=0; iel<fNelem; ++iel) {
      //      cout << "Material::ctor: el#" << iel << " A: " << a[iel] << " Z: " << z[iel] << endl;
      elem.fW = w[iel];
      elem.fA = a[iel];
      elem.fZ = z[iel];
      if(nelements<0) {
	 fA += elem.fW;
	 elem.fW/=elem.fA;
      } else
	 fA += elem.fA*elem.fW;
      totw += elem.fW;
      fZ += elem.fZ*elem.fW;
      fElements.push_back(elem);
   }
   totw = 1/totw;
   fA *= totw;
   fZ *= totw;
   for(vector<Element>::iterator el = fElements.begin(); el!=fElements.end(); ++el) el->fW*=totw;
   fIndex = fMatDB.size();
   fMatDB.push_back(this);
}

template <typename T> int Material::GetRatioBW(vector<T> &rbw) const {
   rbw.clear();
   for(vector<Element>::const_iterator el = fElements.begin(); el!=fElements.end(); ++el) 
      rbw.push_back(el->fW*el->fA/fA);
   return fNelem;
 }


}
} // End global namespace

#endif
