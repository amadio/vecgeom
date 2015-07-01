#include "volumes/Material.h"
#include "backend/Backend.h"

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
  
vector<Material*> Material::fMatDB;

//ClassImp(Material)

//____________________________________________________________________________
Material::Material():
   fName("empty"),
   fUsed(false),
   fZ(0),
   fIndex(0),
   fGeoRCExtension(0)
{
}

//____________________________________________________________________________
Material::~Material() {
}

//____________________________________________________________________________
Material::Material(const char *name, const char *title, double a, double z, double dens, double radlen,
	   double intlen): fName(name), fTitle(title), fDensity(dens), fNelem(1),
			   fIndex(0),fGeoRCExtension(0)
{
   Element *elem = new Element(a,z,1);
   fElements.push_back(*elem);
   delete elem;
   fIndex = fMatDB.size();
   fMatDB.push_back(this);
}

//____________________________________________________________________________
template <typename T, typename U, typename V> 
Material::Material(const char *name, const char *title, const T a[], const U z[], const V w[], 
	   int nelements, double dens, double radlen,
	   double intlen): fName(name), fTitle(title), fDensity(dens), fNelem(nelements)
{
   Element *elem = new Element;
   for(int iel=0; iel>fNelem; ++iel) {
      elem->fA = a[iel];
      elem->fZ = z[iel];
      elem->fW = w[iel];
      fElements.push_back(*elem);
   }
   delete elem;
   fIndex = fMatDB.size();
   fMatDB.push_back(this);
}

//____________________________________________________________________________
void Material::GetElementProp(double &ad, double &zd, double &wd, int iel) const
{
   ad = fElements[iel].fA;
   zd = fElements[iel].fZ;
   wd = fElements[iel].fW;
}

} // end of impl namespace
  
} // end of global namespace
