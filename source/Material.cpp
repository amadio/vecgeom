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
   fDensity(0),
   fZ(0),
   fA(0),
   fIndex(0),
   xsecPtr(0)
{
}

//____________________________________________________________________________
Material::~Material() {
}

//____________________________________________________________________________
Material::Material(const char *name, double a, double z, double dens, double radlen,
		   double intlen): fName(name), fUsed(false), fDensity(dens), fZ(z), fA(a), fNelem(1),
			   fIndex(0),xsecPtr(0)
{
   fElements.push_back(Element(a,z,1));
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

ostream& operator<<(ostream& os, const Material &mat)
{
    os << mat.fName << " Z: " << mat.fZ << " A: " << mat.fA ;
    return os;
}


} // end of impl namespace
  
} // end of global namespace
