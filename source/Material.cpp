#include "volumes/Material.h"
#include "backend/Backend.h"
#include <iostream>

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
  

  std::vector<Material*> Material::gMatDB;



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


Material::~Material() {
}


Material::Material(const char *name, double a, double z, double dens, double radlen,
          double intlen): fName(name), fUsed(false), fDensity(dens), fZ(z), fA(a), fNelem(1),
          fIndex(0),xsecPtr(0)
{
   fElements.push_back(Element(a,z,1));
   fIndex = gMatDB.size();
   gMatDB.push_back(this);
}


void Material::GetElementProp(double &ad, double &zd, double &wd, int iel) const
{
   ad = fElements[iel].fA;
   zd = fElements[iel].fZ;
   wd = fElements[iel].fW;
}


void Material::Dump() const {
    std::cout << "Material::Dump() function to be implemented\n";
}

} // end of impl namespace
  
} // end of global namespace
