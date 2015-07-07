#include "volumes/Medium.h"

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
    
vector<Medium*> Medium::fMedDB;

//ClassImp(Medium)

//____________________________________________________________________________
Medium::Medium():
   fName("empty"),
   fUsed(false),
   fMat(NULL)
{
   memset(fParams,0,20*sizeof(double));
}

//____________________________________________________________________________
Medium::~Medium() {
}

//____________________________________________________________________________
Medium::Medium(const char *name, Material *mat, double params[20])
   : fName(name), fUsed(false), fMat(mat)
{
   memcpy(fParams,params,20*sizeof(double));
   fMedDB.push_back(this);
}

} // end of impl namespace
  
} // end of global namespace

