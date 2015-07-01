#include "volumes/Medium.h"

namespace vecgeom {
  inline namespace VECGEOM_IMPL_NAMESPACE {
    


vector<Medium*> Medium::fMedDB;

//ClassImp(Medium)

//____________________________________________________________________________
Medium::Medium():
   fName("empty"),
   fTitle("empty"),
   fUsed(false),
   fMat(NULL)
{
}

//____________________________________________________________________________
Medium::~Medium() {
}

//____________________________________________________________________________
Medium::Medium(const char *name, const char *title, Material *mat)
   : fName(name), fTitle(title), fUsed(false), fMat(mat)
{
}

} // end of impl namespace
  
} // end of global namespace

