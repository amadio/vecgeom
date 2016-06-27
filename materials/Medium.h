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

#include "base/Global.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Medium;);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Material;

class Medium {

public:
  Medium();
  Medium(const char *name, Material *mat, double params[20]);
  virtual ~Medium();

  // Getters and setters
  Material *GetMaterial() const { return fMat; }
  std::string Name() const { return fName; }
  bool IsUsed() const { return fUsed; }
  void Used(bool used = true) { fUsed = used; }
  const char *GetName() const { return fName.c_str(); }
  void Dump() const { std::cout << "To be implemented" << std::endl; }
  static std::vector<Medium *> &GetMedia() { return fMedDB; }

private:
  Medium(const Medium &);            // Not implemented
  Medium &operator=(const Medium &); // Not implemented

  static std::vector<Medium *> fMedDB;

  std::string fName;  // name of the material
  bool fUsed;         // whether the material is used or not
  Material *fMat;     // Material
  double fParams[20]; // Parameters

  //   ClassDef(Medium,1)  //Medium
};
}
} // End global namespace

#endif
