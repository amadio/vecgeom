/// \file Scale3D.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)
#include "base/Scale3D.h"

#include <sstream>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const Scale3D Scale3D::kIdentity =
    Scale3D();

VECGEOM_CUDA_HEADER_BOTH
Scale3D::Scale3D(Precision sx, Precision sy, Precision sz) 
        :fScale(sx,sy,sz), fInvScale(), fSclLocal(1.), fSclMaster(1.) {
  Update();
}

VECGEOM_CUDA_HEADER_BOTH
Scale3D::Scale3D(Vector3D<Precision> const &scale)
        :fScale(scale), fInvScale(), fSclLocal(1.), fSclMaster(1.) {
  Update();
}

VECGEOM_CUDA_HEADER_BOTH
Scale3D::Scale3D(Scale3D const &other)
        :fScale(other.fScale), fInvScale(other.fInvScale), 
         fSclLocal(other.fSclLocal), fSclMaster(other.fSclMaster) { } 

VECGEOM_CUDA_HEADER_BOTH
Scale3D& Scale3D::operator=(Scale3D const &other) {
  fScale = other.fScale;
  fInvScale = other.fInvScale;
  fSclLocal = other.fSclLocal;
  fSclMaster = other.fSclMaster;
  return *this;
}

VECGEOM_CUDA_HEADER_BOTH
void Scale3D::Update() {
  // User-defined scales should be positive
  assert( ((fScale[0]>0) && (fScale[1]>0) && (fScale[2]>0)) );
  fInvScale.Set(1./fScale[0], 1./fScale[1], 1./fScale[2]);
  fSclLocal = Min(fInvScale[0], fInvScale[1]);
  fSclLocal = Min(fSclLocal, fInvScale[2]);
  fSclMaster = Min(fScale[0], fScale[1]);
  fSclMaster = Min(fSclMaster, fScale[2]);    
}

VECGEOM_CUDA_HEADER_BOTH
void Scale3D::SetScale(Vector3D<Precision> const &scale) { 
  fScale = scale;
  Update();
}

VECGEOM_CUDA_HEADER_BOTH
void Scale3D::SetScale(Precision sx, Precision sy, Precision sz) {
  fScale.Set(sx,sy,sz);
  Update();
}
    
VECGEOM_CUDA_HEADER_BOTH
Precision Scale3D::TransformDistance(Precision dist,
                                Vector3D<Precision> &dir) const {
  Precision scale = Sqrt(dir[0]*dir[0]*fInvScale[0]*fInvScale[0] +
                         dir[1]*dir[1]*fInvScale[1]*fInvScale[1] +
                         dir[2]*dir[2]*fInvScale[2]*fInvScale[2]);
  return ( scale*dist );
}
    
VECGEOM_CUDA_HEADER_BOTH
Precision Scale3D::InverseTransformDistance(Precision dist,
                                Vector3D<Precision> &dir) const {
  Precision scale = Sqrt(dir[0]*dir[0]*fScale[0]*fScale[0] +
                         dir[1]*dir[1]*fScale[1]*fScale[1] +
                         dir[2]*dir[2]*fScale[2]*fScale[2]);
  return ( scale*dist );
}

std::ostream& operator<<(std::ostream& os,
                         Scale3D const &scale) {
  os << "Scale " << scale.Scale();
  return os;
}

} // End impl namespace
} // End global namespace

