/*
 * CutTubeStruct.h
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */
#ifndef VECGEOM_CUTTUBESTRUCT_H_
#define VECGEOM_CUTTUBESTRUCT_H_

#include <VecCore/VecCore>
#include "base/Vector3D.h"

#include "TubeStruct.h"
#include "CutPlanes.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// a plain and lightweight struct to encapsulate data members of a cut tube
template <typename T = double>
struct CutTubeStruct {
  T fDz;                     //< Z half length
  TubeStruct<T> fTubeStruct; //< Tube parameters
  CutPlanes fCutPlanes;      //< Cut planes

  T fCosPhi1; //< Cosine of phi
  T fSinPhi1; //< Sine of phi
  T fCosPhi2; //< Cosine of phi+dphi
  T fSinPhi2; //< Sine of phi+dphi

  // constructors

  VECGEOM_CUDA_HEADER_BOTH
  CutTubeStruct() : fTubeStruct(0., 0., 0., 0., 0.), fCutPlanes() {}

  VECGEOM_CUDA_HEADER_BOTH
  CutTubeStruct(T const &rmin, T const &rmax, T const &z, T const &sphi, T const &dphi, Vector3D<T> const &bottomNormal,
                Vector3D<T> const &topNormal)
      : fDz(z), fTubeStruct(rmin, rmax, kInfLength, sphi, dphi), fCutPlanes()
  {
    fCutPlanes.Set(0, bottomNormal, Vector3D<T>(0., 0., -z));
    fCutPlanes.Set(1, topNormal, Vector3D<T>(0., 0., z));
    fCosPhi1 = vecCore::math::Cos(sphi);
    fSinPhi1 = vecCore::math::Sin(sphi);
    fCosPhi2 = vecCore::math::Cos(sphi + dphi);
    fSinPhi2 = vecCore::math::Sin(sphi + dphi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  TubeStruct<T> const &GetTubeStruct() const { return fTubeStruct; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  CutPlanes const &GetCutPlanes() const { return fCutPlanes; }
};
}
} // end global namespace

#endif
