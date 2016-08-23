/*
 * ScaledShapeStruct.h
 *
 *  Created on: 22.08.2016
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_SCALEDSHAPESTRUCT_H_
#define VECGEOM_VOLUMES_SCALEDSHAPESTRUCT_H_
#include "base/Global.h"
#include "base/Scale3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

class VPlacedVolume;
/**
 * A plain struct without member functions to encapsulate just the parameters
 * of the scaled shape
 */
template <typename T = double>
struct ScaledShapeStruct {
  VPlacedVolume const *fPlaced; /// Need a placed volume for the navigation interface
  Scale3D fScale;               /// Scale object

  VECGEOM_CUDA_HEADER_BOTH
  ScaledShapeStruct() : fPlaced(nullptr), fScale() {}

  VECGEOM_CUDA_HEADER_BOTH
  ScaledShapeStruct(VPlacedVolume const *placed, Precision sx, Precision sy, Precision sz)
      : fPlaced(placed), fScale(sx, sy, sz)
  {
  }
};
}
} // end

#endif
