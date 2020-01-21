/*
 * OrbStruct.h
 *
 *  Created on: 11.07.2016
 *      Author: rasehgal
 */

#ifndef VECGEOM_VOLUMES_ORBSTRUCT_H_
#define VECGEOM_VOLUMES_ORBSTRUCT_H_
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// An Orb struct without member functions to encapsulate just the parameters
template <typename T = double>
struct OrbStruct {
  T fR; //<the radius of Orb

  VECCORE_ATT_HOST_DEVICE
  OrbStruct() : fR(0.) {}

  VECCORE_ATT_HOST_DEVICE
  OrbStruct(const T r) : fR(r) {}
};
}
} // end

#endif
