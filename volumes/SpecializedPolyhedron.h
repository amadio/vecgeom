/// \file SpecializedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

#include "base/Global.h"

#include "volumes/kernel/PolyhedronImplementation.h"
#include "volumes/PlacedPolyhedron.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT,Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>

//#ifdef GENERATE_POLYHEDRA_SPECIALIZATIONS 
using SpecializedPolyhedron = ScalarShapeImplementationHelper< PolyhedronImplementation<transCodeT, rotCodeT,innerRadiiT, phiCutoutT> >;

using SimplePolyhedron = SpecializedPolyhedron<translation::kGeneric, rotation::kGeneric, Polyhedron::EInnerRadii::kGeneric,
   Polyhedron::EPhiCutout::kGeneric>;

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

