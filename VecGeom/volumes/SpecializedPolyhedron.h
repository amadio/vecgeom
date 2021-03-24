/// \file SpecializedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/PolyhedronImplementation.h"
#include "VecGeom/volumes/PlacedPolyhedron.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, Polyhedron::EInnerRadii innerRadiiT,
          Polyhedron::EPhiCutout phiCutoutT>
using SpecializedPolyhedron =
    SpecializedVolImplHelper<PolyhedronImplementation<innerRadiiT, phiCutoutT>, transCodeT, rotCodeT>;

using SimplePolyhedron = SpecializedPolyhedron<translation::kGeneric, rotation::kGeneric,
                                               Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout::kGeneric>;

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
