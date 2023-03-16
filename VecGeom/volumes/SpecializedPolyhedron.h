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

template <Polyhedron::EInnerRadii innerRadiiT, Polyhedron::EPhiCutout phiCutoutT>
using SpecializedPolyhedron = SpecializedVolImplHelper<PolyhedronImplementation<innerRadiiT, phiCutoutT>>;

using SimplePolyhedron = SpecializedPolyhedron<Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout::kGeneric>;

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
