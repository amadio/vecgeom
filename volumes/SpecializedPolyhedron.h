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
using SpecializedPolyhedron = ScalarShapeImplementationHelper< PolyhedronImplementation<transCodeT, rotCodeT,innerRadiiT, phiCutoutT> >;


using SimplePolyhedron = SpecializedPolyhedron<translation::kGeneric, rotation::kGeneric, Polyhedron::EInnerRadii::kGeneric,
   Polyhedron::EPhiCutout::kGeneric>;

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_


/*
//Polycone
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedPolycone = ScalarShapeImplementationHelper<PolyconeImplementation<transCodeT, rotCodeT> >;

using SimplePolycone = SpecializedPolycone<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace
 */


/*
//Tube
 namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
using SpecializedTube = ShapeImplementationHelper<TubeImplementation<transCodeT, rotCodeT, tubeTypeT> >;

using SimpleTube = SpecializedTube<translation::kGeneric, rotation::kGeneric, TubeTypes::UniversalTube>;

} } // End global namespace
 */

/*
 //Sphere
 namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedSphere = ShapeImplementationHelper<SphereImplementation<transCodeT, rotCodeT> >;

using SimpleSphere = SpecializedSphere<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace
 */
