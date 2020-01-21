/// @file: SpecializedTrapezoid.h
/// @author Guilherme Lima (lima@fnal.gov)
//
//  2016-07-22 Guilherme Lima  Created
//

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TrapezoidImplementation.h"
#include "VecGeom/volumes/PlacedTrapezoid.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTrapezoid.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTrapezoid = SIMDSpecializedVolImplHelper<TrapezoidImplementation, transCodeT, rotCodeT>;

using SimpleTrapezoid = SpecializedTrapezoid<translation::kGeneric, rotation::kGeneric>;

} // inline NS
} // vecgeom NS

#endif
