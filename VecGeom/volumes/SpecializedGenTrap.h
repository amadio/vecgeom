/*
 * SpecializedGenTrap.h
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_
#define VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenTrapImplementation.h"
#include "VecGeom/volumes/PlacedGenTrap.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

#include <stdio.h>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedGenTrap = SIMDSpecializedVolImplHelper<GenTrapImplementation, transCodeT, rotCodeT>;

using SimpleGenTrap = SpecializedGenTrap<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif /* VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_ */
