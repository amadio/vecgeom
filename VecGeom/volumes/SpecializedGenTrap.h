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

using SpecializedGenTrap = SpecializedVolImplHelper<GenTrapImplementation>;

using SimpleGenTrap = SpecializedGenTrap;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_ */
