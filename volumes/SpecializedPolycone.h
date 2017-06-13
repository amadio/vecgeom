/*
 * SpecializedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_

/*
#include "base/Global.h"

#include "volumes/kernel/PolyconeImplementation.h"
#include "volumes/PlacedPolycone.h"
#include "volumes/ScalarShapeImplementationHelper.h"
*/

#include "base/Global.h"

#include "volumes/kernel/PolyconeImplementation.h"
#include "volumes/PlacedPolycone.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedPolycone.h"
///#include "volumes/ScalarShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>

using SpecializedPolycone = LoopSpecializedVolImplHelper<PolyconeImplementation, transCodeT, rotCodeT>;

using SimplePolycone = SpecializedPolycone<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif /* VECGEOM_VOLUMES_SPECIALIZEDPOLYCONE_H_ */
