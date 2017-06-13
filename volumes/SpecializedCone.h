/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "base/Global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedCone.h"

#include <stdio.h>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename coneTypeT>
using SpecializedCone    = SIMDSpecializedVolImplHelper<ConeImplementation<coneTypeT>, transCodeT, rotCodeT>;
using SimpleCone         = SpecializedCone<translation::kGeneric, rotation::kGeneric, ConeTypes::UniversalCone>;
using SimpleUnplacedCone = SpecializedCone<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>;

} // End impl namespace
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
