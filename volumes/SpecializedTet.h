/// @file SpecializedTet.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTET_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTET_H_

#include "base/Global.h"

#include "volumes/kernel/TetImplementation.h"
#include "volumes/PlacedTet.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedTet.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTet = SIMDSpecializedVolImplHelper<TetImplementation, transCodeT, rotCodeT>;

using SimpleTet = SpecializedTet<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTET_H_
