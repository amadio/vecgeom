/// @file SpecializedGenericPolycone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_

#include "base/Global.h"

#include "volumes/kernel/GenericPolyconeImplementation.h"
#include "volumes/PlacedGenericPolycone.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedGenericPolycone.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedGenericPolycone = LoopSpecializedVolImplHelper<GenericPolyconeImplementation, transCodeT, rotCodeT>;

using SimpleGenericPolycone = SpecializedGenericPolycone<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_
