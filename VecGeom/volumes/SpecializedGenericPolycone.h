/// @file SpecializedGenericPolycone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/GenericPolyconeImplementation.h"
#include "VecGeom/volumes/PlacedGenericPolycone.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedGenericPolycone.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedGenericPolycone = SpecializedVolImplHelper<GenericPolyconeImplementation, transCodeT, rotCodeT>;

using SimpleGenericPolycone = SpecializedGenericPolycone<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDGENERICPOLYCONE_H_
