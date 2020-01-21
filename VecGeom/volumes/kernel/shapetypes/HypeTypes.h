/// \file HypeTypes.h
// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
//
// Contains all possible hype types
///

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_

#include <string>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(HypeTypes, struct, UniversalHype, UniversalHype);

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_DECLARE_NS_CONV(HypeTypes, struct, NonHollowHype, UniversalHype);
VECGEOM_DEVICE_DECLARE_NS_CONV(HypeTypes, struct, HollowHype, UniversalHype);
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {
namespace HypeTypes {

#define DEFINE_HYPE_TYPE(name)                      \
  struct name {                                     \
    VECCORE_ATT_HOST_DEVICE                         \
    static char const *toString() { return #name; } \
  }

// A Hype that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_HYPE_TYPE(UniversalHype);

// A Hype with rmin=0
DEFINE_HYPE_TYPE(NonHollowHype);

// A Hype with rmin!=0
DEFINE_HYPE_TYPE(HollowHype);

#undef DEFINE_HYPE_TYPE

// Mapping of Hype types to certain characteristics
enum ETreatmentType { kYes = 0, kNo, kUnknown };

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment {
  static const ETreatmentType value = kYes;
};
template <>
struct NeedsRminTreatment<NonHollowHype> {
  static const ETreatmentType value = kNo;
};

template <>
struct NeedsRminTreatment<UniversalHype> {
  static const ETreatmentType value = kUnknown;
};

template <typename T, typename UnplacedHype>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool checkInnerSurfaceTreatment(const UnplacedHype &hype)
{

  if (NeedsRminTreatment<T>::value != kUnknown)
    return NeedsRminTreatment<T>::value == kYes;
  else
    return hype.InnerSurfaceExists();
}

} // namespace HypeTypes
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_
