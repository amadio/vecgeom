/*
 * ConeTypes.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_

#include <string>
#include "volumes/UnplacedCone.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, UniversalCone, UniversalCone);

#ifndef VECGEOM_NO_SPECIALIZATION

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowCone, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithSmallerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithBiggerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithPiSector, UniversalCone);

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowCone, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithSmallerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithBiggerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithPiSector, UniversalCone);

#endif // VECGEOM_NO_SPECIALIZATION

inline namespace VECGEOM_IMPL_NAMESPACE {
namespace ConeTypes {

#define DEFINE_TRAIT_TYPE(name)                     \
  struct name {                                     \
    static std::string toString() { return #name; } \
  }

// A cone that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_TRAIT_TYPE(UniversalCone);

#ifndef VECGEOM_NO_SPECIALIZATION

// A cone not having rmin or phi sector
DEFINE_TRAIT_TYPE(NonHollowCone);
// A cone without rmin but with a phi sector smaller than pi
DEFINE_TRAIT_TYPE(NonHollowConeWithSmallerThanPiSector);
// A cone without rmin but with a phi sector greater than pi
DEFINE_TRAIT_TYPE(NonHollowConeWithBiggerThanPiSector);
// A cone without rmin but with a phi sector equal to pi
DEFINE_TRAIT_TYPE(NonHollowConeWithPiSector);

// A cone with rmin and no phi sector
DEFINE_TRAIT_TYPE(HollowCone);
// A cone with rmin and a phi sector smaller than pi
DEFINE_TRAIT_TYPE(HollowConeWithSmallerThanPiSector);
// A cone with rmin and a phi sector greater than pi
DEFINE_TRAIT_TYPE(HollowConeWithBiggerThanPiSector);
// A cone with rmin and a phi sector equal to pi
DEFINE_TRAIT_TYPE(HollowConeWithPiSector);

#endif // VECGEOM_NO_SPECIALIZATION

#undef DEFINE_TRAIT_TYPE

// Mapping of cone types to certain characteristics
enum ETreatmentType { kYes = 0, kNo, kUnknown };

// asking for phi treatment
template <typename T>
struct NeedsPhiTreatment {
  static const ETreatmentType value = kYes;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<NonHollowCone> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsPhiTreatment<HollowCone> {
  static const ETreatmentType value = kNo;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<UniversalCone> {
  static const ETreatmentType value = kUnknown;
};

template <typename T>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool checkPhiTreatment(const UnplacedCone &cone)
{
  if (NeedsPhiTreatment<T>::value != kUnknown)
    return NeedsPhiTreatment<T>::value == kYes;
  else
    // could use a direct constant for 2*M_PI here
    return cone.GetDPhi() < 2. * M_PI;
}

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment {
  static const ETreatmentType value = kYes;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<NonHollowCone> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithSmallerThanPiSector> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithBiggerThanPiSector> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithPiSector> {
  static const ETreatmentType value = kNo;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<UniversalCone> {
  static const ETreatmentType value = kUnknown;
};

template <typename T>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool checkRminTreatment(const UnplacedCone &cone)
{
  if (NeedsRminTreatment<T>::value != kUnknown)
    return NeedsRminTreatment<T>::value == kYes;
  else
    return cone.GetRmin1() > 0 || cone.GetRmin2() > 0;
}

// sector size
enum EAngleType { kNoAngle = 0, kSmallerThanPi, kOnePi, kBiggerThanPi, kUnknownAngle };

template <typename T>
struct SectorType {
  static const EAngleType value = kNoAngle;
};

template <>
struct SectorType<UniversalCone> {
  static const EAngleType value = kUnknownAngle;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct SectorType<NonHollowConeWithSmallerThanPiSector> {
  static const EAngleType value = kSmallerThanPi;
};

template <>
struct SectorType<NonHollowConeWithPiSector> {
  static const EAngleType value = kOnePi;
};

template <>
struct SectorType<NonHollowConeWithBiggerThanPiSector> {
  static const EAngleType value = kBiggerThanPi;
};
template <>
struct SectorType<HollowConeWithSmallerThanPiSector> {
  static const EAngleType value = kSmallerThanPi;
};

template <>
struct SectorType<HollowConeWithPiSector> {
  static const EAngleType value = kOnePi;
};

template <>
struct SectorType<HollowConeWithBiggerThanPiSector> {
  static const EAngleType value = kBiggerThanPi;
};

#endif // VECGEOM_NO_SPECIALIZATION

} // end CONETYPES namespace
}
} // End global namespace

#endif /* VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_ */
