/// \file SOA.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_SOA_H_
#define VECGEOM_BASE_SOA_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/AlignedBase.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T, int rows, int columns>
struct SOAData : public AlignedBase {
  T fHead[rows] VECGEOM_ALIGNED;
  SOAData<T, rows, columns - 1> fTail;
};
template <typename T, int rows>
struct SOAData<T, rows, 0> : public AlignedBase {
};

template <typename T, int columns, int rows>
class SOA : public AlignedBase {

private:
  SOAData<T, rows, columns> fData;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
#ifndef VECCORE_CUDA
  static constexpr int ColumnSize();
#else
  static int ColumnSize();
#endif

public:
  typedef T Column_t[rows];

  SOA() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Column_t &operator[](int index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Column_t const &operator[](int index) const;
};

template <typename T, int columns, int rows>
#ifndef VECCORE_CUDA
constexpr int SOA<T, columns, rows>::ColumnSize()
{
#else
int SOA<T, columns, rows>::ColumnSize()
{
#endif
  return sizeof(SOA<T, columns, rows>) / columns;
}

template <typename T, int columns, int rows>
VECCORE_ATT_HOST_DEVICE
typename SOA<T, columns, rows>::Column_t &SOA<T, columns, rows>::operator[](int index)
{
  return *(&fData.fHead + index * ColumnSize());
}

template <typename T, int columns, int rows>
VECCORE_ATT_HOST_DEVICE
typename SOA<T, columns, rows>::Column_t const &SOA<T, columns, rows>::operator[](int index) const
{
  return *(&fData.fHead + index * ColumnSize());
}

} // End inline namespace

} // End global namespace

#endif // VECGEOM_BASE_SOA_H_
