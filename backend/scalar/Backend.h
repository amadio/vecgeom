/// \file scalar/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "base/Global.h"

#include <algorithm>
#include <cstring>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

struct kScalar {
  typedef int int_v;
  typedef Precision precision_v;
  typedef bool bool_v;
  typedef Inside_t inside_v;
  // alternative typedefs ( might supercede above typedefs )
  typedef int Int_t;
  typedef Precision Double_t;
  typedef bool Bool_t;
  typedef int Index_t; // the type of indices

  constexpr static precision_v kOne  = 1.0;
  constexpr static precision_v kZero = 0.0;
  const static bool_v kTrue          = true;
  const static bool_v kFalse         = false;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static VECGEOM_CONSTEXPR_RETURN bool IsEqual()
  {
    return false;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static Precision Convert(Precision const &input) { return input; }
};

template <>
VECGEOM_CUDA_HEADER_BOTH
inline VECGEOM_CONSTEXPR_RETURN bool kScalar::IsEqual<kScalar>()
{
  return true;
}

typedef kScalar::int_v ScalarInt;
typedef kScalar::precision_v ScalarDouble;
typedef kScalar::bool_v ScalarBool;

#ifdef VECGEOM_SCALAR
constexpr size_t kVectorSize = 1;
#define VECGEOM_BACKEND_TYPE vecgeom::kScalar
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) (*(P))
#define VECGEOM_BACKEND_PRECISION_TYPE vecgeom::Precision
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE 1
//#define VECGEOM_BACKEND_PRECISION_NOT_SCALAR
#define VECGEOM_BACKEND_BOOL vecgeom::ScalarBool
#define VECGEOM_BACKEND_INSIDE vecgeom::kScalar::inside_v
#endif

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void CondAssign(const bool cond, Type const &thenval, Type const &elseval, Type *const output)
{
  *output = (cond) ? thenval : elseval;
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void MaskedAssign(const bool cond, Type const &thenval, Type *const output)
{
  *output = (cond) ? thenval : *output;
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void StoreTo(Type const &what, Type *toAddr)
{
  *toAddr = what;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_FORCE_INLINE
bool IsFull(bool const &cond)
{
  return cond;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_FORCE_INLINE
bool Any(bool const &cond)
{
  return cond;
}

#ifndef VECGEOM_USOLIDS
template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void swap(Type &a, Type &b)
{
  std::swap(a, b);
}
#endif

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void copy(Type const *begin, Type const *const end, Type *const target)
{
#ifndef VECGEOM_NVCC_DEVICE
  std::copy(begin, end, target);
#else
  std::memcpy(target, begin, sizeof(Type) * (end - begin));
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void reverse_copy(Type const *const begin, Type const *end, Type *const target)
{
#ifndef VECGEOM_NVCC_DEVICE
  std::reverse_copy(begin, end, target);
#else
  while (--end >= begin)
    *target++ = *end;
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void reverse(Type *begin, Type *end)
{
#ifndef VECGEOM_NVCC_DEVICE
  std::reverse(begin, end);
#else
  while (begin++ < end--)
    swap(begin, end);
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
Type *AlignedAllocate(size_t size)
{
#ifndef VECGEOM_NVCC
  return static_cast<Type *>(_mm_malloc(sizeof(Type) * size, kAlignmentBoundary));
#else
  return new Type[size];
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void AlignedFree(Type *allocated)
{
#ifndef VECGEOM_NVCC
  _mm_free(allocated);
#else
  delete[] allocated;
#endif
}

template <typename IteratorType>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
IteratorType min_element(IteratorType first, IteratorType last)
{
  return std::min_element(first, last);
}

template <typename IteratorType>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
bool all_of(IteratorType first, IteratorType last)
{
  return std::all_of(first, last, [](bool b) { return b; });
}

template <typename InputIterator1, typename InputIterator2>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
bool equal(InputIterator1 first, InputIterator1 last, InputIterator2 target)
{
#ifndef VECGEOM_NVCC_DEVICE
  return std::equal(first, last, target);
#else
  while (first != last) {
    if (*first++ != *target++) return false;
  }
  return true;
#endif
}
}
} // End global namespace

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_
