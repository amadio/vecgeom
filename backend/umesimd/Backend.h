/// \file umesimd/backend.h
/// \author Przemyslaw Karpinski (przemyslaw.karpinski@cern.ch)

#ifndef VECGEOM_BACKEND_UMESIMDBACKEND_H_
#define VECGEOM_BACKEND_UMESIMDBACKEND_H_

#include "base/Global.h"
#include "backend/scalar/Backend.h"

// Use '()' instead of '[]' brackets for writemask syntax
#define USE_PARENTHESES_IN_MASK_ASSIGNMENT
#include <umesimd/UMESimd.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class UmeSimdMask;
class UmeSimdIntegerVector;
class UmeSimdInsideVector;
class UmeSimdPrecisionVector;

struct kUmeSimd {
  typedef UmeSimdMask bool_v;
  typedef UmeSimdIntegerVector int_v;
  typedef UmeSimdPrecisionVector precision_v;
  typedef UmeSimdInsideVector inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  typedef UmeSimdMask Bool_t;
};

#ifdef kVectorSize
#undef kVectorSize
#endif

/* UME::SIMD allows user to control the length of a vector. Since
   vecgeom is not providing this parameter, it is necessary to
   explicitly select vector lengths to be used for different instruction
   set. */
#ifdef VECGEOM_FLOAT_PRECISION
#if defined(__AVX512F__)
constexpr int kVectorSize = 16;
#elif defined(__MIC__)
constexpr int kVectorSize = 16;
#elif defined(__AVX__)
constexpr int kVectorSize = 8;
#elif defined(__SSE__)
constexpr int kVectorSize = 4;
#else // Default fallback to scalar emulation
constexpr int kVectorSize = 1;
#endif
#else
#if defined(__AVX512F__)
constexpr int kVectorSize = 8;
#elif defined(__MIC__)
constexpr int kVectorSize = 8;
#elif defined(__AVX__)
constexpr int kVectorSize = 4;
#elif defined(__SSE__)
constexpr int kVectorSize = 2;
#else // Default fallback to scalar emulation
constexpr int kVectorSize = 1;
#endif
#endif

#ifdef VECGEOM_FLOAT_PRECISION
class UmeSimdMask : public UME::SIMD::SIMDVecMask<kVectorSize> {
public:
  UmeSimdMask() : UME::SIMD::SIMDVecMask<kVectorSize>() {}
  UmeSimdMask(int mm) : UME::SIMD::SIMDVecMask<kVectorSize>(bool(mm)) {}
  UmeSimdMask(UME::SIMD::SIMDVecMask<kVectorSize> const &m) : UME::SIMD::SIMDVecMask<kVectorSize>(m) {}

  const static int Size = kVectorSize;
};

class UmeSimdIntegerVector : public UME::SIMD::SIMDVec_i<int32_t, kVectorSize> {
public:
  UmeSimdIntegerVector() : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>() {}
  UmeSimdIntegerVector(const int i) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(i) {}
  UmeSimdIntegerVector(const int i[kVectorSize]) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(&i[0]) {}
  UmeSimdIntegerVector(UME::SIMD::SIMDVec_i<int32_t, kVectorSize> const &mm)
      : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(mm)
  {
  }

  const static int Size = kVectorSize;
};

class UmeSimdInsideVector : public UME::SIMD::SIMDVec_i<int32_t, kVectorSize> {
public:
  UmeSimdInsideVector() : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>() {}
  UmeSimdInsideVector(const int32_t i) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(i) {}
  UmeSimdInsideVector(const int32_t i[kVectorSize]) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(&i[0]) {}
  UmeSimdInsideVector(UME::SIMD::SIMDVec_i<int32_t, kVectorSize> const &mm)
      : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(mm)
  {
  }

  const static int Size = kVectorSize;
};

class UmeSimdPrecisionVector : public UME::SIMD::SIMDVec_f<float, kVectorSize> {
public:
  UmeSimdPrecisionVector() : UME::SIMD::SIMDVec_f<float, kVectorSize>() {}
  UmeSimdPrecisionVector(const float d) : UME::SIMD::SIMDVec_f<float, kVectorSize>(d) {}
  UmeSimdPrecisionVector(const int i) : UME::SIMD::SIMDVec_f<float, kVectorSize>(float(i)) {}
  UmeSimdPrecisionVector(const float d[kVectorSize]) : UME::SIMD::SIMDVec_f<float, kVectorSize>(&d[0]) {}
  UmeSimdPrecisionVector(UME::SIMD::SIMDVec_f<float, kVectorSize> const &m)
      : UME::SIMD::SIMDVec_f<float, kVectorSize>(m)
  {
  }

  const static int Size = kVectorSize;

  inline UmeSimdPrecisionVector operator*(UmeSimdPrecisionVector const &val) const { return mul(val); }
  inline UmeSimdPrecisionVector operator*(Precision const val) const { return mul(val); }
};

#else
class UmeSimdMask : public UME::SIMD::SIMDVecMask<kVectorSize> {
public:
  UmeSimdMask() : UME::SIMD::SIMDVecMask<kVectorSize>() {}
  UmeSimdMask(int mm) : UME::SIMD::SIMDVecMask<kVectorSize>(bool(mm)) {}
  UmeSimdMask(UME::SIMD::SIMDVecMask<kVectorSize> const &m) : UME::SIMD::SIMDVecMask<kVectorSize>(m) {}

  const static int Size = kVectorSize;
};

class UmeSimdIntegerVector : public UME::SIMD::SIMDVec_i<int64_t, kVectorSize> {
public:
  UmeSimdIntegerVector() : UME::SIMD::SIMDVec_i<int64_t, kVectorSize>() {}
  UmeSimdIntegerVector(const long int i) : UME::SIMD::SIMDVec_i<int64_t, kVectorSize>(i) {}
  // Cop UmeSimdIntegerVector(const long int i[kVectorSize]) : UME::SIMD::SIMDVec_i<int64_t, kVectorSize>(&i[0]) {}
  UmeSimdIntegerVector(UME::SIMD::SIMDVec_i<int64_t, kVectorSize> const &mm)
      : UME::SIMD::SIMDVec_i<int64_t, kVectorSize>(mm)
  {
  }

  const static int Size = kVectorSize;
};

class UmeSimdInsideVector : public UME::SIMD::SIMDVec_i<int32_t, kVectorSize> {
public:
  UmeSimdInsideVector() : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>() {}
  UmeSimdInsideVector(const int i) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(i) {}
  UmeSimdInsideVector(const int i[kVectorSize]) : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(&i[0]) {}
  UmeSimdInsideVector(UME::SIMD::SIMDVec_i<int32_t, kVectorSize> const &mm)
      : UME::SIMD::SIMDVec_i<int32_t, kVectorSize>(mm)
  {
  }

  const static int Size = kVectorSize;
};

class UmeSimdPrecisionVector : public UME::SIMD::SIMDVec_f<double, kVectorSize> {
public:
  UmeSimdPrecisionVector() : UME::SIMD::SIMDVec_f<double, kVectorSize>() {}
  UmeSimdPrecisionVector(const double d) : UME::SIMD::SIMDVec_f<double, kVectorSize>(d) {}
  UmeSimdPrecisionVector(const int i) : UME::SIMD::SIMDVec_f<double, kVectorSize>(double(i)) {}
  UmeSimdPrecisionVector(const long int i) : UME::SIMD::SIMDVec_f<double, kVectorSize>(double(i)) {}
  UmeSimdPrecisionVector(const double d[kVectorSize]) : UME::SIMD::SIMDVec_f<double, kVectorSize>(&d[0]) {}
  UmeSimdPrecisionVector(UME::SIMD::SIMDVec_f<double, kVectorSize> const &m)
      : UME::SIMD::SIMDVec_f<double, kVectorSize>(m)
  {
  }

  const static int Size = kVectorSize;

  inline UmeSimdPrecisionVector operator*(UmeSimdPrecisionVector const &val) const { return mul(val); }
  inline UmeSimdPrecisionVector operator*(Precision const val) const { return mul(val); }
};
#endif

typedef kUmeSimd::int_v UmesimdInteger_v;
typedef kUmeSimd::precision_v UmesimdPrecision_v;
typedef kUmeSimd::bool_v UmesimdBool_v;
typedef kUmeSimd::inside_v UmesimdInside_v;

#define VECGEOM_BACKEND_TYPE kUmeSimd
#define VECGEOM_BACKEND_PRECISION_TYPE UmesimdPrecision_v
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE UmesimdPrecision_v::Size
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) vecgeom::UmesimdPrecision_v(P)
#define VECGEOM_BACKEND_BOOL UmesimdBool_v
#define VECGEOM_BACKEND_INSIDE UmesimdInside_v

VECGEOM_FORCE_INLINE
bool IsFull(UmesimdBool_v const &cond)
{
  return cond.hland();
}

VECGEOM_FORCE_INLINE
bool Any(UmesimdBool_v const &cond)
{
  return cond.hlor();
}

VECGEOM_FORCE_INLINE
bool IsEmpty(UmesimdBool_v const &cond)
{
  return !cond.hlor();
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdPrecisionVector const &what, float *toAddr)
{
  what.store(toAddr);
}

VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdIntegerVector const &what, int *toAddr)
{
  what.store((int32_t *)toAddr);
}

#else
VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdPrecisionVector const &what, double *toAddr)
{
  what.store(toAddr);
}

VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdIntegerVector const &what, int64_t *toAddr)
{
  what.store((int64_t *)toAddr);
}
#endif

VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdMask const &what, bool *toAddr)
{
  what.store(toAddr);
}

VECGEOM_FORCE_INLINE
void StoreTo(UmeSimdInsideVector const &what, Inside_t *toAddr)
{
  what.store((int32_t *)toAddr);
}

VECGEOM_FORCE_INLINE
void CondAssign(UmeSimdMask const &cond, UmeSimdPrecisionVector const &thenval, UmeSimdPrecisionVector const &elseval,
                UmeSimdPrecisionVector *const output)
{
  output->assign(elseval);
  output->assign(cond, thenval);
}

VECGEOM_FORCE_INLINE
void CondAssign(UmeSimdMask const &cond, UmeSimdIntegerVector const &thenval, UmeSimdIntegerVector const &elseval,
                UmeSimdIntegerVector *const output)
{
  output->assign(elseval);
  output->assign(cond, thenval);
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
void CondAssign(UmeSimdMask const &cond, int const thenval, int const elseval, int *output)
{
  UmesimdInteger_v t0(thenval);
  UmesimdInteger_v t1(elseval);
  UmesimdInteger_v t2 = t0.blend(cond, t1);
  t2.store((int32_t *)output);
}

#else
VECGEOM_FORCE_INLINE
void CondAssign(UmeSimdMask const &cond, int const thenval, int const elseval, int *output)
{
  UmesimdInteger_v t0(thenval);
  UmesimdInteger_v t1(elseval);
  UmesimdInteger_v t2 = t0.blend(cond, t1);
  t2.store((int64_t *)output);
}

#endif

VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, UmesimdInside_v const &thenval, UmesimdInside_v *const output)
{
  output->assign(cond, thenval);
}

VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, Inside_t const &thenval, UmesimdInside_v *const output)
{
  output->assign(cond, thenval);
}

VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, UmesimdBool_v const &thenval, UmesimdBool_v *const output)
{
  output->assign(cond, thenval);
}

VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, UmesimdPrecision_v const &thenval, UmesimdPrecision_v *const output)
{
  output->assign(cond, thenval);
}

VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, UmesimdPrecision_v const &thenval, double *const output)
{
  thenval.store(cond, output);
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, Inside_t thenval, Inside_t *const output)
{
  UmesimdInteger_v t0(thenval);
  t0.store(cond, (int32_t *)output);
}
#else
VECGEOM_FORCE_INLINE
void MaskedAssign(UmesimdBool_v const &cond, Inside_t thenval, Inside_t *const output)
{
  UmesimdInteger_v t0(thenval);
  t0.store(cond, (int64_t *)output);
}
#endif

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator-(UmeSimdPrecisionVector const &val1, Precision const &val2)
{
  return val1.sub(val2);
}
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator-(Precision const &val1, UmeSimdPrecisionVector const &val2)
{
  return val2.subfrom(val1);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator+(UmeSimdPrecisionVector const &val1, Precision const &val2)
{
  return val1.add(val2);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator+(Precision const &val1, UmeSimdPrecisionVector const &val2)
{
  return val2.add(val1);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator*(Precision const &val1, UmeSimdPrecisionVector const &val2)
{
  return val2.mul(val1);
}
/*
UmeSimdPrecisionVector operator *(UmeSimdPrecisionVector const & val1, Precision const &val2) {
  return val1.mul(val2);
}*/

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator/(Precision const &val1, UmeSimdPrecisionVector const &val2)
{
  return val2.rcp(val1);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector operator/(UmeSimdPrecisionVector const &val1, Precision const &val2)
{
  return val1.div(val2);
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Abs(UME::SIMD::SIMDVec_f<float, kVectorSize> const &val)
{
  return val.abs();
}
#else
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Abs(UME::SIMD::SIMDVec_f<double, kVectorSize> const &val)
{
  return val.abs();
}
#endif

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Abs(UmeSimdPrecisionVector const &val)
{
  return val.abs();
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Sqrt(UME::SIMD::SIMDVec_f<float, kVectorSize> &val)
{
  return val.sqrt();
}
#else
VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Sqrt(UME::SIMD::SIMDVec_f<double, kVectorSize> const &val)
{
  return val.sqrt();
}
#endif

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Sqrt(UmeSimdPrecisionVector const &val)
{
  return val.sqrt();
}

#ifdef VECGEOM_FLOAT_PRECISION
VECGEOM_FORCE_INLINE
UmeSimdMask IsInf(UME::SIMD::SIMDVec_f<float, kVectorSize> &val)
{
  return val.isinf();
}
#else
VECGEOM_FORCE_INLINE
UmeSimdMask IsInf(UME::SIMD::SIMDVec_f<double, kVectorSize> const &val)
{
  return val.isinf();
}
#endif

VECGEOM_FORCE_INLINE
UmeSimdMask IsInf(UmeSimdPrecisionVector const &val)
{
  return val.isinf();
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Max(UmeSimdPrecisionVector const &val1, UmeSimdPrecisionVector const &val2)
{
  return val1.max(val2);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector Min(UmeSimdPrecisionVector const &val1, UmeSimdPrecisionVector const &val2)
{
  return val1.min(val2);
}

VECGEOM_FORCE_INLINE
UmeSimdPrecisionVector ATan2(UmeSimdPrecisionVector const &val1, UmeSimdPrecisionVector const &val2)
{
  return val1.atan2(val2);
}

} // End inline namespace
} // End global namespace

// setting up type traits to be able to interact
// with the VecCore API
namespace vecCore {

template <>
struct TypeTraits<vecgeom::UmeSimdMask> {
  using IndexType  = int;
  using ScalarType = Bool_s;
};

template <>
struct TypeTraits<vecgeom::cxx::UmeSimdPrecisionVector> {
  using ScalarType = double;
  using MaskType   = UME::SIMD::SIMDVecMask<vecgeom::kVectorSize>;
  using IndexType  = vecgeom::cxx::UmeSimdIntegerVector;
};

template <>
struct TypeTraits<vecgeom::cxx::UmeSimdInsideVector> {
  using ScalarType = int;
  using MaskType   = UME::SIMD::SIMDVecMask<vecgeom::kVectorSize>;
  using IndexType  = vecgeom::cxx::UmeSimdIntegerVector;
};
}

// some other TEMPORARY additions until we've made the full transition
// to VecCore
// Will be deleted together with this file
namespace vecCore {
namespace math {

#define UMESIMD_REAL_FUNC(f, name) \
  VECCORE_FORCE_INLINE vecgeom::UmeSimdPrecisionVector f(const vecgeom::UmeSimdPrecisionVector &x) { return x.name(); }

UMESIMD_REAL_FUNC(Abs, abs)
UMESIMD_REAL_FUNC(Exp, exp)
UMESIMD_REAL_FUNC(Log, log)
UMESIMD_REAL_FUNC(Sin, sin)
UMESIMD_REAL_FUNC(Cos, cos)
UMESIMD_REAL_FUNC(Tan, tan)
UMESIMD_REAL_FUNC(Atan, atan)
UMESIMD_REAL_FUNC(Sqrt, sqrt)
UMESIMD_REAL_FUNC(Round, round)
UMESIMD_REAL_FUNC(Floor, floor)
UMESIMD_REAL_FUNC(Ceil, ceil)

#undef UMESIMD_REAL_FUNC

} // end namespace math

VECGEOM_FORCE_INLINE
bool MaskFull(const vecgeom::UmeSimdMask &cond)
{
  return cond.hland();
}

VECGEOM_FORCE_INLINE
bool MaskEmpty(const vecgeom::UmeSimdMask &cond)
{
  return !cond.hlor();
}

// overload VecCore backend interface functions for VecGeom UMESIMD types

VECCORE_FORCE_INLINE
void MaskedAssign(vecgeom::UmeSimdInsideVector &dest, const vecgeom::UmeSimdMask &mask,
                  const vecgeom::UmeSimdInsideVector &src)
{
  dest.assign(mask, src);
}

VECCORE_FORCE_INLINE
void MaskedAssign(vecgeom::UmeSimdIntegerVector &dest, const vecgeom::UmeSimdMask &mask,
                  const vecgeom::UmeSimdIntegerVector &src)
{
  dest.assign(mask, src);
}

VECCORE_FORCE_INLINE
void MaskedAssign(vecgeom::UmeSimdPrecisionVector &dest, const UME::SIMD::SIMDVecMask<vecgeom::kVectorSize> &mask,
                  const vecgeom::UmeSimdPrecisionVector &src)
{
  dest.assign(mask, src);
}

VECCORE_FORCE_INLINE
vecgeom::UmeSimdInsideVector Blend(const vecgeom::UmeSimdMask &mask, const vecgeom::UmeSimdInsideVector &tval,
                                   const vecgeom::UmeSimdInsideVector &fval)
{
  return tval.blend(mask, fval);
}

VECCORE_FORCE_INLINE
vecgeom::UmeSimdIntegerVector Blend(const vecgeom::UmeSimdMask &mask, const vecgeom::UmeSimdIntegerVector &tval,
                                    const vecgeom::UmeSimdIntegerVector &fval)
{
  return tval.blend(mask, fval);
}

VECCORE_FORCE_INLINE
vecgeom::UmeSimdPrecisionVector Blend(const UME::SIMD::SIMDVecMask<vecgeom::kVectorSize> &mask,
                                      const vecgeom::UmeSimdPrecisionVector &tval,
                                      const vecgeom::UmeSimdPrecisionVector &fval)
{
  return tval.blend(mask, fval);
}

} // end namespace VecCore

#endif // VECGEOM_BACKEND_VCBACKEND_H_
