/// \file Vector2D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR2D_H_
#define VECGEOM_BASE_VECTOR2D_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/backend/scalar/Backend.h"
#include "VecGeom/base/AlignedBase.h"

#include <algorithm>
#include <ostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class Vector2D;);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, Vector2D, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename Type>
class Vector2D : public AlignedBase {

private:
  Type vec[2];

  typedef Vector2D<Type> VecType;

public:
  VECCORE_ATT_HOST_DEVICE
  Vector2D();

  VECCORE_ATT_HOST_DEVICE
  Vector2D(const Type x, const Type y);

  VECCORE_ATT_HOST_DEVICE
  Vector2D(Vector2D const &other);

  VECCORE_ATT_HOST_DEVICE
  VecType operator=(VecType const &other);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type operator[](const int index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &x();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type x() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &y();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type y() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(const Type x, const Type y);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Cross(VecType const &other) const { return vec[0] * other.vec[1] - vec[1] * other.vec[0]; }

#define VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(OPERATOR) \
  VECCORE_ATT_HOST_DEVICE                             \
  VECGEOM_FORCE_INLINE                                \
  VecType &operator OPERATOR(const VecType &other)    \
  {                                                   \
    vec[0] OPERATOR other.vec[0];                     \
    vec[1] OPERATOR other.vec[1];                     \
    return *this;                                     \
  }                                                   \
  VECCORE_ATT_HOST_DEVICE                             \
  VECGEOM_FORCE_INLINE                                \
  VecType &operator OPERATOR(const Type &scalar)      \
  {                                                   \
    vec[0] OPERATOR scalar;                           \
    vec[1] OPERATOR scalar;                           \
    return *this;                                     \
  }
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(+=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(-=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(*=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(/=)
#undef VECTOR2D_TEMPLATE_INPLACE_BINARY_OP
};

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Vector2D<Type>::Vector2D()
{
  vec[0] = 0;
  vec[1] = 0;
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Vector2D<Type>::Vector2D(const Type x, const Type y)
{
  vec[0] = x;
  vec[1] = y;
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Vector2D<Type>::Vector2D(Vector2D const &other)
{
  vec[0] = other.vec[0];
  vec[1] = other.vec[1];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Vector2D<Type> Vector2D<Type>::operator=(Vector2D<Type> const &other)
{
  vec[0] = other.vec[0];
  vec[1] = other.vec[1];
  return *this;
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type &Vector2D<Type>::operator[](const int index)
{
  return vec[index];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type Vector2D<Type>::operator[](const int index) const
{
  return vec[index];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type &Vector2D<Type>::x()
{
  return vec[0];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type Vector2D<Type>::x() const
{
  return vec[0];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type &Vector2D<Type>::y()
{
  return vec[1];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
Type Vector2D<Type>::y() const
{
  return vec[1];
}

template <typename Type>
VECCORE_ATT_HOST_DEVICE
void Vector2D<Type>::Set(const Type x, const Type y)
{
  vec[0] = x;
  vec[1] = y;
}

template <typename Type>
std::ostream &operator<<(std::ostream &os, Vector2D<Type> const &vec)
{
  os << "(" << vec[0] << ", " << vec[1] << ")";
  return os;
}

#define VECTOR2D_BINARY_OP(OPERATOR, INPLACE)                                                              \
  template <typename Type>                                                                                 \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(const Vector2D<Type> &lhs, \
                                                                                const Vector2D<Type> &rhs) \
  {                                                                                                        \
    Vector2D<Type> result(lhs);                                                                            \
    result INPLACE rhs;                                                                                    \
    return result;                                                                                         \
  }                                                                                                        \
  template <typename Type, typename ScalarType>                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(Vector2D<Type> const &lhs, \
                                                                                const ScalarType rhs)      \
  {                                                                                                        \
    Vector2D<Type> result(lhs);                                                                            \
    result INPLACE rhs;                                                                                    \
    return result;                                                                                         \
  }                                                                                                        \
  template <typename Type, typename ScalarType>                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(const ScalarType rhs,      \
                                                                                Vector2D<Type> const &lhs) \
  {                                                                                                        \
    Vector2D<Type> result(rhs);                                                                            \
    result INPLACE lhs;                                                                                    \
    return result;                                                                                         \
  }
VECTOR2D_BINARY_OP(+, +=)
VECTOR2D_BINARY_OP(-, -=)
VECTOR2D_BINARY_OP(*, *=)
VECTOR2D_BINARY_OP(/, /=)
#undef VECTOR2D_BINARY_OP

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_VECTOR2D_H_
