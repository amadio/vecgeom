/// \file Container3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_CONTAINER3D_H_
#define VECGEOM_BASE_CONTAINER3D_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/Vector3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Implementation>
class Container3D;

template <template <typename> class ImplementationType, typename T>
class Container3D<ImplementationType<T>> {

protected:
  VECCORE_ATT_HOST_DEVICE
  Container3D() {}

  VECCORE_ATT_HOST_DEVICE
  ~Container3D() {}

private:
  typedef ImplementationType<T> Implementation;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Implementation &implementation() { return *static_cast<Implementation *>(this); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Implementation &implementation_const() const { return *static_cast<Implementation const *>(this); }

  typedef T value_type;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t size() const { return implementation_const().size(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t capacity() const { return implementation_const().capacity(); }

  VECGEOM_FORCE_INLINE
  void resize() { implementation().resize(); }

  VECGEOM_FORCE_INLINE
  void reserve() { implementation().reserve(); }

  VECGEOM_FORCE_INLINE
  void clear() { implementation().clear(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<value_type> operator[](size_t index) const { return implementation_const().operator[](index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type const &x(size_t index) const { return implementation_const().x(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type &x(size_t index) { return implementation().x(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type const &y(size_t index) const { return implementation_const().y(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type &y(size_t index) { return implementation().y(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type z(size_t index) const { return implementation_const().z(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  value_type &z(size_t index) { return implementation().z(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void set(size_t index, value_type xval, value_type yval, value_type zval)
  {
    implementation().set(index, xval, yval, zval);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void set(size_t index, Vector3D<value_type> const &vec) { implementation().set(index, vec); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void push_back(const value_type xval, const value_type yval, const value_type zval)
  {
    implementation().push_back(xval, yval, zval);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void push_back(Vector3D<value_type> const &vec) { implementation().push_back(vec); }
};
}
} // End global namespace

#endif // VECGEOM_BASE_CONTAINER3D_H_
