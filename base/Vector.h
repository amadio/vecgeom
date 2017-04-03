/// \file Vector.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include "base/Global.h"
#include <initializer_list>
#ifdef VECGEOM_ENABLE_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class Vector;);

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Internal {
template <typename T>
struct AllocTrait {
  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T &obj) { obj.~T(); };
};

template <typename T>
struct AllocTrait<T *> {
  VECCORE_ATT_HOST_DEVICE
  static void Destroy(T *&) {}
};
}

template <typename Type>
class Vector {

private:
  Type *fData;
  size_t fSize, fMemorySize;
  bool fAllocated;

public:
  using value_type = Type;

  VECCORE_ATT_HOST_DEVICE
  Vector() : Vector(5) {}

  VECCORE_ATT_HOST_DEVICE
  Vector(const int maxsize) : fData(nullptr), fSize(0), fMemorySize(0), fAllocated(true) { reserve(maxsize); }

  VECCORE_ATT_HOST_DEVICE
  Vector(Type *const vec, const int sz) : fData(vec), fSize(sz), fMemorySize(sz), fAllocated(false) {}

  VECCORE_ATT_HOST_DEVICE
  Vector(Type *const vec, const int sz, const int maxsize)
      : fData(vec), fSize(sz), fMemorySize(maxsize), fAllocated(false)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  Vector(Vector const &other) : fSize(other.fSize), fMemorySize(other.fMemorySize), fAllocated(true)
  {
    fData = new Type[fMemorySize];
    for (size_t i = 0; i < fSize; ++i)
      fData[i]    = other.fData[i];
  }

  VECCORE_ATT_HOST_DEVICE
  Vector &operator=(Vector const &other)
  {
    if (&other != this) {
      reserve(other.fMemorySize);
      for (size_t i = 0; i < other.fSize; ++i)
        push_back(other.fData[i]);
    }
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  Vector(std::initializer_list<Type> entries)
  {
    fSize       = entries.size();
    fData       = new Type[fSize];
    fAllocated  = true;
    fMemorySize = entries.size() * sizeof(Type);
    for (auto itm : entries)
      this->push_back(itm);
  }

  VECCORE_ATT_HOST_DEVICE
  ~Vector()
  {
    if (fAllocated) delete[] fData;
  }

  VECCORE_ATT_HOST_DEVICE
  void clear()
  {
    if (fAllocated) {
      for (size_t i = 0; i < fSize; ++i)
        Internal::AllocTrait<Type>::Destroy(fData[i]);
    }
    fSize = 0;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index) { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &operator[](const int index) const { return fData[index]; }

  VECCORE_ATT_HOST_DEVICE
  void push_back(const Type item)
  {
    if (fSize == fMemorySize) {
      assert(fAllocated && "Trying to push on a 'fixed' size vector (memory "
                           "not allocated by Vector itself)");
      reserve(fMemorySize << 1);
    }
    new (&fData[fSize]) Type(item);
    fSize++;
  }

  typedef Type *iterator;
  typedef Type const *const_iterator;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator begin() const { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator end() const { return &fData[fSize]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const_iterator cbegin() const { return &fData[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const_iterator cend() const { return &fData[fSize]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t size() const { return fSize; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t capacity() const { return fMemorySize; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void resize(size_t newsize, Type value)
  {
    if (newsize <= fSize) {
      for (size_t i = newsize; i < fSize; ++i) {
        Internal::AllocTrait<Type>::Destroy(fData[i]);
      }
      fSize = newsize;
    } else {
      if (newsize > fMemorySize) {
        reserve(newsize);
      }
      for (size_t i = fSize; i < newsize; ++i)
        push_back(value);
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void reserve(size_t newsize)
  {
    if (newsize <= fMemorySize) {
      // Do nothing ...
    } else {
      Type *newdata = (Type *)new char[newsize * sizeof(Type)];
      for (size_t i = 0; i < fSize; ++i)
        new (&newdata[i]) Type(fData[i]);
      if (fAllocated) delete[] fData;
      fData       = newdata;
      fMemorySize = newsize;
      fAllocated  = true;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  iterator erase(const_iterator position)
  {
    iterator where = (begin() + (position - cbegin()));
    if (where + 1 != end()) {
      auto last = cend();
      for (auto c = where; (where + 1) != last; ++c)
        *c = (*c + 1);
    }
    --fSize;
    if (fSize) Internal::AllocTrait<Type>::Destroy(fData[fSize]);
    return where;
  }

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::Vector<CudaType_t<Type>>> CopyToGpu(DevicePtr<CudaType_t<Type>> const gpu_ptr_arr,
                                                      DevicePtr<cuda::Vector<CudaType_t<Type>>> const gpu_ptr) const
  {
    gpu_ptr.Construct(gpu_ptr_arr, size());
    return gpu_ptr;
  }
#endif

private:
};
}
} // End global namespace

#endif // VECGEOM_BASE_CONTAINER_H_
