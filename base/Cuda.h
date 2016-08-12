#ifndef VECGEOM_CUDA_H
#define VECGEOM_CUDA_H

#include <memory>

// Keep compact version of the macros.
// clang-format off

#if (defined(__CUDACC__) || defined(__NVCC__))
  // Compiling with nvcc
  #define VECGEOM_NVCC
  #ifdef __CUDA_ARCH__
    // Compiling device code
    #define VECGEOM_NVCC_DEVICE
  #endif
  #define VECGEOM_IMPL_NAMESPACE cuda
  #define VECGEOM_NAMESPACE ::vecgeom
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
  #define VECGEOM_ALIGNED __align__((64))
  #define VECGEOM_HOST_FORWARD_DECLARE(X) namespace cxx { X }
  #define VECGEOM_DEVICE_FORWARD_DECLARE(X) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV(classOrStruct,X) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,classOrStruct,X,Def) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(classOrStruct,X,ArgType) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(classOrStruct,X,ArgType1,ArgType2) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1t_2v(classOrStruct,X,ArgType1,ArgType2,Def2,ArgType3,Def3) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3) class __QuietSemi
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_4v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3,ArgType4,Def4) class __QuietSemi
  #undef VECGEOM_VC
  #undef VECGEOM_CILK
  #undef VECGEOM_ROOT
  #undef VECGEOM_USOLIDS
  #undef VECGEOM_GEANT4
  #undef VECGEOM_BENCHMARK
#else
  // Not compiling with NVCC
  #define VECGEOM_IMPL_NAMESPACE cxx
  #define VECGEOM_NAMESPACE ::vecgeom
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
  #ifdef VECGEOM_CUDA
    // CUDA is enabled, but currently compiling regular C++ code.
    // This enables methods that interface between C++ and CUDA environments
    #define VECGEOM_CUDA_INTERFACE
  #endif
  namespace vecgeom {
     template <typename DataType> struct kCudaType;
     template <typename DataType> using CudaType_t = typename kCudaType<DataType>::type_t;
     template <> struct kCudaType<float> { using type_t = float; };
     template <> struct kCudaType<double> { using type_t = double; };
     template <> struct kCudaType<int> { using type_t = int; };
  }
  #define VECGEOM_HOST_FORWARD_DECLARE(X)  class __QuietSemi
  #define VECGEOM_DEVICE_FORWARD_DECLARE(X)  namespace cuda { X }  class __QuietSemi

  #define VECGEOM_DEVICE_DECLARE_CONV(classOrStruct,X)                 \
     namespace cuda { classOrStruct X; }                               \
     inline namespace cxx  { classOrStruct X; }                        \
     template <> struct kCudaType<cxx::X> { using type_t = cuda::X; }

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(classOrStruct,X,ArgType) \
     namespace cuda { template <ArgType Arg> classOrStruct X; }         \
     inline namespace cxx  { template <ArgType Arg> classOrStruct X; }  \
     template <ArgType Arg> struct kCudaType<cxx::X<Arg> >              \
     { using type_t = cuda::X<CudaType_t<Arg> >; }

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(classOrStruct,X,ArgType1,ArgType2)   \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }        \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; } \
     template <ArgType1 Arg1,ArgType2 Arg2> struct kCudaType<cxx::X<Arg1,Arg2> >       \
     { using type_t = cuda::X<CudaType_t<Arg1>, CudaType_t<Arg2> >; }

#ifdef VECGEOM_CUDA_VOLUME_SPECIALIZATION

  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,classOrStruct,X,Def)               \
     namespace cuda { namespace NS { classOrStruct X; } }                      \
     inline namespace cxx { namespace NS { classOrStruct X; } }                \
     template <> struct kCudaType<cxx::NS::X> { using type_t = cuda::NS::X; }

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }                \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }         \
     template <ArgType1 Arg1,ArgType2 Arg2> struct kCudaType<cxx::X<Arg1,Arg2> >               \
     { using type_t = cuda::X<Arg1,Arg2 >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }       \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >        \
     { using type_t = cuda::X<Arg1, Arg2, CudaType_t<Arg3> >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1t_2v(classOrStruct,X,ArgType1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }       \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >        \
     { using type_t = cuda::X<CudaType_t<Arg1>, Arg2, Arg3 >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }                \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }         \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >          \
     { using type_t = cuda::X<Arg1,Arg2,Arg3 >; }

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_4v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3,ArgType4,Def4) \
    namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }                 \
    inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }          \
    template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> struct kCudaType<cxx::X<Arg1,Arg2,Arg3,Arg4> >      \
    { using type_t = cuda::X<Arg1,Arg2,Arg3,Arg4 >; }

#else // VECGEOM_CUDA_VOLUME_SPECIALIZATION

  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,classOrStruct,X,Def)                 \
     namespace cuda { namespace NS { classOrStruct Def; } }                      \
     inline namespace cxx { namespace NS { classOrStruct X; } }                  \
     template <> struct kCudaType<cxx::NS::X> { using type_t = cuda::NS::Def; }

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }                \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }         \
     template <ArgType1 Arg1,ArgType2 Arg2> struct kCudaType<cxx::X<Arg1,Arg2> >               \
     { using type_t = cuda::X<Def1, Def2 >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }       \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >        \
     { using type_t = cuda::X<Def1, Def2, CudaType_t<Arg3> >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1t_2v(classOrStruct,X,ArgType1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }       \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >        \
     { using type_t = cuda::X<CudaType_t<Arg1>, Def2, Def3 >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }                \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }         \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> >          \
     { using type_t = cuda::X<Def1,Def2,Def3 >; }
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_4v(classOrStruct,X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3,ArgType4,Def4) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }                \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }         \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> struct kCudaType<cxx::X<Arg1,Arg2,Arg3,Arg4> >     \
     { using type_t = cuda::X<Def1,Def2,Def3,Def4 >; }

#endif // VECGEOM_CUDA_VOLUME_SPECIALIZATION

/* Instead of multiple macro, when we have auto expansion of Template pack we could use:
template <typename... Arguments>
struct kCudaType<cxx::BoxImplementation<Arguments...>  >
   { using type_t = typename cuda::BoxImplementation<CudaType_t<Arguments...> >; };
*/
#endif

// clang-format on

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC
using std::unique_ptr;
#else
template <typename T>
class unique_ptr {
  T *fValue;

public:
  VECGEOM_CUDA_HEADER_BOTH
  unique_ptr(T *in) : fValue(in) {}

  VECGEOM_CUDA_HEADER_BOTH
  ~unique_ptr() { delete fValue; }

  VECGEOM_CUDA_HEADER_BOTH
  T *operator->() { return fValue; }
};

template <typename T>
class unique_ptr<T[]> {
  T *fValue;

public:
  VECGEOM_CUDA_HEADER_BOTH
  unique_ptr(T *in) : fValue(in) {}

  VECGEOM_CUDA_HEADER_BOTH
  ~unique_ptr() { delete[] fValue; }

  VECGEOM_CUDA_HEADER_BOTH
  T &operator[](size_t idx) { return fValue[idx]; }
};
#endif
}
}

#endif
