/// \file vectorlorentz.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch) + fca

#ifndef VECGEOM_BASE_VECTORLorentz_H_
#define VECGEOM_BASE_VECTORLorentz_H_

#include "base/Global.h"
#include "base/Vector3D.h"

#include "backend/Backend.h"
#ifndef VECGEOM_NVCC
  #if (defined(VECGEOM_VC) || defined(VECGEOM_VC_ACCELERATION))
    #include <Vc/Vc>
  #endif
#endif
#include "base/AlignedBase.h"
#include "base/Vector3D.h"

#include <cstdlib>
#include <ostream>
#include <string>
#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( template <typename Type> class VectorLorentz; )

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Lorentz dimensional vector class supporting most arithmetic operations.
 * @details If vector acceleration is enabled, the scalar template instantiation
 *          will use vector instructions for operations when possible.
 */
template <typename Type>
class VectorLorentz : public AlignedBase {

  typedef VectorLorentz<Type> VecType;

private:

  Type vec[4];

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz(const Type a, const Type b, const Type c, const Type d) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
    vec[3] = d;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz() {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
    vec[3] = 0;
  }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz(const Type a) {
    vec[0] = a;
    vec[1] = a;
    vec[2] = a;
    vec[3] = a;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz(VectorLorentz<TypeOther> const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
    vec[3] = other[3];
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
     VectorLorentz(Vector3D<TypeOther> const &other, Type t) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
    vec[3] = t;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz& operator=(VectorLorentz const &other) {
    if(this != &other) {
       vec[0] = other[0];
       vec[1] = other[1];
       vec[2] = other[2];
       vec[3] = other[3];
    }
    return *this;
  }


  /**
   * Constructs a vector from an std::string of the same format as output by the
   * "<<"-operator for outstreams.
   * @param str String formatted as "(%d, %d, %d %d)".
   */
  VECGEOM_CUDA_HEADER_HOST
  VectorLorentz(std::string const &str) {
    int begin = str.find("(")+1, end = str.find(",")-1;
    vec[0] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(",", begin)-1;
    vec[1] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(",", begin)-1;
    vec[2] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(")", begin)-1;
    vec[3] = std::atof(str.substr(begin, end-begin+1).c_str());
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return vec[index];
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return vec[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& x() { return vec[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& x() const { return vec[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& y() { return vec[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& y() const { return vec[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& z() { return vec[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& z() const { return vec[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& t() { return vec[3]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& t() const { return vec[3]; }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(Type const &a, Type const &b, Type const &c, Type const &d) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
    vec[3] = d;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(const Type a) {
    Set(a, a, a, a);
  }

  /// \return the length squared perpendicular to z direction
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Perp2() const {
    return vec[0]*vec[0]+vec[1]*vec[1];
  }

  /// \return the length perpendicular to z direction
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Perp() const {
    return Sqrt(Perp2());
  }

  template <typename Type2>
  ///The dot product of two VectorLorentz<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  Type Dot(VectorLorentz<Type> const &left, VectorLorentz<Type2> const &right) {
     return left[0]*right[0] + left[1]*right[1] + left[2]*right[2] - left[3]*right[3];
  }

  template <typename Type2>
  /// The dot product of two VectorLorentz<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Dot(VectorLorentz<Type2> const &right) const {
    return Dot(*this, right);
  }

  // For UVector3 compatibility. Is equal to normal multiplication.
  // TODO: check if there are implicit dot products in USolids...
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType MultiplyByComponents(VecType const &other) const {
    return *this * other;
  }

  /// \return Squared magnitude of the vector.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Mag2() const {
    return Dot(*this, *this);
  }

  /// \return Magnitude of the vector.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Mag() const {
    return Sqrt(Mag2());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Length() const {
    return Mag();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Length2() const {
    return Mag2();
  }

  /// \return Azimuthal angle between -pi and pi.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Phi() const {
    //Type output = 0;
    //vecgeom::MaskedAssign(vec[0] != 0. || vec[1] != 0.,
    //                      ATan2(vec[1], vec[0]), &output);
    //return output;
    return ATan2(vec[1], vec[0]);
  }

  /// \return Polar angle between 0 and pi.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Theta() const {
    return ACos(vec[2]/SpaceVector<Type>().Mag());
  }

  /// Maps each vector entry to a function that manipulates the entry type.
  /// \param f A function of type "Type f(const Type&)" to map over entries.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Map(Type (*f)(const Type&)) {
    vec[0] = f(vec[0]);
    vec[1] = f(vec[1]);
    vec[2] = f(vec[2]);
    vec[3] = f(vec[3]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz<Type> Abs() const {
    return VectorLorentz<Type>(vecgeom::Abs(vec[0]),
                          vecgeom::Abs(vec[1]),
                          vecgeom::Abs(vec[2]),
                          vecgeom::Abs(vec[3]));
  }

  template <typename BoolType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void MaskedAssign(VectorLorentz<BoolType> const &condition,
                    VectorLorentz<Type> const &value) {
    vec[0] = (condition[0]) ? value[0] : vec[0];
    vec[1] = (condition[1]) ? value[1] : vec[1];
    vec[2] = (condition[2]) ? value[2] : vec[2];
    vec[3] = (condition[3]) ? value[3] : vec[3];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
     static VecType FromCylindrical(Type r, Type phi, Type z, Type t) {
     return VecType(r*cos(phi), r*sin(phi), z, t);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& FixZeroes() {
    for (int i = 0; i < 4; ++i) {
      vecgeom::MaskedAssign(vecgeom::Abs(vec[i]) < kTolerance, 0., &vec[i]);
    }
    return *this;
  }

  // Inplace binary operators

  #define VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    vec[0] OPERATOR other.vec[0]; \
    vec[1] OPERATOR other.vec[1]; \
    vec[2] OPERATOR other.vec[2]; \
    vec[3] OPERATOR other.vec[3]; \
    return *this; \
  } \
  template <typename OtherType> \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VectorLorentz<OtherType> &other) { \
    vec[0] OPERATOR other[0]; \
    vec[1] OPERATOR other[1]; \
    vec[2] OPERATOR other[2]; \
    vec[3] OPERATOR other[3]; \
    return *this; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const Type &scalar) { \
    vec[0] OPERATOR scalar; \
    vec[1] OPERATOR scalar; \
    vec[2] OPERATOR scalar; \
    vec[3] OPERATOR scalar; \
    return *this; \
  }
  VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP(+=)
  VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP(-=)
  VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP(*=)
  VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP(/=)

  #undef VECTORLorentz_TEMPLATE_INPLACE_BINARY_OP

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  operator bool() const {
    return vec[0] && vec[1] && vec[2] && vec[3];
  }

  template <typename TypeOther> 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<TypeOther> SpaceVector() const {
     return Vector3D<TypeOther>(vec[0], vec[1], vec[2]);
  }

  template <typename TypeOther> 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType Boost(const Vector3D<TypeOther> &beta) const {
    double beta2 = beta.Mag2();
    double gamma = Sqrt(1./(1-beta2));
    double bdotv = beta.Dot(this->SpaceVector());
    return VectorLorentz(this->SpaceVector() + 
			 ((gamma-1)/beta2*bdotv-gamma*vec[3]) * beta,
			 gamma*(vec[3]-bdotv));
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Beta2() const {
     return (SpaceVector<Type>()/vec[4]).Mag2();}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Beta() const {
     return Sqrt(Beta2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Gamma2() const {
     return 1/(1-Beta2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Gamma() const {
     return Sqrt(Gamma2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Rapidity() const {
     return 0.5*Log((vec[4]+vec[3])/(vec[4]-vec[3]));}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type PseudoRapidity() const {
     return -Log(Tan(0.5*Theta()));}

};

template <typename T>
std::ostream& operator<<(std::ostream& os, VectorLorentz<T> const &vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ")";
  return os;
}

#ifdef VECGEOM_VC_ACCELERATION

/// This is a template specialization of class VectorLorentz<double> or
/// VectorLorentz<float> that can provide internal vectorization of common vector
/// operations.
template <>
class VectorLorentz<Precision> : public AlignedBase {

  typedef VectorLorentz<Precision> VecType;
  typedef VectorLorentz<bool> BoolType;
  typedef Vc::Vector<Precision> Base_t;

private:

  Vc::Memory<Vc::Vector<Precision>, 4> mem;

public:

  Precision* AsArray() {
    return &mem[0];
  }

 VectorLorentz(const Precision a, const Precision b, const Precision c, const Precision d) : mem() {
    mem[0] = a;
    mem[1] = b;
    mem[2] = c;
    mem[3] = d;
  }

  // Performance issue in Vc with: mem = a;
  VECGEOM_INLINE
  VectorLorentz(const Precision a) : VectorLorentz(a, a, a, a) {}

  VECGEOM_INLINE
  VectorLorentz() : VectorLorentz(0, 0, 0, 0) {}

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
     VectorLorentz(Vector3D<TypeOther> const &other, Precision t) {
    mem[0] = other[0];
    mem[1] = other[1];
    mem[2] = other[2];
    mem[3] = t;
  }

  VECGEOM_INLINE
    VectorLorentz(VectorLorentz const &other) : mem() {
    //for( int i=0; i < 1 + 3/Base_t::Size; i++ )
//      {
         //Base_t v1 = other.mem.vector(i);
         //this->mem.vector(i)=v1;
       //}
     mem[0]=other.mem[0];
     mem[1]=other.mem[1];
     mem[2]=other.mem[2];
     mem[3]=other.mem[3];
  }

  VECGEOM_INLINE
  VectorLorentz & operator=( VectorLorentz const & rhs )
   {
      if(this != &rhs) {
	 //for( int i=0; i < 1 + 3/Base_t::Size; i++ )
	 //{
         //Base_t v1 = rhs.mem.vector(i);
         //this->mem.vector(i)=v1;
	 //}
	 // the following line must not be used: this is a bug in Vc
	 // this->mem = rhs.mem;
	 mem[0]=rhs.mem[0];
	 mem[1]=rhs.mem[1];
	 mem[2]=rhs.mem[2];
	 mem[3]=rhs.mem[3];
      }
      return *this;
   }

  VectorLorentz(std::string const &str) : mem() {
    int begin = str.find("(")+1, end = str.find(",")-1;
    mem[0] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(",", begin)-1;
    mem[1] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(",", begin)-1;
    mem[2] = std::atof(str.substr(begin, end-begin+1).c_str());
    begin = end + 2;
    end = str.find(")", begin)-1;
    mem[3] = std::atof(str.substr(begin, end-begin+1).c_str());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision& operator[](const int index) {
    return mem[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision& operator[](const int index) const {
    return mem[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision& x() { return mem[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision& x() const { return mem[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision& y() { return mem[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision& y() const { return mem[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision& z() { return mem[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision& z() const { return mem[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision& t() { return mem[3]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision& t() const { return mem[4]; }

  VECGEOM_CUDA_HEADER_BOTH
    void Set(const Precision in_x, const Precision in_y, const Precision in_z, const Precision in_t) {
    mem[0] = in_x;
    mem[1] = in_y;
    mem[2] = in_z;
    mem[3] = in_t;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Set(const Precision in_x) {
     Set(in_x, in_x, in_x, in_x);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Length() const {
    return sqrt(mem[0]*mem[0] + mem[1]*mem[1] + mem[2]*mem[2]-mem[3]*mem[3]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Mag2() const {
      return Dot(*this,*this);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Mag() const {
    return Sqrt(Mag2());
  }

  // TODO: study if we gain from internal vectorization here.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Perp2() const {
    return mem[0]*mem[0] + mem[1]*mem[1];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Perp() const {
    return Sqrt(Perp2());
  }

  VECGEOM_INLINE
  void Map(Precision (*f)(const Precision&)) {
    mem[0] = f(mem[0]);
    mem[1] = f(mem[1]);
    mem[2] = f(mem[2]);
    mem[3] = f(mem[3]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void MaskedAssign(VectorLorentz<bool> const &condition,
                    VectorLorentz<Precision> const &value) {
    mem[0] = (condition[0]) ? value[0] : mem[0];
    mem[1] = (condition[1]) ? value[1] : mem[1];
    mem[2] = (condition[2]) ? value[2] : mem[2];
    mem[3] = (condition[3]) ? value[3] : mem[3];
  }

  /// \return Azimuthal angle between -pi and pi.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Phi() const {
    return (mem[0] != 0. || mem[1] != 0.) ? ATan2(mem[1], mem[0]) : 0.;
  }

  /// \return Polar angle between 0 and pi.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Theta() const {
     Precision theta = ATan2(mem[2],Sqrt(mem[0]*mem[0]+mem[1]*mem[1]));
     if(theta < 0) theta += 3.14159265358979323846264338327950;
     return theta;
  }

  /// \return The dot product of two VectorLorentz<Precision> objects.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  Precision Dot(VectorLorentz<Precision> const &left,
                VectorLorentz<Precision> const &right) {
    // TODO: This function should be internally vectorized (if proven to be
    //       beneficial)

    // To avoid to initialize the padding component, we can not use mem.vector's
    // multiplication and addition since it would accumulate also the (random) padding component
    return left.mem[0]*right.mem[0] + left.mem[1]*right.mem[1] + left.mem[2]*right.mem[2] - left.mem[3]*right.mem[3];
  }

  /// \return The dot product with another VectorLorentz<Precision> object.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Dot(VectorLorentz<Precision> const &right) const {
    return Dot(*this, right);
  }

  // For UVector3 compatibility. Is equal to normal multiplication.
  // TODO: check if there are implicit dot products in USolids...
  VECGEOM_INLINE
  VecType MultiplyByComponents(VecType const &other) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VectorLorentz<Precision>& FixZeroes() {
    for (int i = 0; i < 4; ++i) {
      if (std::abs(mem.scalar(i)) < kTolerance) mem.scalar(i) = 0;
    }
    return *this;
  }

  // Inplace binary operators

  #define VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VecType& operator OPERATOR(const VecType &other) { \
    for (unsigned i = 0; i < 1 + 4/Vc::Vector<Precision>::Size; ++i) { \
      this->mem.vector(i) OPERATOR other.mem.vector(i); \
    } \
    return *this; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VecType& operator OPERATOR(const Precision &scalar) { \
    for (unsigned i = 0; i < 1 + 4/Vc::Vector<Precision>::Size; ++i) { \
      this->mem.vector(i) OPERATOR scalar; \
    } \
    return *this; \
  }
  VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP(+=)
  VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP(-=)
  VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP(*=)
  VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP(/=)
  #undef VECTORLorentz_ACCELERATED_INPLACE_BINARY_OP

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
     static VecType FromCylindrical(Precision r, Precision phi, Precision z, Precision t) {
     return VecType(r*cos(phi), r*sin(phi), z, t);
  }

  template <typename TypeOther> 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<TypeOther> SpaceVector() const {
     return Vector3D<TypeOther>(mem[0], mem[1], mem[2]);
  }

  template <typename TypeOther> 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType Boost(const Vector3D<TypeOther> &beta) const {
    Precision beta2 = beta.Mag2();
    Precision gamma = Sqrt(1./(1-beta2));
    Precision bdotv = beta.Dot(SpaceVector<Precision>());
    return VectorLorentz(SpaceVector<Precision>() + 
			 ((gamma-1)/beta2*bdotv-gamma*mem[3]) * beta,
			 gamma*(mem[3]-bdotv));
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Beta2() const {
     return (SpaceVector<Precision>()/mem[4]).Mag2();}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Beta() const {
     return Sqrt(Beta2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Gamma2() const {
     return 1/(1-Beta2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Gamma() const {
     return Sqrt(Gamma2());}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Rapidity() const {
     return 0.5*Log((mem[4]+mem[3])/(mem[4]-mem[3]));}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision PseudoRapidity() const {
     return -Log(tan(0.5*Theta()));}

};
#else
//#pragma message "using normal VectorLorentz.h"
#endif // VECGEOM_VC_ACCELERATION


#define VECTORLorentz_BINARY_OP(OPERATOR, INPLACE) \
template <typename Type, typename OtherType> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
VectorLorentz<Type> operator OPERATOR(const VectorLorentz<Type> &lhs, \
                                 const VectorLorentz<OtherType> &rhs) { \
  VectorLorentz<Type> result(lhs); \
  result INPLACE rhs; \
  return result; \
} \
template <typename Type, typename ScalarType> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
VectorLorentz<Type> operator OPERATOR(VectorLorentz<Type> const &lhs, \
                                 const ScalarType rhs) { \
  VectorLorentz<Type> result(lhs); \
  result INPLACE rhs; \
  return result; \
} \
template <typename Type, typename ScalarType> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
VectorLorentz<Type> operator OPERATOR(const ScalarType lhs, \
                                 VectorLorentz<Type> const &rhs) { \
  VectorLorentz<Type> result(lhs); \
  result INPLACE rhs; \
  return result; \
}
VECTORLorentz_BINARY_OP(+, +=)
VECTORLorentz_BINARY_OP(-, -=)
VECTORLorentz_BINARY_OP(*, *=)
VECTORLorentz_BINARY_OP(/, /=)
#undef VECTORLorentz_BINARY_OP

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
bool operator==(
    VectorLorentz<Precision> const &lhs,
    VectorLorentz<Precision> const &rhs) {
  return Abs(lhs[0] - rhs[0]) < kTolerance &&
         Abs(lhs[1] - rhs[1]) < kTolerance &&
         Abs(lhs[2] - rhs[2]) < kTolerance &&
         Abs(lhs[3] - rhs[3]) < kTolerance;
}

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
VectorLorentz<bool> operator!=(
    VectorLorentz<Precision> const &lhs,
    VectorLorentz<Precision> const &rhs) {
  return !(lhs == rhs);
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
VectorLorentz<Type> operator-(VectorLorentz<Type> const &vec) {
   return VectorLorentz<Type>(-vec[0], -vec[1], -vec[2], ~vec[3]);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
VectorLorentz<bool> operator!(VectorLorentz<bool> const &vec) {
   return VectorLorentz<bool>(!vec[0], !vec[1], !vec[2], !vec[3]);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#define VECTORLorentz_SCALAR_BOOLEAN_LOGICAL_OP(OPERATOR) \
VECGEOM_CUDA_HEADER_BOTH \
VECGEOM_INLINE \
VectorLorentz<bool> operator OPERATOR(VectorLorentz<bool> const &lhs, \
                                 VectorLorentz<bool> const &rhs) { \
  return VectorLorentz<bool>(lhs[0] OPERATOR rhs[0], \
                        lhs[1] OPERATOR rhs[1], \
			lhs[2] OPERATOR rhs[2],	\
                        lhs[3] OPERATOR rhs[3]); \
}
VECTORLorentz_SCALAR_BOOLEAN_LOGICAL_OP(&&)
VECTORLorentz_SCALAR_BOOLEAN_LOGICAL_OP(||)
#undef VECTORLorentz_SCALAR_BOOLEAN_LOGICAL_OP
#pragma GCC diagnostic pop

#ifdef VECGEOM_VC

VECGEOM_INLINE
VectorLorentz<VcBool> operator!(VectorLorentz<VcBool> const &vec) {
   return VectorLorentz<VcBool>(!vec[0], !vec[1], !vec[2], !vec[3]);
}

VECGEOM_INLINE
VcBool operator==(
    VectorLorentz<VcPrecision> const &lhs,
    VectorLorentz<VcPrecision> const &rhs) {
  return Abs(lhs[0] - rhs[0]) < kTolerance &&
         Abs(lhs[1] - rhs[1]) < kTolerance &&
         Abs(lhs[2] - rhs[2]) < kTolerance &&
         Abs(lhs[3] - rhs[3]) < kTolerance;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#define VECTORLorentz_VC_BOOLEAN_LOGICAL_OP(OPERATOR) \
VECGEOM_INLINE \
VectorLorentz<VcBool> operator OPERATOR( \
    VectorLorentz<VcBool> const &lhs, \
    VectorLorentz<VcBool> const &rhs) { \
  return VectorLorentz<VcBool>(lhs[0] OPERATOR rhs[0], \
                          lhs[1] OPERATOR rhs[1], \
                          lhs[2] OPERATOR rhs[2], \
                          lhs[3] OPERATOR rhs[3]); \
}
VECTORLorentz_VC_BOOLEAN_LOGICAL_OP(&&)
VECTORLorentz_VC_BOOLEAN_LOGICAL_OP(||)
#undef VECTORLorentz_VC_BOOLEAN_LOGICAL_OP
#pragma GCC diagnostic pop

#endif // VECGEOM_VC


#ifdef VECGEOM_VC_ACCELERATION

   /* not sure it makes sense
VectorLorentz<Precision> VectorLorentz<Precision>::Normalized() const {
  return VectorLorentz<Precision>(*this) * (1. / Length());
}
   */

VectorLorentz<Precision> VectorLorentz<Precision>::MultiplyByComponents(
    VecType const &other) const {
  return (*this) * other;
}

#endif

} // End inline namespace

} // End global namespace

#endif // VECGEOM_BASE_VECTORLorentz_H_
