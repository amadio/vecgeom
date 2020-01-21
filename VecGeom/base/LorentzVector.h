/// \file LorentzVector.h
/// \author Federico Carminati (based on existing Vector3D.h)

#ifndef VECGEOM_BASE_LORENTZVECTOR_H_
#define VECGEOM_BASE_LORENTZVECTOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/RNG.h"

#include <cstdlib>
#include <ostream>
#include <string>
#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename T> class LorentzVector;);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
class LorentzRotation;

/**
 * @brief Lorentz dimensional vector class supporting most arithmetic operations.
 * @details If vector acceleration is enabled, the scalar template instantiation
 *          will use vector instructions for operations when possible.
 */
template <typename T>
class LorentzVector : public AlignedBase {

  typedef LorentzVector<T> VecType;

private:
  T fVec[4];

public:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector(const T a, const T b, const T c, const T d) : fVec{a, b, c, d} {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector() : fVec{0, 0, 0, 0} {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector(const T a) : fVec{a, a, a, a} {}

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  LorentzVector(LorentzVector<U> const &other) : fVec{other[0], other[1], other[2], other[3]}
  {
  }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  LorentzVector(Vector3D<U> const &other, T t) : fVec{other[0], other[1], other[2], t}
  {
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector &operator=(LorentzVector const &other)
  {
    if (this != &other) {
      fVec[0] = other[0];
      fVec[1] = other[1];
      fVec[2] = other[2];
      fVec[3] = other[3];
    }
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  operator Vector3D<T> &() { return reinterpret_cast<Vector3D<T> &>(*this); }

  /**
   * Constructs a vector from an std::string of the same format as output by the
   * "<<"-operator for outstreams.
   * @param str String formatted as "(%d, %d, %d %d)".
   */
  VECCORE_ATT_HOST
  LorentzVector(std::string const &str)
  {
    int begin = str.find("(") + 1, end = str.find(",") - 1;
    fVec[0] = std::atof(str.substr(begin, end - begin + 1).c_str());
    begin   = end + 2;
    end     = str.find(",", begin) - 1;
    fVec[1] = std::atof(str.substr(begin, end - begin + 1).c_str());
    begin   = end + 2;
    end     = str.find(",", begin) - 1;
    fVec[2] = std::atof(str.substr(begin, end - begin + 1).c_str());
    begin   = end + 2;
    end     = str.find(")", begin) - 1;
    fVec[3] = std::atof(str.substr(begin, end - begin + 1).c_str());
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &operator[](const int index) { return fVec[index]; }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &operator[](const int index) const { return fVec[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &x() { return fVec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &x() const { return fVec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &y() { return fVec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &y() const { return fVec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &z() { return fVec[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &z() const { return fVec[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &t() { return fVec[3]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &t() const { return fVec[3]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &px() { return fVec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &px() const { return fVec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &py() { return fVec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &py() const { return fVec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &pz() { return fVec[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const &pz() const { return fVec[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T &e() { return fVec[3]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const e() const { return fVec[3]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const m2() const { return Mag2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const m() const { return Mag(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const mt2() const { return e() * e() - pz() * pz(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T const mt() const
  {
    T mmm = mt2();
    return mmm < 0 ? -Sqrt(-mmm) : Sqrt(mmm);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(T const &a, T const &b, T const &c, T const &d)
  {
    fVec[0] = a;
    fVec[1] = b;
    fVec[2] = c;
    fVec[3] = d;
  }

  template <typename U, typename V>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Set(const Vector3D<U> &v, V t)
  {
    fVec[0] = v[0];
    fVec[1] = v[1];
    fVec[2] = v[2];
    fVec[3] = t;
  }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Set(const Vector3D<U> &v)
  {
    fVec[0] = v[0];
    fVec[1] = v[1];
    fVec[2] = v[2];
  }

  VECCORE_ATT_HOST_DEVICE
  void Set(const T a) { Set(a, a, a, a); }

  /// \return the length squared perpendicular to z direction
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Perp2() const { return fVec[0] * fVec[0] + fVec[1] * fVec[1]; }

  /// \return the length perpendicular to z direction
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Perp() const { return Sqrt(Perp2()); }

  /// \set the length perpendicular to z direction
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T SetPerp(T perp) const
  {
    T factor = perp / NonZero(Perp());
    fVec[0] *= factor;
    fVec[1] *= factor;
  }

  /// The dot product of two LorentzVector<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static T Dot(LorentzVector<T> const &left, LorentzVector<U> const &right)
  {
    return -left[0] * right[0] - left[1] * right[1] - left[2] * right[2] + left[3] * right[3];
  }

  /// The dot product of two LorentzVector<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T Dot(LorentzVector<U> const &right) const
  {
    return Dot(*this, right);
  }

  // For UVector3 compatibility. It is equal to normal multiplication.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType MultiplyByComponents(VecType const &other) const { return *this * other; }

  /// \return Squared magnitude of the vector.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Mag2() const { return Dot(*this, *this); }

  /// \return Magnitude of the vector.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Mag() const { return Sqrt(Mag2()); }

  /// \return Azimuthal angle between -pi and pi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Phi() const { return ATan2(fVec[1], fVec[0]); }

  /// \return Polar angle between 0 and pi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Theta() const { return SpaceVector<T>().Theta(); }

  /// Maps each vector entry to a function that manipulates the entry type.
  /// \param f A function of type "T f(const T&)" to map over entries.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Map(T (*f)(const T &))
  {
    fVec[0] = f(fVec[0]);
    fVec[1] = f(fVec[1]);
    fVec[2] = f(fVec[2]);
    fVec[3] = f(fVec[3]);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> Abs() const
  {
    using vecCore::math::Abs;
    return LorentzVector<T>(Abs(fVec[0]), Abs(fVec[1]), Abs(fVec[2]), Abs(fVec[3]));
  }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void MaskedAssign(LorentzVector<U> const &condition, LorentzVector<T> const &value)
  {
    vecCore::MaskedAssign(fVec[0], condition[0], value[0]);
    vecCore::MaskedAssign(fVec[1], condition[1], value[1]);
    vecCore::MaskedAssign(fVec[2], condition[2], value[2]);
    vecCore::MaskedAssign(fVec[3], condition[3], value[3]);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VecType FromCylindrical(T r, T phi, T z, T t) { return VecType(r * cos(phi), r * sin(phi), z, t); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType &FixZeroes()
  {
    using vecCore::math::Abs;
    for (int i = 0; i < 4; ++i) {
      vecCore::MaskedAssign(fVec[i], Abs(fVec[i]) < kTolerance, T(0.0));
    }
    return *this;
  }

// Inplace binary operators

#define LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(OPERATOR)                                               \
  VECCORE_ATT_HOST_DEVICE                                                                                \
  VECGEOM_FORCE_INLINE                                                                                   \
  VecType &operator OPERATOR(const VecType &other)                                                       \
  {                                                                                                      \
    fVec[0] OPERATOR other.fVec[0];                                                                      \
    fVec[1] OPERATOR other.fVec[1];                                                                      \
    fVec[2] OPERATOR other.fVec[2];                                                                      \
    fVec[3] OPERATOR other.fVec[3];                                                                      \
    return *this;                                                                                        \
  }                                                                                                      \
  template <typename V>                                                                                  \
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE VecType &operator OPERATOR(const LorentzVector<V> &other) \
  {                                                                                                      \
    fVec[0] OPERATOR other[0];                                                                           \
    fVec[1] OPERATOR other[1];                                                                           \
    fVec[2] OPERATOR other[2];                                                                           \
    fVec[3] OPERATOR other[3];                                                                           \
    return *this;                                                                                        \
  }                                                                                                      \
  VECCORE_ATT_HOST_DEVICE                                                                                \
  VECGEOM_FORCE_INLINE                                                                                   \
  VecType &operator OPERATOR(const T &scalar)                                                            \
  {                                                                                                      \
    fVec[0] OPERATOR scalar;                                                                             \
    fVec[1] OPERATOR scalar;                                                                             \
    fVec[2] OPERATOR scalar;                                                                             \
    fVec[3] OPERATOR scalar;                                                                             \
    return *this;                                                                                        \
  }
  LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(+=)
  LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(-=)
  LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(*=)
  LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(/=)
  LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP(*)

#undef LORENTZVECTOR_TEMPLATE_INPLACE_BINARY_OP

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  operator bool() const { return fVec[0] && fVec[1] && fVec[2] && fVec[3]; }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<U> SpaceVector() const
  {
    return Vector3D<U>(fVec[0], fVec[1], fVec[2]);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<T> vect() const { return Vector3D<T>(fVec[0], fVec[1], fVec[2]); }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VecType Boost(const Vector3D<U> &beta)
  {
    U beta2 = beta.Mag2();
    if (beta2 <= 1e-16) return LorentzVector(*this);
    U gamma  = Sqrt(1. / (1 - beta2));
    U gamma2 = (gamma - 1.0) / beta2;
    U bdotv  = beta.Dot(SpaceVector<U>());
    fVec[0] += (gamma2 * bdotv + gamma * fVec[3]) * beta[0];
    fVec[1] += (gamma2 * bdotv + gamma * fVec[3]) * beta[1];
    fVec[2] += (gamma2 * bdotv + gamma * fVec[3]) * beta[2];
    fVec[3] = gamma * (fVec[3] + bdotv);
    return *this;
  }

  Vector3D<T> BoostVector() const
  {
    if (fVec[3] == 0) {
      if (vect().Mag2() < kTolerance) {
        return Vector3D<T>(0);
      } else {
        std::cerr << "LorentzVector::boostVector() - "
                  << "boostVector computed for LorentzVector with t=0 -- infinite result" << std::endl;
        return Vector3D<T>(InfinityLength<T>());
      }
    }
    if (m2() <= 0) {
      std::cerr << "LorentzVector::boostVector() - "
                << "boostVector computed for a non-timelike LorentzVector " << std::endl;
      // result will make analytic sense but is physically meaningless
    }
    return (1. / fVec[3]) * vect();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Beta2() const { return (SpaceVector<T>() / fVec[3]).Mag2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Beta() const { return Sqrt(Beta2()); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Gamma2() const { return 1 / (1 - Beta2()); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Gamma() const { return Sqrt(Gamma2()); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T Rapidity() const { return 0.5 * Log((fVec[3] + fVec[2]) / (fVec[3] - fVec[2])); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  T PseudoRapidity() const { return -Log(Tan(0.5 * Theta())); }

  LorentzVector<T> &operator*=(const LorentzRotation<T> &m1);

  LorentzVector<T> &transform(const LorentzRotation<T> &m1);

  /**
   * This method generates a two body decay
   @param masses [input] a vector of lenght two with the masses of the products
   @param t1     [out]   LorentzVector of the first decay product
   @param t2     [out]   LorentzVector of the second decay product
   @param func   [in]    function defining the angular distribution of the particles if not uniform
  */
  template <typename U, typename V, typename W>
  VECCORE_ATT_HOST_DEVICE
  void TwoBodyDecay(const U masses[], LorentzVector<V> &t1, LorentzVector<W> &t2,
                    void (*func)(double &mthe, double &mphi) = 0) const
  {
    // Make two body decay
    double theta = 0;
    double phi   = 0;
    if (func)
      func(theta, phi);
    else {
      phi   = RNG::Instance().uniform() * kTwoPi;
      theta = RNG::Instance().uniform() * kPi;
    }
    double mass2                          = Mag2();
    if (mass2 > -1e-6 && mass2 < 0) mass2 = 0;
    double dmass                          = masses[0] + masses[1];
    if (mass2 * (1. + 1.e-7) < dmass * dmass || (mass2 == 0 && dmass == 0)) {
      if (dmass != 0)
        std::cout << "WARNING::LorentzVector::TwoBodyDecay: daughters' mass square " << dmass * dmass
                  << " larger than parent " << mass2 << ", no decay " << std::endl;
      t1.Set(0, 0, 0, masses[0]);
      t2.Set(0, 0, 0, masses[1]);
      return;
    }
    const double mass = sqrt(mass2);

    double en1 = 0.5 * (mass * mass + masses[0] * masses[0] - masses[1] * masses[1]) / mass;
    double en2 = 0.5 * (mass * mass - masses[0] * masses[0] + masses[1] * masses[1]) / mass;
    if ((en1 - masses[0] < -1e-10) || (en2 - masses[1] < -1e-10)) {
      std::cout << "WARNING::LorentzVector::TwoBodyDecay en<0  en1-masses[0] " << en1 - masses[0] << " en2-masses[1] "
                << en2 - masses[1] << std::endl;
    }
    double p1 = 0;
    if (en1 < masses[0])
      en1 = masses[0];
    else
      p1      = sqrt((en1 + masses[0]) * (en1 - masses[0]));
    double p2 = 0;
    if (en2 < masses[1])
      en2 = masses[1];
    else
      p2     = sqrt((en2 + masses[1]) * (en2 - masses[1]));
    double p = 0.5 * (p1 + p2);
    // debug
    if (fabs(p1 - p2) / mass > 1e-4)
      std::cout << "WARNING::LorentzVector::TwoBodyDecay: momentum imbalance err " << fabs(p1 - p2) / mass << " p1-p2 "
                << p1 - p2 << " p1 " << p1 << " p2 " << p2 << " mass " << mass << std::endl;
    if (fabs(mass - en1 - en2) / (mass + en1 + en2) / 3 > 1e-7)
      std::cout << "WARNING::LorentzVector::TwoBodyDecay: mass imbalance mass - en1 - en2 " << mass - en1 - en2
                << std::endl;
    // debug

    // Set the two vectors
    const Vector3D<T> space(p1 * sin(theta) * cos(phi), p1 * sin(theta) * sin(phi), p1 * cos(theta));
    t1.Set(space, en1);
    t2.Set(-space, en2);

    if (func) {
      // This part only if there is a matrix element, otherwise isotropic is isotropic
      const Transformation3D tr(SpaceVector<T>(), false);
      tr.Transform(space, static_cast<Vector3D<T> &>(t1));
      tr.Transform(-space, static_cast<Vector3D<T> &>(t2));
    }

    Vector3D<T> beta(-fVec[0] / fVec[3], -fVec[1] / fVec[3], -fVec[2] / fVec[3]);
    t1 = t1.Boost(beta);
    t2 = t2.Boost(beta);
    // dedbug
    static const LorentzVector<T> metric(-1, -1, -1, 1);
    const LorentzVector<T> vdiff = (*this) - t1 - t2;
    const double mdiff =
        sqrt(vdiff.Dot(vdiff * metric) / (Dot((*this) * metric) + t1.Dot(t1 * metric) + t2.Dot(t2 * metric))) / 3;
    if (mdiff > 1e-7)
      std::cout << "WARNING::LorentzVector::TwoBodyDecay error " << mdiff
                << " in momentum energy balance this=" << (*this) << " difference=" << (*this) - t1 - t2 << std::endl;
    // debug
  }

  /**
   * This method generate recursively the phase space via two body decays
   * The method is robust, however it provides a biased distribution
   * this will be fixed in due time, for the moment it is good enough
   @param[in] masses the masses of the products
   @param[out] daughters a vector of LorentzVectors with the decay products
  */
  template <typename U>
  VECCORE_ATT_HOST_DEVICE
  void PhaseSpace(const Vector<U> &masses, Vector<LorentzVector<T>> &daughters)
  {
    // Relativistic phase space
    LorentzVector<T> pa(*this);
    LorentzVector<T> d1;
    LorentzVector<T> d2;

    //
    int ndec = masses.size();
    daughters.clear();
    double mass2 = 0;
    for (int j = 0; j < ndec; ++j)
      mass2 += masses[j];
    if (mass2 > Mag()) {
      std::cout << "LorentzVector::PhaseSpace: cannot decay: parent mass " << Mag() << " sum of daughter mass " << mass2
                << std::endl;
      return;
    }
    for (int j = 0; j < ndec - 1; ++j) {
      mass2 = masses[j + 1];
      if (j < ndec - 2) {
        for (int i = j + 2; i < ndec; ++i)
          mass2 += masses[i];
        double dm          = 0;
        double pam2        = pa.Mag2();
        if (pam2 < 0) pam2 = 0;
        double freen       = sqrt(pam2) - masses[j] - mass2;
        if (mass2 > 0)
          dm = freen *
               std::pow(RNG::Instance().uniform(), masses[j] / (mass2 - 0.5 * (masses[ndec - 1] + masses[ndec - 2])));
        while (dm < 1e-7 * freen)
          dm = freen * RNG::Instance().uniform();
        mass2 += dm;
      }
      double vmass[2] = {masses[j], mass2};
      pa.TwoBodyDecay(vmass, d1, d2);
      pa = d2;
      daughters.push_back(d1);
    }
    daughters.push_back(d2);
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, LorentzVector<T> const &vec)
{
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ")";
  return os;
}

#define LORENTZVECTOR_BINARY_OP(OPERATOR, INPLACE)                                                             \
  template <typename T, typename V>                                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE LorentzVector<T> operator OPERATOR(const LorentzVector<T> &lhs, \
                                                                                  const LorentzVector<V> &rhs) \
  {                                                                                                            \
    LorentzVector<T> result(lhs);                                                                              \
    result INPLACE rhs;                                                                                        \
    return result;                                                                                             \
  }                                                                                                            \
  template <typename T, typename V>                                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE LorentzVector<T> operator OPERATOR(LorentzVector<T> const &lhs, \
                                                                                  const V rhs)                 \
  {                                                                                                            \
    LorentzVector<T> result(lhs);                                                                              \
    result INPLACE rhs;                                                                                        \
    return result;                                                                                             \
  }                                                                                                            \
  template <typename T, typename V>                                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE LorentzVector<T> operator OPERATOR(const V lhs,                 \
                                                                                  LorentzVector<T> const &rhs) \
  {                                                                                                            \
    LorentzVector<T> result(lhs);                                                                              \
    result INPLACE rhs;                                                                                        \
    return result;                                                                                             \
  }
LORENTZVECTOR_BINARY_OP(+, +=)
LORENTZVECTOR_BINARY_OP(-, -=)
LORENTZVECTOR_BINARY_OP(*, *=)
LORENTZVECTOR_BINARY_OP(/, /=)
#undef LORENTZVECTOR_BINARY_OP

VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool operator==(LorentzVector<Precision> const &lhs, LorentzVector<Precision> const &rhs)
{
  return Abs(lhs[0] - rhs[0]) < kTolerance && Abs(lhs[1] - rhs[1]) < kTolerance && Abs(lhs[2] - rhs[2]) < kTolerance &&
         Abs(lhs[3] - rhs[3]) < kTolerance;
}

VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
LorentzVector<bool> operator!=(LorentzVector<Precision> const &lhs, LorentzVector<Precision> const &rhs)
{
  return !(lhs == rhs);
}

template <typename T>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
LorentzVector<T> operator-(LorentzVector<T> const &vec)
{
  return LorentzVector<T>(-vec[0], -vec[1], -vec[2], -vec[3]);
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
LorentzVector<bool> operator!(LorentzVector<bool> const &vec)
{
  return LorentzVector<bool>(!vec[0], !vec[1], !vec[2], !vec[3]);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#define LORENTZVECTOR_SCALAR_BOOLEAN_LOGICAL_OP(OPERATOR)                                               \
  VECCORE_ATT_HOST_DEVICE                                                                               \
  VECGEOM_FORCE_INLINE                                                                                  \
  LorentzVector<bool> operator OPERATOR(LorentzVector<bool> const &lhs, LorentzVector<bool> const &rhs) \
  {                                                                                                     \
    return LorentzVector<bool>(lhs[0] OPERATOR rhs[0], lhs[1] OPERATOR rhs[1], lhs[2] OPERATOR rhs[2],  \
                               lhs[3] OPERATOR rhs[3]);                                                 \
  }
LORENTZVECTOR_SCALAR_BOOLEAN_LOGICAL_OP(&&)
LORENTZVECTOR_SCALAR_BOOLEAN_LOGICAL_OP(||)
#undef LORENTZVECTOR_SCALAR_BOOLEAN_LOGICAL_OP
#pragma GCC diagnostic pop

static const LorentzVector<double> LorentzMetric(-1, -1, -1, 1);

} // End inline namespace
} // End global namespace

#endif // VECGEOM_BASE_LORENTZVECTOR_H_
