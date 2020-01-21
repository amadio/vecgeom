/// \file LorentzVector.h
/// \author Andrei Gheata (based on CLHEP/Vector/LorentzRotation.h)

#ifndef VECGEOM_BASE_LORENTZROTATION_H_
#define VECGEOM_BASE_LORENTZROTATION_H_

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/LorentzVector.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename T> class LorentzRotation;);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Lorentz rotation class for performing rotations and boosts on Lorentz vectors
 * @details If vector acceleration is enabled, the scalar template instantiation
 *          will use vector instructions for operations when possible.
 */
template <typename T>
class LorentzRotation : public AlignedBase {

  typedef LorentzVector<T> VecType;

private:
  T fxx, fxy, fxz, fxt, fyx, fyy, fyz, fyt, fzx, fzy, fzz, fzt, ftx, fty, ftz, ftt; // The matrix elements.

public:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzRotation()
      : fxx(1.0), fxy(0.0), fxz(0.0), fxt(0.0), fyx(0.0), fyy(1.0), fyz(0.0), fyt(0.0), fzx(0.0), fzy(0.0), fzz(1.0),
        fzt(0.0), ftx(0.0), fty(0.0), ftz(0.0), ftt(1.0)
  {
  }

  template <typename U>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  LorentzRotation(LorentzRotation<U> const &r)
      : fxx(r.fxx), fxy(r.fxy), fxz(r.fxz), fxt(r.fxt), fyx(r.fyx), fyy(r.fyy), fyz(r.fyz), fyt(r.fyt), fzx(r.fzx),
        fzy(r.fzy), fzz(r.fzz), fzt(r.fzt), ftx(r.ftx), fty(r.fty), ftz(r.ftz), ftt(r.ftt)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzRotation &operator=(LorentzRotation const &r)
  {
    if (this != &r) {
      fxx = r.fxx;
      fxy = r.fxy;
      fxz = r.fxz;
      fxt = r.mxt;
      fyx = r.fyx;
      fyy = r.fyy;
      fyz = r.fyz;
      fyt = r.myt;
      fzx = r.fzx;
      fzy = r.fzy;
      fzz = r.fzz;
      fzt = r.mzt;
      ftx = r.ftx;
      fty = r.fty;
      ftz = r.ftz;
      ftt = r.mtt;
    }
    return *this;
  }

  LorentzVector<T> vectorMultiplication(const LorentzVector<T> &p) const
  {
    T x(p.x());
    T y(p.y());
    T z(p.z());
    T t(p.t());
    return LorentzVector<T>(fxx * x + fxy * y + fxz * z + fxt * t, fyx * x + fyy * y + fyz * z + fyt * t,
                            fzx * x + fzy * y + fzz * z + fzt * t, ftx * x + fty * y + ftz * z + ftt * t);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> col1() const { return LorentzVector<T>(fxx, fyx, fzx, ftx); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> col2() const { return LorentzVector<T>(fxy, fyy, fzy, fty); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> col3() const { return LorentzVector<T>(fxz, fyz, fzz, ftz); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> col4() const { return LorentzVector<T>(fxt, fyt, fzt, ftt); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> row1() const { return LorentzVector<T>(fxx, fxy, fxz, fxt); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> row2() const { return LorentzVector<T>(fyx, fyy, fyz, fyt); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> row3() const { return LorentzVector<T>(fzx, fzy, fzz, fzt); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzVector<T> row4() const { return LorentzVector<T>(ftx, fty, ftz, ftt); }

  VECCORE_ATT_HOST_DEVICE
  LorentzRotation<T> &rotateX(T delta)
  {
    T c1                  = Cos(delta);
    T s1                  = Sin(delta);
    LorentzVector<T> rowy = row2();
    LorentzVector<T> rowz = row3();
    LorentzVector<T> r2   = c1 * rowy - s1 * rowz;
    LorentzVector<T> r3   = s1 * rowy + c1 * rowz;
    fyx                   = r2.x();
    fyy                   = r2.y();
    fyz                   = r2.z();
    fyt                   = r2.t();
    fzx                   = r3.x();
    fzy                   = r3.y();
    fzz                   = r3.z();
    fzt                   = r3.t();
    return *this;
  }

  LorentzRotation<T> &rotateY(T delta)
  {
    T c1                  = std::cos(delta);
    T s1                  = std::sin(delta);
    LorentzVector<T> rowx = row1();
    LorentzVector<T> rowz = row3();
    LorentzVector<T> r1   = c1 * rowx + s1 * rowz;
    LorentzVector<T> r3   = -s1 * rowx + c1 * rowz;
    fxx                   = r1.x();
    fxy                   = r1.y();
    fxz                   = r1.z();
    fxt                   = r1.t();
    fzx                   = r3.x();
    fzy                   = r3.y();
    fzz                   = r3.z();
    fzt                   = r3.t();
    return *this;
  }

  LorentzRotation<T> &rotateZ(T delta)
  {
    T c1                  = std::cos(delta);
    T s1                  = std::sin(delta);
    LorentzVector<T> rowx = row1();
    LorentzVector<T> rowy = row2();
    LorentzVector<T> r1   = c1 * rowx - s1 * rowy;
    LorentzVector<T> r2   = s1 * rowx + c1 * rowy;
    fxx                   = r1.x();
    fxy                   = r1.y();
    fxz                   = r1.z();
    fxt                   = r1.t();
    fyx                   = r2.x();
    fyy                   = r2.y();
    fyz                   = r2.z();
    fyt                   = r2.t();
    return *this;
  }

  LorentzRotation<T> &boostX(T beta)
  {
    T b2                  = beta * beta;
    T g1                  = 1.0 / Sqrt(1.0 - b2);
    T bg                  = beta * g1;
    LorentzVector<T> rowx = row1();
    LorentzVector<T> rowt = row4();
    LorentzVector<T> r1   = g1 * rowx + bg * rowt;
    LorentzVector<T> r4   = bg * rowx + g1 * rowt;
    fxx                   = r1.x();
    fxy                   = r1.y();
    fxz                   = r1.z();
    fxt                   = r1.t();
    ftx                   = r4.x();
    fty                   = r4.y();
    ftz                   = r4.z();
    ftt                   = r4.t();
    return *this;
  }

  LorentzRotation<T> &boostY(T beta)
  {
    T b2                  = beta * beta;
    T g1                  = 1.0 / std::sqrt(1.0 - b2);
    T bg                  = beta * g1;
    LorentzVector<T> rowy = row2();
    LorentzVector<T> rowt = row4();
    LorentzVector<T> r2   = g1 * rowy + bg * rowt;
    LorentzVector<T> r4   = bg * rowy + g1 * rowt;
    fyx                   = r2.x();
    fyy                   = r2.y();
    fyz                   = r2.z();
    fyt                   = r2.t();
    ftx                   = r4.x();
    fty                   = r4.y();
    ftz                   = r4.z();
    ftt                   = r4.t();
    return *this;
  }

  LorentzRotation<T> &boostZ(T beta)
  {
    T b2                  = beta * beta;
    T g1                  = 1.0 / std::sqrt(1.0 - b2);
    T bg                  = beta * g1;
    LorentzVector<T> rowz = row3();
    LorentzVector<T> rowt = row4();
    LorentzVector<T> r3   = g1 * rowz + bg * rowt;
    LorentzVector<T> r4   = bg * rowz + g1 * rowt;
    ftx                   = r4.x();
    fty                   = r4.y();
    ftz                   = r4.z();
    ftt                   = r4.t();
    fzx                   = r3.x();
    fzy                   = r3.y();
    fzz                   = r3.z();
    fzt                   = r3.t();
    return *this;
  }

  /** @brief Matrix multiplication. Note a *= b; <=> a = a * b; while a.transform(b); <=> a = b * a; */
  // LorentzRotation & operator *= (const LorentzRotation & r);
  // LorentzRotation & transform   (const LorentzRotation & r);

protected:
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LorentzRotation(T rxx, T rxy, T rxz, T rxt, T ryx, T ryy, T ryz, T ryt, T rzx, T rzy, T rzz, T rzt, T rtx, T rty,
                  T rtz, T rtt)
      : fxx(rxx), fxy(rxy), fxz(rxz), fxt(rxt), fyx(ryx), fyy(ryy), fyz(ryz), fyt(ryt), fzx(rzx), fzy(rzy), fzz(rzz),
        fzt(rzt), ftx(rtx), fty(rty), ftz(rtz), ftt(rtt)
  {
  }
};

} // End inline namespace
} // End global namespace

#endif // VECGEOM_BASE_LORENTZVECTOR_H_
