/*
 * Wedge.cpp
 *
 *  Created on: 28.03.2015
 *      Author: swenzel
 */
#include "base/Global.h"
#include "volumes/Wedge.h"
#include <iostream>
#include <iomanip>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
Wedge::Wedge(Precision angle, Precision zeroangle)
    : // fSPhi(zeroangle),
      fDPhi(angle),
      fAlongVector1(), fAlongVector2()
{
  // check input
  assert(angle > 0.0 && angle <= kTwoPi);

  // initialize angles
  fAlongVector1.x() = std::cos(zeroangle);
  fAlongVector1.y() = std::sin(zeroangle);
  fAlongVector2.x() = std::cos(zeroangle + angle);
  fAlongVector2.y() = std::sin(zeroangle + angle);

  fNormalVector1.x() = -std::sin(zeroangle);
  fNormalVector1.y() = std::cos(zeroangle); // not the + sign
  fNormalVector2.x() = std::sin(zeroangle + angle);
  fNormalVector2.y() = -std::cos(zeroangle + angle); // note the - sign
}
}
}
