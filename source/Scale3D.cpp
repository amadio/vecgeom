/// \file Scale3D.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)
#include "base/Scale3D.h"

#include <sstream>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const Scale3D Scale3D::kIdentity =
    Scale3D();

std::ostream& operator<<(std::ostream& os,
                         Scale3D const &scale) {
  os << "Scale " << scale.Scale();
  return os;
}

} // End impl namespace
} // End global namespace
