#include "VecGeom/volumes/Plane.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

std::ostream &operator<<(std::ostream &os, Plane const &plane)
{
  os << "{" << plane.GetNormal() << ", " << plane.GetDistance() << "}\n";
  return os;
}

} // End inline implementation namespace

} // End global namespace
