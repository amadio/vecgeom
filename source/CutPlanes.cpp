#include "VecGeom/volumes/CutPlanes.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void CutPlanes::Set(int index, Vector3D<Precision> const &normal, Vector3D<Precision> const &x0)
{
  fCutPlanes[index].Set(normal, x0);
}

VECCORE_ATT_HOST_DEVICE
void CutPlanes::Set(int index, Vector3D<Precision> const &normal, Precision distance)
{
  fCutPlanes[index].Set(normal, distance);
}

std::ostream &operator<<(std::ostream &os, CutPlanes const &planes)
{
  for (int i = 0; i < 2; ++i) {
    os << "{" << planes.GetCutPlane(0) << ", " << planes.GetCutPlane(1) << "}\n";
  }
  return os;
}

} // End inline implementation namespace

} // End global namespace
