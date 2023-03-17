#include "VecGeom/volumes/PlacedAssembly.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedAssembly::PrintType() const
{
  printf("PlacedAssembly");
}

void PlacedAssembly::PrintType(std::ostream &s) const
{
  s << "PlacedAssembly";
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // End namespace vecgeom
