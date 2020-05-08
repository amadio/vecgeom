/// \file NavStateFwd.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 12.03.2014
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class NavStatePath;
class NavStateIndex;

#ifdef VECGEOM_USE_NAVINDEX
typedef NavStateIndex NavigationState;
#else
typedef NavStatePath NavigationState;
#endif

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
