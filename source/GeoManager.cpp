/// \file GeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/GeoManager.h"
#include "volumes/PlacedVolume.h"
#include <dlfcn.h>
#include "navigation/NavigationState.h"

#include <dlfcn.h>
#include "navigation/NavigationState.h"

#include <stdio.h>
#include <list>
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void GeoManager::RegisterLogicalVolume(LogicalVolume *const logical_volume) {
  fLogicalVolumesMap[logical_volume->id()] = logical_volume;
}

void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume) {
  fPlacedVolumesMap[placed_volume->id()] = placed_volume;
}

void GeoManager::DeregisterLogicalVolume(const int id) {
  fLogicalVolumesMap.erase(id);
}

void GeoManager::DeregisterPlacedVolume(const int id) {
  fPlacedVolumesMap.erase(id);
}

void GeoManager::CloseGeometry() {
    Assert( GetWorld() != NULL, "world volume not set" );
    // cache some important variables of this geometry
    GetMaxDepthVisitor depthvisitor;
    visitAllPlacedVolumes( GetWorld(), &depthvisitor, 1 );
    fMaxDepth = depthvisitor.getMaxDepth();

    GetTotalNodeCountVisitor totalcountvisitor;
    visitAllPlacedVolumes( GetWorld(), &totalcountvisitor, 1 );
    fTotalNodeCount = totalcountvisitor.GetTotalNodeCount();

    // get a consistent state for index - placed volumes lookups
    for( auto element : fPlacedVolumesMap ){
        fVolumeToIndexMap[element.second] = element.first;
    }
}


void GeoManager::LoadGeometryFromSharedLib( std::string libname ){
    void *handle;
    handle = dlopen(libname.c_str(), RTLD_NOW);
    if (!handle){
        std::cerr << "Error loading geometry shared lib: " << dlerror() << "\n";
    }

    // the create detector "function type":
    typedef VPlacedVolume const * (*CreateFunc_t)();

    // find entry symbol to geometry creation
    // TODO: get rid of hard coded name
    CreateFunc_t create = (CreateFunc_t) dlsym(handle,"_Z16generateDetectorv");

    if (create != nullptr ){
      // call the create function and set the geometry world
      SetWorld( create() );

      // close the geometry
      // TODO: This step often necessitates extensive computation and could be done
      // as part of the shared lib load itself
      CloseGeometry();
    }
    else {
      std::cerr << "Loading geometry from shared lib failed\n";
    }
}


VPlacedVolume* GeoManager::FindPlacedVolume(const int id) {
  auto iterator = fPlacedVolumesMap.find(id);
  return (iterator != fPlacedVolumesMap.end()) ? iterator->second : NULL;
}

VPlacedVolume* GeoManager::FindPlacedVolume(char const *const label) {
  VPlacedVolume *output = NULL;
  bool multiple = false;
  for (auto v = fPlacedVolumesMap.begin(), v_end = fPlacedVolumesMap.end();
       v != v_end; ++v) {
    if (v->second->GetLabel() == label) {
      if (!output) {
        output = v->second;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindPlacedVolume: Multiple placed volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", v->second->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

LogicalVolume* GeoManager::FindLogicalVolume(const int id) {
  auto iterator = fLogicalVolumesMap.find(id);
  return (iterator != fLogicalVolumesMap.end()) ? iterator->second : NULL;
}

LogicalVolume* GeoManager::FindLogicalVolume(char const *const label) {
  LogicalVolume *output = NULL;
  bool multiple = false;
  for (auto v = fLogicalVolumesMap.begin(), v_end = fLogicalVolumesMap.end();
       v != v_end; ++v) {

    const std::string& fullname = v->second->GetLabel();
    if (fullname.compare(label)==0) {
      if (!output) {
        output = v->second;
      } else {
        if (!multiple) {
          multiple = true;
          printf("GeoManager::FindLogicalVolume: Multiple logical volumes with "
                 "identifier \"%s\" found: [%i], ", label, output->id());
        } else {
          printf(", ");
        }
        printf("[%i]", v->second->id());
      }
    }
  }
  if (multiple) printf(". Returning first occurence.\n");
  return output;
}

void GeoManager::Clear()
{
    fLogicalVolumesMap.clear();
    fPlacedVolumesMap.clear();
    fVolumeCount=0; fWorld=NULL;
    fMaxDepth=-1;
}


template<typename Container>
class GetPathsForLogicalVolumeVisitor : public GeoVisitorWithAccessToPath<Container>
{
private:
    LogicalVolume const * fReferenceLogicalVolume;
    int fMaxDepth;
public:
    GetPathsForLogicalVolumeVisitor(
      Container &c, LogicalVolume const * lv, int maxd)
      : GeoVisitorWithAccessToPath<Container>(c), fReferenceLogicalVolume(lv), fMaxDepth(maxd)
    {}

    void apply( NavigationState * state, int /* level */ )
    {
        if( state->Top()->GetLogicalVolume() == fReferenceLogicalVolume ){
            // the current state is a good one;

            // make a copy and store it in the container for this visitor
            NavigationState * copy = NavigationState::MakeCopy( *state );

            this->c_.push_back( copy );
        }
    }
};



template<typename Visitor>
void
GeoManager::visitAllPlacedVolumesWithContext( VPlacedVolume const * currentvolume, Visitor * visitor, NavigationState * state, int level ) const
{
   if( currentvolume != NULL )
   {
      state->Push( currentvolume );
      visitor->apply( state, level );
      int size = currentvolume->daughters().size();
      for( int i=0; i<size; ++i )
      {
         visitAllPlacedVolumesWithContext( currentvolume->daughters().operator[](i), visitor, state, level+1 );
      }
      state->Pop();
   }
}

template<typename Container>
void GeoManager::getAllPathForLogicalVolume( LogicalVolume const * lvol, Container & c ) const
{
   NavigationState * state = NavigationState::MakeInstance(getMaxDepth());
   c.clear();
   state->Clear();

   // instantiate the visitor
   GetPathsForLogicalVolumeVisitor<Container> pv(c, lvol, getMaxDepth());

   // now walk the placed volume hierarchy
   visitAllPlacedVolumesWithContext( GetWorld(), &pv, state );
   NavigationState::ReleaseInstance( state );
}

// init symbols for getAllPathsForLogicalVolume
void initSymbols(){
    std::list<NavigationState  * > l;
    std::vector<NavigationState  * > v;
    GeoManager::Instance().getAllPathForLogicalVolume( nullptr, l);
    GeoManager::Instance().getAllPathForLogicalVolume( nullptr, v);
}



} } // End global namespace
