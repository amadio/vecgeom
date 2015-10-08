/// \file GeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/GeoManager.h"
#include "volumes/PlacedVolume.h"
#include <dlfcn.h>
#include "navigation/NavigationState.h"
#include "navigation/ABBoxNavigator.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/UnplacedScaledShape.h"
#include "volumes/LogicalVolume.h"

#include <dlfcn.h>
#include "navigation/NavigationState.h"

#include <stdio.h>
#include <list>
#include <vector>
#include <set>
#include <functional>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VPlacedVolume *GeoManager::gCompactPlacedVolBuffer=nullptr;

void GeoManager::RegisterLogicalVolume(LogicalVolume *const logical_volume) {
  if(!fIsClosed) fLogicalVolumesMap[logical_volume->id()] = logical_volume;
  else{
      std::cerr << "Logical Volume created after geometry is closed --> will not be registered\n";
  }
}

void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume) {
  if(!fIsClosed) fPlacedVolumesMap[placed_volume->id()] = placed_volume;
  else{
      std::cerr << "PlacedVolume created after geometry is closed --> will not be registered\n";
  }
}

void GeoManager::DeregisterLogicalVolume(const int id) {
  if (fLogicalVolumesMap.find(id) != fLogicalVolumesMap.end()) {
    if (fIsClosed) {
      std::cerr << "deregistering an object from GeoManager while geometry is closed\n";
    }
    fLogicalVolumesMap.erase(id);
  }
}

void GeoManager::DeregisterPlacedVolume(const int id) {
  if (fPlacedVolumesMap.find(id) != fPlacedVolumesMap.end()) {
    if (fIsClosed) {
      std::cerr << "deregistering an object from GeoManager while geometry is closed\n";
    }
    fPlacedVolumesMap.erase(id);
  }
}


void GeoManager::CompactifyMemory() {
    // this function will compactify the memory a-posteriori
    // it might be worth investigating other methods that do this directly
    // ( for instance via specialized allocators )


    // ---------------------------------
    // start with just the placedvolumes

    // do a check on a fundamental hypothesis :
    // all placed volume objects have the same size ( so that we can compactify them in an array )
    for( auto v : fPlacedVolumesMap ){
        if( v.second->memory_size() != fWorld->memory_size() )
            std::cerr << "Fatal Warning : placed volume instances have non-uniform size \n";
    }

    unsigned int pvolumecount = fPlacedVolumesMap.size();

    std::vector<VPlacedVolume const *> pvolumes;
    getAllPlacedVolumes(pvolumes);
    // make it a set ( to get rid of potential duplicates )
    std::set<VPlacedVolume const *> pvolumeset(pvolumes.begin(), pvolumes.end());

    std::vector<LogicalVolume const *> lvolumes;
    GetAllLogicalVolumes(lvolumes);
    std::set<LogicalVolume const *> lvolumeset(lvolumes.begin(), lvolumes.end());

    std::cerr << pvolumecount << " vs " << pvolumeset.size() << "\n";
    std::cerr << fLogicalVolumesMap.size() << " vs " << lvolumeset.size() << "\n";

    // conversion map to repair pointers from old to new
    std::map<VPlacedVolume const *, VPlacedVolume const *> conversionmap;

    // allocate the buffer ( consider alignment issues later )
    // BIG NOTE HERE: we cannot call new VPlacedVolume[pvolumecount] as it is a pure virtual class
    // this also means: our mechanism will only work if none of the derived classes of VPlacedVolumes
    // adds a data member and we have to find a way to check or forbid this
    // ( a runtime check is done above )
    gCompactPlacedVolBuffer = (VPlacedVolume *) malloc(pvolumecount*sizeof(VPlacedVolume));

//    // the first element in the buffer has to be the world
//    buffer[0] = *fWorld; // copy assignment of PlacedVolumes
//    // fix the index to pointer map
//    fPlacedVolumesMap[fWorld->id()] = &buffer[0];
//    conversionmap[ fWorld ] = &buffer[0];
//    // free memory ( we should really be doing this with smart pointers --> check CUDA ! )
//    // delete fWorld;
//    // fix the global world pointer
//    fWorld = &buffer[0];

    // go through rest of volumes
    // TODO: we could take an influence on the order here ( to place certain volumes next to each other )
    for( auto v : fPlacedVolumesMap ){
            unsigned int volumeindex = v.first;
            gCompactPlacedVolBuffer[volumeindex] = *v.second;
            fPlacedVolumesMap[volumeindex] = &gCompactPlacedVolBuffer[volumeindex];
            conversionmap[ v.second ] = &gCompactPlacedVolBuffer[volumeindex];
         //   delete v.second;
    }

    // a little reusable lambda for the pointer conversion
    std::function<VPlacedVolume const *(VPlacedVolume const *)> ConvertOldToNew = [&] ( VPlacedVolume const * old ){
        if(conversionmap.find(old) == conversionmap.cend()){
            std::cerr << "CANNOT CONVERT ... probably already done" << std::endl;
            return old;
        }
        return conversionmap[old];
    };

   // fix pointers to placed volumes referenced in all logical volumes
    for( auto v : fLogicalVolumesMap ){
        LogicalVolume * lvol = v.second;
	auto ndaughter = lvol->GetDaughtersp()->size();
        for( decltype(ndaughter) i = 0; i < ndaughter; ++i){
            lvol->GetDaughtersp()->operator[](i) = ConvertOldToNew( lvol->GetDaughtersp()->operator[](i) );
        }
    }

    for( auto v : fLogicalVolumesMap ){
        UnplacedBooleanVolume *bvol;
        if( (bvol = const_cast<UnplacedBooleanVolume *>(dynamic_cast<UnplacedBooleanVolume const *>( v.second->GetUnplacedVolume())))){
            bvol->SetLeft( ConvertOldToNew( bvol->GetLeft() ) );
            bvol->SetRight( ConvertOldToNew( bvol->GetRight() ));
        }
	     // same for scaled shape volume
        UnplacedScaledShape *svol;
        if( (svol = const_cast<UnplacedScaledShape *>(dynamic_cast<UnplacedScaledShape const *>( v.second->GetUnplacedVolume())))){
            svol->SetPlaced( ConvertOldToNew( svol->GetPlaced() ) );
        }	
    }
    // cleanup conversion map ... automatically done

    // fix reference to World in GeoManager ( and everywhere else )
    fWorld = ConvertOldToNew(fWorld);
}


void GeoManager::CloseGeometry() {
    Assert( GetWorld() != NULL, "world volume not set" );
    if(fIsClosed)
      {
	std::cerr << "geometry is already closed; I cannot close it again (very likely this message signifies a substational error !!!\n";
      }
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

    CompactifyMemory();
    vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();

    fIsClosed=true;
}


void GeoManager::LoadGeometryFromSharedLib( std::string libname, bool close ){
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
      VPlacedVolume const * world = create();
      world->PrintType();
      SetWorld( world );

      // close the geometry
      // TODO: This step often necessitates extensive computation and could be done
      // as part of the shared lib load itself
      if( close ) CloseGeometry();
      else{
	std::cerr << "Geometry left open for further manipulation; Please close later\n";
      }
    }
    else {
      std::cerr << "Loading geometry from shared lib failed\n";
    }
    //    dlclose(handle);
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
  fVolumeCount = 0;
  fTotalNodeCount = 0;
  fWorld = nullptr;
  fPlacedVolumesMap.clear();
  fLogicalVolumesMap.clear();
  fVolumeToIndexMap.clear();
  fMaxDepth = -1;
  fIsClosed=false;
  // should we also reset the global static id counts?
  LogicalVolume::gIdCount=0;
  VPlacedVolume::g_id_count=0;
  // delete compact buffer for placed volumes
  if(GeoManager::gCompactPlacedVolBuffer != nullptr) {
      free(gCompactPlacedVolBuffer);
      gCompactPlacedVolBuffer=nullptr;
  }
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
      int size = currentvolume->GetDaughters().size();
      for( int i=0; i<size; ++i )
      {
         visitAllPlacedVolumesWithContext( currentvolume->GetDaughters().operator[](i), visitor, state, level+1 );
      }
      state->Pop();
   }
}

template<typename Container>
__attribute__((noinline))
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
int initSymbols(){
    std::list<NavigationState  * > l;
    std::vector<NavigationState  * > v;
    GeoManager::Instance().getAllPathForLogicalVolume( nullptr, l);
    GeoManager::Instance().getAllPathForLogicalVolume( nullptr, v);
    return l.size() + v.size();
}



} } // End global namespace
