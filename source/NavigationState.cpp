/// \file NavigationState.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 17.04.2014

#include "VecGeom/navigation/NavigationState.h"

#include <iostream>
#include <list>
#include <sstream>

#ifdef VECGEOM_ROOT
#include "VecGeom/management/RootGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> NavigationState::GlobalToLocal(Vector3D<Precision> const &globalpoint, int tolevel) const
{
  Vector3D<Precision> tmp = globalpoint;
  Vector3D<Precision> current;
  for (int level = 0; level < tolevel; ++level) {
    Transformation3D const *m = At(level)->GetTransformation();
    current                   = m->Transform(tmp);
    tmp                       = current;
  }
  return tmp;
}

VECCORE_ATT_HOST_DEVICE
void NavigationState::TopMatrix(int tolevel, Transformation3D &global_matrix) const
{
  for (int i = 1; i < tolevel; ++i) {
    global_matrix.MultiplyFromRight(*(At(i)->GetTransformation()));
  }
}

// returning a "delta" transformation that can transform
// coordinates given in reference frame of this->Top() to the reference frame of other->Top()
// simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
VECCORE_ATT_HOST_DEVICE
void NavigationState::DeltaTransformation(NavigationState const &other, Transformation3D &delta) const
{
  Transformation3D g2;
  Transformation3D g1;
  other.TopMatrix(g2);
  this->TopMatrix(g1);
  g1.Inverse(delta);
  g2.SetProperties();
  delta.SetProperties();
  delta.FixZeroes();
  delta.MultiplyFromRight(g2);
  delta.FixZeroes();
}

/**
 * function that transforms a global point to local point in reference frame of deepest volume in current navigation
 * state
 * ( equivalent to using a global matrix )
 */
VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> NavigationState::GlobalToLocal(Vector3D<Precision> const &globalpoint) const
{
  Vector3D<Precision> tmp = globalpoint;
  Vector3D<Precision> current;
  for (int level = 0; level < fCurrentLevel; ++level) {
    Transformation3D const *m = At(level)->GetTransformation();
    current                   = m->Transform(tmp);
    tmp                       = current;
  }
  return tmp;
}

size_t FindIndexWithinMother(VPlacedVolume const *mother, VPlacedVolume const *daughter)
{
  for (size_t d = 0; d < mother->GetDaughters().size(); ++d) {
    if (mother->GetDaughters()[d] == daughter) return d;
  }
  assert(false && "did not find index of a daughter volume within mother");
  return static_cast<uint>(-1);
}

VPlacedVolume const *GetDaughterWithinMother(VPlacedVolume const *mother, uint index)
{
  if (index < (uint)mother->GetDaughters().size()) return mother->GetDaughters()[index];

  return NULL;
}

void NavigationState::GetPathAsListOfIndices(std::list<uint> &indices) const
{
  indices.clear();
  if (IsOutside()) return;
  for (uint level = fCurrentLevel - 1; level > 0; --level) {
    indices.push_front(FindIndexWithinMother(At(level - 1), At(level)));
  }
  indices.push_front(0);
}

VECCORE_ATT_HOST_DEVICE
void NavigationState::Print() const
{
// printf("VariableSizeObj: fPath=%p (%l bytes)\n", fPath, sizeof(fPath));
#ifndef VECCORE_CUDA
  printf("NavState: Level=%i/%i,  onBoundary=%s, path=<", fCurrentLevel, GetMaxLevel(),
         (fOnBoundary ? "true" : "false"));
  for (int i = 0; i < fCurrentLevel; ++i)
    printf("/%s", ToPlacedVolume(fPath[i]) ? ToPlacedVolume(fPath[i])->GetLabel().c_str() : "NULL");
  printf(">\n");
#else
  printf("NavState: Level=%i/%i,  onBoundary=%s, topVol=<%p>, this=%p\n", fCurrentLevel, GetMaxLevel(),
         (fOnBoundary ? "true" : "false"), Top(), (const void *)this);
#endif
}

void NavigationState::ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices)
{
  // clear current nav state
  fCurrentLevel = indices.size();
  if (indices.size() > 0) {
    fPath[0] = ToIndex(world);
    // have to disregard first one;
    // then iterate through list
    int counter = 0;
    for (auto x : indices) {
      if (counter > 0) fPath[counter] = ToIndex(GetDaughterWithinMother(At(counter - 1), x));
      counter++;
    }
  }
}

#ifdef VECGEOM_ROOT
TGeoBranchArray *NavigationState::ToTGeoBranchArray() const
{
// attention: the counting of levels is different: fLevel=0 means that
// this is a branch which is filled at level zero

// my counting is a bit different: it tells the NUMBER of levels which are filled
#if ROOT_VERSION_CODE >= ROOT_VERSION(5, 34, 23)
  TGeoBranchArray *tmp = TGeoBranchArray::MakeInstance(GetMaxLevel());
#else
  TGeoBranchArray *tmp = new TGeoBranchArray(GetMaxLevel());
#endif
  // gain access to array
  TGeoNode **array   = tmp->GetArray();
  RootGeoManager &mg = RootGeoManager::Instance();
  TGeoNavigator *nav = gGeoManager->GetCurrentNavigator();
  tmp->InitFromNavigator(nav);

  // tmp->
  for (int i = 0; i < fCurrentLevel; ++i)
    array[i] = const_cast<TGeoNode *>(mg.tgeonode(ToPlacedVolume(fPath[i])));
  // assert( tmp->GetCurrentNode() == mg.tgeonode( Top() ));

  /*
  std::list<uint> ilist;
  GetPathAsListOfIndices( ilist );
  int counter=0;
  for( auto x : ilist ) {
      if(counter>0)
          tmp->AddLevel(x);
      counter++;
  }
  */

  return tmp;
}

NavigationState &NavigationState::operator=(TGeoBranchArray const &other)
{
  // attention: the counting of levels is different: fLevel=0 means that
  // this is a branch which is filled at level zero
  this->fCurrentLevel = other.GetLevel() + 1;
  assert(fCurrentLevel <= GetMaxLevel());

  RootGeoManager &mg = RootGeoManager::Instance();

  for (int i = 0; i < fCurrentLevel; ++i)
    fPath[i] = ToIndex(mg.GetPlacedVolume(other.GetNode(i)));

  // other things like onboundary I don't care
  fOnBoundary = false;

  return *this;
}

#endif

std::string NavigationState::RelativePath(NavigationState const &other) const
{
  int lastcommonlevel = -1;
  int maxlevel        = Min(GetCurrentLevel(), other.GetCurrentLevel());
  std::stringstream str;
  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  auto filledlevel1 = GetCurrentLevel() - 1;
  auto filledlevel2 = other.GetCurrentLevel() - 1;

  // paths are the same
  if (filledlevel1 == lastcommonlevel && filledlevel2 == lastcommonlevel) {
    return std::string("");
  }

  // emit only ups
  if (filledlevel1 > lastcommonlevel && filledlevel2 == lastcommonlevel) {
    for (int i = 0; i < filledlevel1 - lastcommonlevel; ++i) {
      str << "/up";
    }
    return str.str();
  }

  // emit only downs
  if (filledlevel1 == lastcommonlevel && filledlevel2 > lastcommonlevel) {
    for (int i = lastcommonlevel + 1; i <= filledlevel2; ++i) {
      str << "/down";
      str << "/" << other.ValueAt(i);
    }
    return str.str();
  }

  // mixed case: first up; then down
  if (filledlevel1 > lastcommonlevel && filledlevel2 > lastcommonlevel) {
    // emit ups
    int level = filledlevel1;
    for (; level > lastcommonlevel + 1; --level) {
      str << "/up";
    }

    level = lastcommonlevel + 1;
    // emit horiz ( exists when there is a turning point )
    int delta = other.ValueAt(level) - this->ValueAt(level);
    if (delta != 0) str << "/horiz/" << delta;

    level++;
    // emit downs with index
    for (; level <= filledlevel2; ++level) {
      str << "/down/" << other.ValueAt(level);
    }
  }
  return str.str();
}
}
} // End global namespace
