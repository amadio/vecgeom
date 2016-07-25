/// \file NavigationState.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 12.03.2014

#ifndef VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
#define VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_

#include "backend/Backend.h"
#include "base/VariableSizeObj.h"
#include "base/Transformation3D.h"
#include "volumes/PlacedVolume.h"
#include "management/GeoManager.h"
#ifdef VECGEOM_CUDA
#include "management/CudaManager.h"
#endif
#include "base/Global.h"

#ifdef VECGEOM_ROOT
#include "management/RootGeoManager.h"
#endif

#include <iostream>
#include <string>

class TGeoBranchArray;

// gcc 4.8.2's -Wnon-virtual-dtor is broken and turned on by -Weffc++, we
// need to disable it for SOA3D

#if __GNUC__ < 3 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 8)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Weffc++"
#define GCC_DIAG_POP_NEEDED
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// the NavStateIndex type determines is used
// to calculate addresses of PlacedVolumes
// a short type should be used in case the number of PlacedVolumes can
// be counted with 16bits
// TODO: consider putting uint16 + uint32 types
#ifdef VECGEOM_USE_INDEXEDNAVSTATES
typedef unsigned short NavStateIndex_t;
// typedef unsigned long NavStateIndex_t;
#else
typedef VPlacedVolume const *NavStateIndex_t;
#endif

// helper functionality to convert from NavStateIndex_t to *PlacedVolumes and back
// the template abstraction also allows to go back to pointers as NavStateIndex_t
// via a template specialization
// T stands for NavStateIndex_t
template <typename T>
struct Index2PVolumeConverter {
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(T index)
  {
#ifdef VECGEOM_NVCC_DEVICE
    // checking here for NVCC_DEVICE since the global variable globaldevicegeomgata::gCompact...
    // is marked __device__ and can only be compiled within device compiler passes
    assert(vecgeom::globaldevicegeomdata::GetCompactPlacedVolBuffer() != nullptr);
    return &vecgeom::globaldevicegeomdata::GetCompactPlacedVolBuffer()[index];
#else
#ifndef VECGEOM_NVCC
    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
#else
    // this is the case when we compile with nvcc for host side
    // (failed preveously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    assert(false && "reached unimplement code");
    (void)index; // avoid unused parameter warning.
    return nullptr;
#endif
#endif
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static T ToIndex(VPlacedVolume const *pvol) { return pvol->id(); }
};

// template specialization when we directly save VPlacedVolume pointers into the NavStates
template <>
struct Index2PVolumeConverter<VPlacedVolume const *> {
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(VPlacedVolume const *pvol) { return pvol; }
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToIndex(VPlacedVolume const *pvol) { return pvol; }
};

/**
 * a class describing a current geometry state
 * likely there will be such an object for each
 * particle/track currently treated
 */
class NavigationState : protected veccore::VariableSizeObjectInterface<NavigationState, NavStateIndex_t>,
                        private Index2PVolumeConverter<NavStateIndex_t> {
public:
  using Value_t        = NavStateIndex_t;
  using Base_t         = veccore::VariableSizeObjectInterface<NavigationState, Value_t>;
  using VariableData_t = veccore::VariableSizeObj<Value_t>;

private:
  friend Base_t;

  // Required by VariableSizeObjectInterface
  VECGEOM_CUDA_HEADER_BOTH
  VariableData_t &GetVariableData() { return fPath; }
  VECGEOM_CUDA_HEADER_BOTH
  const VariableData_t &GetVariableData() const { return fPath; }

  unsigned char
      fCurrentLevel; // value indicating the next free slot in the fPath array ( ergo the current geometry depth )
  // we choose unsigned char in order to save memory, thus supporting geometry depths up to 255 which seems large enough

  bool fOnBoundary; // flag indicating whether track is on boundary of the "Top()" placed volume

  // a member to cache some state information across state usages
  // one particular example could be a calculated index for this state
  // if fCache == -1 it means the we have to recalculate it; otherwise we don't
  short fCache;

  // pointer data follows; has to be last
  veccore::VariableSizeObj<Value_t> fPath;

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  NavigationState(size_t nvalues);

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  NavigationState(size_t new_size, NavigationState &other)
      : fCurrentLevel(other.fCurrentLevel), fOnBoundary(other.fOnBoundary), fCache(-1), fPath(new_size, other.fPath)
  {
    // Raw memcpy of the content to another existing state.
    //
    // in case NavigationState was a virtual class: change to
    // std::memcpy(other->DataStart(), DataStart(), DataSize());

    if (new_size > other.fPath.fN) {
      memset(fPath.GetValues() + other.fPath.fN, 0, new_size - other.fPath.fN);
    }
  }

  // some private management methods
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void InitInternalStorage();

private:
  // The data start should point to the address of the first data member,
  // after the virtual table
  // the purpose is probably for the Copy function
  const void *DataStart() const { return (const void *)&fCurrentLevel; }
  const void *ObjectStart() const { return (const void *)this; }
  void *DataStart() { return (void *)&fCurrentLevel; }
  void *ObjectStart() { return (void *)this; }

  // The actual size of the data for an instance, excluding the virtual table
  size_t DataSize() const { return SizeOf() + (size_t)ObjectStart() - (size_t)DataStart(); }

public:
  // replaces the volume pointers from CPU volumes in fPath
  // to the equivalent pointers on the GPU
  // uses the CudaManager to do so
  void ConvertToGPUPointers();

  // replaces the pointers from GPU volumes in fPath
  // to the equivalent pointers on the CPU
  // uses the CudaManager to do so
  void ConvertToCPUPointers();

  // Enumerate the part of the private interface, we want to expose.
  using Base_t::MakeCopy;
  using Base_t::MakeCopyAt;
  using Base_t::ReleaseInstance;
  using Base_t::SizeOf;
  using Base_t::SizeOfAlignAware;

  // Enumerate functions from converter which we want to use
  // ( without retyping of the struct name )
  using Index2PVolumeConverter<NavStateIndex_t>::ToIndex;
  using Index2PVolumeConverter<NavStateIndex_t>::ToPlacedVolume;

  // produces a compact navigation state object of a certain depth
  // the caller can give a memory address where the object will
  // be placed
  // the caller has to make sure that the size of the external memory
  // is >= sizeof(NavigationState) + sizeof(VPlacedVolume*)*maxlevel
  //
  // Methods MakeInstance(), MakeInstanceAt(), MakeCopy() and MakeCopyAt() are provided by
  // VariableSizeObjectInterface

  VECGEOM_CUDA_HEADER_BOTH
  static NavigationState *MakeInstance(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return Base_t::MakeInstance(maxlevel + 1);
  }

  VECGEOM_CUDA_HEADER_BOTH
  static NavigationState *MakeInstanceAt(int maxlevel, void *addr)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return Base_t::MakeInstanceAt(maxlevel + 1, addr);
  }

  // returns the size in bytes of a NavigationState object with internal
  // path depth maxlevel
  VECGEOM_CUDA_HEADER_BOTH
  static size_t SizeOfInstance(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return VariableSizeObjectInterface::SizeOf(maxlevel + 1);
  }

  // returns the size in bytes of a NavigationState object with internal
  // path depth maxlevel -- including space needed for padding to next aligned object
  // of same kind
  VECGEOM_CUDA_HEADER_BOTH
  static size_t SizeOfInstanceAlignAware(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return VariableSizeObjectInterface::SizeOfAlignAware(maxlevel + 1);
  }

  VECGEOM_CUDA_HEADER_BOTH
  int GetObjectSize() const { return SizeOf(GetMaxLevel()); }

  VECGEOM_CUDA_HEADER_BOTH
  int SizeOf() const { return NavigationState::SizeOfInstance(GetMaxLevel()); }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  NavigationState &operator=(NavigationState const &rhs);

  // functions useful to "serialize" navigationstate
  // the Vector-of-Indices basically describes the path on the tree taken from top to bottom
  // an index corresponds to a daughter
  void GetPathAsListOfIndices(std::list<uint> &indices) const;
  void ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices);

  VECGEOM_CUDA_HEADER_BOTH
  void CopyTo(NavigationState *other) const
  {
    // Raw memcpy of the content to another existing state.
    //
    // in case NavigationState was a virtual class: change to
    // std::memcpy(other->DataStart(), DataStart(), DataSize());
    bool alloc = other->fPath.fSelfAlloc;
    // std::memcpy(other, this, this->SizeOf());
    // we only need to copy to relevant depth
    // GetCurrentLevel indicates the 'next' level, i.e. currentLevel==0 is empty
    // fCurrentLevel = maxlevel+1 is full
    // SizeOfInstance expect [0,maxlevel] and add +1 to its params
    std::memcpy(other, this, NavigationState::SizeOfInstance(this->GetCurrentLevel() - 1));

    other->fPath.fSelfAlloc = alloc;
  }

  // copies a fixed and predetermined number of bytes
  // might be useful for specialized navigators which know the depth + SizeOf in advance
  // N is number of bytes to be copied and can be obtained by a prior call to constexpr NavigationState::SizeOf( ... );
  template <size_t N>
  void CopyToFixedSize(NavigationState *other) const
  {
    bool alloc = other->fPath.fSelfAlloc;
    for (size_t i = 0; i < N; ++i) {
      ((char *)other)[i] = ((char *)this)[i];
    }
    other->fPath.fSelfAlloc = alloc;
  }

#ifdef VECGEOM_ROOT
  TGeoBranchArray *ToTGeoBranchArray() const;
  NavigationState &operator=(TGeoBranchArray const &rhs);
#endif

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  ~NavigationState();

  // what else: operator new etc...

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  unsigned char GetMaxLevel() const { return fPath.fN - 1; }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  unsigned char GetCurrentLevel() const { return fCurrentLevel; }

  // better to use pop and push
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Push(VPlacedVolume const *);

  // a push version operating on IndexTypes
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void PushIndexType(NavStateIndex_t);

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  VPlacedVolume const *Top() const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  VPlacedVolume const *At(int level) const { return ToPlacedVolume(fPath[level]); }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Value_t ValueAt(int level) const { return fPath[level]; }

  // direct write access to the path
  // (no one should ever call this function unless you know what you are doing)
  // TODO: consider making this private + friend or so
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SetValueAt(int level, Value_t v) { fPath[level] = v; }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void TopMatrix(Transformation3D &) const;

  // returning a "delta" transformation that can transform
  // coordinates given in reference frame of this->Top() to the reference frame of other->Top()
  // simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
  VECGEOM_CUDA_HEADER_BOTH
  void DeltaTransformation(NavigationState const &other, Transformation3D & /* delta */) const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &, int tolevel) const;

  VECGEOM_CUDA_HEADER_BOTH
  void TopMatrix(int tolevel, Transformation3D &) const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Pop();

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  int Distance(NavigationState const &) const;

  // returns a string representation of a (relative) sequence of operations/moves
  // that transforms this navigation state into the other navigation state
  // example:
  // state1 = /0/1/1/
  // state2 = /0/2/2/3
  // results in string
  // "/up/horiz/1/down/2/down/3" with 4 operations "up", "horiz", "down", "down"
  // the sequence of moves is the following
  // up: /0/1/1 --> /0/1/
  // horiz/1 : 0/1 --> /0/2 ( == /0/(1+1) )   "we are hopping from daughter 1 to 2 (which corresponds to a step of 1)"
  // down/2 : /0/2 --> /0/2/2   "going further down 2nd daughter"
  // down/3 : /0/2/2/3 --> /0/2/2/3  "going further down 2nd daughter"
  std::string RelativePath(NavigationState const & /*other*/) const;

  // clear all information
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Clear();

  VECGEOM_CUDA_HEADER_BOTH
  void Print() const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void Dump() const;

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  bool HasSamePathAsOther(NavigationState const &other) const
  {
    if (other.fCurrentLevel != fCurrentLevel) return false;
    for (int i = fCurrentLevel - 1; i >= 0; --i) {
      if (fPath[i] != other.fPath[i]) return false;
    }
    return true;
  }

  void printValueSequence(std::ostream & = std::cerr) const;

#ifdef VECGEOM_ROOT
  VECGEOM_FORCE_INLINE
  void printVolumePath(std::ostream & = std::cerr) const;

  /**
   * returns the number of FILLED LEVELS such that
   * state.GetNode( state.GetLevel() ) == state.Top()
   */
  VECGEOM_FORCE_INLINE
  unsigned char GetLevel() const { return fCurrentLevel - 1; }

  TGeoNode const *GetNode(int level) const { return RootGeoManager::Instance().tgeonode(ToPlacedVolume(fPath[level])); }
#endif

  /**
    function returning whether the point (current navigation state) is outside the detector setup
  */
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  bool IsOutside() const { return !(fCurrentLevel > 0); }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  bool IsOnBoundary() const { return fOnBoundary; }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SetBoundaryState(bool b) { fOnBoundary = b; }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  short GetCacheValue() const { return fCache; }

  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void SetCacheValue(short v) { fCache = v; }

#ifdef VECGEOM_ROOT
  /**
   * function return the ROOT TGeoNode object which is equivalent to calling Top()
   * function included for convenience; to make porting Geant-V easier; we should eventually get rid of this function
   */
  VECGEOM_FORCE_INLINE
  TGeoNode const *GetCurrentNode() const { return RootGeoManager::Instance().tgeonode(this->Top()); }
#endif

  // void GetGlobalMatrixFromPath( Transformation3D *const m ) const;
  // Transformation3D const * GetGlobalMatrixFromPath() const;
}; // end of class

NavigationState &NavigationState::operator=(NavigationState const &rhs)
{
  if (this != &rhs) {
    fCurrentLevel = rhs.fCurrentLevel;
    fOnBoundary   = rhs.fOnBoundary;
    fCache        = rhs.fCache;

    // Use memcpy.  Potential truncation if this is smaller than rhs.
    fPath = rhs.fPath;
  }
  return *this;
}

/*
NavigationState::NavigationState( NavigationState const & rhs ) :
        fMaxlevel(rhs.fMaxlevel),
        fCurrentLevel(rhs.fCurrentLevel),
        fOnBoundary(rhs.fOnBoundary),
        fPath(&fBuffer[0])
{
   InitInternalStorage();
   std::memcpy(fPath, rhs.fPath, sizeof(*fPath)*rhs.fCurrentLevel );
}
*/

// private implementation of standard constructor
NavigationState::NavigationState(size_t nvalues) : fCurrentLevel(0), fOnBoundary(false), fCache(-1), fPath(nvalues)
{
  // clear the buffer
  std::memset(fPath.GetValues(), 0, nvalues * sizeof(NavStateIndex_t));
}

VECGEOM_CUDA_HEADER_BOTH
NavigationState::~NavigationState()
{
}

void NavigationState::Pop()
{
  if (fCurrentLevel > 0) {
    fPath[--fCurrentLevel] = 0;
  }
}

void NavigationState::Clear()
{
  fCurrentLevel = 0;
  fOnBoundary   = false;
  fCache        = -1;
}

void NavigationState::Push(VPlacedVolume const *v)
{
#ifdef DEBUG
  assert(fCurrentLevel < GetMaxLevel());
   assert( fCurrentLevel < 2<<sizeof(char) - 1;
#endif
   fPath[fCurrentLevel++] = ToIndex( v );
}

void NavigationState::PushIndexType(NavStateIndex_t v)
{
#ifdef DEBUG
  assert(fCurrentLevel < GetMaxLevel());
   assert( fCurrentLevel < 2<<sizeof(char) - 1;
#endif
   fPath[fCurrentLevel++] = v;
}

VPlacedVolume const *NavigationState::Top() const
{
  return (fCurrentLevel > 0) ? ToPlacedVolume(fPath[fCurrentLevel - 1]) : nullptr;
}

// calculates the global matrix to transform from global coordinates
// to the frame of the top volume in the state
// input: a reference to a transformation object ( which should be initialized to identity )
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void NavigationState::TopMatrix(Transformation3D &global_matrix) const
{
  // this could be actually cached in case the path does not change ( particle stays inside a volume )
  for (int i = 1; i < fCurrentLevel; ++i) {
    global_matrix.MultiplyFromRight(*(ToPlacedVolume(fPath[i])->GetTransformation()));
  }
}

VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void NavigationState::Dump() const
{
  const unsigned int *ptr = (const unsigned int *)this;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
  printf("NavState::Dump(): data: %p(%lu) : %p(%lu) : %p(%lu)\n", (void *)&fCurrentLevel, sizeof(fCurrentLevel),
         (void *)&fOnBoundary, sizeof(fOnBoundary), (void *)&fPath, sizeof(fPath));
  for (unsigned int i = 0; i < 20; ++i) {
    printf("%p: ", (void *)ptr);
    for (unsigned int j = 0; j < 8; ++j) {
      printf(" %08x ", *ptr);
      ptr++;
    }
    printf("\n");
  }
#pragma GCC diagnostic pop
}

/**
 * encodes the geometry path as a concatenated string of ( Value_t ) present in fPath
 */
inline void NavigationState::printValueSequence(std::ostream &stream) const
{
  for (int i = 0; i < fCurrentLevel; ++i) {
    stream << "/" << fPath[i];
  }
}

#ifdef VECGEOM_ROOT
VECGEOM_FORCE_INLINE
/**
 * prints the path of the track as a verbose string ( like TGeoBranchArray in ROOT )
 * (uses internal root representation for the moment)
 */
void NavigationState::printVolumePath(std::ostream &stream) const
{
  for (int i = 0; i < fCurrentLevel; ++i) {
    stream << "/" << RootGeoManager::Instance().tgeonode(ToPlacedVolume(fPath[i]))->GetName();
  }
}
#endif

/**
 * calculates if other navigation state takes a different branch in geometry path or is on same branch
 * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing ))
 */
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
int NavigationState::Distance(NavigationState const &other) const
{
  int lastcommonlevel = -1;
  int maxlevel        = Min(GetCurrentLevel(), other.GetCurrentLevel());

  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  return (GetCurrentLevel() - lastcommonlevel) + (other.GetCurrentLevel() - lastcommonlevel) - 2;
}

inline void NavigationState::ConvertToGPUPointers()
{
#if !defined(VECGEOM_NVCC) && defined(VECGEOM_CUDA)
  for (int i = 0; i < fCurrentLevel; ++i) {
    fPath[i] = ToIndex((vecgeom::cxx::VPlacedVolume *)vecgeom::CudaManager::Instance()
                           .LookupPlaced(ToPlacedVolume(fPath[i]))
                           .GetPtr());
  }
#endif
}

inline void NavigationState::ConvertToCPUPointers()
{
#if !defined(VECGEOM_NVCC) && defined(VECGEOM_CUDA)
  for (int i = 0; i < fCurrentLevel; ++i)
    fPath[i] = ToIndex(vecgeom::CudaManager::Instance().LookupPlacedCPUPtr((const void *)ToPlacedVolume(fPath[i])));
#endif
}
}
} // End global namespace

#if defined(GCC_DIAG_POP_NEEDED)
#pragma GCC diagnostic pop
#undef GCC_DIAG_POP_NEEDED
#endif

#endif // VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
