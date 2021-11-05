// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \author created by Sandro Wenzel

#include "VecGeom/management/FlatVoxelManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/navigation/SimpleABBoxSafetyEstimator.h"
#include <thread>
#include <future>
#include <random> // C++11 random numbers
#include <sstream>
#include <set>

// this is for serialization
#ifdef VECGEOM_ROOT
#include "TFile.h"
#include "TTree.h"
#endif

// for timing measurement
#include "VecGeom/base/Stopwatch.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void FlatVoxelManager::InitStructure(LogicalVolume const *lvol)
{
  auto numregisteredlvols = GeoManager::Instance().GetRegisteredVolumesCount();
  if (fStructureHolder.size() != numregisteredlvols) {
    fStructureHolder.resize(numregisteredlvols, nullptr);
  }
  if (fStructureHolder[lvol->id()] != nullptr) {
    RemoveStructure(lvol);
  }
  fStructureHolder[lvol->id()] = BuildStructure(lvol);
}

// #define EXTREMELOOKUP

FlatVoxelHashMap<int, false> *FlatVoxelManager::BuildSafetyVoxels(LogicalVolume const *vol)
{
#ifdef EXTREMELOOKUP
  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation
  int Nx      = 500;
  int Ny      = 500;
  int Nz      = 500;
  auto voxels = new FlatVoxelHashMap<float, true>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);

  // fill the values ... we will sample this using a point cloud so as to avoid
  // filling useless values (completely inside volumes)
  // We should then try to fill the remaining holes via some kind of connecting/clustering
  // algorithm

  int numtasks = std::thread::hardware_concurrency();
  // int npointstotal{1000000000};
  int npointstotal{1000000};
  int pointspertask = npointstotal / numtasks;

  std::vector<std::mt19937> engines(numtasks);
  std::vector<SOA3D<float> *> points(numtasks);
  std::vector<std::vector<long> *> keyspertask(numtasks);
  // reserve space and init engines
  for (int t = 0; t < numtasks; ++t) {
    engines[t].seed(t * 13 + 11);
    points[t] = new SOA3D<float>(pointspertask);
  }

  Stopwatch timer;
  timer.Start();
  std::vector<std::future<void>> futures;
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([&engines, &points, &keyspertask, vol, t, voxels] {
      bool foundPoints = volumeUtilities::FillUncontainedPoints(*vol, engines[t], *points[t]);
      if (foundPoints) {
        keyspertask[t] = new std::vector<long>;
        voxels->getKeys(*points[t], *keyspertask[t]);
      } else {
        keyspertask[t] = nullptr;
      }
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  auto elapsed = timer.Stop();
  std::cout << "Sampling points and keys took " << elapsed << "s \n";

  timer.Start();
  // merge all keys
  std::vector<long> allkeys;
  for (int t = 0; t < numtasks; ++t) {
    if (keyspertask[t]) std::copy(keyspertask[t]->begin(), keyspertask[t]->end(), std::back_inserter(allkeys));
  }
  // Do we need to delete keyspertask[t] ??  JA

  if (allkeys.begin() == allkeys.end() || allkeys.size() <= 1) {
    // No points found -- the daughthers fill the current volume
    return nullptr;
  }
  std::sort(allkeys.begin(), allkeys.end());

  // get rid of duplicates easily since they are sorted
  std::vector<long> sortedkeys;
  for (auto &k : allkeys) {
    if (sortedkeys.size() == 0) {
      sortedkeys.push_back(k);
    } else if (k != sortedkeys.back()) {
      sortedkeys.push_back(k);
    }
  }
  std::cout << "Generating unique keys took " << timer.Stop() << "s \n";
  std::cout << "We have " << sortedkeys.size() << " sorted unique keys; fraction "
            << sortedkeys.size() / (1. * Nx * Ny * Nz) << " estimated volume "
            << sortedkeys.size() * safetyvoxels->getVoxelVolume() << "\n";

  timer.Start();
  // make a vector where to collect safeties
  std::vector<float> safeties(sortedkeys.size());
  const auto safetyestimator = static_cast<SimpleABBoxSafetyEstimator const *>(SimpleABBoxSafetyEstimator::Instance());
  std::vector<std::future<void>> safetyfutures;

  std::cerr << " Calculating safeties ... in parallel ";
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &safeties, safetyvoxels, &sortedkeys, safetyestimator, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }
      const float voxelhalfdiagonal = 0.5 * safetyvoxels->getVoxelDiagonal();

      for (int i = startindex; i < endindex; ++i) {
        auto k                        = sortedkeys[i];
        Vector3D<float> midpointfloat = safetyvoxels->keyToPos(k);
        Vector3D<double> midpointdouble(midpointfloat.x(), midpointfloat.y(), midpointfloat.z());
        auto safety = safetyestimator->TreatSafetyToIn(midpointdouble, vol, 1E20);
        safeties[i] = safety - voxelhalfdiagonal;
      }
    });
    safetyfutures.push_back(std::move(fut));
  }
  std::for_each(safetyfutures.begin(), safetyfutures.end(), [](std::future<void> &fut) { fut.wait(); });
  std::cout << "Generating safeties took " << timer.Stop() << "s \n";

  auto filename = createName(vol, Nx, Ny, Nz);
  dumpToTFile(filename.c_str(), *points[0], sortedkeys, safeties);

  // finally register safeties in voxel map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto k      = sortedkeys[i];
    auto safety = safeties[i];
    voxels->addPropertyForKey(k, safety);
  }
  std::cout << " done \n";

  auto structure     = new VoxelStructure();
  structure->fVoxels = voxels;
  structure->fVol    = vol;

  return structure;
#else

  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation

  const auto &daughters   = vol->GetDaughters();
  const size_t ndaughters = daughters.size();
  //  a good guess is by the number of daughters and their average extent/dimensions
  std::cout << " Setting up safety voxels for " << vol->GetName() << " with " << ndaughters << " daughters \n";

  int Nx = std::max(4., 2 * std::sqrt(1. * ndaughters));
  int Ny = std::max(4., 2 * std::sqrt(1. * ndaughters));
  int Nz = std::max(4., 2 * std::sqrt(1. * ndaughters));
  //  int Nx = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));
  //  int Ny = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));
  //  int Nz = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));

  // first of all look if we have a cached version sitting in some file
  FlatVoxelHashMap<int, false> *safetyvoxels = nullptr;
  safetyvoxels = FlatVoxelHashMap<int, false>::readFromTFile(createName(vol, Nx, Ny, Nz).c_str());
  if (safetyvoxels) {
    std::cout << "Found cached version of safety voxels for volume " << vol->GetName() << "\n";
    return safetyvoxels;
  }
  safetyvoxels = new FlatVoxelHashMap<int, false>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);

  int numtasks = std::thread::hardware_concurrency();
  int npointstotal{1000 * Nx * Ny * Nz};
  int pointspertask = npointstotal / numtasks;

  std::vector<std::mt19937> engines(numtasks);
  std::vector<SOA3D<float> *> points(numtasks);
  std::vector<std::vector<long> *> keyspertask(numtasks);
  // reserve space and init engines
  for (int t = 0; t < numtasks; ++t) {
    engines[t].seed(t * 13 + 11);
    points[t] = new SOA3D<float>(pointspertask);
  }

  Stopwatch timer;
  timer.Start();
  std::vector<std::future<void>> futures;
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([&engines, &points, &keyspertask, vol, t, safetyvoxels] {
      bool foundPoints = volumeUtilities::FillUncontainedPoints(*vol, engines[t], *points[t]);
      if (foundPoints) {
        keyspertask[t] = new std::vector<long>;
        safetyvoxels->getKeys(*points[t], *keyspertask[t]);
        // delete points[t]; points[t] = nullptr;  // JA 2021.03.04 17:15 CEST ???
      } else {
        keyspertask[t] = nullptr;
        std::cerr << " WARNING: Found 0 uncontained points for " << vol->GetName()
                  << " -- expect problems in estimating safety. \n";
      }
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  auto elapsed = timer.Stop();
  std::cout << "Sampling points and keys took " << elapsed << "s \n";

  timer.Start();
  // merge all keys
  std::vector<long> allkeys;
  for (int t = 0; t < numtasks; ++t) {
    if (keyspertask[t]) std::copy(keyspertask[t]->begin(), keyspertask[t]->end(), std::back_inserter(allkeys));
  }

  if (allkeys.begin() == allkeys.end() || allkeys.size() <= 1) {
    // No points found -- the daughthers fill the current volume
    return nullptr;
  }

  std::sort(allkeys.begin(), allkeys.end());

  // get rid of duplicates easily since they are sorted
  std::vector<long> sortedkeys;
  for (auto &k : allkeys) {
    if (sortedkeys.size() == 0) {
      sortedkeys.push_back(k);
    } else if (k != sortedkeys.back()) {
      sortedkeys.push_back(k);
    }
  }
  std::cout << "Generating unique keys took " << timer.Stop() << " s \n";

  std::cout << " We have " << sortedkeys.size() << " sorted unique keys; fraction "
            << sortedkeys.size() / (1. * Nx * Ny * Nz) << " estimated volume "
            << sortedkeys.size() * safetyvoxels->getVoxelVolume() << "\n";
  size_t minSize = 50;
  if( sortedkeys.size() < minSize ){
     std::cerr << " ** Keys are few -- not creating acceleration structure. \n";
     return nullptr;
  }
  //
  timer.Start();
  // make a vector where to collect the safetycandidates
  std::vector<std::vector<int>> safetycandidates(sortedkeys.size());

  std::vector<std::future<void>> safetyfutures;
#ifdef VOXEL_DEBUG
  std::cerr << " Calculating safety candidates ... in parallel \n";
#endif
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &safetycandidates, safetyvoxels, &sortedkeys, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }

      int size{0};
      auto abboxcorners = ABBoxManager::Instance().GetABBoxes(vol, size);

      std::vector<Vector3D<float>> voxelsurfacepoints;
      for (int i = startindex; i < endindex; ++i) {
        auto k       = sortedkeys[i];

        // ---
        // step 1 is to check intersections of this voxel with all object bounding boxes
        // ---
        Vector3D<float> keylower;
        Vector3D<float> keyupper;
        safetyvoxels->Extent(k, keylower, keyupper);
#ifdef VOXEL_DEBUG
        bool verbose = false;
        if (verbose) {
          std::cerr << "KEY LOWER EXTENT " << keylower << "\n";
          std::cerr << "KEY UPPER EXTENT " << keyupper << "\n";
        }
#endif
        // painful but we could speed up with SIMD
        for (int boxindex = 0; boxindex < size; ++boxindex) {
          const auto &boxlower = abboxcorners[2 * boxindex];
          const auto &boxupper = abboxcorners[2 * boxindex + 1];

          if (volumeUtilities::IntersectionExist(keylower, keyupper, boxlower, boxupper)) {
            safetycandidates[i].push_back(boxindex);
          }
        }

        // ---
        // step 2 is to determine which other candidates are possible --- for instance
        // important when this voxel is in empty space
        // (WE NEED TO BE CAREFUL ABOUT TOPOLOGICALLY WEIRD CASES)
        // ---
        std::set<int> othersafetycandidates;
        std::set<int> insidedaughtercandidates;

        voxelsurfacepoints.clear();
        volumeUtilities::GenerateRegularSurfacePointsOnBox(keylower, keyupper, 10, voxelsurfacepoints);
        for (const auto &sp : voxelsurfacepoints) {
          // surface point in double precission (needed for some interfaces)
          Vector3D<Precision> spdouble(sp.x(), sp.y(), sp.z());
#ifdef VOXEL_DEBUG
          if (verbose) {
            std::cerr << "CHECKING SURFACE POINT " << sp << "\n";
          }
#endif
          const auto inmother = vol->GetUnplacedVolume()->Contains(spdouble);
          if (!inmother) {
            othersafetycandidates.insert(-1);
            continue;
          }

          // we can use the knowledge about intersecting bounding boxes to query
          // daughter insection quickly
          auto daughters     = vol->GetDaughters();
          bool inanydaughter = false;
          for (const auto &boxindex : safetycandidates[i]) {
            const auto &boxlower = abboxcorners[2 * boxindex];
            const auto &boxupper = abboxcorners[2 * boxindex + 1];
            bool inboundingbox{false};
            ABBoxImplementation::ABBoxContainsKernelGeneric(boxlower, boxupper, sp, inboundingbox);
            if (inboundingbox) {
              // ASSUMING BOXINDEX == DAUGHTERINDEX !!
              if (daughters[boxindex]->Contains(spdouble)) {
                inanydaughter = true;
                // if(verbose) std::cerr << "used to ignore surface point " << sp << " inside daughter vol " << boxindex
                // << "\n";

                insidedaughtercandidates.insert(boxindex);
                // Should only be inside one daughter volume
                break;
              }
            }
          }

          if (!inanydaughter) {
            // get safetytoout as reference length scale
            const auto safetyout    = vol->GetUnplacedVolume()->SafetyToOut(spdouble);
            const auto safetyoutsqr = safetyout * safetyout;
#ifdef VOXEL_DEBUG
            if (verbose) {
              std::cerr << "POINT OK; MOTHER SAFETY " << safetyout << "\n";
            }
#endif
            // get all intersecting objects within this distance
            // which are not yet part of candidates ... we are using
            // the simple safety estimator for this (could use better algorithms) but in a SIMD way

            int size{0};
            // fetches the SIMDized bounding box representations
            ABBoxManager::ABBoxContainer_v bboxes = ABBoxManager::Instance().GetABBoxes_v(vol, size);

            using IdDistPair_t = ABBoxManager::BoxIdDistancePair_t;
            char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
            IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&stackspace);

            // calculate squared bounding box safeties in vectorized way which are within range of safety to mother
            auto ncandidates =
                SimpleABBoxSafetyEstimator::GetSafetyCandidates_v(spdouble, bboxes, size, boxsafetylist, safetyoutsqr);

            int bestcandidate = -1; // -1 means mother
            // final safety for this surfacepoint
            float finalsafetysqr = safetyoutsqr;
            for (size_t candidateindex = 0; candidateindex < ncandidates; ++candidateindex) {
              const auto volid          = boxsafetylist[candidateindex].first;
              const auto safetytoboxsqr = boxsafetylist[candidateindex].second;

              const auto candidatesafety = daughters[volid]->SafetyToIn(spdouble);
              const auto candsafetysqr   = candidatesafety * candidatesafety;
#ifdef VOXEL_DEBUG
              if (verbose) {
                std::cerr << "DAUGH " << volid << " squared box saf " << safetytoboxsqr
                          << " cands " << candidatesafety << "\n";
              }
#endif
              // we take the larger of boxsafety or candidatesafety as the safety for this object
              const auto thiscandidatesafetysqr = std::max<Precision>(candsafetysqr, safetytoboxsqr);

              // if this safety is smaller than the previously known safety
              if (thiscandidatesafetysqr <= finalsafetysqr) {
#ifdef VOXEL_DEBUG
                if (verbose) {
                  std::cerr << "Updating best cand from " << bestcandidate << " to " << volid
                            << " safety-sq = " << thiscandidatesafetysqr << " for sp = " << sp << "\n";
                }
#endif
                bestcandidate  = volid;
                finalsafetysqr = thiscandidatesafetysqr;
              }
            }
#ifdef VOXEL_DEBUG            
            if (verbose) { std::cerr << "Inserting best candidate " << bestcandidate << "\n"; }
#endif
            othersafetycandidates.insert(bestcandidate);
          } // if not in daughter
        }   // loop over surface points of voxel

        for (const auto &other : othersafetycandidates) {
          // we add the other candidates to the list of existing candidates
          auto iter = std::find(safetycandidates[i].begin(), safetycandidates[i].end(), other);
          if (iter == safetycandidates[i].end()) {
            safetycandidates[i].push_back(other);
          }
        }

#ifdef CHECK_IN_DAUGHTER_CANDIDATES
        // Check whether any 'in-daughter' candidates are not yet seen
        for (const auto &other : insidedaughtercandidates) {
          // we add the other candidates to the list of existing candidates
          auto iter = std::find(safetycandidates[i].begin(), safetycandidates[i].end(), other);
          if (iter == safetycandidates[i].end()) {
            // if(verbose)
            std::cerr << "used to ignore 'inside daughter' vol " << other << "\n";
            safetycandidates[i].push_back(other);
          }
        }
#endif
        std::sort(safetycandidates[i].begin(), safetycandidates[i].end());
      } // loop over keys/voxels
    });
    safetyfutures.push_back(std::move(fut));
  }
  std::for_each(safetyfutures.begin(), safetyfutures.end(), [](std::future<void> &fut) { fut.wait(); });
  std::cout << "Generating safeties took " << timer.Stop() << "s \n";
  // bool verboseAdd= false;
  // finally register safety or locate candidates in voxel hash map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto key = sortedkeys[i];
    for (const auto &cand : safetycandidates[i]) {
      // if( verboseAdd ) { std::cout << "Adding cand " << cand << " to key " << key << "\n"; }
      safetyvoxels->addPropertyForKey(key, cand);
    }
  }
  std::cout << " done \n";

  // safetyvoxels->print();
  // create cache
  safetyvoxels->dumpToTFile(createName(vol, Nx, Ny, Nz).c_str());

  return safetyvoxels;
#endif // extreme lookup
}

FlatVoxelHashMap<int, false> *FlatVoxelManager::BuildLocateVoxels(LogicalVolume const *vol)
{
  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation

  const auto &daughters   = vol->GetDaughters();
  const size_t ndaughters = daughters.size();
  //  a good guess is by the number of daughters and their average extent/dimensions
  std::cout << "Setting up locate voxels for " << vol->GetName() << " with " << ndaughters << " daughters \n";

  int Nx            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  int Ny            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  int Nz            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  auto locatevoxels = new FlatVoxelHashMap<int, false>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);
  size_t numkeys    = Nx * Ny * Nz;

  int numtasks = std::thread::hardware_concurrency();

  // here we simply iterate over all keys
  Stopwatch timer;
  timer.Start();
  //
  std::vector<long> sortedkeys;
  for (size_t i = 0; i < numkeys; ++i) {
    sortedkeys.push_back(i);
  }
  std::cout << "Generating unique keys took " << timer.Stop() << "s \n";

  //
  timer.Start();
  // make a vector where to collect the locatecandidates
  std::vector<std::vector<int>> locatecandidates(sortedkeys.size());
  std::vector<std::future<void>> futures;

  std::cout << " Calculating locate candidates ... in parallel ";
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &locatecandidates, locatevoxels, &sortedkeys, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }
      int size{0};
      auto abboxcorners = ABBoxManager::Instance().GetABBoxes(vol, size);

      for (int i = startindex; i < endindex; ++i) {
        auto k = sortedkeys[i];
        Vector3D<float> keylower;
        Vector3D<float> keyupper;
        locatevoxels->Extent(k, keylower, keyupper);

        // painful but we could speed up with SIMD
        for (int boxindex = 0; boxindex < size; ++boxindex) {
          const auto &boxlower = abboxcorners[2 * boxindex];
          const auto &boxupper = abboxcorners[2 * boxindex + 1];

          if (volumeUtilities::IntersectionExist(keylower, keyupper, boxlower, boxupper)) {
            locatecandidates[i].push_back(boxindex);
          }
        }
      } // loop over keys/voxels
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  std::cout << "Generating locate voxels took " << timer.Stop() << "s \n";

  // finally register safety or locate candidates in voxel hash map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto key = sortedkeys[i];
    for (const auto &cand : locatecandidates[i]) {
      locatevoxels->addPropertyForKey(key, cand);
    }
  }
  // locatevoxels->print();

  return locatevoxels;
}

FlatVoxelManager::VoxelStructure *FlatVoxelManager::BuildStructure(LogicalVolume const *vol)
{
  auto structure                      = new VoxelStructure();
  structure->fVoxelToCandidate        = BuildSafetyVoxels(vol);
  structure->fVoxelToLocateCandidates = BuildLocateVoxels(vol);
  structure->fVol                     = vol;

  return structure;
}

void FlatVoxelManager::RemoveStructure(LogicalVolume const *lvol)
{
  // FIXME: take care of memory deletion within acceleration structure
  if (fStructureHolder[lvol->id()]) delete fStructureHolder[lvol->id()];
}

// save to TFile
void FlatVoxelManager::dumpToTFile(const char *name, std::vector<float> const &xs, std::vector<float> const &ys,
                                   std::vector<float> const &zs, std::vector<long> const &keys,
                                   std::vector<float> const &safeties)
{
#ifdef VECGEOM_ROOT
  TFile f(name, "RECREATE");
  f.WriteObjectAny((void *)&xs, "std::vector<float>", "XS");
  f.WriteObjectAny((void *)&ys, "std::vector<float>", "YS");
  f.WriteObjectAny((void *)&zs, "std:s:vector<float>", "ZS");
  f.WriteObjectAny((void *)&keys, "std::vector<long>", "KEYS");
  f.WriteObjectAny((void *)&safeties, "std::vector<float>", "SAF");
  f.Close();
#endif
}

void FlatVoxelManager::dumpToTFile(const char *name, SOA3D<float> const &points, std::vector<long> const &keys,
                                   std::vector<float> const &safeties)
{
#ifdef VECGEOM_ROOT
  size_t npoints = points.size();
  std::vector<float> xs(points.x(), points.x() + npoints);
  std::vector<float> ys(points.y(), points.y() + npoints);
  std::vector<float> zs(points.z(), points.z() + npoints);

  TFile f(name, "RECREATE");
  TTree tree("t", "t");
  tree.Branch("XS", &xs);
  tree.Branch("YS", &ys);
  tree.Branch("ZS", &zs);
  std::vector<long> *keysptr = (std::vector<long> *)&keys;
  tree.Branch("KEYS", keysptr);
  std::vector<float> *sptr = (std::vector<float> *)&safeties;
  tree.Branch("SAF", sptr);
  tree.Fill();
  tree.Write();
  f.Close();
#endif
}

// read from TFile; return true if successful
bool FlatVoxelManager::readFromTFile(const char *name, std::vector<float> &xs, std::vector<float> &ys,
                                     std::vector<float> &zs, std::vector<long> &keys, std::vector<float> &safeties)
{
  // TFile f(name);
  // std::vector<float> *lxs = nullptr;
  // std::vector<float> *lys = nullptr;
  // std::vector<float> *lzs = nullptr;
  // std::vector<long>  *lkeys = nullptr;
  // std::vector<float> *lsafeties = nullptr;

  // if (!f.IsZombie()) {
  //  lxs = (std::vector<float> *)f.GetObjectChecked("XS","std::vector<float>");
  //  lys = (std::vector<float> *)f.GetObjectChecked("YS","std::vector<float>");
  //  lzs = (std::vector<float> *)f.GetObjectChecked("ZS","std::vector<float>");
  //  lkeys = (std::vector<long> *)f.GetObjectChecked("KEYS","std::vector<long>");
  //  lsafeties = (std::vector<float> *)f.GetObjectChecked("SAF","std::vector<float>");
  //  return true;
  // }
  return false;
}

// create a name for the backup file
std::string FlatVoxelManager::createName(LogicalVolume const *vol, int kx, int ky, int kz)
{
  const auto volname    = vol->GetName();
  const auto ndaughters = vol->GetDaughtersp()->size();
  std::stringstream str;
  str << volname << "_ND_" << ndaughters << "_KX_" << kx << "_KY_" << ky << "_KZ_" << kz << ".root";
  return str.str();
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
