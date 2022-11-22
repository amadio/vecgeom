/*
 * volume_utilities.h
 *
 *  Created on: Mar 24, 2014
 *      Author: swenzel
 */

#ifndef VOLUME_UTILITIES_H_
#define VOLUME_UTILITIES_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/management/GeoManager.h"

#include "VecGeom/management/GeoManager.h"

#include <cstdio>
#include <random>
#include <vector>
#include <random>
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace volumeUtilities {

/**
 * @brief Is the trajectory through a point along a direction hitting a volume?
 * @details If ROOT is available and VECGEOM_TEST_BENCHMARK is set, use
 *    ROOT to calculate it, otherwise use VecGeom utilities.
 * @param point is the starting point
 * @param dir is the direction of the trajectory
 * @param volume is the shape under test
 * @return true/false whether the trajectory hits the volume
 */
VECGEOM_FORCE_INLINE
bool IsHittingVolume(Vector3D<Precision> const &point, Vector3D<Precision> const &dir, VPlacedVolume const &volume)
{
  assert(!volume.Contains(point));
#if defined(USEROOTFORHITDETECTION)
  std::shared_ptr<TGeoShape const> rootshape(volume.ConvertToRoot());
  Transformation3D const *m  = volume.GetTransformation();
  Vector3D<Precision> rpoint = m->Transform(point);
  Vector3D<Precision> rdir   = m->TransformDirection(dir);
  return rootshape->DistFromOutside((double *)&rpoint[0], (double *)&rdir[0], 3, vecgeom::kInfLength) < 1E20;
#else
  return volume.DistanceToIn(point, dir, vecgeom::kInfLength) < vecgeom::kInfLength;
#endif
}

// utility function to check if track hits any daughter of input logical volume
inline bool IsHittingAnyDaughter(Vector3D<Precision> const &point, Vector3D<Precision> const &dir,
                                 LogicalVolume const &volume)
{
  for (size_t daughter = 0; daughter < volume.GetDaughters().size(); ++daughter) {
    if (IsHittingVolume(point, dir, *volume.GetDaughters()[daughter])) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Returns a random point, based on a sampling rectangular volume.
 * @details Mostly used for benchmarks and navigation tests
 * @param size is a Vector3D containing the rectangular dimensions of the sampling volume
 * @param scale an optional scale factor (default is 1)
 * @return a random output point
 */
VECGEOM_FORCE_INLINE
Vector3D<Precision> SamplePoint(Vector3D<Precision> const &size, const Precision scale = 1)
{
  const Vector3D<Precision> ret(scale * (1. - 2. * RNG::Instance().uniform()) * size[0],
                                scale * (1. - 2. * RNG::Instance().uniform()) * size[1],
                                scale * (1. - 2. * RNG::Instance().uniform()) * size[2]);
  return ret;
}

/**
 * @brief Returns a random point, based on a sampling rectangular volume.
 * @details Mostly used for benchmarks and navigation tests
 * @param size is a Vector3D containing the rectangular dimensions of the sampling volume
 * @param scale an optional scale factor (default is 1)
 * @return a random output point
 */
template <typename RngEngine>
VECGEOM_FORCE_INLINE
Vector3D<Precision> SamplePoint(Vector3D<Precision> const &size, RngEngine &rngengine, const Precision scale = 1)
{
  std::uniform_real_distribution<double> dist(0, 2.);
  const Vector3D<Precision> ret(scale * (1. - dist(rngengine)) * size[0], scale * (1. - dist(rngengine)) * size[1],
                                scale * (1. - dist(rngengine)) * size[2]);
  return ret;
}

/**
 *  @brief Returns a random, normalized direction vector.
 *  @details Mostly used for benchmarks, when a direction is needed.
 *  @return a random, normalized direction vector
 */
VECGEOM_FORCE_INLINE
Vector3D<Precision> SampleDirection()
{

  Vector3D<Precision> dir((1. - 2. * RNG::Instance().uniform()), (1. - 2. * RNG::Instance().uniform()),
                          (1. - 2. * RNG::Instance().uniform()));

  const Precision inverse_norm = 1. / std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  dir *= inverse_norm;

  return dir;
}

/**
 *  @brief Fills a container with random normalized directions.
 *  @param dirs is the output container, provided by the caller
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillRandomDirections(TrackContainer &dirs)
{
  dirs.resize(dirs.capacity());
  for (int i = 0, iMax = dirs.capacity(); i < iMax; ++i) {
    dirs.set(i, SampleDirection());
  }
}

/**
 * @brief Fills a container with biased normalized directions.
 * @details Directions are randomly assigned first, and then the
 *    fraction of hits is measured and compared to suggested bias.
 *    Then some directions will be modified as needed, to force the
 *    sample as a whole to have the suggested hit bias (@see bias).
 * @param volume provided must have daughter volumes.  Those daughters
 *    are used to determine the hit bias (@see bias).
 * @param points provided, and not modified.
 * @param bias is a real number in the range [0,1], which suggests the
 *    fraction of points hitting any of the daughter volumes.
 * @param dirs is the output directions container, provided by the
 *    caller.
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillBiasedDirections(VPlacedVolume const &volume, TrackContainer const &points, Precision bias,
                          TrackContainer &dirs)
{
  assert(bias >= 0. && bias <= 1.);

  if (bias > 0. && volume.GetDaughters().size() == 0) {
    printf("\nFillBiasedDirections ERROR:\n bias=%f requested, but no daughter volumes found.\n", bias);
    //// should throw exception, but for now just abort
    // printf("FillBiasedDirections: aborting...\n");
    // exit(1);
    ///== temporary: reset bias to zero
    bias = 0.0;
  }

  const int size = dirs.capacity();
  int n_hits     = 0;
  std::vector<bool> hit(size, false);

  // Randomize directions
  FillRandomDirections(dirs);

  // Check hits
  for (int track = 0; track < size; ++track) {
    if (IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume())) {
      n_hits++;
      hit[track] = true;
    }
  }

  // Remove hits until threshold
  printf("VolumeUtilities: FillBiasedDirs: nhits/size = %i/%i and requested bias=%f\n", n_hits, size, bias);
  int tries    = 0;
  int maxtries = 10000 * size;
  while (static_cast<Precision>(n_hits) / static_cast<Precision>(size) > bias) {
    // while (n_hits > 0) {
    tries++;
    if (tries % 1000000 == 0) {
      printf("%s line %i: Warning: %i tries to reduce bias... volume=%s. Please check.\n", __FILE__, __LINE__, tries,
             volume.GetLabel().c_str());
    }

    int track         = static_cast<int>(static_cast<Precision>(size) * RNG::Instance().uniform());
    int internaltries = 0;
    while (hit[track]) {
      if (internaltries % 2) {
        dirs.set(track, SampleDirection());
      } else {
        // try inversing direction
        dirs.set(track, -dirs[track]);
      }
      internaltries++;
      if (!IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume())) {
        n_hits--;
        hit[track] = false;
        //	  tries = 0;
      }
      if (internaltries % 100 == 0) {
        // printf("%s line %i: Warning: %i tries to reduce bias... current bias %d volume=%s. Please check.\n",
        // __FILE__,
        //       __LINE__, internaltries, n_hits, volume.GetLabel().c_str());
        // try another track
        break;
      }
    }
  }

  // crosscheck
  {
    int crosscheckhits = 0;
    for (int track = 0; track < size; ++track)
      if (IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume())) crosscheckhits++;
    assert(crosscheckhits == n_hits && "problem with hit count == 0");
    (void)crosscheckhits; // silence set but not unused warnings when asserts are disabled
  }

  // Add hits until threshold
  tries = 0;
  while (static_cast<Precision>(n_hits) / static_cast<Precision>(size) < bias && tries < maxtries) {
    int track = static_cast<int>(static_cast<Precision>(size) * RNG::Instance().uniform());
    while (!hit[track] && tries < maxtries) {
      ++tries;
      if (tries % 1000000 == 0) {
        printf("%s line %i: Warning: %i tries to increase bias... volume=%s, current bias=%i/%i=%f.  Please check.\n",
               __FILE__, __LINE__, tries, volume.GetLabel().c_str(), n_hits, size,
               static_cast<Precision>(n_hits) / static_cast<Precision>(size));
      }

      // SW: a potentially much faster algorithm is the following:
      // sample a daughter to hit ( we can adjust the sampling probability according to Capacity or something; then
      // generate point on surface of daughter )
      // set direction accordingly
      uint selecteddaughter              = (uint)RNG::Instance().uniform() * volume.GetDaughters().size();
      VPlacedVolume const *daughter      = volume.GetDaughters()[selecteddaughter];
      Vector3D<Precision> pointonsurface = daughter->GetUnplacedVolume()->SamplePointOnSurface();
      // point is in reference frame of daughter so need to transform it back
      Vector3D<Precision> dirtosurfacepoint =
          daughter->GetTransformation()->InverseTransform(pointonsurface) - points[track];
      dirtosurfacepoint.Normalize();
      dirs.set(track, dirtosurfacepoint);

      // the brute force and simple sampling technique is the following
      // dirs.set(h, SampleDirection());
      if (IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume())) {
        n_hits++;
        hit[track] = true;
        tries      = 0;
      }
    }
  }

  // crosscheck
  {
    int crosscheckhits = 0;
    for (int p = 0; p < size; ++p)
      if (IsHittingAnyDaughter(points[p], dirs[p], *volume.GetLogicalVolume())) crosscheckhits++;
    assert(crosscheckhits == n_hits && "problem with hit count");
    (void)crosscheckhits; // silence set but not unused warnings when asserts are disabled
  }

  if (tries == maxtries) {
    printf("WARNING: NUMBER OF DIRECTORY SAMPLING TRIES EXCEEDED MAXIMUM; N_HITS %d; ACHIEVED BIAS %lf \n", n_hits,
           n_hits / (1. * size));
  }
}

/**
 * @brief Same as previous function, but now taking a LogicalVolume as input.
 * @detail Delegates the filling to the other function (@see FillBiasedDirections).
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillBiasedDirections(LogicalVolume const &volume, TrackContainer const &points, const Precision bias,
                          TrackContainer &dirs)
{
  VPlacedVolume const *const placed = volume.Place();
  FillBiasedDirections(*placed, points, bias, dirs);
  delete placed;
}

VECGEOM_FORCE_INLINE
Precision UncontainedCapacity(VPlacedVolume const &volume)
{
  Precision momCapacity = const_cast<VPlacedVolume &>(volume).Capacity();
  Precision dauCapacity = 0.;
  unsigned int kk       = 0;
  for (Vector<Daughter>::const_iterator j = volume.GetDaughters().cbegin(), jEnd = volume.GetDaughters().cend();
       j != jEnd; ++j, ++kk) {
    dauCapacity += const_cast<VPlacedVolume *>(*j)->Capacity();
  }
  return momCapacity - dauCapacity;
}

/**
 * @brief Fills the volume with 3D points which are _not_ contained in
 *    any daughters of the input mother volume.
 * @details Requires a proper bounding box from the input volume.
 *    Point coordinates are local to input mother volume.
 * @param volume is the input mother volume containing all output points.
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE bool FillUncontainedPoints(VPlacedVolume const &volume, TrackContainer &points)
{
  static double lastUncontCap = 0.0;
  double uncontainedCapacity  = UncontainedCapacity(volume);
  if (uncontainedCapacity != lastUncontCap) {
    std::cout << "Uncontained capacity for " << volume.GetLabel() << ":" << uncontainedCapacity << " units\n";
    lastUncontCap = uncontainedCapacity;
  }
  if (uncontainedCapacity <= 1000 * kTolerance) {
    std::cout << "\nVolUtil: FillUncontPts: WARNING: Volume provided <" << volume.GetLabel()
              << "> does not have uncontained capacity!  Method returns false.\n";
    return false;
  }

  const int size = points.capacity();
  points.resize(points.capacity());

  Vector3D<Precision> lower, upper, offset;
  volume.GetUnplacedVolume()->Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  int totaltries = 0;
  for (int i = 0; i < size; ++i) {
    bool contained;
    Vector3D<Precision> point;
    totaltries = 0;
    do {
      // ensure that point is contained in mother volume
      do {
        ++totaltries;
        if (totaltries % 10000 == 0) {
          printf("%s line %i: Warning: %i tries to find uncontained points... volume=%s.  Please check.\n", __FILE__,
                 __LINE__, totaltries, volume.GetLabel().c_str());
        }
        if (totaltries % 5000000 == 0) {
          double ratio = 1.0 * i / totaltries;
          printf("Progress : %i tries ( succeeded = %i , ratio %f %% ) to find uncontained points... volume=%s.\n",
                 totaltries, i, 100. * ratio, volume.GetLabel().c_str());
        }

        point = offset + SamplePoint(dim);
      } while (!volume.UnplacedContains(point));
      points.set(i, point);

      contained = false;
      int kk    = 0;
      for (Vector<Daughter>::const_iterator j = volume.GetDaughters().cbegin(), jEnd = volume.GetDaughters().cend();
           j != jEnd; ++j, ++kk) {
        if ((*j)->Contains(points[i])) {
          contained = true;
          break;
        }
      }
    } while (contained);
  }
  return true;
}

template <typename TrackContainer>
VECGEOM_FORCE_INLINE bool FillUncontainedPoints(LogicalVolume const &volume, TrackContainer &points)
{
  VPlacedVolume const *const placed = volume.Place();
  bool good                         = FillUncontainedPoints(*placed, points);
  delete placed;

  return good;
}

// *** The following functions allow to give an external generator
// *** which should make these functions usable in parallel

/**
 * @brief Fills the volume with 3D points which are _not_ contained in
 *    any daughters of the input mother volume.
 * @details Requires a proper bounding box from the input volume.
 *    Point coordinates are local to input mother volume.
 * @param volume is the input mother volume containing all output points.
 * @param points is the output container, provided by the caller.
 */
template <typename RandomEngine, typename TrackContainer>
VECGEOM_FORCE_INLINE bool FillUncontainedPoints(VPlacedVolume const &volume, RandomEngine &rngengine,
                                                TrackContainer &points)
{
  static double lastUncontCap = 0.0;
  double uncontainedCapacity  = UncontainedCapacity(volume);
  if (uncontainedCapacity != lastUncontCap) {
    printf("Uncontained capacity for %s: %g units\n", volume.GetLabel().c_str(), uncontainedCapacity);
    lastUncontCap = uncontainedCapacity;
  }
  double totalcapacity = const_cast<VPlacedVolume &>(volume).Capacity();

  std::cout << "\nVolUtil: FillUncontPts: Volume <" << volume.GetLabel() << "  capacities: total =  " << totalcapacity
            << " uncontained = " << uncontainedCapacity << "\n";

  if (uncontainedCapacity <= 1000 * kTolerance) {
    // double checkUC= UncontainedCapacity(volume); // Rerun - for debugging ...
    std::cout << "\nVolUtil: FillUncontPts: ERROR: Volume provided <" << volume.GetLabel()
              << "> does not have uncontained capacity!  "
              << "    Value = " << uncontainedCapacity << " \n"
              << "      contained = " << totalcapacity
        // << "    check = " << checkUC << " \n"
        ;
    // if( checkUC < 0 ) { assert(false); }
    return false;
    // TODO --- try to find points anyway, and decide if real points were found
  }

  const int size = points.capacity();
  points.resize(points.capacity());

  Vector3D<Precision> lower, upper, offset;
  volume.GetUnplacedVolume()->Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  const int maxtries = 100 * 1000 * 1000;

  int tries = 0; // count total trials ...
  int i;
  for (i = 0; i < size; ++i) {
    bool contained;
    Vector3D<Precision> point;
    do {
      // ensure that point is contained in mother volume
      int onego = 0;
      do {
        ++tries;
        onego++;
        if (onego % 100000 == 0) {
          printf("%s line %i: Warning: %i tries ( success = %i ) to find uncontained points... volume=%s.  Please "
                 "check.\n",
                 __FILE__, __LINE__, tries, i, volume.GetLabel().c_str());
        }
        if (tries % 5000000 == 0) {
          double ratio = 1.0 * i / tries;
          printf("Progress : %i tries ( succeeded = %i , ratio %f %% ) to find uncontained points... volume=%s.\n",
                 tries, i, 100.0 * ratio, volume.GetLabel().c_str());
        }

        point = offset + SamplePoint(dim, rngengine);
      } while (!volume.UnplacedContains(point));
      points.set(i, point);

      contained = false;
      int kk    = 0;
      for (Vector<Daughter>::const_iterator j = volume.GetDaughters().cbegin(), jEnd = volume.GetDaughters().cend();
           j != jEnd; ++j, ++kk) {
        if ((*j)->Contains(points[i])) {
          contained = true;
          break;
        }
      }
    } while (contained && tries < maxtries);

    if (tries >= maxtries) break;
  }
  std::cout << " FillUncontained:  trials " << tries << " for num points = " << i << " ( out of " << size
            << " requested - "  << " success ratio = " << (i * 1.0) / tries << "\n";
  return (i > 0);
}

template <typename RandomEngine, typename TrackContainer>
VECGEOM_FORCE_INLINE bool FillUncontainedPoints(LogicalVolume const &volume, RandomEngine &rngengine,
                                                TrackContainer &points)
{
  VPlacedVolume const *const placed = volume.Place();
  bool good                         = FillUncontainedPoints(*placed, rngengine, points);
  delete placed;

  return good;
}

/**
 * @brief Fill a container structure (SOA3D or AOS3D) with random
 *    points contained in a volume. Points are returned in the reference
 *    frame of the volume (and not in the mother containing this volume)
 * @details Input volume must have a valid bounding box, which is used
 *    for sampling.
 * @param volume containing all points
 * @param points is the output container, provided by the caller.
 * returns if successful or not
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
bool FillRandomPoints(VPlacedVolume const &volume, TrackContainer &points)
{
  const int size = points.capacity();
  points.resize(points.capacity());

  int tries = 0;

  Vector3D<Precision> lower, upper, offset;
  volume.GetUnplacedVolume()->Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  for (int i = 0; i < size; ++i) {
    Vector3D<Precision> point;
    do {
      ++tries;
      if (tries % 1000000 == 0) {
        printf("%s line %i: Warning: %i tries to find contained points... volume=%s.  Please check.\n", __FILE__,
               __LINE__, tries, volume.GetLabel().c_str());
      }
      if (tries > 100000000) {
        printf("%s line %i: giving up\n", __FILE__, __LINE__);
        return false;
      }
      point = offset + SamplePoint(dim);
    } while (!volume.UnplacedContains(point));
    points.set(i, point);
  }
  return true;
}

/**
 * @brief Fill a container structure (SOA3D or AOS3D) with random
 *    points contained in a volume. Points are returned in the reference
 *    frame of the volume (and not in the mother containing this volume)
 * @details Input volume must have a valid bounding box, which is used
 *    for sampling.
 * @param volume containing all points
 * @param points is the output container, provided by the caller.
 * returns if successful or not
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
bool FillRandomPoints(VUnplacedVolume const &volume, TrackContainer &points)
{
  const int size = points.capacity();
  points.resize(points.capacity());

  int tries = 0;

  Vector3D<Precision> lower, upper, offset;
  volume.Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  for (int i = 0; i < size; ++i) {
    Vector3D<Precision> point;
    do {
      ++tries;
      if (tries % 1000000 == 0) {
        printf("%s line %i: Warning: %i tries to find contained points... in UnplacedVolume. Please check.\n", __FILE__,
               __LINE__, tries);
      }
      if (tries > 100000000) {
        printf("%s line %i: giving up\n", __FILE__, __LINE__);
        return false;
      }
      point = offset + SamplePoint(dim);
    } while (!volume.Contains(point));
    points.set(i, point);
  }
  return true;
}

/**
 * @brief Fills the volume with 3D points which are to be contained in
 *    any daughters of the input mother volume.
 * @details Requires a proper bounding box from the input volume.
 * @param volume is the input mother volume containing all output points.
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillContainedPoints(VPlacedVolume const &volume, const double bias, TrackContainer &points,
                         const bool placed = true)
{

  const int size = points.capacity();
  points.resize(points.capacity());

  Vector3D<Precision> lower, upper, offset;
  volume.Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  int insideCount = 0;
  std::vector<bool> insideVector(size, false);
  for (int i = 0; i < size; ++i) {
    points.set(i, offset + SamplePoint(dim));
    // measure bias, which is the fraction of points contained in daughters
    for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(), v_end = volume.GetDaughters().cend();
         v != v_end; ++v) {
      bool inside = (placed) ? (*v)->Contains(points[i]) : (*v)->UnplacedContains(points[i]);
      if (inside) {
        ++insideCount;
        insideVector[i] = true;
      }
    }
  }

  // remove contained points to reduce bias as needed
  int i     = 0;
  int totaltries = 0;
  while (static_cast<double>(insideCount) / static_cast<double>(size) > bias) {
    while (!insideVector[i])
      ++i;
    bool contained;
    do {
      ++totaltries;
      if (totaltries % 1000000 == 0) {
        printf("%s line %i: Warning: %i totaltries to reduce bias... volume=%s.  Please check.\n", __FILE__, __LINE__,
               totaltries, volume.GetLabel().c_str());
      }

      points.set(i, offset + SamplePoint(dim));
      contained = false;
      for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(), v_end = volume.GetDaughters().end();
           v != v_end; ++v) {
        bool inside = (placed) ? (*v)->Contains(points[i]) : (*v)->UnplacedContains(points[i]);
        if (inside) {
          contained = true;
          break;
        }
      }
    } while (contained);
    insideVector[i] = false;
    // tries           = 0;
    --insideCount;
    ++i;
  }

  int tries;
  // add contained points to increase bias as needed
  i     = 0;
  tries = 0;
  SOA3D<Precision> daughterpoint(1); // a "container" to be reused;
  while (static_cast<double>(insideCount) / static_cast<double>(size) < bias) {
    while (insideVector[i])
      ++i;
    bool contained = false;
    do {
      ++tries;
      if (tries % 1000000 == 0) {
        printf("%s line %i: Warning: %i tries to increase bias... volume=%s.  Please check.\n", __FILE__, __LINE__,
               tries, volume.GetLabel().c_str());
      }

      auto ndaughters = volume.GetDaughters().size();
      if (ndaughters == 1) {
        // a faster procedure for just 1 daughter --> can directly sample in daughter
        auto daughter = volume.GetDaughters().operator[](0);
        FillRandomPoints(*daughter, daughterpoint);
        points.set(i, placed ? volume.GetTransformation()->InverseTransform(daughterpoint[0]) : daughterpoint[0]);
        contained = true;
      } else {
        const Vector3D<Precision> sample = offset + SamplePoint(dim);
        for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(), v_end = volume.GetDaughters().cend();
             v != v_end; ++v) {
          bool inside = (placed) ? (*v)->Contains(sample) : (*v)->UnplacedContains(sample);
          if (inside) {
            points.set(i, sample);
            contained = true;
            break;
          }
        }
      }

    } while (!contained);
    insideVector[i] = true;
    tries           = 0;
    ++insideCount;
    ++i;
  }
}

template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillContainedPoints(VPlacedVolume const &volume, TrackContainer &points, const bool placed = true)
{
  FillContainedPoints<TrackContainer>(volume, 1, points, placed);
}

/**
 * @brief Fills a container structure (SOA3D or AOS3D) with random
 *    points contained inside a box defined by the two input corners.
 * @param lowercorner, uppercorner define the sampling box
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillRandomPoints(Vector3D<Precision> const &lowercorner, Vector3D<Precision> const &uppercorner,
                      TrackContainer &points)
{
  const int size = points.capacity();
  points.resize(points.capacity());
  Vector3D<Precision> dim    = (uppercorner - lowercorner) / 2.;
  Vector3D<Precision> offset = (uppercorner + lowercorner) / 2.;
  for (int i = 0; i < size; ++i) {
    points.set(i, offset + SamplePoint(dim));
  }
}

/**
 * @brief Fills a container structure (SOA3D or AOS3D) with random
 *    points contained inside a box defined by the two input corners, but
 *    not contained in an ecluded volume. This can be useful to sample
 *    the space in a bounding box not pertaining to the volume.
 * @param lowercorner, uppercorner define the sampling box
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer, typename ExcludedVol, bool exlu = true>
VECGEOM_FORCE_INLINE
void FillRandomPoints(Vector3D<Precision> const &lowercorner, Vector3D<Precision> const &uppercorner,
                      ExcludedVol const &vol, TrackContainer &points)
{
  const int size = points.capacity();
  points.resize(points.capacity());
  Vector3D<Precision> dim    = (uppercorner - lowercorner) / 2.;
  Vector3D<Precision> offset = (uppercorner + lowercorner) / 2.;
  for (int i = 0; i < size; ++i) {
    Vector3D<Precision> p;
    do {
      p = offset + SamplePoint(dim);
    } while (!(exlu ^ vol.Contains(p))); // XNOR
    points.set(i, p);
  }
}

/**
 * @brief Fills a (SOA3D or AOS3D) container with random points inside
 *    a box at the origin
 * @param dim is a Vector3D with w,y,z half-lengths defining the sampling box
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_FORCE_INLINE
void FillRandomPoints(Vector3D<Precision> const &dim, TrackContainer &points)
{
  FillRandomPoints(Vector3D<Precision>(-dim.x(), -dim.y(), -dim.z()), Vector3D<Precision>(dim.x(), dim.y(), dim.z()),
                   points);
}

/**
 * @brief Generates _uncontained_ global points and directions based
 *   on a logical volume.
 *
 * @details Points and direction coordinates are based on the global
 *   reference frame.  The positions have to be within a given logical
 *   volume, and not within any daughters of that logical volume.
 *
 * The function also returns the generated points in local reference
 *   frame of the logical volume.
 *
 * @param fraction: is the fraction with which the directions should
 *   hit a daughtervolume
 * @param np: number of particles
 *
 */
template <typename TrackContainer>
inline void FillGlobalPointsAndDirectionsForLogicalVolume(LogicalVolume const *lvol, TrackContainer &localpoints,
                                                          TrackContainer &globalpoints, TrackContainer &directions,
                                                          Precision fraction, int np)
{

  // we need to generate a list of all the paths ( or placements ) which reference
  // the logical volume as their deepest node

  std::list<NavigationState *> allpaths;
  GeoManager::Instance().getAllPathForLogicalVolume(lvol, allpaths);

  NavigationState *s1       = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  NavigationState *s2       = NavigationState::MakeInstance(GeoManager::Instance().getMaxDepth());
  int virtuallyhitsdaughter = 0;
  int reallyhitsdaughter    = 0;
  if (allpaths.size() > 0) {
    // get one representative of such a logical volume
    VPlacedVolume const *pvol = allpaths.front()->Top();

    // generate points which are in lvol but not in its daughters
    bool good = FillUncontainedPoints(*pvol, localpoints);
    // assert(good);
    if (!good) {
      std::cerr << "FATAL ERROR> FillUncontainedPoints failed for volume " << pvol->GetName() << std::endl;
      exit(1);
    }

    // now have the points in the local reference frame of the logical volume
    FillBiasedDirections(*lvol, localpoints, fraction, directions);

    // transform points to global frame
    globalpoints.resize(globalpoints.capacity());
    int placedcount = 0;

    while (placedcount < np) {
      std::list<NavigationState *>::iterator iter = allpaths.begin();
      while (placedcount < np && iter != allpaths.end()) {
        // this is matrix linking local and global reference frame
        Transformation3D m;
        (*iter)->TopMatrix(m);

        bool hitsdaughter = IsHittingAnyDaughter(localpoints[placedcount], directions[placedcount], *lvol);
        if (hitsdaughter) virtuallyhitsdaughter++;
        globalpoints.set(placedcount, m.InverseTransform(localpoints[placedcount]));
        directions.set(placedcount, m.InverseTransformDirection(directions[placedcount]));

        // do extensive cross tests
        s1->Clear();
        s2->Clear();
        GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), globalpoints[placedcount], *s1, true);
        assert(s1->Top()->GetLogicalVolume() == lvol);
        Precision step = vecgeom::kInfLength;
        auto nav       = s1->Top()->GetLogicalVolume()->GetNavigator();
        nav->FindNextBoundaryAndStep(globalpoints[placedcount], directions[placedcount], *s1, *s2, vecgeom::kInfLength,
                                     step);
#ifdef DEBUG
        if (!hitsdaughter) assert(s1->Distance(*s2) > s2->GetCurrentLevel() - s1->GetCurrentLevel());
#endif
        if (hitsdaughter)
          if (s1->Distance(*s2) == s2->GetCurrentLevel() - s1->GetCurrentLevel()) {
            reallyhitsdaughter++;
          }

        placedcount++;
        iter++;
      }
    }
  } else {
    // an error message
    printf("VolumeUtilities: FillGlobalPointsAndDirectionsForLogicalVolume()... ERROR condition detected.\n");
  }
  printf(" really hits %d, virtually hits %d ", reallyhitsdaughter, virtuallyhitsdaughter);
  NavigationState::ReleaseInstance(s1);
  NavigationState::ReleaseInstance(s2);
  std::list<NavigationState *>::iterator iter = allpaths.begin();
  while (iter != allpaths.end()) {
    NavigationState::ReleaseInstance(*iter);
    ++iter;
  }
}

// same as above; logical volume is given by name
template <typename TrackContainer>
inline void FillGlobalPointsAndDirectionsForLogicalVolume(std::string const &name, TrackContainer &localpoints,
                                                          TrackContainer &globalpoints, TrackContainer &directions,
                                                          Precision fraction, int np)
{

  LogicalVolume const *vol = GeoManager::Instance().FindLogicalVolume(name.c_str());
  if (vol != NULL)
    FillGlobalPointsAndDirectionsForLogicalVolume(vol, localpoints, globalpoints, directions, fraction, np);
}

// same as above; logical volume is given by id
template <typename TrackContainer>
inline void FillGlobalPointsAndDirectionsForLogicalVolume(int id, TrackContainer &localpoints,
                                                          TrackContainer &globalpoints, TrackContainer &directions,
                                                          Precision fraction, int np)
{

  LogicalVolume const *vol = GeoManager::Instance().FindLogicalVolume(id);
  if (vol != NULL)
    FillGlobalPointsAndDirectionsForLogicalVolume(vol, localpoints, globalpoints, directions, fraction, np);
}

/**
 * @brief Generates _uncontained_ global points based
 *   on a logical volume.
 *
 * @details Points coordinates are based on the global
 *   reference frame.  The positions have to be within a given logical
 *   volume, and optionally not within any daughters of that logical volume.
 *
 * * @param np: number of particles
 *
 */
template <typename TrackContainer>
inline void FillGlobalPointsForLogicalVolume(LogicalVolume const *lvol, TrackContainer &localpoints,
                                             TrackContainer &globalpoints, int np, bool maybeindaughters = false)
{

  // we need to generate a list of all the paths ( or placements ) which reference
  // the logical volume as their deepest node

  std::list<NavigationState *> allpaths;
  GeoManager::Instance().getAllPathForLogicalVolume(lvol, allpaths);

  if (allpaths.size() > 0) {
    // get one representative of such a logical volume
    VPlacedVolume const *pvol = allpaths.front()->Top();

    if (maybeindaughters) {
      FillContainedPoints(*pvol, localpoints);
    } else {
      // generate points which are in lvol but not in its daughters
      bool good = FillUncontainedPoints(*pvol, localpoints);
      // assert(good);
      if (!good) {
        std::cerr << "FATAL ERROR> FillUncontainedPoints failed for volume " << pvol->GetName() << std::endl;
        exit(1);
      }
    }

    // transform points to global frame
    globalpoints.resize(globalpoints.capacity());
    int placedcount = 0;

    while (placedcount < np) {
      std::list<NavigationState *>::iterator iter = allpaths.begin();
      while (placedcount < np && iter != allpaths.end()) {
        // this is matrix linking local and global reference frame
        Transformation3D m;
        (*iter)->TopMatrix(m);

        globalpoints.set(placedcount, m.InverseTransform(localpoints[placedcount]));

        placedcount++;
        iter++;
      }
    }
  } else {
    // an error message
    printf("VolumeUtilities: FillGlobalPointsForLogicalVolume()... ERROR condition detected.\n");
  }

  std::list<NavigationState *>::iterator iter = allpaths.begin();
  while (iter != allpaths.end()) {
    NavigationState::ReleaseInstance(*iter);
    ++iter;
  }
}

// same as above; logical volume is given by name
template <typename TrackContainer>
inline void FillGlobalPointsForLogicalVolume(std::string const &name, TrackContainer &localpoints,
                                             TrackContainer &globalpoints, int np)
{

  LogicalVolume const *vol = GeoManager::Instance().FindLogicalVolume(name.c_str());
  if (vol != NULL) FillGlobalPointsForLogicalVolume(vol, localpoints, globalpoints, np);
}

// same as above; logical volume is given by id
template <typename TrackContainer>
inline void FillGlobalPointsForLogicalVolume(int id, TrackContainer &localpoints, TrackContainer &globalpoints, int np)
{

  LogicalVolume const *vol = GeoManager::Instance().FindLogicalVolume(id);
  if (vol != NULL) FillGlobalPointsForLogicalVolume(vol, localpoints, globalpoints, np);
}

inline Precision GetRadiusInRing(Precision rmin, Precision rmax)
{

  // Generate radius in annular ring according to uniform area
  if (rmin <= 0.) {
    return rmax * std::sqrt(RNG::Instance().uniform());
  }
  if (rmin != rmax) {
    Precision rmin2 = rmin * rmin;
    Precision rmax2 = rmax * rmax;
    return std::sqrt(rmin2 + RNG::Instance().uniform() * (rmax2 - rmin2));
  }
  return rmin;
}

/** This function will detect whether two aligned boxes intersects or not.
 *  returns a boolean, true if intersection exist, else false
 *
 *  Since the boxes are already aligned so we don't need Transformation matrices
 *  for the intersection detection algorithm.
 *                                  _
 *  input : 1. lowercornerFirstBox   |__ Extent of First Aligned UnplacedBox in mother's reference frame.
 *          2. uppercornerFirstBox  _|
 *                                  _
 *          3. lowercornerSecondBox  |__ Extent of Second Aligned UnplacedBox in mother's reference frame.
 *          4. uppercornerSecondBox _|
 *
 *  output :  Return a boolean, true if intersection exists, otherwise false.
 *
 */
VECGEOM_FORCE_INLINE
bool IntersectionExist(Vector3D<Precision> const lowercornerFirstBox, Vector3D<Precision> const uppercornerFirstBox,
                       Vector3D<Precision> const lowercornerSecondBox, Vector3D<Precision> const uppercornerSecondBox)
{

  // Simplest algorithm
  // Needs to handle a total of 6 cases

  // Case 1: First Box is on left of Second Box
  if (uppercornerFirstBox.x() < lowercornerSecondBox.x()) return false;

  // Case 2: First Box is on right of Second Box
  if (lowercornerFirstBox.x() > uppercornerSecondBox.x()) return false;

  // Case 3: First Box is back side
  if (uppercornerFirstBox.y() < lowercornerSecondBox.y()) return false;

  // Case 4: First Box is front side
  if (lowercornerFirstBox.y() > uppercornerSecondBox.y()) return false;

  // Case 5: First Box is below the Second Box
  if (uppercornerFirstBox.z() < lowercornerSecondBox.z()) return false;

  // Case 6: First Box is above the Second Box
  if (lowercornerFirstBox.z() > uppercornerSecondBox.z()) return false;

  return true; // boxes overlap
}

/** This function will detect whether two boxes in arbitrary orientation intersects or not.
 *  returns a boolean, true if intersection exist, else false
 *
 *  Logic is implemented using Separation Axis Theorem (SAT) for 3D
 *                                  _
 *  input : 1. lowercornerFirstBox   |__ Extent of First UnplacedBox in mother's reference frame.
 *          2. uppercornerFirstBox  _|
 *                                  _
 *          3. lowercornerSecondBox  |__ Extent of Second UnplacedBox in mother's reference frame.
 *          4. uppercornerSecondBox _|
 *                                  _
 *          5. transformFirstBox     |__ Transformation matrix of First and Second Unplaced Box
 *          6. transformSecondBox   _|
 *
 *  output :  Return a boolean, true if intersection exists, otherwise false.
 */
VECGEOM_FORCE_INLINE
bool IntersectionExist(Vector3D<Precision> const lowercornerFirstBox, Vector3D<Precision> const uppercornerFirstBox,
                       Vector3D<Precision> const lowercornerSecondBox, Vector3D<Precision> const uppercornerSecondBox,
                       Transformation3D const *transformFirstBox, Transformation3D const *transformSecondBox, bool aux)
{

  // Required variables
  Precision halfAx, halfAy, halfAz; // Half lengths of box A
  Precision halfBx, halfBy, halfBz; // Half lengths of box B

  halfAx = std::fabs(uppercornerFirstBox.x() - lowercornerFirstBox.x()) / 2.;
  halfAy = std::fabs(uppercornerFirstBox.y() - lowercornerFirstBox.y()) / 2.;
  halfAz = std::fabs(uppercornerFirstBox.z() - lowercornerFirstBox.z()) / 2.;

  halfBx = std::fabs(uppercornerSecondBox.x() - lowercornerSecondBox.x()) / 2.;
  halfBy = std::fabs(uppercornerSecondBox.y() - lowercornerSecondBox.y()) / 2.;
  halfBz = std::fabs(uppercornerSecondBox.z() - lowercornerSecondBox.z()) / 2.;

  Vector3D<Precision> pA = transformFirstBox->InverseTransform(Vector3D<Precision>(0, 0, 0));
  Vector3D<Precision> pB = transformSecondBox->InverseTransform(Vector3D<Precision>(0, 0, 0));
  Vector3D<Precision> T  = pB - pA;

  Vector3D<Precision> Ax = transformFirstBox->InverseTransformDirection(Vector3D<Precision>(1., 0., 0.));
  Vector3D<Precision> Ay = transformFirstBox->InverseTransformDirection(Vector3D<Precision>(0., 1., 0.));
  Vector3D<Precision> Az = transformFirstBox->InverseTransformDirection(Vector3D<Precision>(0., 0., 1.));

  Vector3D<Precision> Bx = transformSecondBox->InverseTransformDirection(Vector3D<Precision>(1., 0., 0.));
  Vector3D<Precision> By = transformSecondBox->InverseTransformDirection(Vector3D<Precision>(0., 1., 0.));
  Vector3D<Precision> Bz = transformSecondBox->InverseTransformDirection(Vector3D<Precision>(0., 0., 1.));

  /** Needs to handle total 15 cases for 3D.
   *   Literature can be found at following link
   *   http://www.jkh.me/files/tutorials/Separating%20Axis%20Theorem%20for%20Oriented%20Bounding%20Boxes.pdf
   */

  // Case 1:
  // L = Ax
  // std::cout<<" 1 : "<<std::fabs(T.Dot(Ax))<<" :: 2 : "<<(halfAx + std::fabs(halfBx*Ax.Dot(Bx)) +
  // std::fabs(halfBy*Ax.Dot(By)) + std::fabs(halfBz*Ax.Dot(Bz)) )<<std::endl;
  if (std::fabs(T.Dot(Ax)) >
      (halfAx + std::fabs(halfBx * Ax.Dot(Bx)) + std::fabs(halfBy * Ax.Dot(By)) + std::fabs(halfBz * Ax.Dot(Bz)))) {
    return false;
  }

  // Case 2:
  // L = Ay
  if (std::fabs(T.Dot(Ay)) >
      (halfAy + std::fabs(halfBx * Ay.Dot(Bx)) + std::fabs(halfBy * Ay.Dot(By)) + std::fabs(halfBz * Ay.Dot(Bz)))) {
    return false;
  }

  // Case 3:
  // L = Az
  if (std::fabs(T.Dot(Az)) >
      (halfAz + std::fabs(halfBx * Az.Dot(Bx)) + std::fabs(halfBy * Az.Dot(By)) + std::fabs(halfBz * Az.Dot(Bz)))) {
    return false;
  }

  // Case 4:
  // L = Bx
  if (std::fabs(T.Dot(Bx)) >
      (halfBx + std::fabs(halfAx * Ax.Dot(Bx)) + std::fabs(halfAy * Ay.Dot(Bx)) + std::fabs(halfAz * Az.Dot(Bx)))) {
    return false;
  }

  // Case 5:
  // L = By
  if (std::fabs(T.Dot(By)) >
      (halfBy + std::fabs(halfAx * Ax.Dot(By)) + std::fabs(halfAy * Ay.Dot(By)) + std::fabs(halfAz * Az.Dot(By)))) {
    return false;
  }

  // Case 6:
  // L = Bz
  if (std::fabs(T.Dot(Bz)) >
      (halfBz + std::fabs(halfAx * Ax.Dot(Bz)) + std::fabs(halfAy * Ay.Dot(Bz)) + std::fabs(halfAz * Az.Dot(Bz)))) {
    return false;
  }

  // Case 7:
  // L = Ax X Bx
  if ((std::fabs(T.Dot(Az) * Ay.Dot(Bx) - T.Dot(Ay) * Az.Dot(Bx))) >
      (std::fabs(halfAy * Az.Dot(Bx)) + std::fabs(halfAz * Ay.Dot(Bx)) + std::fabs(halfBy * Ax.Dot(Bz)) +
       std::fabs(halfBz * Ax.Dot(By)))) {
    return false;
  }

  // Case 8:
  // L = Ax X By
  if ((std::fabs(T.Dot(Az) * Ay.Dot(By) - T.Dot(Ay) * Az.Dot(By))) >
      (std::fabs(halfAy * Az.Dot(By)) + std::fabs(halfAz * Ay.Dot(By)) + std::fabs(halfBx * Ax.Dot(Bz)) +
       std::fabs(halfBz * Ax.Dot(Bx)))) {
    return false;
  }

  // Case 9:
  // L = Ax X Bz
  if ((std::fabs(T.Dot(Az) * Ay.Dot(Bz) - T.Dot(Ay) * Az.Dot(Bz))) >
      (std::fabs(halfAy * Az.Dot(Bz)) + std::fabs(halfAz * Ay.Dot(Bz)) + std::fabs(halfBx * Ax.Dot(By)) +
       std::fabs(halfBy * Ax.Dot(Bx)))) {
    return false;
  }

  // Case 10:
  // L = Ay X Bx
  if ((std::fabs(T.Dot(Ax) * Az.Dot(Bx) - T.Dot(Az) * Ax.Dot(Bx))) >
      (std::fabs(halfAx * Az.Dot(Bx)) + std::fabs(halfAz * Ax.Dot(Bx)) + std::fabs(halfBy * Ay.Dot(Bz)) +
       std::fabs(halfBz * Ay.Dot(By)))) {
    return false;
  }

  // Case 11:
  // L = Ay X By
  if ((std::fabs(T.Dot(Ax) * Az.Dot(By) - T.Dot(Az) * Ax.Dot(By))) >
      (std::fabs(halfAx * Az.Dot(By)) + std::fabs(halfAz * Ax.Dot(By)) + std::fabs(halfBx * Ay.Dot(Bz)) +
       std::fabs(halfBz * Ay.Dot(Bx)))) {
    return false;
  }

  // Case 12:
  // L = Ay X Bz
  if ((std::fabs(T.Dot(Ax) * Az.Dot(Bz) - T.Dot(Az) * Ax.Dot(Bz))) >
      (std::fabs(halfAx * Az.Dot(Bz)) + std::fabs(halfAz * Ax.Dot(Bz)) + std::fabs(halfBx * Ay.Dot(By)) +
       std::fabs(halfBy * Ay.Dot(Bx)))) {
    return false;
  }

  // Case 13:
  // L = Az X Bx
  if ((std::fabs(T.Dot(Ay) * Ax.Dot(Bx) - T.Dot(Ax) * Ay.Dot(Bx))) >
      (std::fabs(halfAx * Ay.Dot(Bx)) + std::fabs(halfAy * Ax.Dot(Bx)) + std::fabs(halfBy * Az.Dot(Bz)) +
       std::fabs(halfBz * Az.Dot(By)))) {
    return false;
  }

  // Case 14:
  // L = Az X By
  if ((std::fabs(T.Dot(Ay) * Ax.Dot(By) - T.Dot(Ax) * Ay.Dot(By))) >
      (std::fabs(halfAx * Ay.Dot(By)) + std::fabs(halfAy * Ax.Dot(By)) + std::fabs(halfBx * Az.Dot(Bz)) +
       std::fabs(halfBz * Az.Dot(Bx)))) {
    return false;
  }

  // Case 15:
  // L = Az X Bz
  if ((std::fabs(T.Dot(Ay) * Ax.Dot(Bz) - T.Dot(Ax) * Ay.Dot(Bz))) >
      (std::fabs(halfAx * Ay.Dot(Bz)) + std::fabs(halfAy * Ax.Dot(Bz)) + std::fabs(halfBx * Az.Dot(By)) +
       std::fabs(halfBy * Az.Dot(Bx)))) {
    return false;
  }

  return true;
}

/// generates regularly spaced surface points on each face of a box
/// npointsperline : number of points on each 1D line (there will be a total of
/// 6 * pointsperline * pointsperline + 1 non-degenerate points with the corner points being
/// included degenerate
template <typename T>
void GenerateRegularSurfacePointsOnBox(Vector3D<T> const &lower, Vector3D<T> const &upper, int pointsperline,
                                       std::vector<Vector3D<T>> &points)
{
  const auto lengthvector = upper - lower;
  const auto delta        = lengthvector / (1. * pointsperline);

  // face y-z at x =y -L and x = +L
  for (int ny = 0; ny < pointsperline; ++ny) {
    const auto y = lower.y() + delta.y() * ny;
    for (int nz = 0; nz < pointsperline; ++nz) {
      const auto z = lower.z() + delta.z() * nz;
      Vector3D<T> p1(lower.x(), y, z);
      Vector3D<T> p2(upper.x(), y, z);
      points.push_back(p1);
      points.push_back(p2);
    }
  }
  // face x-z at y=-L and y=+L
  for (int nx = 0; nx < pointsperline; ++nx) {
    const auto x = lower.x() + delta.x() * nx;
    for (int nz = 0; nz < pointsperline; ++nz) {
      const auto z = lower.z() + delta.z() * nz;
      Vector3D<T> p1(x, lower.y(), z);
      Vector3D<T> p2(x, upper.y(), z);
      points.push_back(p1);
      points.push_back(p2);
    }
  }
  // face x-y at z=-L and z=+L
  for (int nx = 0; nx < pointsperline; ++nx) {
    const auto x = lower.x() + delta.x() * nx;
    for (int ny = 0; ny < pointsperline; ++ny) {
      const auto y = lower.y() + delta.y() * ny;
      Vector3D<T> p1(x, y, lower.z());
      Vector3D<T> p2(x, y, upper.z());
      points.push_back(p1);
      points.push_back(p2);
    }
  }
  points.push_back(upper);
}

} // end namespace volumeUtilities
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUME_UTILITIES_H_ */
