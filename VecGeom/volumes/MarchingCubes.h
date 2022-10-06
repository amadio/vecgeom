/// \file volumes/MarchingCubes.h
/// \author Guilherme Amadio

// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/**
 * @brief Solid mesh generation via the marching cubes algorithm.
 *
 * @details The marching cubes algorithm allows the creation of triangle
 * models from an implicit signed distance function or using 3D mesh data.
 * In VecGeom, the methods <tt>Contains</tt>, <tt>DistanceToIn</tt> and
 * <tt>DistanceToOut</tt> of a <tt>VUnplacedVolume</tt> are used to compute
 * the mesh for any unplaced volume type in VecGeom, including booleans.
 *
 * More details about the marching cubes can be found in the original paper[1]
 * and on Wikipedia[2]. This implementation uses tables adapted from a
 * popular page[3] that also discusses the marching cubes algorithm.
 *
 * 1. https://doi.org/10.1145/37402.37422
 * 2. https://en.wikipedia.org/wiki/Marching_cubes
 * 3. http://paulbourke.net/geometry/polygonise/
 *
 */

#ifndef VECGEOM_VOLUMES_MARCHING_CUBES_H_
#define VECGEOM_VOLUMES_MARCHING_CUBES_H_

#include "VecGeom/volumes/SolidMesh.h"
#include "VecGeom/volumes/UnplacedVolume.h"

namespace vecgeom {
namespace cxx {

/**
 * Build a triangle mesh of @p v using the marching cubes algorithm.
 * @param[in] v Unplaced volume for which to create the mesh.
 * @param[in] layers Minimum number of layers in each direction.
 * @returns A pointer to a SolidMesh instance (owned by the caller).
 * @remark The meshing grid is always composed of square voxels, so the
 * number of layers is proportional to the size of the shape in other
 * directions. For example, if a box with size 10, 20, 30 is meshed with
 * 20 @p layers, then it will have 20 layers on the x-axis, 40 in the
 * y-axis and 60 in the z-axis.
 */
SolidMesh *MarchingCubes(VUnplacedVolume const * const v, int layers);

/**
 * Build a triangle mesh of @p v using the marching cubes algorithm.
 * @param[in] v Unplaced volume for which to create the mesh.
 * @param[in] h Voxel size of the meshing grid (uniform in all directions).
 * @returns A pointer to a SolidMesh instance (owned by the caller).
 */
SolidMesh *MarchingCubes(VUnplacedVolume const * const v, Precision h);

} // namespace cxx
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_MARCHING_CUBES_H_
