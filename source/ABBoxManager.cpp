/*
 * ABBoxManager.cpp
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#include "management/ABBoxManager.h"
#include "volumes/UnplacedBox.h"

#ifdef VECGEOM_VC
//#include "backend/vc/Backend.h"
#include "backend/vcfloat/Backend.h"
#else
#include "backend/scalarfloat/Backend.h"
#endif

#include <cassert>

namespace vecgeom {
inline namespace cxx {


void ABBoxManager::ComputeABBox(VPlacedVolume const *pvol, ABBox_t *lowerc, ABBox_t *upperc) {
  // idea: take the 8 corners of the bounding box in the reference frame of pvol
  // transform those corners and keep track of minimum and maximum extent
  // TODO: could make this code shorter with a more complex Vector3D class
  Vector3D<Precision> lower, upper;
  pvol->Extent(lower, upper);
  Vector3D<Precision> delta = upper - lower;
  Precision minx, miny, minz, maxx, maxy, maxz;
  minx = kInfinity;
  miny = kInfinity;
  minz = kInfinity;
  maxx = -kInfinity;
  maxy = -kInfinity;
  maxz = -kInfinity;
  Transformation3D const *transf = pvol->GetTransformation();
  for (int x = 0; x <= 1; ++x)
    for (int y = 0; y <= 1; ++y)
      for (int z = 0; z <= 1; ++z) {
        Vector3D<Precision> corner;
        corner.x() = lower.x() + x * delta.x();
        corner.y() = lower.y() + y * delta.y();
        corner.z() = lower.z() + z * delta.z();
        Vector3D<Precision> transformedcorner = transf->InverseTransform(corner);
        minx = std::min(minx, transformedcorner.x());
        miny = std::min(miny, transformedcorner.y());
        minz = std::min(minz, transformedcorner.z());
        maxx = std::max(maxx, transformedcorner.x());
        maxy = std::max(maxy, transformedcorner.y());
        maxz = std::max(maxz, transformedcorner.z());
      }
// put some margin around these boxes
  *lowerc = Vector3D<Precision>(minx - 1E-3, miny - 1E-3, minz - 1E-3);
  *upperc = Vector3D<Precision>(maxx + 1E-3, maxy + 1E-3, maxz + 1E-3);

#ifdef CHECK
  // do some tests on this stuff
  delta = (*upperc - *lowerc) / 2.;
  Vector3D<Precision> boxtranslation = (*lowerc + *upperc) / 2.;
  UnplacedBox box(delta);
  Transformation3D tr(boxtranslation.x(), boxtranslation.y(), boxtranslation.z());
  VPlacedVolume const *boxplaced = LogicalVolume("", &box).Place(&tr);
  // no point on the surface of the aligned box should be inside the volume
  std::cerr << "lower " << *lowerc;
  std::cerr << "upper " << *upperc;
  int contains = 0;
  for (int i = 0; i < 10000; ++i) {
    Vector3D<Precision> p = box.GetPointOnSurface() + boxtranslation;
    std::cerr << *lowerc << " " << *upperc << " " << p << "\n";
    if (pvol->Contains(p))
      contains++;
  }
  if (contains > 10) {
    Visualizer visualizer;
    visualizer.AddVolume(*pvol, *pvol->GetTransformation());
    visualizer.AddVolume(*boxplaced, tr);
    visualizer.Show();
  }
  std::cerr << "## wrong points " << contains << "\n";
#endif
}

void ABBoxManager::InitABBoxes(LogicalVolume const *lvol) {
  if (fVolToABBoxesMap[lvol->id()] != nullptr) {
    // remove old boxes first
    RemoveABBoxes(lvol);
  }
  uint ndaughters = lvol->GetDaughtersp()->size();
  ABBox_t *boxes = new ABBox_t[2 * ndaughters];
  fVolToABBoxesMap[lvol->id()] = boxes;

  // same for the vector part
  int extra = (ndaughters % Real_vSize > 0) ? 1 : 0;
  int size = 2 * (ndaughters / Real_vSize + extra);
  ABBox_v *vectorboxes = new ABBox_v[size];
  fVolToABBoxesMap_v[lvol->id()] = vectorboxes;

  // calculate boxes by iterating over daughters
  for (uint d = 0; d < ndaughters; ++d) {
    auto pvol = lvol->GetDaughtersp()->operator[](d);
    ComputeABBox(pvol, &boxes[2 * d], &boxes[2 * d + 1]);
#ifdef CHECK
    // do some tests on this stuff
    Vector3D<Precision> lower = boxes[2 * d];
    Vector3D<Precision> upper = boxes[2 * d + 1];

    Vector3D<Precision> delta = (upper - lower) / 2.;
    Vector3D<Precision> boxtranslation = (lower + upper) / 2.;
    UnplacedBox box(delta);
    Transformation3D tr(boxtranslation.x(), boxtranslation.y(), boxtranslation.z());
    VPlacedVolume const *boxplaced = LogicalVolume("", &box).Place(&tr);
//                   int contains = 0;
//                   for(int i=0;i<10000;++i)
//                   {
//                       Vector3D<Precision> p =  box.GetPointOnSurface() + boxtranslation;
//                       std::cerr << *lowerc << " " << * upperc << " " << p << "\n";
//                       if( pvol->Contains( p ) ) contains++;
//                   }
//                   if( contains > 10){
#endif
  }

  // initialize vector version of Container
  int index = 0;
  unsigned int assignedscalarvectors = 0;
  for (uint i = 0; i < ndaughters; i += Real_vSize) {
    Vector3D<Real_v> lower;
    Vector3D<Real_v> upper;
// assign by components ( this will be much more generic with new VecCore )
#ifdef VECGEOM_VC
    for (uint k = 0; k < Real_vSize; ++k) {
      if (2 * (i + k) < 2 * ndaughters) {
        lower.x()[k] = boxes[2 * (i + k)].x();
        lower.y()[k] = boxes[2 * (i + k)].y();
        lower.z()[k] = boxes[2 * (i + k)].z();
        upper.x()[k] = boxes[2 * (i + k) + 1].x();
        upper.y()[k] = boxes[2 * (i + k) + 1].y();
        upper.z()[k] = boxes[2 * (i + k) + 1].z();
        assignedscalarvectors += 2;
      } else {
        // filling in bounding boxes of zero size
        // better to put some irrational number than 0?
        lower.x()[k] = -vecgeom::kInfinity;
        lower.y()[k] = -vecgeom::kInfinity;
        lower.z()[k] = -vecgeom::kInfinity;
        upper.x()[k] = -vecgeom::kInfinity;
        upper.y()[k] = -vecgeom::kInfinity;
        upper.z()[k] = -vecgeom::kInfinity;
      }
    }
    vectorboxes[index++] = lower;
    vectorboxes[index++] = upper;
  }
#else
    lower.x() = boxes[2 * i].x();
    lower.y() = boxes[2 * i].y();
    lower.z() = boxes[2 * i].z();
    upper.x() = boxes[2 * i + 1].x();
    upper.y() = boxes[2 * i + 1].y();
    upper.z() = boxes[2 * i + 1].z();
    assignedscalarvectors += 2;

    vectorboxes[index++] = lower;
    vectorboxes[index++] = upper;
  }
#endif
  assert(index == size);
  assert(assignedscalarvectors == 2 * ndaughters);
}

void ABBoxManager::RemoveABBoxes(LogicalVolume const *lvol) {
  if( fVolToABBoxesMap[lvol->id()]!=nullptr ) delete[] fVolToABBoxesMap[lvol->id()];
}

}}
