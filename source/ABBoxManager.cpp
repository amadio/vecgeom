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

/** Splitted Aligned bounding boxes
 *
 *  This function will calculate the "numOfSlices" num of aligned bounding
 *  boxes of "numOfSlices" divisions of Bounding box of Placed Volume
 *
 *  input : 1. *pvol : A pointer to the Placed Volume.
 *  	    2. numOfSlices : that user want
 *
 *  output : lowerc : A STL vector containing the lower extent of the newly
 *  				  calculated "numOfSlices" num of Aligned Bounding boxes
 *
 *           upperc : A STL vector containing the upper extent of the newly
 *  				  calculated "numOfSlices" num of Aligned Bounding boxes
 *
 */
void ABBoxManager::ComputeSplittedABBox(VPlacedVolume const *pvol, std::vector<ABBox_t> &lowerc,
                                        std::vector<ABBox_t> &upperc, int numOfSlices) {

  // idea: Split the Placed Bounding Box of volume into the numOfSlices.
  //		  Then pass each placed slice to the ComputABBox function,
  //		  Get the coordinates of lower and upper corner of splittedABBox,
  //		  store these coordinates into the vector of coordinates provided
  //		  by the calling function.

  Vector3D<Precision> tmpLower, tmpUpper;
  pvol->Extent(tmpLower, tmpUpper);
  Vector3D<Precision> delta = tmpUpper - tmpLower;
  // chose the largest dimension for splitting
  int dim = 0;                                        // 0 for x, 1 for y,  2 for z //default considering X is largest
  if (delta.y() > delta.x() && delta.y() > delta.z()) // if y is largest
    dim = 1;
  if (delta.z() > delta.x() && delta.z() > delta.y()) // if z is largest
    dim = 2;

  Precision splitDx = 0., splitDy = 0., splitDz = 0.;
  splitDx = delta.x();
  splitDy = delta.y();
  splitDz = delta.z();

  // Only one will execute, considering slicing only in one dimension
  Precision val = 0.;

  if (dim == 0) {
    splitDx = delta.x() / numOfSlices;
    val = -delta.x() / 2 + splitDx / 2;
  }
  if (dim == 1) {
    splitDy = delta.y() / numOfSlices;
    val = -delta.y() / 2 + splitDy / 2;
  }
  if (dim == 2) {
    splitDz = delta.z() / numOfSlices;
    val = -delta.z() / 2 + splitDz / 2;
  }

  // Precision minx, miny, minz, maxx, maxy, maxz;
  Transformation3D const *transf = pvol->GetTransformation();

  // Actual Stuff of slicing
  for (int i = 0; i < numOfSlices; i++) {
    // TODO :  Try to create sliced placed box.
    // Needs to modifiy translation parameters, without touching rotation
    // parameters

    Transformation3D transf2;
    Vector3D<Precision> transVec(0., 0., 0.);
    if (dim == 0) {
      transVec = transf->InverseTransform(Vector3D<Precision>(val, 0., 0.));
      val += splitDx;
    }
    if (dim == 1) {
      transVec = transf->InverseTransform(Vector3D<Precision>(0., val, 0.));
      val += splitDy;
    }
    if (dim == 2) {
      transVec = transf->InverseTransform(Vector3D<Precision>(0., 0., val));
      val += splitDz;
    }

    transf2.SetTranslation(transVec);
    transf2.SetRotation(transf->Rotation()[0], transf->Rotation()[1], transf->Rotation()[2], transf->Rotation()[3],
                        transf->Rotation()[4], transf->Rotation()[5], transf->Rotation()[6], transf->Rotation()[7],
                        transf->Rotation()[8]);
    transf2.SetProperties();

    Vector3D<Precision> lower1(0., 0., 0.), upper1(0., 0., 0.);
    UnplacedBox newBox2(splitDx / 2., splitDy / 2., splitDz / 2.);
    VPlacedVolume const *newBoxPlaced2 = LogicalVolume("", &newBox2).Place(&transf2);
    ABBoxManager::Instance().ComputeABBox(newBoxPlaced2, &lower1, &upper1);
    lowerc.push_back(lower1);
    upperc.push_back(upper1);
  }
}

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
