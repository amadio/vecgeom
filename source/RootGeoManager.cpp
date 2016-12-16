#include "base/Transformation3D.h"
#include "base/Stopwatch.h"
#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "navigation/NavigationState.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedRootVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedRootVolume.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedPolyhedron.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/UnplacedTorus2.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/UnplacedScaledShape.h"
#include "volumes/UnplacedGenTrap.h"
#include "volumes/UnplacedSExtruVolume.h"
#include "volumes/PlanarPolygon.h"
#include "volumes/UnplacedAssembly.h"
#include "volumes/UnplacedCutTube.h"
#include "materials/Medium.h"
#include "materials/Material.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoPara.h"
#include "TGeoParaboloid.h"
#include "TGeoPgon.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoTorus.h"
#include "TGeoArb8.h"
#include "TGeoPcon.h"
#include "TGeoXtru.h"
#include "TGeoShapeAssembly.h"
#include "TGeoScaledShape.h"
#include "TGeoEltu.h"

#include <iostream>
#include <list>

namespace vecgeom {

void RootGeoManager::LoadRootGeometry()
{
  Clear();
  GeoManager::Instance().Clear();
  TGeoNode const *const world_root = ::gGeoManager->GetTopNode();
  // Convert() will recursively convert daughters
  Stopwatch timer;
  timer.Start();
  fWorld = Convert(world_root);
  timer.Stop();
  if (fVerbose) {
    std::cout << "*** Conversion of ROOT -> VecGeom finished (" << timer.Elapsed() << " s) ***\n";
  }
  GeoManager::Instance().SetWorld(fWorld);
  timer.Start();
  GeoManager::Instance().CloseGeometry();
  timer.Stop();
  if (fVerbose) {
    std::cout << "*** Closing VecGeom geometry finished (" << timer.Elapsed() << " s) ***\n";
  }
  // fix the world --> close geometry might have changed it ( "compactification" )
  // this is very ugly of course: some observer patter/ super smart pointer might be appropriate
  fWorld = GeoManager::Instance().GetWorld();

  // setup fast lookup table
  fTGeoNodeVector.resize(VPlacedVolume::GetIdCount(), nullptr);
  auto iter = fPlacedVolumeMap.begin();
  for (; iter != fPlacedVolumeMap.end(); ++iter) {
    fTGeoNodeVector[iter->first] = iter->second;
  }
}

void RootGeoManager::LoadRootGeometry(std::string filename)
{
  if (::gGeoManager != NULL) delete ::gGeoManager;
  TGeoManager::Import(filename.c_str());
  LoadRootGeometry();
}

void RootGeoManager::ExportToROOTGeometry(VPlacedVolume const *topvolume, std::string filename)
{
  if (gGeoManager != nullptr && gGeoManager->IsClosed()) {
    std::cerr << "will not export to ROOT file; gGeoManager already initialized and closed\n";
    return;
  }
  TGeoNode *world = Convert(topvolume);
  ::gGeoManager->SetTopVolume(world->GetVolume());
  ::gGeoManager->CloseGeometry();
  ::gGeoManager->CheckOverlaps();
  ::gGeoManager->Export(filename.c_str());

  // setup fast lookup table
  fTGeoNodeVector.resize(VPlacedVolume::GetIdCount(), nullptr);
  auto iter = fPlacedVolumeMap.begin();
  for (; iter != fPlacedVolumeMap.end(); ++iter) {
    fTGeoNodeVector[iter->first] = iter->second;
  }
}

// a helper function to convert ROOT assembly constructs into a flat list of nodes
// allows parsing of more complex ROOT geometries ( until VecGeom supports assemblies natively )
void FlattenAssemblies(TGeoNode *node, std::list<TGeoNode *> &nodeaccumulator, TGeoHMatrix const *globalmatrix,
                       int currentdepth, int &count /* keeps track of number of flattenened nodes */, int &maxdepth)
{
  maxdepth = currentdepth;
  if (RootGeoManager::Instance().GetFlattenAssemblies() &&
      nullptr != dynamic_cast<TGeoVolumeAssembly *>(node->GetVolume())) {
    // it is an assembly --> so modify the matrix
    TGeoVolumeAssembly *assembly = dynamic_cast<TGeoVolumeAssembly *>(node->GetVolume());
    for (int i = 0, Nd = assembly->GetNdaughters(); i < Nd; ++i) {
      TGeoHMatrix nextglobalmatrix = *globalmatrix;
      nextglobalmatrix.Multiply(assembly->GetNode(i)->GetMatrix());
      FlattenAssemblies(assembly->GetNode(i), nodeaccumulator, &nextglobalmatrix, currentdepth + 1, count, maxdepth);
    }
  } else {
    if (currentdepth == 0) // can keep original node ( it was not an assembly )
      nodeaccumulator.push_back(node);
    else { // need a new flattened node with a different transformation
      TGeoMatrix *newmatrix   = new TGeoHMatrix(*globalmatrix);
      TGeoNodeMatrix *newnode = new TGeoNodeMatrix(node->GetVolume(), newmatrix);

      // need a new name for the flattened node
      std::string *newname = new std::string(node->GetName());
      *newname += "_assemblyinternalcount_" + std::to_string(count);
      newnode->SetName(newname->c_str());
      nodeaccumulator.push_back(newnode);
      count++;
    }
  }
}

bool RootGeoManager::PostAdjustTransformation(Transformation3D *tr, TGeoNode const *node,
                                              Transformation3D *adjustment) const
{
  // post-fixing the placement ...
  // unfortunately, in ROOT there is a special case in which a placement transformation
  // is hidden (or "misused") inside the Box shape via the origin property
  //
  // --> In this case we need to adjust the transformation for this placement
  bool adjust(false);
  if (node->GetVolume()->GetShape()->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const *>(node->GetVolume()->GetShape());
    auto o                    = box->GetOrigin();
    if (o[0] != 0. || o[1] != 0. || o[2] != 0.) {
      if (fVerbose) {
        std::cerr << "Warning: **********************************************************\n";
        std::cerr << "Warning: Found a box " << node->GetName() << " with non-zero origin\n";
        std::cerr << "Warning: **********************************************************\n";
      }
      *adjustment = Transformation3D::kIdentity;
      adjustment->SetTranslation(o[0], o[1], o[2]);
      adjustment->SetProperties();
      tr->MultiplyFromRight(*adjustment);
      tr->SetProperties();
      adjust = true;
    }
  }

  // other special cases may follow ...
  return adjust;
}

VPlacedVolume *RootGeoManager::Convert(TGeoNode const *const node)
{
  if (fPlacedVolumeMap.Contains(node)) return const_cast<VPlacedVolume *>(GetPlacedVolume(node));
  // convert node transformation
  Transformation3D const *const transformation = Convert(node->GetMatrix());
  // possibly adjust transformation
  Transformation3D adjustmentTr;
  bool adjusted = PostAdjustTransformation(const_cast<Transformation3D *>(transformation), node, &adjustmentTr);

  LogicalVolume *const logical_volume = Convert(node->GetVolume());
  VPlacedVolume *const placed_volume  = logical_volume->Place(node->GetName(), transformation);

  int remaining_daughters = 0;
  {
    // All or no daughters should have been placed already
    remaining_daughters = node->GetNdaughters() - logical_volume->GetDaughters().size();
    assert(remaining_daughters <= 0 || remaining_daughters == (int)node->GetNdaughters());
  }

  // we have to convert here assemblies to list of normal nodes
  std::list<TGeoNode *> flattenenednodelist;
  int assemblydepth   = 0;
  int flatteningcount = 0;
  for (int i = 0; i < remaining_daughters; ++i) {
    TGeoHMatrix trans = *node->GetDaughter(i)->GetMatrix();
    FlattenAssemblies(node->GetDaughter(i), flattenenednodelist, &trans, 0, flatteningcount, assemblydepth);
  }

  if ((int)flattenenednodelist.size() > remaining_daughters) {
    std::cerr << "INFO: flattening of assemblies (depth " << assemblydepth << ") resulted in "
              << flattenenednodelist.size() << " daughters vs " << remaining_daughters << "\n";
  }

  //
  for (auto &n : flattenenednodelist) {
    auto placed = Convert(n);
    logical_volume->PlaceDaughter(placed);

    // fixup placements in case the mother was shifted
    if (adjusted) {
      Transformation3D inv;
      adjustmentTr.Inverse(inv);
      inv.SetProperties();
      Transformation3D *placedtr = const_cast<Transformation3D *>(placed->GetTransformation());
      inv.MultiplyFromRight(*placedtr);
      inv.SetProperties();
      placedtr->CopyFrom(inv);
    }
  }

  fPlacedVolumeMap.Set(node, placed_volume->id());
  return placed_volume;
}

TGeoNode *RootGeoManager::Convert(VPlacedVolume const *const placed_volume)
{
  if (fPlacedVolumeMap.Contains(placed_volume->id()))
    return const_cast<TGeoNode *>(fPlacedVolumeMap[placed_volume->id()]);

  TGeoVolume *geovolume = Convert(placed_volume, placed_volume->GetLogicalVolume());
  TGeoNode *node        = new TGeoNodeMatrix(geovolume, NULL);
  fPlacedVolumeMap.Set(node, placed_volume->id());

  // only need to do daughterloop once for every logical volume.
  // So only need to check if
  // logical volume already done ( if it already has the right number of daughters )
  auto remaining_daughters = placed_volume->GetDaughters().size() - geovolume->GetNdaughters();
  assert(remaining_daughters == 0 || remaining_daughters == placed_volume->GetDaughters().size());

  // do daughters
  for (size_t i = 0; i < remaining_daughters; ++i) {
    // get placed daughter
    VPlacedVolume const *daughter_placed = placed_volume->GetDaughters().operator[](i);

    // RECURSE DOWN HERE
    TGeoNode *daughternode = Convert(daughter_placed);

    // get matrix of daughter
    TGeoMatrix *geomatrixofdaughter = Convert(daughter_placed->GetTransformation());

    // add node to the TGeoVolume; using the TGEO API
    // unfortunately, there is not interface allowing to add an existing
    // nodepointer directly
    geovolume->AddNode(daughternode->GetVolume(), i, geomatrixofdaughter);
  }

  return node;
}

Transformation3D *RootGeoManager::Convert(TGeoMatrix const *const geomatrix)
{
  // if (fTransformationMap.Contains(geomatrix)) return const_cast<Transformation3D *>(fTransformationMap[geomatrix]);

  Double_t const *const t = geomatrix->GetTranslation();
  Double_t const *const r = geomatrix->GetRotationMatrix();
  Transformation3D *const transformation =
      new Transformation3D(t[0], t[1], t[2], r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);

  //  transformation->FixZeroes();
  // transformation->SetProperties();
  fTransformationMap.Set(geomatrix, transformation);
  return transformation;
}

TGeoMatrix *RootGeoManager::Convert(Transformation3D const *const trans)
{
  if (fTransformationMap.Contains(trans)) return const_cast<TGeoMatrix *>(fTransformationMap[trans]);

  TGeoMatrix *const geomatrix = trans->ConvertToTGeoMatrix();

  fTransformationMap.Set(geomatrix, trans);
  return geomatrix;
}

LogicalVolume *RootGeoManager::Convert(TGeoVolume const *const volume)
{
  if (fLogicalVolumeMap.Contains(volume)) return const_cast<LogicalVolume *>(fLogicalVolumeMap[volume]);

  VUnplacedVolume const *unplaced;
  if (!volume->IsAssembly()) {
    unplaced = Convert(volume->GetShape());
  } else {
    unplaced = ConvertAssembly(volume);
  }
  LogicalVolume *const logical_volume = new LogicalVolume(volume->GetName(), unplaced);
  Medium const *const medium          = Convert(volume->GetMedium());
  const_cast<LogicalVolume *>(logical_volume)->SetTrackingMediumPtr((void *)medium);

  fLogicalVolumeMap.Set(volume, logical_volume);
  return logical_volume;
}

Medium *RootGeoManager::Convert(TGeoMedium const *const medium)
{
  // Check whether medium is already there
  std::vector<Medium *> media = Medium::GetMedia();
  for (auto m = media.begin(); m != media.end(); ++m) {
    //      std::cout << "Name " << (*m)->Name() << " " << medium->GetName() << std::endl;
    if ((*m)->Name() == std::string(medium->GetName())) return (*m);
  }

  //   std::cout << "Adding Medium #" << media.size() << " " << medium->GetName() << std::endl;
  // Medium not there. We add it

  Material *vmat = Convert(medium->GetMaterial());

  double pars[20];
  for (int i   = 0; i < 20; ++i)
    pars[i]    = medium->GetParam(i);
  Medium *vmed = new Medium(medium->GetName(), vmat, pars);
  return vmed;
}

Material *RootGeoManager::Convert(TGeoMaterial const *const material)
{
  auto materials = Material::GetMaterials();
  for (auto m = materials.begin(); m != materials.end(); ++m)
    if ((*m)->GetName() == std::string(material->GetName())) return (*m);
  Material *vmat = 0;
  int nelem      = material->GetNelements();

  //   std::cout << "Adding Material #" << materials.size() << " "<< material->GetName() << std::endl;
  if (nelem < 2) {
    vmat = new Material(material->GetName(), material->GetA(), material->GetZ(), material->GetDensity(),
                        material->GetRadLen(), material->GetIntLen());
  } else {
    double *a = new double[nelem];
    double *z = new double[nelem];
    double *w = new double[nelem];
    //      cout << "nelem " << nelem << endl;
    for (int i = 0; i < nelem; ++i) {
      double aa;
      double zz;
      const_cast<TGeoMaterial *>(material)->GetElementProp(aa, zz, w[i], i);
      // cout << "Elem props: A: " << aa << " Z: " << zz << endl;
      a[i] = aa;
      z[i] = zz;
    }
    vmat = new Material(material->GetName(), a, z, w, nelem, material->GetDensity(), material->GetRadLen(),
                        material->GetIntLen());
    delete[] a;
    delete[] z;
    delete[] w;
  }
  vmat->Used();
  //   cout << *vmat << endl;
  return vmat;
}

// the inverse: here we need both the placed volume and logical volume as input
// they should match
TGeoVolume *RootGeoManager::Convert(VPlacedVolume const *const placed_volume, LogicalVolume const *const logical_volume)
{
  assert(placed_volume->GetLogicalVolume() == logical_volume);

  if (fLogicalVolumeMap.Contains(logical_volume)) return const_cast<TGeoVolume *>(fLogicalVolumeMap[logical_volume]);

  TGeoVolume *geovolume = new TGeoVolume(logical_volume->GetLabel().c_str(), /* the name */
                                         placed_volume->ConvertToRoot(), 0   /* NO MATERIAL FOR THE MOMENT */
                                         );

  fLogicalVolumeMap.Set(geovolume, logical_volume);
  return geovolume;
}

VUnplacedVolume *RootGeoManager::Convert(TGeoShape const *const shape)
{

  if (fUnplacedVolumeMap.Contains(shape)) return const_cast<VUnplacedVolume *>(fUnplacedVolumeMap[shape]);

  VUnplacedVolume *unplaced_volume = NULL;

  // THE BOX
  if (shape->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const *>(shape);
    unplaced_volume           = new UnplacedBox(box->GetDX(), box->GetDY(), box->GetDZ());
  }

  // THE TUBE
  if (shape->IsA() == TGeoTube::Class()) {
    TGeoTube const *const tube = static_cast<TGeoTube const *>(shape);
    unplaced_volume            = new GenericUnplacedTube(tube->GetRmin(), tube->GetRmax(), tube->GetDz(), 0., kTwoPi);
  }

  // THE TUBESEG
  if (shape->IsA() == TGeoTubeSeg::Class()) {
    TGeoTubeSeg const *const tube = static_cast<TGeoTubeSeg const *>(shape);
    unplaced_volume =
        new GenericUnplacedTube(tube->GetRmin(), tube->GetRmax(), tube->GetDz(), kDegToRad * tube->GetPhi1(),
                                kDegToRad * (tube->GetPhi2() - tube->GetPhi1()));
  }

  // THE CONESEG
  if (shape->IsA() == TGeoConeSeg::Class()) {
    TGeoConeSeg const *const cone = static_cast<TGeoConeSeg const *>(shape);
    unplaced_volume =
        new UnplacedCone(cone->GetRmin1(), cone->GetRmax1(), cone->GetRmin2(), cone->GetRmax2(), cone->GetDz(),
                         kDegToRad * cone->GetPhi1(), kDegToRad * (cone->GetPhi2() - cone->GetPhi1()));
  }

  // THE CONE
  if (shape->IsA() == TGeoCone::Class()) {
    TGeoCone const *const cone = static_cast<TGeoCone const *>(shape);
    unplaced_volume = new UnplacedCone(cone->GetRmin1(), cone->GetRmax1(), cone->GetRmin2(), cone->GetRmax2(),
                                       cone->GetDz(), 0., kTwoPi);
  }

  // THE PARABOLOID
  if (shape->IsA() == TGeoParaboloid::Class()) {
    TGeoParaboloid const *const p = static_cast<TGeoParaboloid const *>(shape);
    unplaced_volume               = new UnplacedParaboloid(p->GetRlo(), p->GetRhi(), p->GetDz());
  }

  // THE PARALLELEPIPED
  if (shape->IsA() == TGeoPara::Class()) {
    TGeoPara const *const p = static_cast<TGeoPara const *>(shape);
    unplaced_volume =
        new UnplacedParallelepiped(p->GetX(), p->GetY(), p->GetZ(), p->GetAlpha(), p->GetTheta(), p->GetPhi());
  }

  // Polyhedron/TGeoPgon
  if (shape->IsA() == TGeoPgon::Class()) {
    TGeoPgon const *pgon = static_cast<TGeoPgon const *>(shape);
    unplaced_volume      = new UnplacedPolyhedron(pgon->GetPhi1() * kDegToRad, // phiStart
                                             pgon->GetDphi() * kDegToRad,      // phiEnd
                                             pgon->GetNedges(),                // sideCount
                                             pgon->GetNz(),                    // zPlaneCount
                                             pgon->GetZ(),                     // zPlanes
                                             pgon->GetRmin(),                  // rMin
                                             pgon->GetRmax()                   // rMax
                                             );
  }

  // TRD2
  if (shape->IsA() == TGeoTrd2::Class()) {
    TGeoTrd2 const *const p = static_cast<TGeoTrd2 const *>(shape);
    unplaced_volume         = new UnplacedTrd(p->GetDx1(), p->GetDx2(), p->GetDy1(), p->GetDy2(), p->GetDz());
  }

  // TRD1
  if (shape->IsA() == TGeoTrd1::Class()) {
    TGeoTrd1 const *const p = static_cast<TGeoTrd1 const *>(shape);
    unplaced_volume         = new UnplacedTrd(p->GetDx1(), p->GetDx2(), p->GetDy(), p->GetDz());
  }

  // TRAPEZOID
  if (shape->IsA() == TGeoTrap::Class()) {
    TGeoTrap const *const p = static_cast<TGeoTrap const *>(shape);
    if (!TGeoTrapIsDegenerate(p)) {
      unplaced_volume = new UnplacedTrapezoid(p->GetDz(), p->GetTheta() * kDegToRad, p->GetPhi() * kDegToRad,
                                              p->GetH1(), p->GetBl1(), p->GetTl1(), p->GetAlpha1() * kDegToRad,
                                              p->GetH2(), p->GetBl2(), p->GetTl2(), p->GetAlpha2() * kDegToRad);
    } else {
      std::cerr << "Warning: this trap is degenerate -- will convert it to a generic trap!!\n";
      unplaced_volume = ToUnplacedGenTrap((TGeoArb8 const *)p);
    }
  }

  // THE SPHERE | ORB
  if (shape->IsA() == TGeoSphere::Class()) {
    // make distinction
    TGeoSphere const *const p = static_cast<TGeoSphere const *>(shape);
    if (p->GetRmin() == 0. && p->GetTheta2() - p->GetTheta1() == 180. && p->GetPhi2() - p->GetPhi1() == 360.) {
      unplaced_volume = new UnplacedOrb(p->GetRmax());
    } else {
      unplaced_volume = new UnplacedSphere(p->GetRmin(), p->GetRmax(), p->GetPhi1() * kDegToRad,
                                           (p->GetPhi2() - p->GetPhi1()) * kDegToRad, p->GetTheta1() * kDegToRad,
                                           (p->GetTheta2() - p->GetTheta1()) * kDegToRad);
    }
  }

  if (shape->IsA() == TGeoCompositeShape::Class()) {
    TGeoCompositeShape const *const compshape = static_cast<TGeoCompositeShape const *>(shape);
    TGeoBoolNode const *const boolnode        = compshape->GetBoolNode();

    // need the matrix;
    Transformation3D const *lefttrans  = Convert(boolnode->GetLeftMatrix());
    Transformation3D const *righttrans = Convert(boolnode->GetRightMatrix());

    auto transformationadjust = [](Transformation3D *tr, TGeoShape const *shape) {
      // TODO: combine this with external method
      Transformation3D adjustment;
      if (shape->IsA() == TGeoBBox::Class()) {
        TGeoBBox const *const box = static_cast<TGeoBBox const *>(shape);
        auto o                    = box->GetOrigin();
        if (o[0] != 0. || o[1] != 0. || o[2] != 0.) {
          adjustment = Transformation3D::kIdentity;
          adjustment.SetTranslation(o[0], o[1], o[2]);
          adjustment.SetProperties();
          tr->MultiplyFromRight(adjustment);
          tr->SetProperties();
        }
      }
    };
    // adjust transformation in case of shifted boxes
    transformationadjust(const_cast<Transformation3D *>(lefttrans), boolnode->GetLeftShape());
    transformationadjust(const_cast<Transformation3D *>(righttrans), boolnode->GetRightShape());

    // unplaced shapes
    VUnplacedVolume const *leftunplaced  = Convert(boolnode->GetLeftShape());
    VUnplacedVolume const *rightunplaced = Convert(boolnode->GetRightShape());

    assert(leftunplaced != nullptr);
    assert(rightunplaced != nullptr);

    // the problem is that I can only place logical volumes
    VPlacedVolume *const leftplaced  = (new LogicalVolume("inner_virtual", leftunplaced))->Place(lefttrans);
    VPlacedVolume *const rightplaced = (new LogicalVolume("inner_virtual", rightunplaced))->Place(righttrans);

    // now it depends on concrete type
    if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoSubtraction) {
      unplaced_volume = new UnplacedBooleanVolume(kSubtraction, leftplaced, rightplaced);
    } else if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoIntersection) {
      unplaced_volume = new UnplacedBooleanVolume(kIntersection, leftplaced, rightplaced);
    } else if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoUnion) {
      unplaced_volume = new UnplacedBooleanVolume(kUnion, leftplaced, rightplaced);
    }
  }

  // THE TORUS
  if (shape->IsA() == TGeoTorus::Class()) {
    // make distinction
    TGeoTorus const *const p = static_cast<TGeoTorus const *>(shape);
    unplaced_volume =
        new UnplacedTorus2(p->GetRmin(), p->GetRmax(), p->GetR(), p->GetPhi1() * kDegToRad, p->GetDphi() * kDegToRad);
  }

  // THE POLYCONE
  if (shape->IsA() == TGeoPcon::Class()) {
    TGeoPcon const *const p = static_cast<TGeoPcon const *>(shape);
    unplaced_volume = new UnplacedPolycone(p->GetPhi1() * kDegToRad, p->GetDphi() * kDegToRad, p->GetNz(), p->GetZ(),
                                           p->GetRmin(), p->GetRmax());
  }

  // THE SCALED SHAPE
  if (shape->IsA() == TGeoScaledShape::Class()) {
    TGeoScaledShape const *const p = static_cast<TGeoScaledShape const *>(shape);
    // First convert the referenced shape
    VUnplacedVolume *referenced_shape = Convert(p->GetShape());
    const double *scale_root          = p->GetScale()->GetScale();
    unplaced_volume = new UnplacedScaledShape(referenced_shape, scale_root[0], scale_root[1], scale_root[2]);
  }

  // THE ELLIPTICAL TUBE AS SCALED TUBE
  if (shape->IsA() == TGeoEltu::Class()) {
    TGeoEltu const *const p = static_cast<TGeoEltu const *>(shape);
    // Create the corresponding unplaced tube, with:
    //   rmin=0, rmax=A, dz=dz, which is scaled with (1., A/B, 1.)
    UnplacedTube *tubeUnplaced = new GenericUnplacedTube(0, p->GetA(), p->GetDZ(), 0, kTwoPi);
    unplaced_volume            = new UnplacedScaledShape(tubeUnplaced, 1., p->GetB() / p->GetA(), 1.);
  }

  // THE ARB8
  if (shape->IsA() == TGeoArb8::Class() || shape->IsA() == TGeoGtra::Class()) {
    TGeoArb8 *p     = (TGeoArb8 *)(shape);
    unplaced_volume = ToUnplacedGenTrap(p);
  }

  // THE SIMPLE XTRU
  if (shape->IsA() == TGeoXtru::Class()) {
    TGeoXtru *p = (TGeoXtru *)(shape);
    // analyse convertability
    if (p->GetNz() == 2) {
      // add check on scaling and distortions
      size_t Nvert = (size_t)p->GetNvert();
      double *x    = new double[Nvert];
      double *y    = new double[Nvert];
      for (size_t i = 0; i < Nvert; ++i) {
        x[i] = p->GetX(i);
        y[i] = p->GetY(i);
      }
      // check in which orientation the polygon in given
      if (PlanarPolygon::GetOrientation(x, y, Nvert) > 0.) {
        // std::cerr << "Points not given in clockwise order ... reordering \n";
        for (size_t i = 0; i < Nvert; ++i) {
          x[Nvert - 1 - i] = p->GetX(i);
          y[Nvert - 1 - i] = p->GetY(i);
        }
      }
      unplaced_volume = new UnplacedSExtruVolume(p->GetNvert(), x, y, p->GetZ()[0], p->GetZ()[1]);
    }
  }

  // THE CUT TUBE
  if (shape->IsA() == TGeoCtub::Class()) {
    TGeoCtub *ctube = (TGeoCtub *)(shape);
    // Create the corresponding unplaced cut tube
    unplaced_volume =
        new UnplacedCutTube(ctube->GetRmin(), ctube->GetRmax(), ctube->GetDz(), kDegToRad * ctube->GetPhi1(),
                            kDegToRad * (ctube->GetPhi2() - ctube->GetPhi1()),
                            Vector3D<Precision>(ctube->GetNlow()[0], ctube->GetNlow()[1], ctube->GetNlow()[2]),
                            Vector3D<Precision>(ctube->GetNhigh()[0], ctube->GetNhigh()[1], ctube->GetNhigh()[2]));
  }

  // New volumes should be implemented here...
  if (!unplaced_volume) {
    if (fVerbose) {
      printf("Unsupported shape for ROOT volume \"%s\" of type %s. "
             "Using ROOT implementation.\n",
             shape->GetName(), shape->ClassName());
    }
    unplaced_volume = new UnplacedRootVolume(shape);
  }

  fUnplacedVolumeMap.Set(shape, unplaced_volume);
  return unplaced_volume;
}

VUnplacedVolume *RootGeoManager::ConvertAssembly(TGeoVolume const *const v)
{
  if (TGeoVolumeAssembly const *va = dynamic_cast<TGeoVolumeAssembly const *>(v)) {
    // std::cerr << "treating volume assembly " << va->GetName() << "\n";
    (void)va;
    return new UnplacedAssembly();
  }
  return nullptr;
}

void RootGeoManager::PrintNodeTable() const
{
  for (auto iter : fPlacedVolumeMap) {
    std::cerr << iter.first << " " << iter.second << "\n";
    TGeoNode const *n = iter.second;
    n->Print();
  }
}

void RootGeoManager::Clear()
{
  fPlacedVolumeMap.Clear();
  fUnplacedVolumeMap.Clear();
  fLogicalVolumeMap.Clear();
  fTransformationMap.Clear();
  // this should be done by smart pointers
  //  for (auto i = fPlacedVolumeMap.begin(); i != fPlacedVolumeMap.end(); ++i) {
  //    delete i->first;
  //  }
  //  for (auto i = fUnplacedVolumeMap.begin(); i != fUnplacedVolumeMap.end(); ++i) {
  //    delete i->first;
  //  }
  //  for (auto i = fLogicalVolumeMap.begin(); i != fLogicalVolumeMap.end(); ++i) {
  //    delete i->first;
  //  }
  //  for (auto i = fTransformationMap.begin(); i != fTransformationMap.end(); ++i) {
  //    delete i->first;
  //  }
  if (GeoManager::Instance().GetWorld() == fWorld) {
    GeoManager::Instance().SetWorld(nullptr);
  }
}

bool RootGeoManager::TGeoTrapIsDegenerate(TGeoTrap const *trap)
{
  bool degeneracy = false;
  // const_cast because ROOT is lacking a const GetVertices() function
  auto const vertices = const_cast<TGeoTrap *>(trap)->GetVertices();
  // check degeneracy within the layers (as vertices do not contains z information)
  for (int layer = 0; layer < 2; ++layer) {
    auto lowerindex = layer * 4;
    auto upperindex = (layer + 1) * 4;
    for (int i = lowerindex; i < upperindex; ++i) {
      auto currentx = vertices[2 * i];
      auto currenty = vertices[2 * i + 1];
      for (int j = lowerindex; j < upperindex; ++j) {
        if (j == i) {
          continue;
        }
        auto otherx = vertices[2 * j];
        auto othery = vertices[2 * j + 1];
        if (otherx == currentx && othery == currenty) {
          degeneracy = true;
        }
      }
    }
  }
  return degeneracy;
}

UnplacedGenTrap *RootGeoManager::ToUnplacedGenTrap(TGeoArb8 const *p)
{
  // Create the corresponding GenTrap
  const double *vertices = const_cast<TGeoArb8 *>(p)->GetVertices();
  Precision verticesx[8], verticesy[8];
  for (auto ivert = 0; ivert < 8; ++ivert) {
    verticesx[ivert] = vertices[2 * ivert];
    verticesy[ivert] = vertices[2 * ivert + 1];
  }
  return new UnplacedGenTrap(verticesx, verticesy, p->GetDz());
}

// lookup the placed volume corresponding to a TGeoNode
VPlacedVolume const *RootGeoManager::Lookup(TGeoNode const *node) const
{
  if (node == nullptr) return nullptr;
  return Index2PVolumeConverter<NavStateIndex_t>::ToPlacedVolume(fPlacedVolumeMap[node]);
}

} // End global namespace
