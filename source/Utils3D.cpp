///
/// \file Utils3D.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include "base/Utils3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

using vecCore::math::Abs;
using vecCore::math::Max;

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, Plane const &hpl)
{
  os << "   plane normal: " << hpl.fNorm << "  distance = " << hpl.fDist;
  return os;
}

std::ostream &operator<<(std::ostream &os, Polygon const &poly)
{
  os << "   polygon (";
  for (size_t i = 0; i < poly.fN; ++i)
    os << i << ":" << poly.GetVertex(i) << "  ";
  os << "  normal: " << poly.fNorm << "  distance = " << poly.fDist;
  return os;
}

std::ostream &operator<<(std::ostream &os, Polyhedron const &polyh)
{
  os << "   polyhedron:\n";
  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    os << "   " << polyh.GetPolygon(i) << std::endl;
  return os;
}
#endif

void Plane::Transform(Transformation3D const &tr)
{
  // Transform normal vector
  Vec_t tempdir;
  tr.InverseTransformDirection(fNorm, tempdir);
  fNorm = tempdir;
  fDist -= fNorm.Dot(tr.Translation());
}

Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, bool convex)
    : fN(n), fConvex(convex), fNorm(), fVert(vertices), fInd(n), fSides(n)
{
  assert(fN > 2);
}

Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, Vec_t const &normal)
    : fN(n), fConvex(true), fHasNorm(true), fNorm(normal), fVert(vertices), fInd(n), fSides(n)
{
  assert(fN > 2 && fNorm.IsNormalized());
}

Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, vector_t<size_t> const &indices, bool convex)
    : fN(n), fConvex(convex), fNorm(), fVert(vertices), fInd(indices), fSides(n)
{
  CheckAndFixDegenerate();
}

void Polygon::CheckAndFixDegenerate()
{

  if (fValid) {
    return;
  }

  vector_t<size_t> validIndices;
  validIndices.push_back(fInd[0]);
  for (size_t i = 1; i < fN; i++) {
    auto diff1 = fVert[fInd[i]] - fVert[validIndices[0]];
    auto diff2 = fVert[fInd[i]] - fVert[validIndices[validIndices.size() - 1]];

    if (diff1.Mag2() > kToleranceSquared && diff2.Mag2() > kToleranceSquared) {
      validIndices.push_back(fInd[i]);
    }
  }

  fN   = validIndices.size();
  fInd = validIndices;
  fSides.resize(fN, 0);
  if (fN > 2) {
    fValid = true;
  }
}

void Polygon::Init()
{

  // Compute sides
  for (size_t i = 0; i < fN - 1; ++i) {
    fSides[i] = GetVertex(i + 1) - GetVertex(i);
    assert(fSides[i].Mag2() > kToleranceSquared);
  }
  fSides[fN - 1] = GetVertex(0) - GetVertex(fN - 1);
  assert(fSides[fN - 1].Mag2() > kToleranceSquared);
  // Compute normal if not already set
  if (!fHasNorm) {
    fNorm = fSides[0].Cross(fSides[1]);
    fNorm.Normalize();
  }
  assert((fSides[0].Cross(fSides[1])).Dot(fNorm) > 0);
  // Compute convexity if not supplied
  if (!fConvex) {
    fConvex = true;
    for (size_t i = 0; i < fN; ++i) {
      for (size_t k = 0; k < fN - 2; ++k) {
        size_t j = (i + k + 2) % fN; // remaining vertices
        if (fSides[i].Cross(GetVertex(j) - GetVertex(i)).Dot(fNorm) < 0) {
          fConvex = false;
          break;
        }
      }
    }
  }
  // Compute distance to origin
  fDist = -fNorm.Dot(GetVertex(0));
}

void Polygon::Transform(Transformation3D const &tr)
{
  // The polygon must be already initialized and the vertices transformed
  Vec_t temp;
  tr.InverseTransformDirection(fNorm, temp);
  fNorm = temp;
  // Compute sides
  for (size_t i = 0; i < fN - 1; ++i)
    fSides[i] = GetVertex(i + 1) - GetVertex(i);
  fSides[fN - 1] = GetVertex(0) - GetVertex(fN - 1);
  // Compute distance to origin
  fDist = -fNorm.Dot(GetVertex(0));
}

void Polyhedron::Transform(Transformation3D const &tr)
{
  // Transform vertices
  for (size_t i = 0; i < fVert.size(); ++i) {
    Vec_t temp;
    tr.InverseTransform(fVert[i], temp);
    fVert[i] = temp;
  }
  for (size_t i = 0; i < fPolys.size(); ++i)
    fPolys[i].Transform(tr);
}

bool Polyhedron::AddPolygon(const Polygon &poly)
{
  if (!poly.fValid) return false;
  fPolys.push_back(poly);
  return true;
}

void FillBoxPolyhedron(Vec_t const &box, Polyhedron &polyh)
{
  polyh.Reset(8, 6);
  vector_t<Vec_t> &vert    = polyh.fVert;
  vector_t<Polygon> &polys = polyh.fPolys;

  vert          = {{-box[0], -box[1], -box[2]}, {-box[0], box[1], -box[2]}, {box[0], box[1], -box[2]},
          {box[0], -box[1], -box[2]},  {-box[0], -box[1], box[2]}, {-box[0], box[1], box[2]},
          {box[0], box[1], box[2]},    {box[0], -box[1], box[2]}};
  polys         = {{4, vert, {0., 0., -1.}}, {4, vert, {0., 0., 1.}}, {4, vert, {-1., 0., 0.}},
           {4, vert, {0., 1., 0.}},  {4, vert, {1., 0., 0.}}, {4, vert, {0., -1., 0.}}};
  polys[0].fInd = {0, 1, 2, 3};
  polys[1].fInd = {4, 7, 6, 5};
  polys[2].fInd = {0, 4, 5, 1};
  polys[3].fInd = {1, 5, 6, 2};
  polys[4].fInd = {2, 6, 7, 3};
  polys[5].fInd = {3, 7, 4, 0};
  for (size_t i = 0; i < 6; ++i)
    polys[i].Init();
}

EPlaneXing_t PlaneXing(Plane const &pl1, Plane const &pl2, Vector3D<double> &point, Vector3D<double> &direction)
{
  direction        = pl1.fNorm.Cross(pl2.fNorm);
  const double det = direction.Mag2();
  if (Abs(det) < kTolerance) {
    // The 2 planes are parallel, let's find the distance between them
    const double d12 = Abs(Abs(pl1.fDist) - Abs(pl2.fDist));
    if (d12 < kTolerance) {
      if (pl1.fDist * pl2.fDist * pl1.fNorm.Dot(pl2.fNorm) > 0.) return kIdentical;
    }
    return kParallel;
  }
  // The planes do intersect
  point = ((pl1.fDist * direction.Cross(pl2.fNorm) - pl2.fDist * direction.Cross(pl1.fNorm))) / det;
  direction.Normalize();
  return kIntersecting;
}

EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line)
{
  using Vec_t = Vector3D<double>;
  using vecCore::math::CopySign;
  using vecCore::math::Max;
  using vecCore::math::Min;
  using vecCore::math::Sqrt;

  Vec_t point, direction;
  EPlaneXing_t crossing = PlaneXing(Plane(poly1.fNorm, poly1.fDist), Plane(poly2.fNorm, poly2.fDist), point, direction);
  if (crossing == kParallel) return kDisjoint;

  if (crossing == kIdentical) {
    // We use the separate axis theorem
    // loop segments of 1
    for (size_t i = 0; i < poly1.fN; ++i) {
      // loop vertices of 2
      bool outside = false;
      for (size_t j = 0; j < poly2.fN; ++j) {
        outside = poly1.fNorm.Dot((poly2.GetVertex(j) - poly1.GetVertex(i)).Cross(poly1.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    // loop segments of 2
    for (size_t i = 0; i < poly2.fN; ++i) {
      // loop vertices of 1
      bool outside = false;
      for (size_t j = 0; j < poly1.fN; ++j) {
        outside = poly2.fNorm.Dot((poly1.GetVertex(j) - poly2.GetVertex(i)).Cross(poly2.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    return kTouching;
  }

  // The polygons do cross each other along a line
  if (!(poly1.fConvex | poly2.fConvex)) return kDisjoint; // cannot solve yet non-convex case
  double smin1 = InfinityLength<double>();
  double smax1 = -InfinityLength<double>();
  for (size_t i = 0; i < poly1.fN; ++i) {
    Vec_t crossdirs      = poly1.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly1.GetVertex(i)).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) continue;
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly1.GetVertex(i)).Cross(poly1.fSides[i]);
      double distsq = CopySign(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      smin1         = Min(smin1, distsq);
      smax1         = Max(smax1, distsq);
    }
  }
  if (smax1 <= smin1) return kDisjoint;

  double smin2 = InfinityLength<double>();
  double smax2 = -InfinityLength<double>();
  for (size_t i = 0; i < poly2.fN; ++i) {
    Vec_t crossdirs      = poly2.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly2.GetVertex(i)).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) continue;
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly2.GetVertex(i)).Cross(poly2.fSides[i]);
      double distsq = CopySign(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      smin2         = Min(smin2, distsq);
      smax2         = Max(smax2, distsq);
    }
  }
  if (smax2 <= smin2) return kDisjoint;
  if (smin2 - smax1 > -kTolerance || smin1 - smax2 > -kTolerance) return kDisjoint;
  if (line != nullptr) {
    double dmin = Max(smin1, smin2);
    double dmax = Min(smax1, smax2);
    assert(dmax - dmin > -kTolerance);
    line->fPts[0] = point + direction * CopySign(Sqrt(Abs(dmin)), dmin);
    line->fPts[1] = point + direction * CopySign(Sqrt(Abs(dmax)), dmax);
  }
  return kOverlapping;
}

EBodyXing_t PolyhedronXing(Polyhedron const &polyh1, Polyhedron const &polyh2, vector_t<Line> &lines)
{
  // We assume PolyhedronCollision was called and the polyhedra do intersect. The polihedra are already transformed.
  using vecCore::math::Max;
  Line line;
  EBodyXing_t result = kDisjoint;
  for (const auto &poly1 : polyh1.fPolys) {
    for (const auto &poly2 : polyh2.fPolys) {
      EBodyXing_t crossing = PolygonXing(poly1, poly2, &line);
      result               = Max(result, crossing);
      if (crossing == kOverlapping) lines.push_back(line);
    }
  }
  return result;
}

EBodyXing_t BoxCollision(Vector3D<double> const &box1, Transformation3D const &tr1, Vector3D<double> const &box2,
                         Transformation3D const &tr2)
{
  // A fast check if the bounding spheres touch
  using Vec_t = Vector3D<double>;
  using vecCore::math::Max;
  using vecCore::math::Min;

  Vec_t orig1 = tr1.Translation();
  Vec_t orig2 = tr2.Translation();
  double r1sq = box1.Mag2();
  double r2sq = box2.Mag2();
  double dsq  = (orig2 - orig1).Mag2();
  if (dsq > r1sq + r2sq + 2. * Sqrt(r1sq * r2sq)) return kDisjoint;

  if (!tr1.HasRotation() && !tr2.HasRotation()) {
    // Aligned boxes case
    Vector3D<double> eps1 = (orig2 - box2) - (orig1 + box1);
    Vector3D<double> eps2 = (orig1 - box1) - (orig2 + box2);
    double deps           = Max(eps1.Max(), eps2.Max());
    if (deps > kTolerance)
      return kDisjoint;
    else if (deps > -kTolerance)
      return kTouching;
    return kOverlapping;
  }
  // General case: use separating plane theorem (3D version of SAT)

  // A lambda computing min for the i component
  // compute matrix to go from 2 to 1
  Transformation3D tr12;
  tr1.Inverse(tr12);
  tr12.MultiplyFromRight(tr2); // Relative transformation of 2 in local coordinates of 1
  // Fill mesh of points for 2
  const Vec_t mesh2[8] = {{-box2[0], -box2[1], -box2[2]}, {-box2[0], box2[1], -box2[2]}, {box2[0], box2[1], -box2[2]},
                          {box2[0], -box2[1], -box2[2]},  {-box2[0], -box2[1], box2[2]}, {-box2[0], box2[1], box2[2]},
                          {box2[0], box2[1], box2[2]},    {box2[0], -box2[1], box2[2]}};
  Vec_t mesh[8];
  for (auto i = 0; i < 8; ++i)
    tr12.InverseTransform(mesh2[i], mesh[i]);

  // Check mesh2 against faces of 1
  const double maxx2 = Max(Max((mesh[0].x(), mesh[1].x()), Max(mesh[2].x(), mesh[3].x())),
                           Max((mesh[4].x(), mesh[5].x()), Max(mesh[6].x(), mesh[7].x())));
  if (maxx2 < -box1[0] - kTolerance) return kDisjoint;
  const double minx2 = Min(Min((mesh[0].x(), mesh[1].x()), Min(mesh[2].x(), mesh[3].x())),
                           Min((mesh[4].x(), mesh[5].x()), Min(mesh[6].x(), mesh[7].x())));
  if (minx2 > box1[0] + kTolerance) return kDisjoint;

  const double maxy2 = Max(Max((mesh[0].y(), mesh[1].y()), Max(mesh[2].y(), mesh[3].y())),
                           Max((mesh[4].y(), mesh[5].y()), Max(mesh[6].y(), mesh[7].y())));
  if (maxy2 < -box1[1] - kTolerance) return kDisjoint;
  const double miny2 = Min(Min((mesh[0].y(), mesh[1].y()), Min(mesh[2].y(), mesh[3].y())),
                           Min((mesh[4].y(), mesh[5].y()), Min(mesh[6].y(), mesh[7].y())));
  if (miny2 > box1[1] + kTolerance) return kDisjoint;

  const double maxz2 = Max(Max((mesh[0].z(), mesh[1].z()), Max(mesh[2].z(), mesh[3].z())),
                           Max((mesh[4].z(), mesh[5].z()), Max(mesh[6].z(), mesh[7].z())));
  if (maxz2 < -box1[2] - kTolerance) return kDisjoint;
  const double minz2 = Min(Min((mesh[0].z(), mesh[1].z()), Min(mesh[2].z(), mesh[3].z())),
                           Min((mesh[4].z(), mesh[5].z()), Min(mesh[6].z(), mesh[7].z())));
  if (minz2 > box1[2] + kTolerance) return kDisjoint;

  // Fill mesh of points for 2
  const Vec_t mesh1[8] = {{-box1[0], -box1[1], -box1[2]}, {-box1[0], box1[1], -box1[2]}, {box1[0], box1[1], -box1[2]},
                          {box1[0], -box1[1], -box1[2]},  {-box1[0], -box1[1], box1[2]}, {-box1[0], box1[1], box1[2]},
                          {box1[0], box1[1], box1[2]},    {box1[0], -box1[1], box1[2]}};

  Transformation3D tr21;
  tr2.Inverse(tr21);
  tr21.MultiplyFromRight(tr1); // Relative transformation of 2 in local coordinates of 1
  for (auto i = 0; i < 8; ++i)
    tr21.InverseTransform(mesh1[i], mesh[i]);

  // Check mesh2 against faces of 1
  const double maxx1 = Max(Max((mesh[0].x(), mesh[1].x()), Max(mesh[2].x(), mesh[3].x())),
                           Max((mesh[4].x(), mesh[5].x()), Max(mesh[6].x(), mesh[7].x())));
  if (maxx1 < -box2[0] - kTolerance) return kDisjoint;
  const double minx1 = Min(Min((mesh[0].x(), mesh[1].x()), Min(mesh[2].x(), mesh[3].x())),
                           Min((mesh[4].x(), mesh[5].x()), Min(mesh[6].x(), mesh[7].x())));
  if (minx1 > box2[0] + kTolerance) return kDisjoint;

  const double maxy1 = Max(Max((mesh[0].y(), mesh[1].y()), Max(mesh[2].y(), mesh[3].y())),
                           Max((mesh[4].y(), mesh[5].y()), Max(mesh[6].y(), mesh[7].y())));
  if (maxy1 < -box2[1] - kTolerance) return kDisjoint;
  const double miny1 = Min(Min((mesh[0].y(), mesh[1].y()), Min(mesh[2].y(), mesh[3].y())),
                           Min((mesh[4].y(), mesh[5].y()), Min(mesh[6].y(), mesh[7].y())));
  if (miny1 > box2[1] + kTolerance) return kDisjoint;

  const double maxz1 = Max(Max((mesh[0].z(), mesh[1].z()), Max(mesh[2].z(), mesh[3].z())),
                           Max((mesh[4].z(), mesh[5].z()), Max(mesh[6].z(), mesh[7].z())));
  if (maxz1 < -box2[2] - kTolerance) return kDisjoint;
  const double minz1 = Min(Min((mesh[0].z(), mesh[1].z()), Min(mesh[2].z(), mesh[3].z())),
                           Min((mesh[4].z(), mesh[5].z()), Min(mesh[6].z(), mesh[7].z())));
  if (minz1 > box2[2] + kTolerance) return kDisjoint;

  return kOverlapping;
}

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
