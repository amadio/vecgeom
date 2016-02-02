/*
 * SecondOrderSurfaceShell.h
 *
 *  Created on: Aug 1, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_SECONDORDERSURFACESHELL_H_
#define VECGEOM_SECONDORDERSURFACESHELL_H_

#include "base/Vector3D.h"
#include "backend/Backend.h"
#include "volumes/kernel/GenericKernels.h"
#include <iostream>

//#define GENTRAPDEB = 1

namespace vecgeom {
/**
 * class providing a (SOA) encapsulation of
 * second order curved surfaces used in generic trap
 */
VECGEOM_DEVICE_FORWARD_DECLARE( class SecondOrderSurfaceShell; )

inline namespace VECGEOM_IMPL_NAMESPACE {

template <int N>
class SecondOrderSurfaceShell
{
private:
    // caching some important values for each of the curved planes
    Precision fxa[N], fya[N], fxb[N], fyb[N], fxc[N], fyc[N], fxd[N], fyd[N];
    Precision ftx1[N], fty1[N], ftx2[N], fty2[N];

    // the cross term ftx1[i]*fty2[i] - ftx2[i]*fty1[i]
    Precision ft1crosst2[N];
    // the term ftx2[i] - ftx1[i];
    Precision fDeltatx[N];
    // the term fty2[i] - fty1[i]
    Precision fDeltaty[N];

    // height of surface (coming from the height of the GenTrap)
    Precision fDz;
    Precision fDz2;  // 0.5/fDz

    // indicate which surfaces are planar
    Precision fiscurved[N];
    
    // flag that all surfaces are planar
    bool fisplanar;
		
    // pre-computed normals
    Vector3D<Precision> fNormals[N];
    
    // pre-computed cross products for normal computation
    Vector3D<Precision> fViCrossHi0[N];
    Vector3D<Precision> fViCrossVj[N];
    Vector3D<Precision> fHi1CrossHi0[N];
    
public:
    
    SecondOrderSurfaceShell( Vector3D<Precision> * vertices, Precision dz ) : fDz(dz), fDz2(0.5/dz) {
				Vector3D<Precision> va, vb, vc, vd;
				for(int i=0;i<N;++i)
        {
            int j = (i+1)%N;
						va = vertices[i];
						va[2] = -dz;
            fxa[i]=vertices[i][0];
            fya[i]=vertices[i][1];
						vb = vertices[i+N];
						vb[2] = dz;
            fxb[i]=vertices[i+N][0];
            fyb[i]=vertices[i+N][1];
						vc = vertices[j];
						vc[2] = -dz;
            fxc[i]=vertices[j][0];
            fyc[i]=vertices[j][1];
						vd = vertices[j+N];
						vd[2] = dz;
            fxd[i]=vertices[N+j][0];
            fyd[i]=vertices[N+j][1];
            ftx1[i]=fDz2*(fxb[i]-fxa[i]);
            fty1[i]=fDz2*(fyb[i]-fya[i]);
            ftx2[i]=fDz2*(fxd[i]-fxc[i]);
            fty2[i]=fDz2*(fyd[i]-fyc[i]);

            ft1crosst2[i]=ftx1[i]*fty2[i] - ftx2[i]*fty1[i];
            fDeltatx[i]=ftx2[i]-ftx1[i];
            fDeltaty[i]=fty2[i]-fty1[i];
            fNormals[i] = Vector3D<Precision>::Cross(vb-va, vc-va);
						// The computation of normals is done also for the curved surfaces, even if they will not be used
	          if (fNormals[i].Mag2() < kTolerance) {
							// points i and i+1/N are overlapping - use i+N and j+N instead
							fNormals[i] = Vector3D<Precision>::Cross(vb-va, vd-vb);
							if (fNormals[i].Mag2() < kTolerance) fNormals[i].Set(0.,0.,1.); // No surface, just a line
						}
						fNormals[i].Normalize();
            // Cross products used for normal computation
            fViCrossHi0[i] = Vector3D<Precision>::Cross(vb-va, vc-va);
            fViCrossVj[i] = Vector3D<Precision>::Cross(vb-va, vd-vc);
            fHi1CrossHi0[i] = Vector3D<Precision>::Cross(vd-vb, vc-va);
#ifdef GENTRAPDEB
						std::cout << "fNormals[" << i << "] = " << fNormals[i] << std::endl;				
#endif
        }

    // analyse planarity and precompute normals
        fisplanar = true;
        for(int i=0;i<N;++i){
//            int j = (i+1)%N;
//            fiscurved[i] = ((Abs(fDeltatx[i]) < kTolerance) && (Abs(fDeltaty[i]) < kTolerance)
//                    && (Abs(ft1crosst2[i]) < kTolerance))? 0 : 1 ;   // this test does not seem to work
            fiscurved[i] = (((Abs(fxc[i]-fxa[i]) < kTolerance) && (Abs(fyc[i]-fya[i]) < kTolerance)) ||
                            ((Abs(fxd[i]-fxb[i]) < kTolerance) && (Abs(fyd[i]-fyb[i]) < kTolerance)) ||
													  (Abs((fxc[i]-fxa[i])*(fyd[i]-fyb[i])-(fxd[i]-fxb[i])*(fyc[i]-fya[i])) < kTolerance))? 0 : 1;
            if (fiscurved[i]) fisplanar = false;                
#ifdef GENTRAPDEB
						std::cout << "fiscurved[" << i << "] = " << fiscurved[i] << std::endl;				
#endif
        }
        std::cout << "fisplanar = " << fisplanar << std::endl;
    }

  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  /**
   * A generic function calculation the distance to a set of curved/planar surfaces
   * The calculations:
   * a) autovectorizes largely then Backend=scalar
   * b) vectorizes by definition when Backend=Vc by use of the Vc library
   * c) unifies the treatment of curved/planar surfaces as much as possible to render internal vectorization possible
   *
   * Possible improvements could be: distinguish code for planar / curved at some places. Might speed up in particular the external vectorization
   * ( need a template specialization for some parts of code )
   * moreover we could get rid of some runtime if statements ( shape specialization ... )
   *
   * things to improve: error handling for boundary cases
   *
   * very likely: this kernel will not be efficient when we have only planar surfaces
   *
   * another possibility relies on the following idea:
   * we always have an even number of planar/curved surfaces. We could organize them in separate substructures...
   */
  typename Backend::precision_v DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &dir ) const {

  typedef typename Backend::precision_v Float_t;
//  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t dist(kInfinity);
  Float_t smin[N],smax[N];
  Vector3D<Float_t> unorm;
  Float_t r = -1.;
  Float_t rz;
  Bool_t inside = (Abs(point.z()) < fDz+kTolerance);
  Float_t cross;
  Float_t vertexX[N];
  Float_t vertexY[N];
  Float_t dzp =fDz+point[2];
  // Point may be on the wrong side - check this
  for (int  i = 0; i < N; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = fxa[i]+ftx1[i]*dzp;
    vertexY[i] = fya[i]+fty1[i]*dzp;
  }
  for (int  i = 0; i < N; i++) {
    int  j = (i + 1) % 4;
    Float_t  DeltaX = vertexX[j]-vertexX[i];
    Float_t  DeltaY = vertexY[j]-vertexY[i];
    cross  = ( point.x() - vertexX[i] ) * DeltaY - ( point.y() - vertexY[i] ) * DeltaX;
    inside &= (cross > -1.e-6);
  }
  // If on the wrong side, return -1.
  Float_t wrongsidedist = -1.;
  MaskedAssign(!inside , wrongsidedist, &dist);
  if ( IsEmpty(inside) ) return dist;
  
  ComputeSminSmax<Backend>(point,dir,smin,smax);

  for (int i=0;i<N;++i){
    // Check if point(s) is(are) on boundary
    Bool_t crtbound = ( Abs(smin[i]) < 10*kTolerance || Abs(smax[i]) < 10*kTolerance);
    if (!IsEmpty(crtbound)) {
      if (fiscurved[i]) UNormal<Backend>(point, i, unorm, rz, r);
      else unorm = fNormals[i];
    }
    // Starting point may be propagated close to boundary
    MaskedAssign(inside && Abs(smin[i])<10*kTolerance && dir.Dot(unorm)<0, kInfinity, &smin[i]);
    MaskedAssign(inside && Abs(smax[i])<10*kTolerance && dir.Dot(unorm)<0, kInfinity, &smax[i]);

    MaskedAssign( inside && (smin[i] > -10*kTolerance) && (smin[i] < dist), Max(smin[i],0.), &dist);
    MaskedAssign( inside && (smax[i] > -10*kTolerance) && (smax[i] < dist), Max(smax[i],0.), &dist);
  }
  return (dist);
} // end of function


/// The type returned is the type corresponding to the backend given
template<typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
/**
* A generic function calculation the distance to a set of curved/planar surfaces
* The calculations:
* a) autovectorizes largely then Backend=scalar
* b) vectorizes by definition when Backend=Vc by use of the Vc library
* c) unifies the treatment of curved/planar surfaces as much as possible to render internal vectorization possible
*
* Possible improvements could be: distinguish code for planar / curved at some places. Might speed up in particular the external vectorization
* ( need a template specialization for some parts of code )
* moreover we could get rid of some runtime if statements ( shape specialization ... )
*
* things to improve: error handling for boundary cases
*
* very likely: this kernel will not be efficient when we have only planar surfaces
*
* another possibility relies on the following idea:
* we always have an even number of planar/curved surfaces. We could organize them in separate substructures...
*/
typename Backend::precision_v DistanceToIn (
       Vector3D<typename Backend::precision_v> const& point,
       Vector3D<typename Backend::precision_v> const &dir,
       typename Backend::bool_v &done) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t smin[N],smax[N];
  ComputeSminSmax<Backend>(point,dir,smin,smax);

  // now we need to analyse which of those distances is good
  // does not vectorize
  Float_t crtdist;
  Vector3D<Float_t> hit;
  Float_t resultdistance(kInfinity);
  Float_t tolerance = 100. * kTolerance;
  Vector3D<Float_t> unorm;
  Float_t r = -1.;
  Float_t rz;
  Bool_t inside = (Abs(point.z()) < fDz-kTolerance);
  Float_t cross;
  Float_t vertexX[N];
  Float_t vertexY[N];
  Float_t dzp =fDz+point[2];
  // Point may be on the wrong side - check this
  for (int  i = 0; i < N; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = fxa[i]+ftx1[i]*dzp;
    vertexY[i] = fya[i]+fty1[i]*dzp;
  }
  for (int  i = 0; i < N; i++) {
    int  j = (i + 1) % 4;
    Float_t  DeltaX = vertexX[j]-vertexX[i];
    Float_t  DeltaY = vertexY[j]-vertexY[i];
    cross  = ( point.x() - vertexX[i] ) * DeltaY - ( point.y() - vertexY[i] ) * DeltaX;
    inside &= (cross > 1.e-6);
  }
  
  // If on the wrong side, return -1.
  Float_t wrongsidedist = -1.;
  MaskedAssign(inside & (!done), wrongsidedist, &resultdistance);
  Bool_t checked = inside | done;
  if (IsFull(checked)) return (resultdistance);
  
  for (int i=0;i<N;++i){
    crtdist=smin[i];
    // Extrapolate with hit distance candidate
    hit = point + crtdist*dir;
    Bool_t crossing = (crtdist > -tolerance) & (Abs(hit.z()) < fDz+kTolerance);
    // Early skip surface if not in Z range
    if ( !IsEmpty(crossing & (!checked)) ) {;
      // Compute local un-normalized outwards normal direction and hit ratio factors
      UNormal<Backend>(hit, i, unorm, rz, r);
      // Distance have to be positive within tolerance, and crossing must be inwards
      crossing &= ( crtdist > -tolerance) & (dir.Dot(unorm)<0.);
      // Propagated hitpoint must be on surface (rz in [0,1] checked already)
      crossing &= (r >= 0.) & (r <= 1.);
      MaskedAssign(crossing && (!checked) && crtdist<resultdistance, Max(crtdist,0.), &resultdistance);
    }  
    // For the particle(s) not crossing at smin, try smax
    if ( !IsFull(crossing | checked) ) {
      // Treat only particles not crossing at smin
      crossing = !crossing;
      crtdist=smax[i];
      hit = point + crtdist*dir;
      crossing &= (Abs(hit.z()) < fDz+kTolerance);
      if ( IsEmpty(crossing) ) continue;
      UNormal<Backend>(hit, i, unorm, rz, r);
      crossing &= ( crtdist > -tolerance) & (dir.Dot(unorm)<0.);
      crossing &= (r >= 0.) & (r <= 1.);
      MaskedAssign(crossing && (!checked) && crtdist<resultdistance, Max(crtdist,0.), &resultdistance);
    }
  }  
  return (resultdistance);
        
} // end distanceToIn function

  /**
   * A generic function calculation for the safety to a set of curved/planar surfaces. 
	   Should be smaller than safmax
   */
//______________________________________________________________________________
  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyToOut(
    Vector3D<typename Backend::precision_v> const &point,
		typename Backend::precision_v const &safmax) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  constexpr Precision eps = 100.*kTolerance;
	
	Float_t safety = safmax;
  Bool_t done = (Abs(safety) < eps);
  if (IsFull(done)) return (safety);
	Float_t safetyface = kInfinity;
	
	// loop lateral surfaces
  // We can use the surface normals to get safety for non-curved surfaces
	Vector3D<Precision> va; // vertex i of lower base
	Vector3D<Float_t> pa;   // same vertex converted to backend type
  int count = 0;
  if (fisplanar) {
	  for (int i=0; i<N; ++i)
	  {
      va.Set(fxa[i], fya[i], -fDz);
	    pa = va;
	    safetyface = (pa - point).Dot(fNormals[i]);
		  MaskedAssign((safetyface<safety) && (!done), safetyface, &safety);
	  }
    return safety;
  }
  
	for (int i=0; i<N; ++i)
	{
	  if ( fiscurved[i] > 0 ) continue;
    count++;
    va.Set(fxa[i], fya[i], -fDz);
		pa = va;
	  safetyface = (pa - point).Dot(fNormals[i]);
		MaskedAssign((safetyface<safety) && (!done), safetyface, &safety);
	}
//  std::cout << "safetyz = " << safmax << std::endl;
//  std::cout << "safetyplanar = " << safety << std::endl;
  if (count<N) {
    safetyface = SafetyCurved<Backend>(point, Backend::kTrue);		
//  std::cout << "safetycurved = " << safetyface << std::endl;
	  MaskedAssign((safetyface<safety) && (!done), safetyface, &safety);
  }  
//  std::cout << "safety = " << safety << std::endl;
	return safety;
	
} // end SafetyToOut	

//______________________________________________________________________________
  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyToIn(
    Vector3D<typename Backend::precision_v> const &point,
		typename Backend::precision_v const &safmax) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;
  constexpr Precision eps = 100.*kTolerance;
	
  Float_t safety = safmax;
  Bool_t done = (Abs(safety) < eps);
  if (IsFull(done)) return (safety);
  Float_t safetyface = kInfinity;

  // loop lateral surfaces
  // We can use the surface normals to get safety for non-curved surfaces
  Vector3D<Precision> va; // vertex i of lower base
  Vector3D<Float_t> pa;   // same vertex converted to backend type
  int count = 0;
  if (fisplanar) {
    for (int i=0; i<N; ++i) {
      va.Set(fxa[i], fya[i], -fDz);
      pa = va;
      safetyface = (point - pa).Dot(fNormals[i]);
      MaskedAssign((safetyface>safety) && (!done), safetyface, &safety);
    }
    return safety;
  }
  
  for (int i=0; i<N; ++i) {
    if ( fiscurved[i] > 0 ) continue;
    count++;
    va.Set(fxa[i], fya[i], -fDz);
    pa = va;
    safetyface = (point - pa).Dot(fNormals[i]);
    MaskedAssign(safetyface > safety, safetyface, &safety);
  }
//  std::cout << "safetyz = " << safmax << std::endl;
//  std::cout << "safetyplanar = " << safety << std::endl;
  if (count<N) {
    safetyface = SafetyCurved<Backend>(point, Backend::kFalse);		
//  std::cout << "safetycurved = " << safetyface << std::endl;
	  MaskedAssign((safetyface>safety) && (!done), safetyface, &safety);
  }  
//  std::cout << "safety = " << safety << std::endl;
	return (safety);
	
} // end SafetyToIn

//______________________________________________________________________________
  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyCurved(
    Vector3D<typename Backend::precision_v> const &point, typename Backend::bool_v in) const {

  typedef typename Backend::precision_v Float_t;
//  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;
	
  Float_t safety = kInfinity;
  Float_t tolerance = 100*kTolerance;
  MaskedAssign(!in, -tolerance, &tolerance);

  //  loop over edges connecting points i with i+4
  Float_t vertexX[N];
  Float_t vertexY[N];
  Float_t dx, dy, dpx, dpy, lsq, u;
  Float_t dx1 = 0.0;
  Float_t dx2 = 0.0;
  Float_t dy1 = 0.0;
  Float_t dy2 = 0.0;
  Float_t dzp =fDz+point[2];
  // vectorizes for scalar backend
  for (int  i = 0; i < N; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = fxa[i]+ftx1[i]*dzp;
    vertexY[i] = fya[i]+fty1[i]*dzp;
  }
  // Check if point is where it is supposed to be
  Bool_t inside = (Abs(point.z()) < fDz+tolerance);
  Float_t cross;
  for (int  i = 0; i < N; i++) {
    int  j = (i + 1) % 4;
    Float_t  DeltaX = vertexX[j]-vertexX[i];
    Float_t  DeltaY = vertexY[j]-vertexY[i];
    cross  = ( point.x() - vertexX[i] ) * DeltaY - ( point.y() - vertexY[i] ) * DeltaX;
    inside &= (cross > -tolerance);
  }
  Bool_t wrong = in & (!inside);
  wrong |= (!in) & inside;
  if ( IsFull(wrong) ) {
    safety = -kTolerance;
    return safety;
  }     
  Float_t umin = 0.0;
  for (int  i = 0; i < N; i++) {
    if (fiscurved[i] == 0) continue;
    int j = (i+1) % N;
    dx = vertexX[j] - vertexX[i];
    dy = vertexY[j] - vertexY[i];
    dpx = point[0] - vertexX[i];
    dpy = point[1] - vertexY[i];
    lsq = dx*dx + dy*dy;
    u = (dpx*dx + dpy*dy)/(lsq+kTiny);
		MaskedAssign(u>1, point[0] - vertexX[j], &dpx);
		MaskedAssign(u>1, point[1] - vertexY[j], &dpy);
    MaskedAssign(u>=0 && u<=1,  dpx - u*dx, &dpx);
    MaskedAssign(u>=0 && u<=1,  dpy - u*dy, &dpy);
    Float_t ssq = dpx*dpx + dpy*dpy; // safety squared
    MaskedAssign(ssq<safety, fxc[i] - fxa[i], &dx1);
    MaskedAssign(ssq<safety, fxd[i] - fxb[i], &dx2);
    MaskedAssign(ssq<safety, fyc[i] - fya[i], &dy1);
    MaskedAssign(ssq<safety, fyd[i] - fyb[i], &dy2);
    MaskedAssign(ssq<safety, u, &umin);
    MaskedAssign(ssq<safety, ssq, &safety);
  }
  MaskedAssign(umin<0 || umin>1, 0.0, &umin);
  dx = dx1 + umin*(dx2-dx1);
  dy = dy1 + umin*(dy2-dy1);
  safety *= 1.- 4.*fDz*fDz/(dx*dx+dy*dy+4.*fDz*fDz);
  safety = Sqrt(safety);
  MaskedAssign(wrong, -safety, &safety);
  return safety;
} // end SafetyFace	

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const *GetNormals() const { return fNormals; } 

//______________________________________________________________________________
  /// Computes un-normalized normal to surface isurf, on the input point
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void UNormal(
    Vector3D<typename Backend::precision_v> const &point, int isurf,
    Vector3D<typename Backend::precision_v> &unorm,
    typename Backend::precision_v &rz, typename Backend::precision_v &r) const {

  // unorm = (vi X hi0) + rz*(vi X vj) + r*(hi1 X hi0)
  //    where: vi, vj are the vectors (AB) and (CD) (see constructor)
  //           hi0 = (AC) and hi1 = (BD)
  //           rz = 0.5*(point.z()+dz)/dz is the vertical ratio 
  //           r = ((AP)-rz*vi) / (hi0+rz(vj-vi)) is the horizontal ratio
  // Any point within the surface range should reurn r and rz in the range [0,1]
  // These can be used as surface crossing criteria
  typedef typename Backend::precision_v Float_t;
  rz = fDz2*(point.z()+fDz);
/*
  Vector3D<Float_t> a(fxa[isurf], fya[isurf], -fDz);
  Vector3D<Float_t> vi(fxb[isurf]-fxa[isurf], fyb[isurf]-fya[isurf], 2*fDz);
  Vector3D<Float_t> vj(fxd[isurf]-fxc[isurf], fyd[isurf]-fyc[isurf], 2*fDz);
  Vector3D<Float_t> hi0(fxc[isurf]-fxa[isurf], fyc[isurf]-fya[isurf], 0.);
*/
  Float_t num = (point.x() - fxa[isurf]) - rz*(fxb[isurf]-fxa[isurf]);
  Float_t denom = (fxc[isurf]-fxa[isurf]) + rz*(fxd[isurf]-fxc[isurf]-fxb[isurf]+fxa[isurf]);
  MaskedAssign(Abs(denom)>1.e-6, num/denom, &r);
  num = (point.y() - fya[isurf]) - rz*(fyb[isurf]-fya[isurf]);
  denom = (fyc[isurf]-fya[isurf]) + rz*(fyd[isurf]-fyc[isurf]-fyb[isurf]+fya[isurf]);
  MaskedAssign(Abs(denom)>1.e-6, num/denom, &r);

  unorm = (Vector3D<Float_t>)fViCrossHi0[isurf] + 
          rz*(Vector3D<Float_t>)fViCrossVj[isurf] + 
          r*(Vector3D<Float_t>)fHi1CrossHi0[isurf];
} // end UNormal  
  
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  /**
   * Function to compute smin and smax crossings with the N lateral surfaces.
   */
  void ComputeSminSmax(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &dir,
    typename Backend::precision_v smin[N], 
    typename Backend::precision_v smax[N]) const {

  typedef typename Backend::precision_v Float_t;

  Float_t dzp =fDz+point[2];
  // calculate everything needed to solve the second order equation
  Float_t a[N],b[N],c[N],d[N];
  Float_t signa[N], inva[N];

	// vectorizes
  for (int i=0;i<N;++i) {
     Float_t xs1 =fxa[i]+ftx1[i]*dzp;
     Float_t ys1 =fya[i]+fty1[i]*dzp;
     Float_t xs2 =fxc[i]+ftx2[i]*dzp;
     Float_t ys2 =fyc[i]+fty2[i]*dzp;
     Float_t dxs =xs2-xs1;
     Float_t dys =ys2-ys1;
     a[i]=(fDeltatx[i]*dir[1]-fDeltaty[i]*dir[0]+ft1crosst2[i]*dir[2])*dir[2];
     b[i]=dxs*dir[1]-dys*dir[0]+(fDeltatx[i]*point[1]-fDeltaty[i]*point[0]+fty2[i]*xs1-fty1[i]*xs2
                           +ftx1[i]*ys2-ftx2[i]*ys1)*dir[2];
     c[i]=dxs*point[1]-dys*point[0] + xs1*ys2-xs2*ys1;
     d[i]=b[i]*b[i]-4*a[i]*c[i];
  }

  // does not vectorize
  for (int i=0;i<N;++i) {
     // zero or one to start with
      signa[i] = 0.;
      MaskedAssign( a[i] < -kTolerance, (Float_t)(-Backend::kOne), &signa[i]);
      MaskedAssign( a[i] > kTolerance, (Float_t)Backend::kOne, &signa[i]);
      inva[i] = c[i]/(b[i]*b[i]);
      MaskedAssign( Abs(a[i]) > kTolerance, 1./(2.*a[i]), &inva[i]);
  }

  // vectorizes
  for (int i=0;i<N;++i) {
     // treatment for curved surfaces. Invalid solutions will be excluded.

     Float_t sqrtd = signa[i]*Sqrt(d[i]);
     // what is the meaning of this??
     smin[i]=(-b[i]-sqrtd)*inva[i];
     smax[i]=(-b[i]+sqrtd)*inva[i]; 
  }
} // end ComputeSminSmax  

}; // end class definition

} // End inline namespace

} // End global namespace

#endif /* SECONDORDERSURFACESHELL_H_ */
