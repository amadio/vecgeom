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

    // indicate which surfaces are planar
    Precision fiscurved[N];
    
    // flag that all surfaces are planar
    bool fisplanar;
		
    // pre-computed normals
    Vector3D<Precision> fNormals[N];

public:
    SecondOrderSurfaceShell( Vector3D<Precision> * vertices, Precision dz ) : fDz(dz) {
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
            Precision dz2 =0.5/fDz;
            ftx1[i]=dz2*(fxb[i]-fxa[i]);
            fty1[i]=dz2*(fyb[i]-fya[i]);
            ftx2[i]=dz2*(fxd[i]-fxc[i]);
            fty2[i]=dz2*(fyd[i]-fyc[i]);

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
//    typedef typename Backend::bool_v Bool_t;

  Float_t dist(kInfinity);
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

     // I am wondering whether we can simplify this a bit ( the minuses might cancel out some terms )
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

  Float_t smin[N],smax[N];
  // vectorizes
//  std::cout<<point<<std::endl;
  for (int i=0;i<N;++i) {
     // treatment for curved surfaces. Invalid solutions will be excluded.

//     Float_t inva = (fiscurved[i]>0) ?  1./(2.*a[i]) : c[i]/(b[i]*b[i]);
//     Float_t inva = 1./(2.*a[i]);
     Float_t sqrtd = signa[i]*Sqrt(d[i]);
     // what is the meaning of this??
     smin[i]=(-b[i]-sqrtd)*inva[i];
     smax[i]=(-b[i]+sqrtd)*inva[i]; 
//     std::cout<<i<< ": a= "<< a[i] << " b= " << b[i] << " c= " << c[i] << " d= "<<d[i]<<"\n";  
//     std::cout<<"  inva= "<< inva[i] << "  smin: " << smin[i] << " smax: " << smax[i] << "\n";
  }

  // does not vectorize
  for (int i=0;i<N;++i){
     {
       MaskedAssign( (smin[i] > 0) && (smin[i] < dist), smin[i], &dist);
       MaskedAssign( (smax[i] > 0) && (smax[i] < dist), smax[i], &dist);
     }
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
       typename Backend::bool_v & done ) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;


  Float_t dzp =fDz+point[2];
  // calculate everything needed to solve the second order equation
  Float_t a[N],b[N],c[N],d[N];
  Float_t signa[N], inva[N];

  // vectorizes
  for (int i=0;i<N;++i)
  {
     Float_t xs1 =fxa[i]+ftx1[i]*dzp;
     Float_t ys1 =fya[i]+fty1[i]*dzp;
     Float_t xs2 =fxc[i]+ftx2[i]*dzp;
     Float_t ys2 =fyc[i]+fty2[i]*dzp;
     Float_t dxs =xs2-xs1;
     Float_t dys =ys2-ys1;
     a[i]=(fDeltatx[i]*dir[1]-fDeltaty[i]*dir[0]+ft1crosst2[i]*dir[2])*dir[2];

     // I am wondering whether we can simplify this a bit ( the minuses might cancel out some terms )
     b[i]=dxs*dir[1]-dys*dir[0]+
             (fDeltatx[i]*point[1]-fDeltaty[i]*point[0]+fty2[i]*xs1-fty1[i]*xs2
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

  Float_t smin[N],smax[N];
  // vectorizes
  for (int i=0;i<N;++i)
  {
    Float_t sqrtd = signa[i]*Sqrt(d[i]);
    smin[i]=(-b[i]-sqrtd)*inva[i];
    smax[i]=(-b[i]+sqrtd)*inva[i];     
  }


  // now we need to analyse which of those distances is good
  // does not vectorize
  Float_t dist[N];
  for (int i=0;i<N;++i){
     dist[i]=kInfinity;
//#define GENTRAPDEB = 1
#ifdef GENTRAPDEB
     std::cerr << "i " << i << " smin " << smin[i] << " smax " << smax[i] << " signa " << signa[i] << "\n";
     std::cerr << "i " << i << " 1./smin " << 1./smin[i] << " 1./smax " << 1./smax[i] << " signa " << signa[i] << "\n";
#endif
//     if( fiscurved[i] > 0 ) {
       MaskedAssign( (smin[i] > 0) , smin[i], &dist[i]);
       MaskedAssign( (smax[i] > 0) && (smax[i] < dist[i]), smax[i], &dist[i] );
//     }
//    else // in planar case smin is the inverse distance and smin == smax
//     {
//       MaskedAssign( smin[i] > 0, 1./smin[i], &dist[i] );
//     }
#ifdef GENTRAPDEB
    std::cerr << "i " << i << " dist[i] " << dist[i] << "\n";
#endif
  }

  // now check if those distances hit
  // this could vectorize depending on backend scalar or vector
  // NOTE: in this algorithm the ray might have hit more than one surface.
  // so we can not early return on the first hit ( unless we sort the distances first )

  // an alternative approach would be to do some prefiltering ( like in the box ) based on safeties and directions
  Float_t resultdistance(kInfinity);
#ifndef GENTRAP_VEC_HITCHECK
  for (int i=0;i<N;++i)
  {
      // put this into a separate kernel
      Float_t zhit = point.z() + dist[i]*dir.z();
      Bool_t isinz = Abs(zhit) < fDz;
#ifdef GENTRAPDEB
          std::cerr << "isinz " << i << ": " << isinz << "\n";
#endif
      if( ! IsEmpty(isinz) )
      {
#ifdef GENTRAPDEB
          std::cerr << "i " << i << " zhit\n ";
#endif
          Float_t leftcmpx   =  fxa[i] + (zhit+fDz)*ftx1[i];
          Float_t rightcmpx  =  fxc[i] + (zhit+fDz)*ftx2[i];
          Float_t xhit = point.x() + dist[i]*dir.x();
#ifdef GENTRAPDEB
                    std::cerr << "zhit " << zhit << " leftcmpx "
                    << leftcmpx << "rightcmpx " << rightcmpx <<
                    " xhit " << xhit << "\n";
#endif
          // check x hit
          Bool_t xok = ((MakeMinusTolerant<true>(leftcmpx) <= xhit) && (xhit <= MakePlusTolerant<true>(rightcmpx)))
                       || ((MakeMinusTolerant<true>(rightcmpx) <= xhit) && (xhit <= MakePlusTolerant<true>(leftcmpx)));
          xok &= isinz;
#ifdef GENTRAPDEB
                    std::cerr << "zhit " << zhit << " leftcmpx "
                             << leftcmpx << "rightcmpx " << rightcmpx <<
                             " xhit " << xhit << " bool " << xok << "\n";
#endif
         if( ! IsEmpty( xok ) )
          {
//          std::cerr << "i " << i << " xhit\n ";
            Float_t leftcmpy  = fya[i] + (zhit+fDz)*fty1[i];
            Float_t rightcmpy = fyc[i] + (zhit+fDz)*fty2[i];
            Float_t yhit = point.y() + dist[i]*dir.y();
            // check y hit
            Bool_t yok = ( (MakeMinusTolerant<true>(leftcmpy) <= yhit)
                         && (yhit <= MakePlusTolerant<true>(rightcmpy)) ) || ( (MakeMinusTolerant<true>(rightcmpy) <= yhit)
                                   && (yhit <= MakePlusTolerant<true>(leftcmpy)) ) ;
            yok &= xok;
#ifdef GENTRAPDEB
              std::cerr << " leftcmpy "
                                          << leftcmpy << "rightcmpy " << rightcmpy <<
                                          " yhit " << yhit << " bool " << yok << "\n";
              std::cerr << " here is what ROOT would do \n";
              Float_t xs1 =fxa[i]+ftx1[i]*dzp;
              Float_t ys1 =fya[i]+fty1[i]*dzp;
              Float_t xs2 =fxc[i]+ftx2[i]*dzp;
              Float_t ys2 =fyc[i]+fty2[i]*dzp;
              Float_t x1=xs1+ftx1[i]*dir[2]*dist[i];
              Float_t x2=xs2+ftx2[i]*dir[2]*dist[i];
              Float_t xp=point.x()+dist[i]*dir.x();
              Float_t y1=ys1+fty1[i]*dir[2]*dist[i];
              Float_t y2=ys2+fty2[i]*dir[2]*dist[i];
              Float_t yp=point.y()+dist[i]*dir.y();
              Float_t zi=(xp-x1)*(xp-x2)+(yp-y1)*(yp-y2);
              std::cerr << "Rlx " << x1 << " Rrx " << x2 << " Rly " << y1 << " Rry " << y2 << " hit " << zi << "\n";
#endif
              // if( ! IsEmpty(yok) ) std::cerr << "i " << i << " yhit\n ";
              // note here: since xok might be a SIMD mask it is not true to assume that xok==true here !!
              Bool_t ok = yok && dist[i] < resultdistance;
             // TODO: still minimize !! ( might hit two planes at same time ) !!!!
              // MaskedAssign( !done && ok, dist[i], &resultdistance);
              MaskedAssign( ok, dist[i], &resultdistance );
              // modify done flag
              // done |= ok;
          }
      }
  } // end of check
#else // if we want hit vectorization
  Float_t zi[N];
  // vectorizes nicely -- this solution seems to be faster then the one above with many checks
  for (int i=0;i<N;++i)
  {
     // put this into a separate kernel
     Float_t zhit = point.z() + dist[i]*dir.z();
     Float_t zfrombottom = zhit+fDz;
     Float_t leftcmpx  = fxa[i] + zfrombottom*ftx1[i];
     Float_t rightcmpx = fxc[i] + zfrombottom*ftx2[i];
     Float_t leftcmpy  = fya[i] + zfrombottom*fty1[i];
     Float_t rightcmpy = fyc[i] + zfrombottom*fty2[i];
     Float_t xhit = point.x() + dist[i]*dir.x();
     Float_t yhit = point.y() + dist[i]*dir.y();
     zi[i]=(xhit-leftcmpx)*(xhit-rightcmpx)+(yhit-leftcmpy)*(yhit-rightcmpy);
  } // end of check

  // final reduction
  for (int i=0;i<N;++i)
  {
      Bool_t ok = zi[i]<0. && dist[i]<resultdistance;
      MaskedAssign( ok, dist[i], &resultdistance );
  }
#endif
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
//  typedef typename Backend::bool_v Bool_t;
	
	Float_t safety = safmax;
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
		  MaskedAssign(safetyface < safety, safetyface, &safety);
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
		MaskedAssign(safetyface < safety, safetyface, &safety);
	}
//  std::cout << "safetyz = " << safmax << std::endl;
//  std::cout << "safetyplanar = " << safety << std::endl;
  if (count<N) {
    safetyface = SafetyCurved<Backend>(point);		
//  std::cout << "safetycurved = " << safetyface << std::endl;
	  MaskedAssign(safetyface < safety, safetyface, &safety);
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
//  typedef typename Backend::bool_v Bool_t;
	
	Float_t safety = safmax;
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
	    safetyface = (point - pa).Dot(fNormals[i]);
		  MaskedAssign(safetyface > safety, safetyface, &safety);
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
		MaskedAssign(safetyface > safety, safetyface, &safety);
	}
//  std::cout << "safetyz = " << safmax << std::endl;
//  std::cout << "safetyplanar = " << safety << std::endl;
  if (count<N) {
    safetyface = SafetyCurved<Backend>(point);		
//  std::cout << "safetycurved = " << safetyface << std::endl;
	  MaskedAssign(safetyface > safety, safetyface, &safety);
  }  
//  std::cout << "safety = " << safety << std::endl;
	return safety;
	
} // end SafetyToIn

//______________________________________________________________________________
  /// The type returned is the type corresponding to the backend given
  template<typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyCurved(
    Vector3D<typename Backend::precision_v> const &point) const {

  typedef typename Backend::precision_v Float_t;
//  typedef typename Backend::int_v Int_t;
//  typedef typename Backend::bool_v Bool_t;
	
	Float_t safety = kInfinity;
  Float_t  cf = 0.5 * (1. - point[2]/fDz);
  // analyse if x-y coordinates of localPoint are within polygon at z-height

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
	return safety;
} // end SafetyFace	


}; // end class definition

} // End inline namespace

} // End global namespace

#endif /* SECONDORDERSURFACESHELL_H_ */
