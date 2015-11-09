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

public:
    SecondOrderSurfaceShell( Vector3D<Precision> * vertices, Precision dz ) : fDz(dz) {
        for(int i=0;i<N;++i)
        {
            int j = (i+1)%N;
            fxa[i]=vertices[i][0];
            fya[i]=vertices[i][1];
            fxb[i]=vertices[i+N][0];
            fyb[i]=vertices[i+N][1];
            fxc[i]=vertices[j][0];
            fyc[i]=vertices[j][1];
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
        }

    // analyse planarity
        for(int i=0;i<N;++i){
            fiscurved[i] = ((Abs(fDeltatx[i]) < kTolerance) && (Abs(fDeltaty[i]) < kTolerance)
                    && (Abs(ft1crosst2[i]) < kTolerance))? 0 : 1 ;
        }

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
  Float_t a[N],b[N],c[N];
  Float_t signa[N];

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
  }

  // does not vectorize
  for (int i=0;i<N;++i) {
     // zero or one to start with
      signa[i]=fiscurved[i];
      if( fiscurved[i] > 0 ) {
          // in this case the sign of a[i] could be negative
          MaskedAssign( a[i] < 0, -Backend::kOne, &signa[i]);
      }
  }

  Float_t smin[N],smax[N];
  // vectorizes
  for (int i=0;i<N;++i) {
     // treatment for curved surfaces
     Float_t d=b[i]*b[i]-4*a[i]*c[i];

     // check if we can have valid solution
     // if (d<0) return TGeoShape::Big();
     Float_t inva = (fiscurved[i]>0) ?  1./(2.*a[i]) : 1./c[i];
     Float_t sqrtd = Sqrt(d);
     Float_t tmp = signa[i]*sqrtd;
     // what is the meaning of this??
     smin[i]=(-b[i]-tmp)*inva;
     smax[i]=(-b[i]+tmp)*inva;
  }

  // does not vectorize
  for (int i=0;i<N;++i){
     //  std::cerr << i << "\t" << smin[i] << "\t" << smax[i] << "\n";
     if( fiscurved[i] > 0 )
     {
       MaskedAssign( (smin[i] > 0) && (smin[i] < dist), smin[i], &dist);
       MaskedAssign( (smax[i] > 0) && (smax[i] < dist), smax[i], &dist);
     }
     else // in planar case smin is the inverse distance and smin == smax
     {
       Float_t s = 1./smin[i];
       MaskedAssign( smin[i] > 0 && s < dist, s, &dist );
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
  Float_t a[N],b[N],c[N];
  Float_t signa[N];

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
  }

  // does not vectorize
  for (int i=0;i<N;++i)
  {
    // zero or one to start with
    signa[i]=fiscurved[i];
    if( fiscurved[i] > 0 )
    {
       MaskedAssign( a[i] < 0, -Backend::kOne, &signa[i]);
    }
  }

  Float_t smin[N],smax[N];
  // vectorizes
  for (int i=0;i<N;++i)
  {
   Float_t d=b[i]*b[i]-4*a[i]*c[i];
   // check if we can have valid solution
   // if (d<0) return TGeoShape::Big();
   Float_t inva = (fiscurved[i]>0) ?  1./(2.*a[i]) : 1./c[i];
   Float_t sqrtd = Sqrt(d);
   Float_t tmp = signa[i]*sqrtd;
   // what is the meaning of this??
   smin[i]=(-b[i]-tmp)*inva;
   smax[i]=(-b[i]+tmp)*inva;
   }


  // now we need to analyse which of those distances is good
  // does not vectorize
  Float_t dist[N];
  for (int i=0;i<N;++i){
     dist[i]=kInfinity;
#ifdef GENTRAPDEB
     std::cerr << "i " << i << " smin " << smin[i] << " smax " << smax[i] << " signa " << signa[i] << "\n";
     std::cerr << "i " << i << " 1./smin " << 1./smin[i] << " 1./smax " << 1./smax[i] << " signa " << signa[i] << "\n";
#endif
     if( fiscurved[i] > 0 ) {
       MaskedAssign( (smin[i] > 0) , smin[i], &dist[i]);
       MaskedAssign( (smax[i] > 0) && (smax[i] < dist[i]), smax[i], &dist[i] );
     }
    else // in planar case smin is the inverse distance and smin == smax
     {
       MaskedAssign( smin[i] > 0, 1./smin[i], &dist[i] );
     }
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
              Bool_t ok = xok && yok && dist[i] < resultdistance;
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

}; // end class definition

} // End inline namespace

} // End global namespace

#endif /* SECONDORDERSURFACESHELL_H_ */
