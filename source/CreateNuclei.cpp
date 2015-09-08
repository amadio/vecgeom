#ifdef __clang__
#pragma clang optimize off
#endif
#include "materials/Nucleus.h"
#include <iostream>
namespace vecgeom {
   inline namespace VECGEOM_IMPL_NAMESPACE {



// this is an auto-generated function ( contact Federico.Carminati@cern.ch )
__attribute__((optimize("O0")))
void Nucleus::CreateNuclei() {
#ifdef VECGEOM_GENERATE_MATERIALS_DB
    Nucleus *nuc=0;

   // Adding 1-H-1-0
   nuc = new Nucleus("H",1,1,0,1.00783,0,0,99.985,0,0,0,0);

   // Adding 1-H-2-0
   nuc = new Nucleus("H",2,1,0,2.0141,0,0,0.015,0,0,0,0);

   // Adding 1-H-3-0
   nuc = new Nucleus("H",3,1,0,3.01605,0,3.88839e+08,0,1.8e-11,1.8e-11,0,0);
   nuc->AddDecay(0,1,0,0.0185995,100);

   // Adding 2-HE-3-0
   nuc = new Nucleus("HE",3,2,0,3.01603,0,0,0.000138,0,0,0,0);

   // Adding 1-H-4-0
   nuc = new Nucleus("H",4,1,0,4.02791,0,0,0,0,0,0,-3);
   nuc->AddDecay(-1,0,0,2.98061,100);

   // Adding 2-HE-4-0
   nuc = new Nucleus("HE",4,2,0,4.0026,0,0,99.9999,0,0,0,0);

   // Adding 3-LI-4-0
   nuc = new Nucleus("LI",4,3,0,4.02718,0,0,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,3.10002,100);

   // Adding 2-HE-5-0
   nuc = new Nucleus("HE",5,2,0,5.01222,0,7.60397e-22,0,0,0,0,-8);
   nuc->AddDecay(-1,0,0,0.89005,50);
   nuc->AddDecay(-4,-2,0,0.89005,50);

   // Adding 3-LI-5-0
   nuc = new Nucleus("LI",5,3,0,5.01254,0,3.04159e-22,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,1.96505,50);
   nuc->AddDecay(-1,-1,0,1.96505,50);

   // Adding 2-HE-6-0
   nuc = new Nucleus("HE",6,2,0,6.01889,0,0.8067,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.50776,100);

   // Adding 3-LI-6-0
   nuc = new Nucleus("LI",6,3,0,6.01512,0,0,7.5,0,0,0,0);

   // Adding 4-BE-6-0
   nuc = new Nucleus("BE",6,4,0,6.01973,0,4.95911e-21,0,0,0,0,0);
   nuc->AddDecay(-2,-2,0,1.37166,100);

   // Adding 2-HE-7-0
   nuc = new Nucleus("HE",7,2,0,7.02803,0,2.85149e-21,0,0,0,0,0);
   nuc->AddDecay(-1,0,0,0.444921,100);

   // Adding 3-LI-7-0
   nuc = new Nucleus("LI",7,3,0,7.016,0,0,92.5,0,0,0,0);

   // Adding 4-BE-7-0
   nuc = new Nucleus("BE",7,4,0,7.01693,0,4.60426e+06,0,2.8e-11,5.3e-11,0,0);
   nuc->AddDecay(0,-1,0,0.8618,100);

   // Adding 5-B-7-0
   nuc = new Nucleus("B",7,5,0,7.02992,0,3.25884e-22,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,2.20442,100);

   // Adding 2-HE-8-0
   nuc = new Nucleus("HE",8,2,0,8.03392,0,0.119,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.6528,84);
   nuc->AddDecay(-1,1,0,8.61902,16);

   // Adding 3-LI-8-0
   nuc = new Nucleus("LI",8,3,0,8.02249,0,0.838,0,0,0,0,0);
   nuc->AddDecay(0,1,0,16.0036,0);
   nuc->AddDecay(-4,-1,0,16.0955,100);

   // Adding 4-BE-8-0
   nuc = new Nucleus("BE",8,4,0,8.00531,0,6.70938e-17,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,0.0918999,100);

   // Adding 5-B-8-0
   nuc = new Nucleus("B",8,5,0,8.02461,0,0.77,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,17.9793,100);

   // Adding 6-C-8-0
   nuc = new Nucleus("C",8,6,0,8.03767,0,1.98364e-21,0,0,0,0,0);
   nuc->AddDecay(-2,-2,0,2.14157,100);

   // Adding 2-HE-9-0
   nuc = new Nucleus("HE",9,2,0,9.04382,0,0,0,0,0,0,-9);
   nuc->AddDecay(-1,0,0,1.14919,100);

   // Adding 3-LI-9-0
   nuc = new Nucleus("LI",9,3,0,9.02679,0,0.1783,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,11.941,49.5);
   nuc->AddDecay(0,1,0,13.6063,50.5);
   nuc->AddDecay(-5,-1,0,12.0329,0);

   // Adding 4-BE-9-0
   nuc = new Nucleus("BE",9,4,0,9.01218,0,0,100,0,0,0,0);

   // Adding 5-B-9-0
   nuc = new Nucleus("B",9,5,0,9.01333,0,8.44885e-19,0,0,0,2,0);
   nuc->AddDecay(-8,-4,0,0.277009,0);
   nuc->AddDecay(-1,-1,0,0.18511,100);

   // Adding 6-C-9-0
   nuc = new Nucleus("C",9,6,0,9.03104,0,0.1265,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,16.4982,100);

   // Adding 3-LI-10-0
   nuc = new Nucleus("LI",10,3,0,10.0359,0,3.80198e-22,0,0,0,0,0);
   nuc->AddDecay(-1,0,0,0.41933,100);

   // Adding 4-BE-10-0
   nuc = new Nucleus("BE",10,4,0,10.0135,0,4.76194e+13,0,1.1e-09,3.3e-08,0,0);
   nuc->AddDecay(0,1,0,0.55591,100);

   // Adding 5-B-10-0
   nuc = new Nucleus("B",10,5,0,10.0129,0,0,19.9,0,0,0,0);

   // Adding 6-C-10-0
   nuc = new Nucleus("C",10,6,0,10.0169,0,19.255,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.6478,100);

   // Adding 7-N-10-0
   nuc = new Nucleus("N",10,7,0,10.0426,0,0,0,0,0,0,-2);

   // Adding 3-LI-11-0
   nuc = new Nucleus("LI",11,3,0,11.0438,0,0.0085,0,0,0,1,0);
   nuc->AddDecay(-5,-1,0,12.6978,0.027);
   nuc->AddDecay(-1,1,0,20.1101,60.783);
   nuc->AddDecay(0,1,0,20.614,39.19);

   // Adding 4-BE-11-0
   nuc = new Nucleus("BE",11,4,0,11.0217,0,13.81,0,0,0,1,0);
   nuc->AddDecay(-4,-1,0,2.84151,3.1);
   nuc->AddDecay(0,1,0,11.5061,96.9);

   // Adding 5-B-11-0
   nuc = new Nucleus("B",11,5,0,11.0093,0,0,80.1,0,0,0,0);

   // Adding 6-C-11-0
   nuc = new Nucleus("C",11,6,0,11.0114,0,1223.4,0,2.4e-11,3.2e-12,0,0);
   nuc->AddDecay(0,-1,0,1.98225,100);

   // Adding 7-N-11-0
   nuc = new Nucleus("N",11,7,0,11.0268,0,6.16538e-22,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,1.97289,100);

   // Adding 4-BE-12-0
   nuc = new Nucleus("BE",12,4,0,12.0269,0,0.0236,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,8.33718,1);
   nuc->AddDecay(0,1,0,11.7076,99);

   // Adding 5-B-12-0
   nuc = new Nucleus("B",12,5,0,12.0144,0,0.0202,0,0,0,2,0);
   nuc->AddDecay(-8,-3,0,6.09421,1.58);
   nuc->AddDecay(0,1,0,13.37,98.42);

   // Adding 6-C-12-0
   nuc = new Nucleus("C",12,6,0,12,0,0,98.9,0,0,0,0);

   // Adding 7-N-12-0
   nuc = new Nucleus("N",12,7,0,12.0186,0,0.011,0,0,0,2,0);
   nuc->AddDecay(-8,-5,0,10.0634,3.44);
   nuc->AddDecay(0,-1,0,17.338,96.56);

   // Adding 8-O-12-0
   nuc = new Nucleus("O",12,8,0,12.0344,0,1.14059e-21,0,0,0,0,0);
   nuc->AddDecay(-2,-2,0,1.78671,100);

   // Adding 4-BE-13-0
   nuc = new Nucleus("BE",13,4,0,13.0377,0,1e-08,0,0,0,0,-8);
   nuc->AddDecay(-1,0,0,2.01064,100);

   // Adding 5-B-13-0
   nuc = new Nucleus("B",13,5,0,13.0178,0,0.01736,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.4373,100);

   // Adding 6-C-13-0
   nuc = new Nucleus("C",13,6,0,13.0034,0,0,1.1,0,0,0,0);

   // Adding 7-N-13-0
   nuc = new Nucleus("N",13,7,0,13.0057,0,597.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.22053,100);

   // Adding 8-O-13-0
   nuc = new Nucleus("O",13,8,0,13.0248,0,0.00858,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,17.7653,100);

   // Adding 4-BE-14-0
   nuc = new Nucleus("BE",14,4,0,14.0428,0,0.00435,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,15.2488,81);
   nuc->AddDecay(-2,1,0,10.3709,5);
   nuc->AddDecay(0,1,0,16.2187,14);

   // Adding 5-B-14-0
   nuc = new Nucleus("B",14,5,0,14.0254,0,0.0138,0,0,0,0,0);
   nuc->AddDecay(0,1,0,20.6438,100);

   // Adding 6-C-14-0
   nuc = new Nucleus("C",14,6,0,14.0032,0,1.80701e+11,0,5.8e-10,5.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.1565,100);

   // Adding 7-N-14-0
   nuc = new Nucleus("N",14,7,0,14.0031,0,0,99.63,0,0,0,0);

   // Adding 8-O-14-0
   nuc = new Nucleus("O",14,8,0,14.0086,0,70.606,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.14311,100);

   // Adding 9-F-14-0
   nuc = new Nucleus("F",14,9,0,14.0361,0,0,0,0,0,0,-9);
   nuc->AddDecay(-1,-1,0,3.20861,0);

   // Adding 5-B-15-0
   nuc = new Nucleus("B",15,5,0,15.0311,0,0.0105,0,0,0,0,0);
   nuc->AddDecay(0,1,0,19.0937,100);

   // Adding 6-C-15-0
   nuc = new Nucleus("C",15,6,0,15.0106,0,2.449,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.77168,100);

   // Adding 7-N-15-0
   nuc = new Nucleus("N",15,7,0,15.0001,0,0,0.366,0,0,0,0);

   // Adding 8-O-15-0
   nuc = new Nucleus("O",15,8,0,15.0031,0,122.24,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.75405,100);

   // Adding 9-F-15-0
   nuc = new Nucleus("F",15,9,0,15.018,0,4.56238e-22,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,1.4815,100);

   // Adding 5-B-16-0
   nuc = new Nucleus("B",16,5,0,16.0399,0,1e-08,0,0,0,0,-8);
   nuc->AddDecay(-1,0,0,0.101178,100);

   // Adding 6-C-16-0
   nuc = new Nucleus("C",16,6,0,16.0147,0,0.747,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.01211,100);

   // Adding 7-N-16-0
   nuc = new Nucleus("N",16,7,0,16.0061,0,7.13,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.419,100);

   // Adding 8-O-16-0
   nuc = new Nucleus("O",16,8,0,15.9949,0,0,99.762,0,0,0,0);

   // Adding 9-F-16-0
   nuc = new Nucleus("F",16,9,0,16.0115,0,1.14059e-20,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,0.535833,100);

   // Adding 10-NE-16-0
   nuc = new Nucleus("NE",16,10,0,16.0258,0,3.73965e-21,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,-0.0772939,100);

   // Adding 5-B-17-0
   nuc = new Nucleus("B",17,5,0,17.0469,0,0.00508,0,0,0,0,0);
   nuc->AddDecay(0,1,0,22.6797,100);

   // Adding 6-C-17-0
   nuc = new Nucleus("C",17,6,0,17.0226,0,0.193,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,7.28329,32);
   nuc->AddDecay(0,1,0,13.1658,68);

   // Adding 7-N-17-0
   nuc = new Nucleus("N",17,7,0,17.0084,0,4.173,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.67983,100);

   // Adding 8-O-17-0
   nuc = new Nucleus("O",17,8,0,16.9991,0,0,0.038,0,0,0,0);

   // Adding 9-F-17-0
   nuc = new Nucleus("F",17,9,0,17.0021,0,64.49,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.76074,100);

   // Adding 10-NE-17-0
   nuc = new Nucleus("NE",17,10,0,17.0177,0,0.1092,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,14.5335,100);

   // Adding 5-B-18-0
   nuc = new Nucleus("B",18,5,0,18.0562,0,2.6e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,0,0,0.535188,100);

   // Adding 6-C-18-0
   nuc = new Nucleus("C",18,6,0,18.0268,0,0.066,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.8072,100);

   // Adding 7-N-18-0
   nuc = new Nucleus("N",18,7,0,18.0141,0,0.624,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.8993,100);

   // Adding 8-O-18-0
   nuc = new Nucleus("O",18,8,0,17.9992,0,0,0.2,0,0,0,0);

   // Adding 9-F-18-0
   nuc = new Nucleus("F",18,9,0,18.0009,0,6586.2,0,4.9e-11,9.3e-11,0,0);
   nuc->AddDecay(0,-1,0,1.65564,100);

   // Adding 10-NE-18-0
   nuc = new Nucleus("NE",18,10,0,18.0057,0,1.672,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.44569,100);

   // Adding 11-NA-18-0
   nuc = new Nucleus("NA",18,11,0,18.0272,0,0,0,0,0,0,-9);
   nuc->AddDecay(-1,-1,0,1.54415,0);

   // Adding 6-C-19-0
   nuc = new Nucleus("C",19,6,0,19.0352,0,0.049,0,0,0,0,0);
   nuc->AddDecay(0,1,0,16.973,100);

   // Adding 7-N-19-0
   nuc = new Nucleus("N",19,7,0,19.017,0,0.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.5282,67);
   nuc->AddDecay(-1,1,0,8.5713,33);

   // Adding 8-O-19-0
   nuc = new Nucleus("O",19,8,0,19.0036,0,26.91,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.81964,100);

   // Adding 9-F-19-0
   nuc = new Nucleus("F",19,9,0,18.9984,0,0,100,0,0,0,0);

   // Adding 10-NE-19-0
   nuc = new Nucleus("NE",19,10,0,19.0019,0,17.34,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.23847,100);

   // Adding 11-NA-19-0
   nuc = new Nucleus("NA",19,11,0,19.0139,0,0,0,0,0,0,-9);
   nuc->AddDecay(0,-1,0,11.1776,100);
   nuc->AddDecay(-1,-2,0,4.76615,0);

   // Adding 6-C-20-0
   nuc = new Nucleus("C",20,6,0,20.0403,0,0.014,0,0,0,0,0);
   nuc->AddDecay(0,1,0,15.7936,100);

   // Adding 7-N-20-0
   nuc = new Nucleus("N",20,7,0,20.0234,0,0.1,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,10.363,61);
   nuc->AddDecay(0,1,0,17.9696,39);

   // Adding 8-O-20-0
   nuc = new Nucleus("O",20,8,0,20.0041,0,13.51,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.81432,100);

   // Adding 9-F-20-0
   nuc = new Nucleus("F",20,9,0,20,0,11,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.02449,100);

   // Adding 10-NE-20-0
   nuc = new Nucleus("NE",20,10,0,19.9924,0,0,90.51,0,0,0,0);

   // Adding 11-NA-20-0
   nuc = new Nucleus("NA",20,11,0,20.0073,0,0.4479,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,13.8869,100);

   // Adding 12-MG-20-0
   nuc = new Nucleus("MG",20,12,0,20.0189,0,0.095,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.7256,97);
   nuc->AddDecay(-1,-2,0,8.53047,3);

   // Adding 7-N-21-0
   nuc = new Nucleus("N",21,7,0,21.0271,0,0.095,0,0,0,0,0);
   nuc->AddDecay(0,1,0,17.1703,16);
   nuc->AddDecay(-1,1,0,13.3638,84);

   // Adding 8-O-21-0
   nuc = new Nucleus("O",21,8,0,21.0087,0,3.42,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.10933,100);

   // Adding 9-F-21-0
   nuc = new Nucleus("F",21,9,0,20.9999,0,4.158,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.68408,100);

   // Adding 10-NE-21-0
   nuc = new Nucleus("NE",21,10,0,20.9938,0,0,0.27,0,0,0,0);

   // Adding 11-NA-21-0
   nuc = new Nucleus("NA",21,11,0,20.9977,0,22.49,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.54733,100);

   // Adding 12-MG-21-0
   nuc = new Nucleus("MG",21,12,0,21.0117,0,0.122,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,10.6646,29.3);
   nuc->AddDecay(0,-1,0,13.0961,70.7);

   // Adding 7-N-22-0
   nuc = new Nucleus("N",22,7,0,22.0344,0,0.024,0,0,0,0,0);
   nuc->AddDecay(0,1,0,22.7966,65);
   nuc->AddDecay(-1,1,0,15.9479,35);

   // Adding 8-O-22-0
   nuc = new Nucleus("O",22,8,0,22.01,0,2.25,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.49054,100);

   // Adding 9-F-22-0
   nuc = new Nucleus("F",22,9,0,22.003,0,4.23,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.8182,100);

   // Adding 10-NE-22-0
   nuc = new Nucleus("NE",22,10,0,21.9914,0,0,9.25,0,0,0,0);

   // Adding 11-NA-22-0
   nuc = new Nucleus("NA",22,11,0,21.9944,0,8.20535e+07,0,3.2e-09,2e-09,0,0);
   nuc->AddDecay(0,-1,0,2.84218,100);

   // Adding 12-MG-22-0
   nuc = new Nucleus("MG",22,12,0,21.9996,0,3.857,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.78543,100);

   // Adding 13-AL-22-0
   nuc = new Nucleus("AL",22,13,0,22.0195,0,0.07,0,0,0,0,-8);
   nuc->AddDecay(-1,-2,0,13.0793,33.3333);
   nuc->AddDecay(0,-1,0,18.5807,33.3333);
   nuc->AddDecay(-2,-3,0,10.6478,33.3333);

   // Adding 14-SI-22-0
   nuc = new Nucleus("SI",22,14,0,22.0345,0,0.006,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,13.9803,50);
   nuc->AddDecay(-1,-2,0,13.9635,50);

   // Adding 8-O-23-0
   nuc = new Nucleus("O",23,8,0,23.0157,0,0.082,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.7513,31);
   nuc->AddDecay(0,1,0,11.2868,69);

   // Adding 9-F-23-0
   nuc = new Nucleus("F",23,9,0,23.0036,0,2.23,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.48331,100);

   // Adding 10-NE-23-0
   nuc = new Nucleus("NE",23,10,0,22.9945,0,37.24,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.37579,100);

   // Adding 11-NA-23-0
   nuc = new Nucleus("NA",23,11,0,22.9898,0,0,100,0,0,0,0);

   // Adding 12-MG-23-0
   nuc = new Nucleus("MG",23,12,0,22.9941,0,11.317,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.05681,100);

   // Adding 13-AL-23-0
   nuc = new Nucleus("AL",23,13,0,23.0073,0,0.47,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,4.66046,0);
   nuc->AddDecay(0,-1,0,12.2399,100);

   // Adding 8-O-24-0
   nuc = new Nucleus("O",24,8,0,24.0204,0,0.061,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.43,42);
   nuc->AddDecay(-1,1,0,7.57365,58);

   // Adding 9-F-24-0
   nuc = new Nucleus("F",24,9,0,24.0081,0,0.34,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.4921,100);

   // Adding 10-NE-24-0
   nuc = new Nucleus("NE",24,10,0,23.9936,0,202.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.47011,100);

   // Adding 11-NA-24-0
   nuc = new Nucleus("NA",24,11,0,23.991,0,53852.4,0,4.3e-10,5.3e-10,0,0);
   nuc->AddDecay(0,1,0,5.5158,100);

   // Adding 11-NA-24-1
   nuc = new Nucleus("NA",24,11,1,23.9915,0.472,0.0202,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.9878,0.05);
   nuc->AddDecay(0,0,-1,0.472,100);

   // Adding 12-MG-24-0
   nuc = new Nucleus("MG",24,12,0,23.985,0,0,78.99,0,0,0,0);

   // Adding 13-AL-24-0
   nuc = new Nucleus("AL",24,13,0,23.9999,0,2.053,0,0,0,1,0);
   nuc->AddDecay(-4,-3,0,4.56186,0.04);
   nuc->AddDecay(0,-1,0,13.8783,99.96);

   // Adding 13-AL-24-1
   nuc = new Nucleus("AL",24,13,1,24.0004,0.426,0.1313,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.426,82);
   nuc->AddDecay(-4,-3,-1,4.98786,0.03);
   nuc->AddDecay(0,-1,-1,14.3043,17.97);

   // Adding 14-SI-24-0
   nuc = new Nucleus("SI",24,14,0,24.0115,0,0.102,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.8099,93);
   nuc->AddDecay(-1,-2,0,8.93843,7);

   // Adding 9-F-25-0
   nuc = new Nucleus("F",25,9,0,25.0121,0,0.059,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.3252,85);
   nuc->AddDecay(-1,1,0,9.14269,15);

   // Adding 10-NE-25-0
   nuc = new Nucleus("NE",25,10,0,24.9978,0,0.602,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.29877,100);

   // Adding 11-NA-25-0
   nuc = new Nucleus("NA",25,11,0,24.99,0,59.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.83521,100);

   // Adding 12-MG-25-0
   nuc = new Nucleus("MG",25,12,0,24.9858,0,0,10,0,0,0,0);

   // Adding 13-AL-25-0
   nuc = new Nucleus("AL",25,13,0,24.9904,0,7.183,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.27685,100);

   // Adding 14-SI-25-0
   nuc = new Nucleus("SI",25,14,0,25.0041,0,0.22,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.7412,100);
   nuc->AddDecay(-1,-2,0,10.4697,0);

   // Adding 15-P-25-0
   nuc = new Nucleus("P",25,15,0,25.0203,0,3e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,0.828476,100);

   // Adding 9-F-26-0
   nuc = new Nucleus("F",26,9,0,26.0196,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,17.8584,0);

   // Adding 10-NE-26-0
   nuc = new Nucleus("NE",26,10,0,26.0005,0,0.23,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.33237,100);

   // Adding 11-NA-26-0
   nuc = new Nucleus("NA",26,11,0,25.9926,0,1.072,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.31201,100);

   // Adding 12-MG-26-0
   nuc = new Nucleus("MG",26,12,0,25.9826,0,0,11.01,0,0,0,0);

   // Adding 13-AL-26-0
   nuc = new Nucleus("AL",26,13,0,25.9869,0,2.33366e+13,0,3.5e-09,1.8e-08,0,0);
   nuc->AddDecay(0,-1,0,4.00419,100);

   // Adding 13-AL-26-1
   nuc = new Nucleus("AL",26,13,1,25.9871,0.228,6.3452,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.23219,100);

   // Adding 14-SI-26-0
   nuc = new Nucleus("SI",26,14,0,25.9923,0,2.234,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.0656,100);

   // Adding 15-P-26-0
   nuc = new Nucleus("P",26,15,0,26.0118,0,0.02,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,12.6001,2);
   nuc->AddDecay(0,-1,0,18.1179,98);

   // Adding 9-F-27-0
   nuc = new Nucleus("F",27,9,0,27.0269,0,0,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,17.9564,0);

   // Adding 10-NE-27-0
   nuc = new Nucleus("NE",27,10,0,27.0076,0,0.032,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,12.6745,100);

   // Adding 11-NA-27-0
   nuc = new Nucleus("NA",27,11,0,26.994,0,0.301,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.56228,0.08);
   nuc->AddDecay(0,1,0,9.00558,99.92);

   // Adding 12-MG-27-0
   nuc = new Nucleus("MG",27,12,0,26.9843,0,567.48,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.6104,100);

   // Adding 13-AL-27-0
   nuc = new Nucleus("AL",27,13,0,26.9815,0,0,100,0,0,0,0);

   // Adding 14-SI-27-0
   nuc = new Nucleus("SI",27,14,0,26.9867,0,4.16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.8119,100);

   // Adding 15-P-27-0
   nuc = new Nucleus("P",27,15,0,26.9992,0,0.26,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.6321,94);
   nuc->AddDecay(-1,-2,0,4.1684,6);

   // Adding 16-S-27-0
   nuc = new Nucleus("S",27,16,0,27.0188,0,0.021,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,18.2601,100);

   // Adding 10-NE-28-0
   nuc = new Nucleus("NE",28,10,0,28.0121,0,0.014,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.3123,100);

   // Adding 11-NA-28-0
   nuc = new Nucleus("NA",28,11,0,27.9989,0,0.0305,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.9851,99.42);
   nuc->AddDecay(-1,1,0,5.48154,0.58);

   // Adding 12-MG-28-0
   nuc = new Nucleus("MG",28,12,0,27.9839,0,75276,0,2.2e-09,1.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.8318,100);

   // Adding 13-AL-28-0
   nuc = new Nucleus("AL",28,13,0,27.9819,0,134.484,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.6422,100);

   // Adding 14-SI-28-0
   nuc = new Nucleus("SI",28,14,0,27.9769,0,0,92.23,0,0,0,0);

   // Adding 15-P-28-0
   nuc = new Nucleus("P",28,15,0,27.9923,0,0.2703,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,14.3317,100);

   // Adding 16-S-28-0
   nuc = new Nucleus("S",28,16,0,28.0044,0,0.125,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,9.16914,0);
   nuc->AddDecay(0,-1,0,11.2343,100);

   // Adding 11-NA-29-0
   nuc = new Nucleus("NA",29,11,0,29.0028,0,0.0449,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.2801,78.5);
   nuc->AddDecay(-1,1,0,9.56641,21.5);

   // Adding 12-MG-29-0
   nuc = new Nucleus("MG",29,12,0,28.9886,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.55428,100);

   // Adding 13-AL-29-0
   nuc = new Nucleus("AL",29,13,0,28.9804,0,393.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.67961,100);

   // Adding 14-SI-29-0
   nuc = new Nucleus("SI",29,14,0,28.9765,0,0,4.67,0,0,0,0);

   // Adding 15-P-29-0
   nuc = new Nucleus("P",29,15,0,28.9818,0,4.142,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.94314,100);

   // Adding 16-S-29-0
   nuc = new Nucleus("S",29,16,0,28.9966,0,0.187,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,13.7931,100);

   // Adding 17-CL-29-0
   nuc = new Nucleus("CL",29,17,0,29.0141,0,2e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,1.78108,100);

   // Adding 11-NA-30-0
   nuc = new Nucleus("NA",30,11,0,30.0092,0,0.048,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,11.1844,30);
   nuc->AddDecay(-4,-1,0,5.73974,5.5e-05);
   nuc->AddDecay(0,1,0,17.4768,68.8299);
   nuc->AddDecay(-2,1,0,7.47071,1.17);

   // Adding 12-MG-30-0
   nuc = new Nucleus("MG",30,12,0,29.9905,0,0.335,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.99015,100);

   // Adding 13-AL-30-0
   nuc = new Nucleus("AL",30,13,0,29.983,0,3.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.56051,100);

   // Adding 14-SI-30-0
   nuc = new Nucleus("SI",30,14,0,29.9738,0,0,3.1,0,0,0,0);

   // Adding 15-P-30-0
   nuc = new Nucleus("P",30,15,0,29.9783,0,149.88,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.23228,100);

   // Adding 16-S-30-0
   nuc = new Nucleus("S",30,16,0,29.9849,0,1.178,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.13781,100);

   // Adding 17-CL-30-0
   nuc = new Nucleus("CL",30,17,0,30.0048,0,3e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,0.313046,100);

   // Adding 11-NA-31-0
   nuc = new Nucleus("NA",31,11,0,31.0136,0,0.017,0,0,0,0,0);
   nuc->AddDecay(-2,1,0,7.18245,0.9);
   nuc->AddDecay(0,1,0,15.879,62.1);
   nuc->AddDecay(-1,1,0,13.4748,37);

   // Adding 12-MG-31-0
   nuc = new Nucleus("MG",31,12,0,30.9965,0,0.23,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.58594,1.7);
   nuc->AddDecay(0,1,0,11.739,98.3);

   // Adding 13-AL-31-0
   nuc = new Nucleus("AL",31,13,0,30.9839,0,0.644,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.9948,100);

   // Adding 14-SI-31-0
   nuc = new Nucleus("SI",31,14,0,30.9754,0,9438,0,1.6e-10,1.1e-10,0,0);
   nuc->AddDecay(0,1,0,1.49201,100);

   // Adding 15-P-31-0
   nuc = new Nucleus("P",31,15,0,30.9738,0,0,100,0,0,0,0);

   // Adding 16-S-31-0
   nuc = new Nucleus("S",31,16,0,30.9796,0,2.572,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.39612,100);

   // Adding 17-CL-31-0
   nuc = new Nucleus("CL",31,17,0,30.9924,0,0.15,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.9804,99.56);
   nuc->AddDecay(-1,-2,0,5.84709,0.44);

   // Adding 11-NA-32-0
   nuc = new Nucleus("NA",32,11,0,32.0196,0,0.0132,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,13.4476,24);
   nuc->AddDecay(0,1,0,19.0994,68);
   nuc->AddDecay(-2,1,0,11.0434,8);

   // Adding 12-MG-32-0
   nuc = new Nucleus("MG",32,12,0,31.9991,0,0.12,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,6.08723,2.4);
   nuc->AddDecay(0,1,0,10.2665,97.6);

   // Adding 13-AL-32-0
   nuc = new Nucleus("AL",32,13,0,31.9881,0,0.033,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.0187,100);

   // Adding 14-SI-32-0
   nuc = new Nucleus("SI",32,14,0,31.9741,0,5.42419e+09,0,5.6e-10,1.1e-07,0,0);
   nuc->AddDecay(0,1,0,0.224409,100);

   // Adding 15-P-32-0
   nuc = new Nucleus("P",32,15,0,31.9739,0,1.23224e+06,0,2.4e-09,3.2e-09,0,0);
   nuc->AddDecay(0,1,0,1.71058,100);

   // Adding 16-S-32-0
   nuc = new Nucleus("S",32,16,0,31.9721,0,0,95.02,0,0,0,0);

   // Adding 17-CL-32-0
   nuc = new Nucleus("CL",32,17,0,31.9857,0,0.298,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.6851,99.983);
   nuc->AddDecay(-4,-3,0,5.73715,0.01);
   nuc->AddDecay(-1,-2,0,3.82126,0.007);

   // Adding 18-AR-32-0
   nuc = new Nucleus("AR",32,18,0,31.9977,0,0.098,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,9.57641,43);
   nuc->AddDecay(0,-1,0,11.1513,57);

   // Adding 11-NA-33-0
   nuc = new Nucleus("NA",33,11,0,33.0274,0,0.0082,0,0,0,0,0);
   nuc->AddDecay(-2,1,0,12.5825,12);
   nuc->AddDecay(-1,1,0,18.2343,52);
   nuc->AddDecay(0,1,0,20.3057,36);

   // Adding 12-MG-33-0
   nuc = new Nucleus("MG",33,12,0,33.0056,0,0.09,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.7092,83);
   nuc->AddDecay(-1,1,0,8.1951,17);

   // Adding 13-AL-33-0
   nuc = new Nucleus("AL",33,13,0,32.9909,0,1e-06,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,11.9875,100);

   // Adding 14-SI-33-0
   nuc = new Nucleus("SI",33,14,0,32.978,0,6.18,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.84529,100);

   // Adding 15-P-33-0
   nuc = new Nucleus("P",33,15,0,32.9717,0,2.18938e+06,0,2.4e-10,1.4e-09,0,0);
   nuc->AddDecay(0,1,0,0.248499,100);

   // Adding 16-S-33-0
   nuc = new Nucleus("S",33,16,0,32.9715,0,0,0.75,0,0,0,0);

   // Adding 17-CL-33-0
   nuc = new Nucleus("CL",33,17,0,32.9775,0,2.511,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.58266,100);

   // Adding 18-AR-33-0
   nuc = new Nucleus("AR",33,18,0,32.9899,0,0.173,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.6221,61.3);
   nuc->AddDecay(-1,-2,0,9.34548,38.7);

   // Adding 19-K-33-0
   nuc = new Nucleus("K",33,19,0,33.0073,0,2.5e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,1.6537,100);

   // Adding 11-NA-34-0
   nuc = new Nucleus("NA",34,11,0,34.0349,0,0.0055,0,0,0,0,0);
   nuc->AddDecay(0,1,0,24.0582,42.5);
   nuc->AddDecay(-2,1,0,17.1622,57.5);

   // Adding 12-MG-34-0
   nuc = new Nucleus("MG",34,12,0,34.0091,0,0.02,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.3132,100);
   nuc->AddDecay(-1,1,0,8.88459,0);

   // Adding 13-AL-34-0
   nuc = new Nucleus("AL",34,13,0,33.9969,0,0.06,0,0,0,0,0);
   nuc->AddDecay(0,1,0,17.0943,73);
   nuc->AddDecay(-1,1,0,9.55883,27);

   // Adding 14-SI-34-0
   nuc = new Nucleus("SI",34,14,0,33.9786,0,2.77,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.60104,100);

   // Adding 15-P-34-0
   nuc = new Nucleus("P",34,15,0,33.9736,0,12.43,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.37416,100);

   // Adding 16-S-34-0
   nuc = new Nucleus("S",34,16,0,33.9679,0,0,4.21,0,0,0,0);

   // Adding 17-CL-34-0
   nuc = new Nucleus("CL",34,17,0,33.9738,0,1.5264,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.49219,100);

   // Adding 17-CL-34-1
   nuc = new Nucleus("CL",34,17,1,33.9739,0.146,1920,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.63819,55.4);
   nuc->AddDecay(0,0,-1,0.146,44.6);

   // Adding 18-AR-34-0
   nuc = new Nucleus("AR",34,18,0,33.9803,0,0.8445,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.06129,100);

   // Adding 19-K-34-0
   nuc = new Nucleus("K",34,19,0,33.9984,0,4e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,0.611132,100);

   // Adding 11-NA-35-0
   nuc = new Nucleus("NA",35,11,0,35.0442,0,0.0015,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,24.6309,0);
   nuc->AddDecay(0,1,0,23.7632,100);

   // Adding 12-MG-35-0
   nuc = new Nucleus("MG",35,12,0,35.0187,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,17.4481,100);

   // Adding 13-AL-35-0
   nuc = new Nucleus("AL",35,13,0,34.9999,0,0.15,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,11.8272,65);
   nuc->AddDecay(0,1,0,14.3017,35);

   // Adding 14-SI-35-0
   nuc = new Nucleus("SI",35,14,0,34.9846,0,0.78,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.4978,100);

   // Adding 15-P-35-0
   nuc = new Nucleus("P",35,15,0,34.9733,0,47.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.98869,100);

   // Adding 16-S-35-0
   nuc = new Nucleus("S",35,16,0,34.969,0,7.56086e+06,0,7.7e-10,1.3e-09,0,0);
   nuc->AddDecay(0,1,0,0.167191,100);

   // Adding 17-CL-35-0
   nuc = new Nucleus("CL",35,17,0,34.9689,0,0,75.77,0,0,0,0);

   // Adding 18-AR-35-0
   nuc = new Nucleus("AR",35,18,0,34.9753,0,1.775,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.96522,100);

   // Adding 19-K-35-0
   nuc = new Nucleus("K",35,19,0,34.988,0,0.19,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.8812,99.63);
   nuc->AddDecay(-1,-2,0,5.9835,0.37);

   // Adding 20-CA-35-0
   nuc = new Nucleus("CA",35,20,0,35.0048,0,0.05,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,15.6068,100);
   nuc->AddDecay(-2,-3,0,10.8652,0);

   // Adding 13-AL-36-0
   nuc = new Nucleus("AL",36,13,0,36.0064,0,1e-06,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,18.317,100);

   // Adding 14-SI-36-0
   nuc = new Nucleus("SI",36,14,0,35.9867,0,0.45,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.38571,10);
   nuc->AddDecay(0,1,0,7.8502,90);

   // Adding 15-P-36-0
   nuc = new Nucleus("P",36,15,0,35.9783,0,5.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.4132,100);

   // Adding 16-S-36-0
   nuc = new Nucleus("S",36,16,0,35.9671,0,0,0.02,0,0,0,0);

   // Adding 17-CL-36-0
   nuc = new Nucleus("CL",36,17,0,35.9683,0,9.49234e+12,0,9.3e-10,6.9e-09,0,0);
   nuc->AddDecay(0,1,0,0.70862,98.1);
   nuc->AddDecay(0,-1,0,1.14211,1.9);

   // Adding 18-AR-36-0
   nuc = new Nucleus("AR",36,18,0,35.9675,0,0,0.337,0,0,0,0);

   // Adding 19-K-36-0
   nuc = new Nucleus("K",36,19,0,35.9813,0,0.342,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.8054,99.9466);
   nuc->AddDecay(-4,-3,0,6.16583,0.0034);
   nuc->AddDecay(-1,-2,0,4.29932,0.05);

   // Adding 20-CA-36-0
   nuc = new Nucleus("CA",36,20,0,35.9931,0,0.1,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,9.32004,20);
   nuc->AddDecay(0,-1,0,10.9859,80);

   // Adding 13-AL-37-0
   nuc = new Nucleus("AL",37,13,0,37.0103,0,0,0,0,0,0,-2);

   // Adding 14-SI-37-0
   nuc = new Nucleus("SI",37,14,0,36.993,0,1e-06,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,12.4705,100);

   // Adding 15-P-37-0
   nuc = new Nucleus("P",37,15,0,36.9796,0,2.31,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.90149,100);

   // Adding 16-S-37-0
   nuc = new Nucleus("S",37,16,0,36.9711,0,303,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.86527,100);

   // Adding 17-CL-37-0
   nuc = new Nucleus("CL",37,17,0,36.9659,0,0,24.23,0,0,0,0);

   // Adding 18-AR-37-0
   nuc = new Nucleus("AR",37,18,0,36.9668,0,3.02746e+06,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.813469,100);

   // Adding 19-K-37-0
   nuc = new Nucleus("K",37,19,0,36.9734,0,1.226,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.1487,100);

   // Adding 20-CA-37-0
   nuc = new Nucleus("CA",37,20,0,36.9859,0,0.175,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,9.78091,76);
   nuc->AddDecay(0,-1,0,11.6387,24);

   // Adding 14-SI-38-0
   nuc = new Nucleus("SI",38,14,0,37.996,0,1e-06,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,10.7215,100);

   // Adding 15-P-38-0
   nuc = new Nucleus("P",38,15,0,37.9845,0,0.64,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.3951,90);
   nuc->AddDecay(-1,1,0,4.35882,10);

   // Adding 16-S-38-0
   nuc = new Nucleus("S",38,16,0,37.9712,0,10218,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.93684,100);

   // Adding 17-CL-38-0
   nuc = new Nucleus("CL",38,17,0,37.968,0,2234.4,0,1.2e-10,7.3e-11,0,0);
   nuc->AddDecay(0,1,0,4.91684,100);

   // Adding 17-CL-38-1
   nuc = new Nucleus("CL",38,17,1,37.9687,0.671,0.715,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.671,100);

   // Adding 18-AR-38-0
   nuc = new Nucleus("AR",38,18,0,37.9627,0,0,0.063,0,0,0,0);

   // Adding 19-K-38-0
   nuc = new Nucleus("K",38,19,0,37.9691,0,458.16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.91308,100);

   // Adding 19-K-38-1
   nuc = new Nucleus("K",38,19,1,37.9692,0.13,0.9239,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.04308,100);

   // Adding 20-CA-38-0
   nuc = new Nucleus("CA",38,20,0,37.9763,0,0.44,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.74272,100);

   // Adding 21-SC-38-0
   nuc = new Nucleus("SC",38,21,0,37.9947,0,0,0,0,0,0,-2);

   // Adding 22-TI-38-0
   nuc = new Nucleus("TI",38,22,0,38.0098,0,0,0,0,0,0,-6);

   // Adding 15-P-39-0
   nuc = new Nucleus("P",39,15,0,38.9864,0,0.16,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,6.14016,41);
   nuc->AddDecay(0,1,0,10.5116,59);

   // Adding 16-S-39-0
   nuc = new Nucleus("S",39,16,0,38.9751,0,11.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.63847,100);

   // Adding 17-CL-39-0
   nuc = new Nucleus("CL",39,17,0,38.968,0,3336,0,8.5e-11,7.6e-11,0,0);
   nuc->AddDecay(0,1,0,3.44203,100);

   // Adding 18-AR-39-0
   nuc = new Nucleus("AR",39,18,0,38.9643,0,8.48318e+09,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.56498,100);

   // Adding 19-K-39-0
   nuc = new Nucleus("K",39,19,0,38.9637,0,0,93.2581,0,0,0,0);

   // Adding 20-CA-39-0
   nuc = new Nucleus("CA",39,20,0,38.9707,0,0.8596,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.53051,100);

   // Adding 21-SC-39-0
   nuc = new Nucleus("SC",39,21,0,38.9848,0,0.001,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,0.601806,100);

   // Adding 15-P-40-0
   nuc = new Nucleus("P",40,15,0,39.9911,0,0.26,0,0,0,0,0);
   nuc->AddDecay(0,1,0,14.5126,70);
   nuc->AddDecay(-1,1,0,6.75313,30);

   // Adding 16-S-40-0
   nuc = new Nucleus("S",40,16,0,39.9755,0,8.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.70821,100);

   // Adding 17-CL-40-0
   nuc = new Nucleus("CL",40,17,0,39.9704,0,81,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.48217,100);

   // Adding 18-AR-40-0
   nuc = new Nucleus("AR",40,18,0,39.9624,0,0,99.6,0,0,0,0);

   // Adding 19-K-40-0
   nuc = new Nucleus("K",40,19,0,39.964,0,4.02715e+16,0.0117,6.2e-09,3e-09,0,0);
   nuc->AddDecay(0,1,0,1.3111,89.28);
   nuc->AddDecay(0,-1,0,1.50487,10.72);

   // Adding 20-CA-40-0
   nuc = new Nucleus("CA",40,20,0,39.9626,0,0,96.941,0,0,0,0);

   // Adding 21-SC-40-0
   nuc = new Nucleus("SC",40,21,0,39.978,0,0.1823,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,5.99139,0.44);
   nuc->AddDecay(-4,-3,0,7.27919,0.02);
   nuc->AddDecay(0,-1,0,14.3197,99.54);

   // Adding 22-TI-40-0
   nuc = new Nucleus("TI",40,22,0,39.9905,0,0.05,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.6762,100);
   nuc->AddDecay(-1,-2,0,11.1371,0);

   // Adding 15-P-41-0
   nuc = new Nucleus("P",41,15,0,40.9948,0,0.12,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.7581,70);
   nuc->AddDecay(-1,1,0,9.93438,30);

   // Adding 16-S-41-0
   nuc = new Nucleus("S",41,16,0,40.98,0,1e-06,0,0,0,0,-3);
   nuc->AddDecay(0,1,0,8.73724,100);

   // Adding 17-CL-41-0
   nuc = new Nucleus("CL",41,17,0,40.9706,0,38.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.72821,100);

   // Adding 18-AR-41-0
   nuc = new Nucleus("AR",41,18,0,40.9645,0,6560.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.49156,100);

   // Adding 19-K-41-0
   nuc = new Nucleus("K",41,19,0,40.9618,0,0,6.7302,0,0,0,0);

   // Adding 20-CA-41-0
   nuc = new Nucleus("CA",41,20,0,40.9623,0,3.24821e+12,0,2.9e-10,1.9e-10,0,0);
   nuc->AddDecay(0,-1,0,0.421391,100);

   // Adding 21-SC-41-0
   nuc = new Nucleus("SC",41,21,0,40.9692,0,0.5963,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.49531,100);

   // Adding 22-TI-41-0
   nuc = new Nucleus("TI",41,22,0,40.9831,0,0.08,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,11.8438,100);
   nuc->AddDecay(0,-1,0,12.9289,0);

   // Adding 15-P-42-0
   nuc = new Nucleus("P",42,15,0,42.0001,0,0.11,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,10.6151,50);
   nuc->AddDecay(0,1,0,17.3268,50);

   // Adding 16-S-42-0
   nuc = new Nucleus("S",42,16,0,41.9815,0,0.56,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,7.74498,100);

   // Adding 17-CL-42-0
   nuc = new Nucleus("CL",42,17,0,41.9732,0,6.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.43483,100);

   // Adding 18-AR-42-0
   nuc = new Nucleus("AR",42,18,0,41.963,0,1.03753e+09,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.599194,100);

   // Adding 19-K-42-0
   nuc = new Nucleus("K",42,19,0,41.9624,0,44496,0,4.3e-10,2e-10,0,0);
   nuc->AddDecay(0,1,0,3.52551,100);

   // Adding 20-CA-42-0
   nuc = new Nucleus("CA",42,20,0,41.9586,0,0,0.647,0,0,0,0);

   // Adding 21-SC-42-0
   nuc = new Nucleus("SC",42,21,0,41.9655,0,0.6813,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.4259,100);

   // Adding 21-SC-42-1
   nuc = new Nucleus("SC",42,21,1,41.9662,0.616,61.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.0419,100);

   // Adding 22-TI-42-0
   nuc = new Nucleus("TI",42,22,0,41.973,0,0.199,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.99999,100);

   // Adding 23-V-42-0
   nuc = new Nucleus("V",42,23,0,41.9912,0,5.5e-08,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,0.255154,100);

   // Adding 15-P-43-0
   nuc = new Nucleus("P",43,15,0,43.0033,0,0.033,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,12.251,100);

   // Adding 16-S-43-0
   nuc = new Nucleus("S",43,16,0,42.9866,0,0.22,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.5473,60);
   nuc->AddDecay(-1,1,0,4.43393,40);

   // Adding 17-CL-43-0
   nuc = new Nucleus("CL",43,17,0,42.9742,0,3.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.94825,100);

   // Adding 18-AR-43-0
   nuc = new Nucleus("AR",43,18,0,42.9657,0,322.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.61623,100);

   // Adding 19-K-43-0
   nuc = new Nucleus("K",43,19,0,42.9607,0,80280,0,2.5e-10,2.6e-10,0,0);
   nuc->AddDecay(0,1,0,1.81455,100);

   // Adding 20-CA-43-0
   nuc = new Nucleus("CA",43,20,0,42.9588,0,0,0.135,0,0,0,0);

   // Adding 21-SC-43-0
   nuc = new Nucleus("SC",43,21,0,42.9612,0,14007.6,0,1.9e-10,1.8e-10,0,0);
   nuc->AddDecay(0,-1,0,2.22083,100);

   // Adding 22-TI-43-0
   nuc = new Nucleus("TI",43,22,0,42.9685,0,0.509,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.86725,100);

   // Adding 23-V-43-0
   nuc = new Nucleus("V",43,23,0,42.9807,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,0,11.2961,0);

   // Adding 16-S-44-0
   nuc = new Nucleus("S",44,16,0,43.9883,0,0.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.11053,70);
   nuc->AddDecay(-1,1,0,5.07752,30);

   // Adding 17-CL-44-0
   nuc = new Nucleus("CL",44,17,0,43.9785,0,0.434,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,12.271,100);

   // Adding 18-AR-44-0
   nuc = new Nucleus("AR",44,18,0,43.9654,0,712.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.54811,100);

   // Adding 19-K-44-0
   nuc = new Nucleus("K",44,19,0,43.9616,0,1327.8,0,8.4e-11,3.7e-11,0,0);
   nuc->AddDecay(0,1,0,5.65895,100);

   // Adding 20-CA-44-0
   nuc = new Nucleus("CA",44,20,0,43.9555,0,0,2.086,0,0,0,0);

   // Adding 21-SC-44-0
   nuc = new Nucleus("SC",44,21,0,43.9594,0,14137.2,0,3.5e-10,3e-10,0,0);
   nuc->AddDecay(0,-1,0,3.65337,100);

   // Adding 21-SC-44-1
   nuc = new Nucleus("SC",44,21,1,43.9597,0.271,210960,0,2.5e-09,2e-09,0,0);
   nuc->AddDecay(0,-1,-1,3.92437,1.2);
   nuc->AddDecay(0,0,-1,0.271,98.8);

   // Adding 22-TI-44-0
   nuc = new Nucleus("TI",44,22,0,43.9597,0,1.54526e+09,0,5.8e-09,1.2e-07,0,0);
   nuc->AddDecay(0,-1,0,0.267437,100);

   // Adding 23-V-44-0
   nuc = new Nucleus("V",44,23,0,43.9744,0,0.09,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,13.7015,100);
   nuc->AddDecay(-4,-3,0,8.57439,0);

   // Adding 24-CR-44-0
   nuc = new Nucleus("CR",44,24,0,43.9855,0,0.053,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.3117,100);

   // Adding 15-P-45-0
   nuc = new Nucleus("P",45,15,0,45.0147,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,18.4964,0);

   // Adding 16-S-45-0
   nuc = new Nucleus("S",45,16,0,44.9948,0,0.082,0,0,0,0,0);
   nuc->AddDecay(0,1,0,14.0794,46);
   nuc->AddDecay(-1,1,0,7.08982,54);

   // Adding 17-CL-45-0
   nuc = new Nucleus("CL",45,17,0,44.9797,0,0.4,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,5.28146,24);
   nuc->AddDecay(0,1,0,10.8101,76);

   // Adding 18-AR-45-0
   nuc = new Nucleus("AR",45,18,0,44.9681,0,21.48,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.88865,100);

   // Adding 19-K-45-0
   nuc = new Nucleus("K",45,19,0,44.9607,0,1038,0,5.4e-11,2.8e-11,0,0);
   nuc->AddDecay(0,1,0,4.20448,100);

   // Adding 20-CA-45-0
   nuc = new Nucleus("CA",45,20,0,44.9562,0,1.41523e+07,0,7.6e-10,2.7e-09,0,0);
   nuc->AddDecay(0,1,0,0.256821,100);

   // Adding 21-SC-45-0
   nuc = new Nucleus("SC",45,21,0,44.9559,0,0,100,0,0,0,0);

   // Adding 21-SC-45-1
   nuc = new Nucleus("SC",45,21,1,44.9559,0.012,0.318,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.012,100);

   // Adding 22-TI-45-0
   nuc = new Nucleus("TI",45,22,0,44.9581,0,11088,0,1.5e-10,1.5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.0625,100);

   // Adding 23-V-45-0
   nuc = new Nucleus("V",45,23,0,44.9658,0,0.547,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.1332,100);

   // Adding 24-CR-45-0
   nuc = new Nucleus("CR",45,24,0,44.9792,0,0.05,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.4616,73);
   nuc->AddDecay(-1,-2,0,10.8473,27);

   // Adding 17-CL-46-0
   nuc = new Nucleus("CL",46,17,0,45.9841,0,0.223,0,0,0,0,0);
   nuc->AddDecay(0,1,0,14.9308,40);
   nuc->AddDecay(-1,1,0,6.85816,60);

   // Adding 18-AR-46-0
   nuc = new Nucleus("AR",46,18,0,45.9681,0,8.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.69817,100);

   // Adding 19-K-46-0
   nuc = new Nucleus("K",46,19,0,45.962,0,105,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.71601,100);

   // Adding 20-CA-46-0
   nuc = new Nucleus("CA",46,20,0,45.9537,0,0,0.004,0,0,0,0);

   // Adding 21-SC-46-0
   nuc = new Nucleus("SC",46,21,0,45.9552,0,7.23946e+06,0,1.5e-09,6.4e-09,0,0);
   nuc->AddDecay(0,1,0,2.3667,100);

   // Adding 21-SC-46-1
   nuc = new Nucleus("SC",46,21,1,45.9553,0.143,18.75,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.143,100);

   // Adding 22-TI-46-0
   nuc = new Nucleus("TI",46,22,0,45.9526,0,0,8,0,0,0,0);

   // Adding 23-V-46-0
   nuc = new Nucleus("V",46,23,0,45.9602,0,0.42237,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.0514,100);

   // Adding 23-V-46-1
   nuc = new Nucleus("V",46,23,1,45.9611,0.801,0.00102,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.801,100);

   // Adding 24-CR-46-0
   nuc = new Nucleus("CR",46,24,0,45.9684,0,0.26,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.603,100);

   // Adding 25-MN-46-0
   nuc = new Nucleus("MN",46,25,0,45.9867,0,0.041,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,17.1009,100);

   // Adding 18-AR-47-0
   nuc = new Nucleus("AR",47,18,0,46.9722,0,0.58,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,9.78939,100);

   // Adding 19-K-47-0
   nuc = new Nucleus("K",47,19,0,46.9617,0,17.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.64192,100);

   // Adding 20-CA-47-0
   nuc = new Nucleus("CA",47,20,0,46.9545,0,391910,0,1.6e-09,2.1e-09,0,0);
   nuc->AddDecay(0,1,0,1.992,100);

   // Adding 21-SC-47-0
   nuc = new Nucleus("SC",47,21,0,46.9524,0,289008,0,5.5e-10,7.3e-10,0,0);
   nuc->AddDecay(0,1,0,0.600086,100);

   // Adding 22-TI-47-0
   nuc = new Nucleus("TI",47,22,0,46.9518,0,0,7.3,0,0,0,0);

   // Adding 23-V-47-0
   nuc = new Nucleus("V",47,23,0,46.9549,0,1956,0,6.3e-11,5e-11,0,0);
   nuc->AddDecay(0,-1,0,2.9278,100);

   // Adding 24-CR-47-0
   nuc = new Nucleus("CR",47,24,0,46.9629,0,0.508,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.4515,100);

   // Adding 25-MN-47-0
   nuc = new Nucleus("MN",47,25,0,46.9761,0,2e-07,0,0,0,0,-8);
   nuc->AddDecay(-1,-2,0,7.52186,50);
   nuc->AddDecay(0,-1,0,12.2894,50);

   // Adding 19-K-48-0
   nuc = new Nucleus("K",48,19,0,47.9655,0,6.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.0902,98.86);
   nuc->AddDecay(-1,1,0,2.1438,1.14);

   // Adding 20-CA-48-0
   nuc = new Nucleus("CA",48,20,0,47.9525,0,0,0.187,0,0,0,0);
   nuc->AddDecay(0,2,0,4.27226,0);

   // Adding 21-SC-48-0
   nuc = new Nucleus("SC",48,21,0,47.9522,0,157212,0,1.7e-09,1.6e-09,0,0);
   nuc->AddDecay(0,1,0,3.99414,100);

   // Adding 22-TI-48-0
   nuc = new Nucleus("TI",48,22,0,47.9479,0,0,73.8,0,0,0,0);

   // Adding 23-V-48-0
   nuc = new Nucleus("V",48,23,0,47.9523,0,1.38011e+06,0,2e-09,2.7e-09,0,0);
   nuc->AddDecay(0,-1,0,4.01237,100);

   // Adding 24-CR-48-0
   nuc = new Nucleus("CR",48,24,0,47.954,0,77616,0,2e-10,2.5e-10,0,0);
   nuc->AddDecay(0,-1,0,1.65925,100);

   // Adding 25-MN-48-0
   nuc = new Nucleus("MN",48,25,0,47.9686,0,0.1581,0,0,0,1,0);
   nuc->AddDecay(-4,-3,0,5.83677,0.0006);
   nuc->AddDecay(0,-1,0,13.5288,99.7194);
   nuc->AddDecay(-1,-2,0,5.4283,0.28);

   // Adding 26-FE-48-0
   nuc = new Nucleus("FE",48,26,0,47.9806,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,0,11.1786,0);

   // Adding 19-K-49-0
   nuc = new Nucleus("K",49,19,0,48.9674,0,1.26,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,5.82337,86);
   nuc->AddDecay(0,1,0,10.97,14);

   // Adding 20-CA-49-0
   nuc = new Nucleus("CA",49,20,0,48.9557,0,522.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.2622,100);

   // Adding 21-SC-49-0
   nuc = new Nucleus("SC",49,21,0,48.95,0,3432,0,8.2e-11,6.1e-11,0,0);
   nuc->AddDecay(0,1,0,2.00577,100);

   // Adding 22-TI-49-0
   nuc = new Nucleus("TI",49,22,0,48.9479,0,0,5.5,0,0,0,0);

   // Adding 23-V-49-0
   nuc = new Nucleus("V",49,23,0,48.9485,0,2.92032e+07,0,1.9e-11,3.2e-11,0,0);
   nuc->AddDecay(0,-1,0,0.601894,100);

   // Adding 24-CR-49-0
   nuc = new Nucleus("CR",49,24,0,48.9513,0,2538,0,6.1e-11,5.9e-11,0,0);
   nuc->AddDecay(0,-1,0,2.63059,100);

   // Adding 25-MN-49-0
   nuc = new Nucleus("MN",49,25,0,48.9596,0,0.384,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.7149,100);

   // Adding 26-FE-49-0
   nuc = new Nucleus("FE",49,26,0,48.9736,0,0.075,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,13.0286,40);
   nuc->AddDecay(-1,-2,0,10.9443,60);

   // Adding 18-AR-50-0
   nuc = new Nucleus("AR",50,18,0,49.9847,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,11.1249,0);

   // Adding 19-K-50-0
   nuc = new Nucleus("K",50,19,0,49.9728,0,0.472,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,7.86611,29);
   nuc->AddDecay(0,1,0,14.2189,71);

   // Adding 20-CA-50-0
   nuc = new Nucleus("CA",50,20,0,49.9575,0,13.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.96602,100);

   // Adding 21-SC-50-0
   nuc = new Nucleus("SC",50,21,0,49.9522,0,102.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.88829,100);

   // Adding 21-SC-50-1
   nuc = new Nucleus("SC",50,21,1,49.9525,0.257,0.35,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.257,97.5);
   nuc->AddDecay(0,1,-1,7.14529,2.5);

   // Adding 22-TI-50-0
   nuc = new Nucleus("TI",50,22,0,49.9448,0,0,5.4,0,0,0,0);

   // Adding 23-V-50-0
   nuc = new Nucleus("V",50,23,0,49.9472,0,4.41504e+24,0.25,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.2082,83);
   nuc->AddDecay(0,1,0,1.0369,17);

   // Adding 24-CR-50-0
   nuc = new Nucleus("CR",50,24,0,49.946,0,0,4.345,0,0,0,0);

   // Adding 25-MN-50-0
   nuc = new Nucleus("MN",50,25,0,49.9542,0,0.28307,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.633,100);

   // Adding 25-MN-50-1
   nuc = new Nucleus("MN",50,25,1,49.9545,0.229,105,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.862,100);

   // Adding 26-FE-50-0
   nuc = new Nucleus("FE",50,26,0,49.963,0,0.15,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,3.56497,0);
   nuc->AddDecay(0,-1,0,8.14995,100);

   // Adding 27-CO-50-0
   nuc = new Nucleus("CO",50,27,0,49.9812,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,0,16.9685,0);

   // Adding 18-AR-51-0
   nuc = new Nucleus("AR",51,18,0,50.9923,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,15.3081,0);

   // Adding 19-K-51-0
   nuc = new Nucleus("K",51,19,0,50.9758,0,0.365,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,8.9895,47);
   nuc->AddDecay(0,1,0,13.394,53);

   // Adding 20-CA-51-0
   nuc = new Nucleus("CA",51,20,0,50.9615,0,10,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.561528,0);
   nuc->AddDecay(0,1,0,7.31413,100);

   // Adding 21-SC-51-0
   nuc = new Nucleus("SC",51,21,0,50.9536,0,12.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.50799,100);

   // Adding 22-TI-51-0
   nuc = new Nucleus("TI",51,22,0,50.9466,0,345.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.4707,100);

   // Adding 23-V-51-0
   nuc = new Nucleus("V",51,23,0,50.944,0,0,99.75,0,0,0,0);

   // Adding 24-CR-51-0
   nuc = new Nucleus("CR",51,24,0,50.9448,0,2.39345e+06,0,3.8e-11,3.7e-11,0,0);
   nuc->AddDecay(0,-1,0,0.752701,100);

   // Adding 25-MN-51-0
   nuc = new Nucleus("MN",51,25,0,50.9482,0,2772,0,9.3e-11,6.8e-11,0,0);
   nuc->AddDecay(0,-1,0,3.2078,100);

   // Adding 26-FE-51-0
   nuc = new Nucleus("FE",51,26,0,50.9568,0,0.305,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.0197,100);

   // Adding 27-CO-51-0
   nuc = new Nucleus("CO",51,27,0,50.9705,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,0,12.7473,0);

   // Adding 19-K-52-0
   nuc = new Nucleus("K",52,19,0,51.9823,0,0.105,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,11.3102,88);
   nuc->AddDecay(0,1,0,15.9858,12);

   // Adding 20-CA-52-0
   nuc = new Nucleus("CA",52,20,0,51.9651,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.94677,100);

   // Adding 21-SC-52-0
   nuc = new Nucleus("SC",52,21,0,51.9566,0,8.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.00826,100);

   // Adding 22-TI-52-0
   nuc = new Nucleus("TI",52,22,0,51.9469,0,102,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.97334,100);

   // Adding 23-V-52-0
   nuc = new Nucleus("V",52,23,0,51.9448,0,225,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.9756,100);

   // Adding 24-CR-52-0
   nuc = new Nucleus("CR",52,24,0,51.9405,0,0,83.789,0,0,0,0);

   // Adding 25-MN-52-0
   nuc = new Nucleus("MN",52,25,0,51.9456,0,483062,0,1.8e-09,1.8e-09,0,0);
   nuc->AddDecay(0,-1,0,4.71179,100);

   // Adding 25-MN-52-1
   nuc = new Nucleus("MN",52,25,1,51.946,0.378,1266,0,6.9e-11,5e-11,0,0);
   nuc->AddDecay(0,-1,-1,5.08979,98.25);
   nuc->AddDecay(0,0,-1,0.378,1.75);

   // Adding 26-FE-52-0
   nuc = new Nucleus("FE",52,26,0,51.9481,0,29790,0,1.4e-09,9.5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.37201,100);

   // Adding 26-FE-52-1
   nuc = new Nucleus("FE",52,26,1,51.9554,6.82,45.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,9.19201,100);

   // Adding 27-CO-52-0
   nuc = new Nucleus("CO",52,27,0,51.9632,0,2e-07,0,0,0,0,-8);
   nuc->AddDecay(-1,-2,0,6.63146,50);
   nuc->AddDecay(0,-1,0,14.0127,50);

   // Adding 28-NI-52-0
   nuc = new Nucleus("NI",52,28,0,51.9757,0,0.038,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,11.6618,50);
   nuc->AddDecay(-1,-2,0,10.2736,50);

   // Adding 19-K-53-0
   nuc = new Nucleus("K",53,19,0,52.9867,0,0.03,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,12.0791,85);
   nuc->AddDecay(0,1,0,15.5398,15);

   // Adding 20-CA-53-0
   nuc = new Nucleus("CA",53,20,0,52.9701,0,0.09,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.48612,30);
   nuc->AddDecay(0,1,0,10.0698,70);

   // Adding 21-SC-53-0
   nuc = new Nucleus("SC",53,21,0,52.9592,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,8.85661,0);

   // Adding 22-TI-53-0
   nuc = new Nucleus("TI",53,22,0,52.9497,0,32.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.02002,100);

   // Adding 23-V-53-0
   nuc = new Nucleus("V",53,23,0,52.9443,0,96.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.43608,100);

   // Adding 24-CR-53-0
   nuc = new Nucleus("CR",53,24,0,52.9407,0,0,9.501,0,0,0,0);

   // Adding 25-MN-53-0
   nuc = new Nucleus("MN",53,25,0,52.9413,0,1.17945e+14,0,3e-11,5.2e-11,0,0);
   nuc->AddDecay(0,-1,0,0.597,100);

   // Adding 26-FE-53-0
   nuc = new Nucleus("FE",53,26,0,52.9453,0,510.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.74259,100);

   // Adding 26-FE-53-1
   nuc = new Nucleus("FE",53,26,1,52.9486,3.04,154.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,3.04,100);

   // Adding 27-CO-53-0
   nuc = new Nucleus("CO",53,27,0,52.9542,0,0.24,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.302,100);

   // Adding 27-CO-53-1
   nuc = new Nucleus("CO",53,27,1,52.9576,3.19,0.247,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,11.492,98.5);
   nuc->AddDecay(-1,-1,-1,1.59089,1.5);

   // Adding 28-NI-53-0
   nuc = new Nucleus("NI",53,28,0,52.9685,0,0.045,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,13.2603,100);

   // Adding 19-K-54-0
   nuc = new Nucleus("K",54,19,0,53.9941,0,0.01,0,0,0,0,0);
   nuc->AddDecay(0,1,0,18.1122,100);
   nuc->AddDecay(-1,1,0,14.3394,0);

   // Adding 20-CA-54-0
   nuc = new Nucleus("CA",54,20,0,53.9747,0,0,0,0,0,0,-6);

   // Adding 22-TI-54-0
   nuc = new Nucleus("TI",54,22,0,53.951,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,4.2792,0);

   // Adding 23-V-54-0
   nuc = new Nucleus("V",54,23,0,53.9464,0,49.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.0416,100);

   // Adding 24-CR-54-0
   nuc = new Nucleus("CR",54,24,0,53.9389,0,0,2.365,0,0,0,0);

   // Adding 25-MN-54-0
   nuc = new Nucleus("MN",54,25,0,53.9404,0,2.69672e+07,0,7.1e-10,1.5e-09,0,0);
   nuc->AddDecay(0,1,0,0.696896,0.001);
   nuc->AddDecay(0,-1,0,1.3771,100);

   // Adding 26-FE-54-0
   nuc = new Nucleus("FE",54,26,0,53.9396,0,0,5.8,0,0,0,0);

   // Adding 27-CO-54-0
   nuc = new Nucleus("CO",54,27,0,53.9485,0,0.19324,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.243,100);

   // Adding 27-CO-54-1
   nuc = new Nucleus("CO",54,27,1,53.9487,0.199,88.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.442,100);

   // Adding 28-NI-54-0
   nuc = new Nucleus("NI",54,28,0,53.9579,0,0.14,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.79906,100);

   // Adding 23-V-55-0
   nuc = new Nucleus("V",55,23,0,54.9472,0,6.54,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.95601,100);

   // Adding 24-CR-55-0
   nuc = new Nucleus("CR",55,24,0,54.9408,0,209.82,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.6031,100);

   // Adding 25-MN-55-0
   nuc = new Nucleus("MN",55,25,0,54.938,0,0,100,0,0,0,0);

   // Adding 26-FE-55-0
   nuc = new Nucleus("FE",55,26,0,54.9383,0,8.60933e+07,0,3.3e-10,9.2e-10,0,0);
   nuc->AddDecay(0,-1,0,0.231602,100);

   // Adding 27-CO-55-0
   nuc = new Nucleus("CO",55,27,0,54.942,0,63108,0,1.1e-09,8.3e-10,0,0);
   nuc->AddDecay(0,-1,0,3.4512,100);

   // Adding 28-NI-55-0
   nuc = new Nucleus("NI",55,28,0,54.9513,0,0.2121,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.6937,100);

   // Adding 29-CU-55-0
   nuc = new Nucleus("CU",55,29,0,54.9655,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,-1,0,13.212,100);
   nuc->AddDecay(-1,-2,0,8.59832,0);

   // Adding 23-V-56-0
   nuc = new Nucleus("V",56,23,0,55.9504,0,1e-06,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,9.13188,100);

   // Adding 24-CR-56-0
   nuc = new Nucleus("CR",56,24,0,55.9406,0,356.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.61692,100);

   // Adding 25-MN-56-0
   nuc = new Nucleus("MN",56,25,0,55.9389,0,9282.6,0,2.6e-10,2e-10,0,0);
   nuc->AddDecay(0,1,0,3.6954,100);

   // Adding 26-FE-56-0
   nuc = new Nucleus("FE",56,26,0,55.9349,0,0,91.727,0,0,0,0);

   // Adding 27-CO-56-0
   nuc = new Nucleus("CO",56,27,0,55.9398,0,6.67613e+06,0,2.5e-09,6.4e-09,0,0);
   nuc->AddDecay(0,-1,0,4.56599,100);

   // Adding 28-NI-56-0
   nuc = new Nucleus("NI",56,28,0,55.9421,0,509760,0,8.6e-10,9.7e-10,0,0);
   nuc->AddDecay(0,-1,0,2.13551,100);

   // Adding 29-CU-56-0
   nuc = new Nucleus("CU",56,29,0,55.9586,0,2e-07,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,15.2987,50);
   nuc->AddDecay(-1,-2,0,8.13378,50);

   // Adding 24-CR-57-0
   nuc = new Nucleus("CR",57,24,0,56.9438,0,21.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.09225,100);

   // Adding 25-MN-57-0
   nuc = new Nucleus("MN",57,25,0,56.9383,0,85.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.69078,100);

   // Adding 26-FE-57-0
   nuc = new Nucleus("FE",57,26,0,56.9354,0,0,2.2,0,0,0,0);

   // Adding 27-CO-57-0
   nuc = new Nucleus("CO",57,27,0,56.9363,0,2.34827e+07,0,2.1e-10,9.4e-10,0,0);
   nuc->AddDecay(0,-1,0,0.835999,100);

   // Adding 28-NI-57-0
   nuc = new Nucleus("NI",57,28,0,56.9398,0,128160,0,8.7e-10,7.7e-10,0,0);
   nuc->AddDecay(0,-1,0,3.26429,100);

   // Adding 29-CU-57-0
   nuc = new Nucleus("CU",57,29,0,56.9492,0,0.1994,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.77021,100);

   // Adding 30-ZN-57-0
   nuc = new Nucleus("ZN",57,30,0,56.9649,0,0.04,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,14.6194,35);
   nuc->AddDecay(-1,-2,0,13.9247,65);

   // Adding 24-CR-58-0
   nuc = new Nucleus("CR",58,24,0,57.9443,0,7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.00822,100);

   // Adding 25-MN-58-0
   nuc = new Nucleus("MN",58,25,0,57.94,0,3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.24678,100);

   // Adding 25-MN-58-1
   nuc = new Nucleus("MN",58,25,1,57.94,0,65.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.24678,100);

   // Adding 26-FE-58-0
   nuc = new Nucleus("FE",58,26,0,57.9333,0,0,0.28,0,0,0,0);

   // Adding 27-CO-58-0
   nuc = new Nucleus("CO",58,27,0,57.9358,0,6.11885e+06,0,7.4e-10,2e-09,0,0);
   nuc->AddDecay(0,-1,0,2.3075,100);

   // Adding 27-CO-58-1
   nuc = new Nucleus("CO",58,27,1,57.9358,0.025,32940,0,2.4e-11,1.7e-11,0,0);
   nuc->AddDecay(0,0,-1,0.025,100);

   // Adding 28-NI-58-0
   nuc = new Nucleus("NI",58,28,0,57.9353,0,0,68.27,0,0,0,0);

   // Adding 29-CU-58-0
   nuc = new Nucleus("CU",58,29,0,57.9445,0,3.204,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.56309,100);

   // Adding 30-ZN-58-0
   nuc = new Nucleus("ZN",58,30,0,57.9546,0,0.08,0,0,0,0,-4);
   nuc->AddDecay(0,-1,0,9.36678,100);

   // Adding 23-V-59-0
   nuc = new Nucleus("V",59,23,0,58.9593,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,9.86098,100);

   // Adding 24-CR-59-0
   nuc = new Nucleus("CR",59,24,0,58.9487,0,0.74,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.70061,100);

   // Adding 25-MN-59-0
   nuc = new Nucleus("MN",59,25,0,58.9404,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.18538,100);

   // Adding 26-FE-59-0
   nuc = new Nucleus("FE",59,26,0,58.9349,0,3.84506e+06,0,1.8e-09,3.5e-09,0,0);
   nuc->AddDecay(0,1,0,1.5651,100);

   // Adding 27-CO-59-0
   nuc = new Nucleus("CO",59,27,0,58.9332,0,0,100,0,0,0,0);

   // Adding 28-NI-59-0
   nuc = new Nucleus("NI",59,28,0,58.9344,0,2.39674e+12,0,6.3e-11,2.2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.0725,100);

   // Adding 29-CU-59-0
   nuc = new Nucleus("CU",59,29,0,58.9395,0,81.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.79959,100);

   // Adding 30-ZN-59-0
   nuc = new Nucleus("ZN",59,30,0,58.9493,0,0.182,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.09408,99.9);
   nuc->AddDecay(-1,-2,0,5.67658,0.1);

   // Adding 24-CR-60-0
   nuc = new Nucleus("CR",60,24,0,59.9497,0,0.57,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.94859,100);

   // Adding 25-MN-60-0
   nuc = new Nucleus("MN",60,25,0,59.9433,0,51,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.63252,100);

   // Adding 25-MN-60-1
   nuc = new Nucleus("MN",60,25,1,59.9436,0.272,1.77,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.90452,88.5);
   nuc->AddDecay(0,0,-1,0.272,11.5);

   // Adding 26-FE-60-0
   nuc = new Nucleus("FE",60,26,0,59.9341,0,4.7304e+13,0,1.1e-07,3.3e-07,0,0);
   nuc->AddDecay(0,1,1,0.178177,100);

   // Adding 27-CO-60-0
   nuc = new Nucleus("CO",60,27,0,59.9338,0,1.66239e+08,0,3.4e-09,2.9e-08,0,0);
   nuc->AddDecay(0,1,0,2.8239,100);

   // Adding 27-CO-60-1
   nuc = new Nucleus("CO",60,27,1,59.9339,0.059,628.02,0,1.7e-12,1.3e-12,0,0);
   nuc->AddDecay(0,1,-1,2.8829,0.24);
   nuc->AddDecay(0,0,-1,0.059,99.76);

   // Adding 28-NI-60-0
   nuc = new Nucleus("NI",60,28,0,59.9308,0,0,26.1,0,0,0,0);

   // Adding 29-CU-60-0
   nuc = new Nucleus("CU",60,29,0,59.9374,0,1422,0,7e-11,6.2e-11,0,0);
   nuc->AddDecay(0,-1,0,6.12689,100);

   // Adding 30-ZN-60-0
   nuc = new Nucleus("ZN",60,30,0,59.9418,0,142.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.15812,100);

   // Adding 24-CR-61-0
   nuc = new Nucleus("CR",61,24,0,60.9541,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,8.80549,100);

   // Adding 25-MN-61-0
   nuc = new Nucleus("MN",61,25,0,60.9446,0,0.71,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.3474,100);

   // Adding 26-FE-61-0
   nuc = new Nucleus("FE",61,26,0,60.9367,0,358.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.9776,100);

   // Adding 27-CO-61-0
   nuc = new Nucleus("CO",61,27,0,60.9325,0,5940,0,7.5e-11,7.5e-11,0,0);
   nuc->AddDecay(0,1,0,1.3217,100);

   // Adding 28-NI-61-0
   nuc = new Nucleus("NI",61,28,0,60.9311,0,0,1.13,0,0,0,0);

   // Adding 29-CU-61-0
   nuc = new Nucleus("CU",61,29,0,60.9335,0,11998.8,0,1.2e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,2.2371,100);

   // Adding 30-ZN-61-0
   nuc = new Nucleus("ZN",61,30,0,60.9395,0,89.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.6371,100);

   // Adding 31-GA-61-0
   nuc = new Nucleus("GA",61,31,0,60.9492,0,0.15,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.99472,100);

   // Adding 32-GE-61-0
   nuc = new Nucleus("GE",61,32,0,60.9638,0,0.04,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,13.1651,80);
   nuc->AddDecay(0,-1,0,13.6189,20);

   // Adding 24-CR-62-0
   nuc = new Nucleus("CR",62,24,0,61.9558,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,7.29563,100);

   // Adding 25-MN-62-0
   nuc = new Nucleus("MN",62,25,0,61.948,0,0.88,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.4326,100);

   // Adding 26-FE-62-0
   nuc = new Nucleus("FE",62,26,0,61.9368,0,68,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.5302,100);

   // Adding 27-CO-62-0
   nuc = new Nucleus("CO",62,27,0,61.9341,0,90,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.31459,100);

   // Adding 27-CO-62-1
   nuc = new Nucleus("CO",62,27,1,61.9341,0.022,834.6,0,4.7e-11,3.7e-11,0,0);
   nuc->AddDecay(0,0,-1,0.022,1);
   nuc->AddDecay(0,1,-1,5.33659,99);

   // Adding 28-NI-62-0
   nuc = new Nucleus("NI",62,28,0,61.9283,0,0,3.59,0,0,0,0);

   // Adding 29-CU-62-0
   nuc = new Nucleus("CU",62,29,0,61.9326,0,584.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.94817,100);

   // Adding 30-ZN-62-0
   nuc = new Nucleus("ZN",62,30,0,61.9343,0,33069.6,0,9.4e-10,6.6e-10,0,0);
   nuc->AddDecay(0,-1,0,1.62703,100);

   // Adding 31-GA-62-0
   nuc = new Nucleus("GA",62,31,0,61.9442,0,0.11612,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.17098,100);

   // Adding 25-MN-63-0
   nuc = new Nucleus("MN",63,25,0,62.9498,0,0.25,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.76139,100);

   // Adding 26-FE-63-0
   nuc = new Nucleus("FE",63,26,0,62.9404,0,6.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.3242,100);

   // Adding 27-CO-63-0
   nuc = new Nucleus("CO",63,27,0,62.9336,0,27.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.6722,100);

   // Adding 28-NI-63-0
   nuc = new Nucleus("NI",63,28,0,62.9297,0,3.15675e+09,0,1.5e-10,5.2e-10,0,0);
   nuc->AddDecay(0,1,0,0.0670013,100);

   // Adding 29-CU-63-0
   nuc = new Nucleus("CU",63,29,0,62.9296,0,0,69.17,0,0,0,0);

   // Adding 30-ZN-63-0
   nuc = new Nucleus("ZN",63,30,0,62.9332,0,2308.2,0,7.9e-11,6.1e-11,0,0);
   nuc->AddDecay(0,-1,0,3.36679,100);

   // Adding 31-GA-63-0
   nuc = new Nucleus("GA",63,31,0,62.9391,0,32.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.52001,100);

   // Adding 32-GE-63-0
   nuc = new Nucleus("GE",63,32,0,62.9496,0,0.095,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,9.77951,100);

   // Adding 25-MN-64-0
   nuc = new Nucleus("MN",64,25,0,63.9537,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,11.8013,0);

   // Adding 26-FE-64-0
   nuc = new Nucleus("FE",64,26,0,63.9411,0,2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.8881,100);

   // Adding 27-CO-64-0
   nuc = new Nucleus("CO",64,27,0,63.9358,0,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.30659,100);

   // Adding 28-NI-64-0
   nuc = new Nucleus("NI",64,28,0,63.928,0,0,0.91,0,0,0,0);

   // Adding 29-CU-64-0
   nuc = new Nucleus("CU",64,29,0,63.9298,0,45720,0,1.2e-10,1.5e-10,0,0);
   nuc->AddDecay(0,-1,0,1.6751,61);
   nuc->AddDecay(0,1,0,0.578903,39);

   // Adding 30-ZN-64-0
   nuc = new Nucleus("ZN",64,30,0,63.9291,0,0,48.6,0,0,0,0);

   // Adding 31-GA-64-0
   nuc = new Nucleus("GA",64,31,0,63.9368,0,157.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.16478,100);

   // Adding 32-GE-64-0
   nuc = new Nucleus("GE",64,32,0,63.9416,0,63.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.41002,100);

   // Adding 25-MN-65-0
   nuc = new Nucleus("MN",65,25,0,64.9561,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,10.3955,100);

   // Adding 26-FE-65-0
   nuc = new Nucleus("FE",65,26,0,64.9449,0,0.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.87638,100);

   // Adding 27-CO-65-0
   nuc = new Nucleus("CO",65,27,0,64.9365,0,1.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.9584,100);

   // Adding 28-NI-65-0
   nuc = new Nucleus("NI",65,28,0,64.9301,0,9061.92,0,1.8e-10,1.3e-10,0,0);
   nuc->AddDecay(0,1,0,2.1367,100);

   // Adding 29-CU-65-0
   nuc = new Nucleus("CU",65,29,0,64.9278,0,0,30.83,0,0,0,0);

   // Adding 30-ZN-65-0
   nuc = new Nucleus("ZN",65,30,0,64.9292,0,2.11041e+07,0,3.9e-09,2.9e-09,0,0);
   nuc->AddDecay(0,-1,0,1.3514,100);

   // Adding 31-GA-65-0
   nuc = new Nucleus("GA",65,31,0,64.9327,0,912,0,3.7e-11,2.9e-11,0,0);
   nuc->AddDecay(0,-1,0,3.2549,100);

   // Adding 32-GE-65-0
   nuc = new Nucleus("GE",65,32,0,64.9394,0,30.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.24222,100);

   // Adding 33-AS-65-0
   nuc = new Nucleus("AS",65,33,0,64.9495,0,0.19,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.35471,100);

   // Adding 34-SE-65-0
   nuc = new Nucleus("SE",65,34,0,64.9647,0,0.05,0,0,0,0,-4);
   nuc->AddDecay(0,-1,0,14.1368,100);

   // Adding 26-FE-66-0
   nuc = new Nucleus("FE",66,26,0,65.946,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,1,0,5.73299,0);

   // Adding 27-CO-66-0
   nuc = new Nucleus("CO",66,27,0,65.9398,0,0.23,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.97659,100);

   // Adding 28-NI-66-0
   nuc = new Nucleus("NI",66,28,0,65.9291,0,196560,0,3e-09,1.9e-09,0,0);
   nuc->AddDecay(0,1,0,0.225304,100);

   // Adding 29-CU-66-0
   nuc = new Nucleus("CU",66,29,0,65.9289,0,305.28,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.6424,100);

   // Adding 30-ZN-66-0
   nuc = new Nucleus("ZN",66,30,0,65.926,0,0,27.9,0,0,0,0);

   // Adding 31-GA-66-0
   nuc = new Nucleus("GA",66,31,0,65.9316,0,34164,0,1.2e-09,7.2e-10,0,0);
   nuc->AddDecay(0,-1,0,5.17498,100);

   // Adding 32-GE-66-0
   nuc = new Nucleus("GE",66,32,0,65.9338,0,8136,0,1e-10,1.3e-10,0,0);
   nuc->AddDecay(0,-1,0,2.1,100);

   // Adding 33-AS-66-0
   nuc = new Nucleus("AS",66,33,0,65.9444,0,0.09577,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.79943,100);

   // Adding 26-FE-67-0
   nuc = new Nucleus("FE",67,26,0,66.95,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,8.74668,100);

   // Adding 27-CO-67-0
   nuc = new Nucleus("CO",67,27,0,66.9406,0,0.42,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.42129,100);

   // Adding 28-NI-67-0
   nuc = new Nucleus("NI",67,28,0,66.9316,0,21,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.55777,100);

   // Adding 29-CU-67-0
   nuc = new Nucleus("CU",67,29,0,66.9277,0,222588,0,3.4e-10,5.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.576935,100);

   // Adding 30-ZN-67-0
   nuc = new Nucleus("ZN",67,30,0,66.9271,0,0,4.1,0,0,0,0);

   // Adding 31-GA-67-0
   nuc = new Nucleus("GA",67,31,0,66.9282,0,281768,0,1.9e-10,2.8e-10,0,0);
   nuc->AddDecay(0,-1,0,1.0004,100);

   // Adding 32-GE-67-0
   nuc = new Nucleus("GE",67,32,0,66.9327,0,1134,0,6.5e-11,4.2e-11,0,0);
   nuc->AddDecay(0,-1,0,4.22277,100);

   // Adding 33-AS-67-0
   nuc = new Nucleus("AS",67,33,0,66.9392,0,42.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.01004,100);

   // Adding 34-SE-67-0
   nuc = new Nucleus("SE",67,34,0,66.9501,0,0.107,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.153,100);

   // Adding 26-FE-68-0
   nuc = new Nucleus("FE",68,26,0,67.9525,0,0.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.59,100);

   // Adding 27-CO-68-0
   nuc = new Nucleus("CO",68,27,0,67.9444,0,0.18,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.6562,100);

   // Adding 28-NI-68-0
   nuc = new Nucleus("NI",68,28,0,67.9318,0,19,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.05603,100);

   // Adding 29-CU-68-0
   nuc = new Nucleus("CU",68,29,0,67.9296,0,31.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.46207,100);

   // Adding 29-CU-68-1
   nuc = new Nucleus("CU",68,29,1,67.9304,0.722,225,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.722,84);
   nuc->AddDecay(0,1,-1,5.18407,16);

   // Adding 30-ZN-68-0
   nuc = new Nucleus("ZN",68,30,0,67.9249,0,0,18.8,0,0,0,0);

   // Adding 31-GA-68-0
   nuc = new Nucleus("GA",68,31,0,67.928,0,4057.74,0,1e-10,8.1e-11,0,0);
   nuc->AddDecay(0,-1,0,2.9211,100);

   // Adding 32-GE-68-0
   nuc = new Nucleus("GE",68,32,0,67.9281,0,2.33988e+07,0,1.3e-09,1.3e-08,0,0);
   nuc->AddDecay(0,-1,0,0.105858,100);

   // Adding 33-AS-68-0
   nuc = new Nucleus("AS",68,33,0,67.9368,0,151.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.10005,100);

   // Adding 34-SE-68-0
   nuc = new Nucleus("SE",68,34,0,67.9419,0,96,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.72902,100);

   // Adding 27-CO-69-0
   nuc = new Nucleus("CO",69,27,0,68.9452,0,0.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.32801,100);

   // Adding 28-NI-69-0
   nuc = new Nucleus("NI",69,28,0,68.9352,0,11.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.36227,100);

   // Adding 29-CU-69-0
   nuc = new Nucleus("CU",69,29,0,68.9294,0,171,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.67493,100);

   // Adding 30-ZN-69-0
   nuc = new Nucleus("ZN",69,30,0,68.9266,0,3384,0,3.1e-11,4.3e-11,0,0);
   nuc->AddDecay(0,1,0,0.906013,100);

   // Adding 30-ZN-69-1
   nuc = new Nucleus("ZN",69,30,1,68.927,0.439,49536,0,3.3e-10,3.4e-10,0,0);
   nuc->AddDecay(0,0,-1,0.439,99.97);
   nuc->AddDecay(0,1,-1,1.34502,0.03);

   // Adding 31-GA-69-0
   nuc = new Nucleus("GA",69,31,0,68.9256,0,0,60.1,0,0,0,0);

   // Adding 32-GE-69-0
   nuc = new Nucleus("GE",69,32,0,68.928,0,140580,0,2.4e-10,3.7e-10,0,0);
   nuc->AddDecay(0,-1,0,2.22729,100);

   // Adding 33-AS-69-0
   nuc = new Nucleus("AS",69,33,0,68.9323,0,912,0,5.7e-11,3.5e-11,0,0);
   nuc->AddDecay(0,-1,0,4.013,100);

   // Adding 34-SE-69-0
   nuc = new Nucleus("SE",69,34,0,68.9396,0,27.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.7831,99.95);
   nuc->AddDecay(-1,-2,0,3.39053,0.05);

   // Adding 35-BR-69-0
   nuc = new Nucleus("BR",69,35,0,68.9499,0,1.5e-07,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,9.61841,100);

   // Adding 27-CO-70-0
   nuc = new Nucleus("CO",70,27,0,69.9498,0,2e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,12.7325,100);

   // Adding 28-NI-70-0
   nuc = new Nucleus("NI",70,28,0,69.9361,0,1e-06,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,3.47538,100);

   // Adding 29-CU-70-0
   nuc = new Nucleus("CU",70,29,0,69.9324,0,4.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.59912,100);

   // Adding 29-CU-70-1
   nuc = new Nucleus("CU",70,29,1,69.9326,0.14,47,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.73912,100);

   // Adding 30-ZN-70-0
   nuc = new Nucleus("ZN",70,30,0,69.9253,0,0,0.6,0,0,0,0);

   // Adding 31-GA-70-0
   nuc = new Nucleus("GA",70,31,0,69.926,0,1268.4,0,3.1e-11,2.6e-11,0,0);
   nuc->AddDecay(0,1,0,1.65578,99.59);
   nuc->AddDecay(0,-1,0,0.6548,0.41);

   // Adding 32-GE-70-0
   nuc = new Nucleus("GE",70,32,0,69.9242,0,0,20.5,0,0,0,0);

   // Adding 33-AS-70-0
   nuc = new Nucleus("AS",70,33,0,69.9309,0,3156,0,1.3e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,6.21996,100);

   // Adding 34-SE-70-0
   nuc = new Nucleus("SE",70,34,0,69.9335,0,2466,0,1.4e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,2.39954,100);

   // Adding 35-BR-70-0
   nuc = new Nucleus("BR",70,35,0,69.9446,0,0.0791,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.3498,100);

   // Adding 35-BR-70-1
   nuc = new Nucleus("BR",70,35,1,69.9446,0,2.2,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,10.3498,100);

   // Adding 27-CO-71-0
   nuc = new Nucleus("CO",71,27,0,70.9517,0,1.5e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,10.9258,100);

   // Adding 28-NI-71-0
   nuc = new Nucleus("NI",71,28,0,70.94,0,1.86,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.8749,100);

   // Adding 29-CU-71-0
   nuc = new Nucleus("CU",71,29,0,70.9326,0,19.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.55737,100);

   // Adding 30-ZN-71-0
   nuc = new Nucleus("ZN",71,30,0,70.9277,0,147,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.81271,100);

   // Adding 30-ZN-71-1
   nuc = new Nucleus("ZN",71,30,1,70.9279,0.158,14256,0,2.4e-10,2.4e-10,0,0);
   nuc->AddDecay(0,0,-1,0.158,0.05);
   nuc->AddDecay(0,1,-1,2.97071,99.95);

   // Adding 31-GA-71-0
   nuc = new Nucleus("GA",71,31,0,70.9247,0,0,39.9,0,0,0,0);

   // Adding 32-GE-71-0
   nuc = new Nucleus("GE",71,32,0,70.925,0,987552,0,1.2e-11,1.1e-11,0,0);
   nuc->AddDecay(0,-1,0,0.229401,100);

   // Adding 32-GE-71-1
   nuc = new Nucleus("GE",71,32,1,70.9252,0.198,0.0204,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.198,100);

   // Adding 33-AS-71-0
   nuc = new Nucleus("AS",71,33,0,70.9271,0,235008,0,4.6e-10,5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.01267,100);

   // Adding 34-SE-71-0
   nuc = new Nucleus("SE",71,34,0,70.9323,0,284.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.79935,100);

   // Adding 35-BR-71-0
   nuc = new Nucleus("BR",71,35,0,70.9392,0,21.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.4999,100);

   // Adding 36-KR-71-0
   nuc = new Nucleus("KR",71,36,0,70.9505,0,0.097,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.493,100);

   // Adding 28-NI-72-0
   nuc = new Nucleus("NI",72,28,0,71.9413,0,2.06,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.2242,100);

   // Adding 29-CU-72-0
   nuc = new Nucleus("CU",72,29,0,71.9357,0,6.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.22206,100);

   // Adding 30-ZN-72-0
   nuc = new Nucleus("ZN",72,30,0,71.9269,0,167400,0,1.4e-09,1.5e-09,0,0);
   nuc->AddDecay(0,1,0,0.458061,100);

   // Adding 31-GA-72-0
   nuc = new Nucleus("GA",72,31,0,71.9264,0,50760,0,1.1e-09,8.4e-10,0,0);
   nuc->AddDecay(0,1,0,4.0011,100);

   // Adding 32-GE-72-0
   nuc = new Nucleus("GE",72,32,0,71.9221,0,0,27.4,0,0,0,0);

   // Adding 33-AS-72-0
   nuc = new Nucleus("AS",72,33,0,71.9268,0,93600,0,1.8e-09,1.3e-09,0,0);
   nuc->AddDecay(0,-1,0,4.35608,100);

   // Adding 34-SE-72-0
   nuc = new Nucleus("SE",72,34,0,71.9271,0,725760,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.33503,100);

   // Adding 35-BR-72-0
   nuc = new Nucleus("BR",72,35,0,71.9365,0,78.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.71449,100);
   nuc->AddDecay(-1,-2,0,1.42372,0);

   // Adding 35-BR-72-1
   nuc = new Nucleus("BR",72,35,1,71.9366,0.101,10.6,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,8.81549,50);
   nuc->AddDecay(0,0,-1,0.101,50);

   // Adding 36-KR-72-0
   nuc = new Nucleus("KR",72,36,0,71.9419,0,17.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.04,100);

   // Adding 28-NI-73-0
   nuc = new Nucleus("NI",73,28,0,72.946,0,0.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.8297,100);

   // Adding 29-CU-73-0
   nuc = new Nucleus("CU",73,29,0,72.9365,0,3.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.25064,100);

   // Adding 30-ZN-73-0
   nuc = new Nucleus("ZN",73,30,0,72.9298,0,23.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.29382,100);

   // Adding 30-ZN-73-1
   nuc = new Nucleus("ZN",73,30,1,72.93,0.196,5.8,0,0,0,0,-8);
   nuc->AddDecay(0,1,-1,4.48982,50);
   nuc->AddDecay(0,0,-1,0.196,50);

   // Adding 31-GA-73-0
   nuc = new Nucleus("GA",73,31,0,72.9252,0,17496,0,2.6e-10,2e-10,0,0);
   nuc->AddDecay(0,1,0,1.59325,0.88);
   nuc->AddDecay(0,1,1,1.52625,99.12);

   // Adding 32-GE-73-0
   nuc = new Nucleus("GE",73,32,0,72.9235,0,0,7.8,0,0,0,0);

   // Adding 32-GE-73-1
   nuc = new Nucleus("GE",73,32,1,72.9235,0.067,0.499,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.067,100);

   // Adding 33-AS-73-0
   nuc = new Nucleus("AS",73,33,0,72.9238,0,6.93792e+06,0,2.6e-10,9.3e-10,0,0);
   nuc->AddDecay(0,-1,0,0.340874,100);

   // Adding 34-SE-73-0
   nuc = new Nucleus("SE",73,34,0,72.9268,0,25740,0,3.9e-10,2.4e-10,0,0);
   nuc->AddDecay(0,-1,0,2.74003,100);

   // Adding 34-SE-73-1
   nuc = new Nucleus("SE",73,34,1,72.9268,0.026,2388,0,4.1e-11,2.7e-11,0,0);
   nuc->AddDecay(0,0,-1,0.026,72.6);
   nuc->AddDecay(0,-1,-1,2.76603,27.4);

   // Adding 35-BR-73-0
   nuc = new Nucleus("BR",73,35,0,72.9318,0,204,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.6564,100);

   // Adding 36-KR-73-0
   nuc = new Nucleus("KR",73,36,0,72.9389,0,27,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.6746,99.32);
   nuc->AddDecay(-1,-2,0,3.7202,0.68);

   // Adding 37-RB-73-0
   nuc = new Nucleus("RB",73,37,0,72.9503,0,1.5e-07,0,0,0,0,-3);
   nuc->AddDecay(-1,-1,0,0.589345,100);

   // Adding 28-NI-74-0
   nuc = new Nucleus("NI",74,28,0,73.9477,0,1.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.09,100);

   // Adding 29-CU-74-0
   nuc = new Nucleus("CU",74,29,0,73.9401,0,1.594,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.88914,100);

   // Adding 30-ZN-74-0
   nuc = new Nucleus("ZN",74,30,0,73.9295,0,96,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.34483,25);
   nuc->AddDecay(0,1,1,2.28483,75);

   // Adding 31-GA-74-0
   nuc = new Nucleus("GA",74,31,0,73.9269,0,487.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.36794,100);

   // Adding 31-GA-74-1
   nuc = new Nucleus("GA",74,31,1,73.927,0.06,9.5,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.42794,25);
   nuc->AddDecay(0,0,-1,0.06,75);

   // Adding 32-GE-74-0
   nuc = new Nucleus("GE",74,32,0,73.9212,0,0,36.5,0,0,0,0);

   // Adding 33-AS-74-0
   nuc = new Nucleus("AS",74,33,0,73.9239,0,1.53533e+06,0,1.3e-09,2.1e-09,0,0);
   nuc->AddDecay(0,-1,0,2.56239,66);
   nuc->AddDecay(0,1,0,1.353,34);

   // Adding 34-SE-74-0
   nuc = new Nucleus("SE",74,34,0,73.9225,0,0,0.9,0,0,0,0);

   // Adding 35-BR-74-0
   nuc = new Nucleus("BR",74,35,0,73.9299,0,1524,0,8.4e-11,6.8e-11,0,0);
   nuc->AddDecay(0,-1,0,6.9067,100);

   // Adding 35-BR-74-1
   nuc = new Nucleus("BR",74,35,1,73.9299,0,2760,0,1.4e-10,1.1e-10,0,0);
   nuc->AddDecay(0,-1,-1,6.9067,100);

   // Adding 36-KR-74-0
   nuc = new Nucleus("KR",74,36,0,73.9333,0,690,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.13766,100);

   // Adding 37-RB-74-0
   nuc = new Nucleus("RB",74,37,0,73.9445,0,0.0649,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.4445,100);

   // Adding 29-CU-75-0
   nuc = new Nucleus("CU",75,29,0,74.9414,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.89187,96.5);
   nuc->AddDecay(-1,1,0,3.06134,3.5);

   // Adding 30-ZN-75-0
   nuc = new Nucleus("ZN",75,30,0,74.9329,0,10.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.99579,100);

   // Adding 31-GA-75-0
   nuc = new Nucleus("GA",75,31,0,74.9265,0,126,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.39165,99.3);
   nuc->AddDecay(0,1,1,3.25165,0.7);

   // Adding 32-GE-75-0
   nuc = new Nucleus("GE",75,32,0,74.9229,0,4966.8,0,4.6e-11,5.4e-11,0,0);
   nuc->AddDecay(0,1,0,1.17651,100);

   // Adding 32-GE-75-1
   nuc = new Nucleus("GE",75,32,1,74.923,0.14,47.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.14,99.97);
   nuc->AddDecay(0,1,-1,1.31651,0.03);

   // Adding 33-AS-75-0
   nuc = new Nucleus("AS",75,33,0,74.9216,0,0,100,0,0,0,0);

   // Adding 34-SE-75-0
   nuc = new Nucleus("SE",75,34,0,74.9225,0,1.03489e+07,0,2.6e-09,1.7e-09,0,0);
   nuc->AddDecay(0,-1,0,0.863602,100);

   // Adding 35-BR-75-0
   nuc = new Nucleus("BR",75,35,0,74.9258,0,5802,0,7.9e-11,8.6e-11,0,0);
   nuc->AddDecay(0,-1,0,3.03001,100);

   // Adding 36-KR-75-0
   nuc = new Nucleus("KR",75,36,0,74.931,0,258,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.8986,100);

   // Adding 37-RB-75-0
   nuc = new Nucleus("RB",75,37,0,74.9386,0,19,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.02013,100);

   // Adding 29-CU-76-0
   nuc = new Nucleus("CU",76,29,0,75.9455,0,0.61,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.3041,97);
   nuc->AddDecay(-1,1,0,3.65847,3);

   // Adding 30-ZN-76-0
   nuc = new Nucleus("ZN",76,30,0,75.9334,0,5.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.16008,100);

   // Adding 31-GA-76-0
   nuc = new Nucleus("GA",76,31,0,75.9289,0,29.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.00993,100);

   // Adding 32-GE-76-0
   nuc = new Nucleus("GE",76,32,0,75.9214,0,0,7.8,0,0,0,0);

   // Adding 33-AS-76-0
   nuc = new Nucleus("AS",76,33,0,75.9224,0,94752,0,1.6e-09,9.2e-10,0,0);
   nuc->AddDecay(0,1,0,2.962,99.98);
   nuc->AddDecay(0,-1,0,0.923302,0.02);

   // Adding 34-SE-76-0
   nuc = new Nucleus("SE",76,34,0,75.9192,0,0,9,0,0,0,0);

   // Adding 35-BR-76-0
   nuc = new Nucleus("BR",76,35,0,75.9245,0,58320,0,4.6e-10,5.8e-10,0,0);
   nuc->AddDecay(0,-1,0,4.96281,100);

   // Adding 35-BR-76-1
   nuc = new Nucleus("BR",76,35,1,75.9247,0.103,1.31,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.06581,0.6);
   nuc->AddDecay(0,0,-1,0.103,99.4);

   // Adding 36-KR-76-0
   nuc = new Nucleus("KR",76,36,0,75.9259,0,53280,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.31139,100);

   // Adding 37-RB-76-0
   nuc = new Nucleus("RB",76,37,0,75.9351,0,39.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.50034,100);

   // Adding 38-SR-76-0
   nuc = new Nucleus("SR",76,38,0,75.9416,0,8.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.08688,100);

   // Adding 29-CU-77-0
   nuc = new Nucleus("CU",77,29,0,76.9473,0,0.469,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.49401,100);

   // Adding 30-ZN-77-0
   nuc = new Nucleus("ZN",77,30,0,76.9371,0,2.08,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.27059,100);

   // Adding 30-ZN-77-1
   nuc = new Nucleus("ZN",77,30,1,76.9379,0.772,1.05,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.04259,50);
   nuc->AddDecay(0,0,-1,0.772,50);

   // Adding 31-GA-77-0
   nuc = new Nucleus("GA",77,31,0,76.9293,0,13.2,0,0,0,0,0);
   nuc->AddDecay(0,1,1,5.17941,100);

   // Adding 32-GE-77-0
   nuc = new Nucleus("GE",77,32,0,76.9235,0,40680,0,3.3e-10,4.5e-10,0,0);
   nuc->AddDecay(0,1,0,2.702,100);

   // Adding 32-GE-77-1
   nuc = new Nucleus("GE",77,32,1,76.9237,0.16,52.9,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.86201,79);
   nuc->AddDecay(0,0,-1,0.16,21);

   // Adding 33-AS-77-0
   nuc = new Nucleus("AS",77,33,0,76.9206,0,139788,0,4e-10,4.2e-10,0,0);
   nuc->AddDecay(0,1,0,0.682892,99.7861);
   nuc->AddDecay(0,1,1,0.520889,0.213868);

   // Adding 34-SE-77-0
   nuc = new Nucleus("SE",77,34,0,76.9199,0,0,7.6,0,0,0,0);

   // Adding 34-SE-77-1
   nuc = new Nucleus("SE",77,34,1,76.9201,0.162,17.36,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.162,100);

   // Adding 35-BR-77-0
   nuc = new Nucleus("BR",77,35,0,76.9214,0,205330,0,9.6e-11,1.3e-10,0,0);
   nuc->AddDecay(0,-1,0,1.36508,100);

   // Adding 35-BR-77-1
   nuc = new Nucleus("BR",77,35,1,76.9215,0.106,256.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.106,100);

   // Adding 36-KR-77-0
   nuc = new Nucleus("KR",77,36,0,76.9247,0,4464,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.06364,100);

   // Adding 37-RB-77-0
   nuc = new Nucleus("RB",77,37,0,76.9304,0,225,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.34451,100);

   // Adding 38-SR-77-0
   nuc = new Nucleus("SR",77,38,0,76.9378,0,9,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,3.71489,0.25);
   nuc->AddDecay(0,-1,0,6.85236,99.75);

   // Adding 29-CU-78-0
   nuc = new Nucleus("CU",78,29,0,77.9518,0,0.342,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.3619,100);

   // Adding 30-ZN-78-0
   nuc = new Nucleus("ZN",78,30,0,77.9386,0,1.47,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.44006,100);

   // Adding 31-GA-78-0
   nuc = new Nucleus("GA",78,31,0,77.9317,0,5.09,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.19996,100);

   // Adding 32-GE-78-0
   nuc = new Nucleus("GE",78,32,0,77.9229,0,5280,0,1.2e-10,1.4e-10,0,0);
   nuc->AddDecay(0,1,0,0.954163,100);

   // Adding 33-AS-78-0
   nuc = new Nucleus("AS",78,33,0,77.9218,0,5442,0,2.1e-10,1.4e-10,0,0);
   nuc->AddDecay(0,1,0,4.20941,100);

   // Adding 34-SE-78-0
   nuc = new Nucleus("SE",78,34,0,77.9173,0,0,23.6,0,0,0,0);

   // Adding 35-BR-78-0
   nuc = new Nucleus("BR",78,35,0,77.9211,0,387.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.57368,99.99);
   nuc->AddDecay(0,1,0,0.706429,0.01);

   // Adding 36-KR-78-0
   nuc = new Nucleus("KR",78,36,0,77.9204,0,0,0.35,0,0,0,0);

   // Adding 37-RB-78-0
   nuc = new Nucleus("RB",78,37,0,77.9281,0,1059.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.22439,100);

   // Adding 37-RB-78-1
   nuc = new Nucleus("RB",78,37,1,77.9283,0.103,344.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.103,10);
   nuc->AddDecay(0,-1,-1,7.32738,90);

   // Adding 38-SR-78-0
   nuc = new Nucleus("SR",78,38,0,77.9322,0,150,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.7622,100);

   // Adding 29-CU-79-0
   nuc = new Nucleus("CU",79,29,0,78.9541,0,0.188,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.6884,45);
   nuc->AddDecay(-1,1,0,6.44071,55);

   // Adding 30-ZN-79-0
   nuc = new Nucleus("ZN",79,30,0,78.9427,0,0.995,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.08954,98.7);
   nuc->AddDecay(-1,1,0,2.19241,1.3);

   // Adding 31-GA-79-0
   nuc = new Nucleus("GA",79,31,0,78.9329,0,2.847,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.30283,0.09);
   nuc->AddDecay(0,1,0,7.00008,94.71);
   nuc->AddDecay(0,1,1,6.81408,5.2);

   // Adding 32-GE-79-0
   nuc = new Nucleus("GE",79,32,0,78.9254,0,18.98,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.14796,100);

   // Adding 32-GE-79-1
   nuc = new Nucleus("GE",79,32,1,78.9256,0.186,39,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.186,4);
   nuc->AddDecay(0,1,-1,4.33396,96);

   // Adding 33-AS-79-0
   nuc = new Nucleus("AS",79,33,0,78.921,0,540.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.28126,1.06);
   nuc->AddDecay(0,1,1,2.18526,98.94);

   // Adding 34-SE-79-0
   nuc = new Nucleus("SE",79,34,0,78.9185,0,2.04984e+13,0,2.9e-09,3.1e-09,0,-4);
   nuc->AddDecay(0,1,0,0.150703,100);

   // Adding 34-SE-79-1
   nuc = new Nucleus("SE",79,34,1,78.9186,0.096,235.2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.096,99.94);
   nuc->AddDecay(0,1,-1,0.246704,0.06);

   // Adding 35-BR-79-0
   nuc = new Nucleus("BR",79,35,0,78.9183,0,0,50.69,0,0,0,0);

   // Adding 35-BR-79-1
   nuc = new Nucleus("BR",79,35,1,78.9186,0.208,4.86,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.208,100);

   // Adding 36-KR-79-0
   nuc = new Nucleus("KR",79,36,0,78.9201,0,126144,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.62578,100);

   // Adding 36-KR-79-1
   nuc = new Nucleus("KR",79,36,1,78.9202,0.13,50,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.13,100);

   // Adding 37-RB-79-0
   nuc = new Nucleus("RB",79,37,0,78.924,0,1374,0,5e-11,3e-11,0,0);
   nuc->AddDecay(0,-1,0,3.64928,100);

   // Adding 38-SR-79-0
   nuc = new Nucleus("SR",79,38,0,78.9297,0,135,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.31818,100);

   // Adding 39-Y-79-0
   nuc = new Nucleus("Y",79,39,0,78.9374,0,14.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.12004,100);
   nuc->AddDecay(-1,-2,0,1.29033,0);

   // Adding 30-ZN-80-0
   nuc = new Nucleus("ZN",80,30,0,79.9444,0,0.545,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.63939,1);
   nuc->AddDecay(0,1,0,7.29039,99);

   // Adding 31-GA-80-0
   nuc = new Nucleus("GA",80,31,0,79.9366,0,1.697,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.38,99.21);
   nuc->AddDecay(-1,1,0,2.34908,0.79);

   // Adding 32-GE-80-0
   nuc = new Nucleus("GE",80,32,0,79.9254,0,29.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.6702,100);

   // Adding 33-AS-80-0
   nuc = new Nucleus("AS",80,33,0,79.9226,0,15.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.64149,100);

   // Adding 34-SE-80-0
   nuc = new Nucleus("SE",80,34,0,79.9165,0,0,49.7,0,0,0,0);

   // Adding 35-BR-80-0
   nuc = new Nucleus("BR",80,35,0,79.9185,0,1060.8,0,3.1e-11,1.7e-11,0,0);
   nuc->AddDecay(0,-1,0,1.8706,8.3);
   nuc->AddDecay(0,1,0,2.00432,91.7);

   // Adding 35-BR-80-1
   nuc = new Nucleus("BR",80,35,1,79.9186,0.086,15913.8,0,1.2e-10,1e-10,0,0);
   nuc->AddDecay(0,0,-1,0.086,100);

   // Adding 36-KR-80-0
   nuc = new Nucleus("KR",80,36,0,79.9164,0,0,2.25,0,0,0,0);

   // Adding 37-RB-80-0
   nuc = new Nucleus("RB",80,37,0,79.9225,0,34,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.72286,100);

   // Adding 38-SR-80-0
   nuc = new Nucleus("SR",80,38,0,79.9245,0,6378,0,3.5e-10,2.1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.868,100);

   // Adding 39-Y-80-0
   nuc = new Nucleus("Y",80,39,0,79.9343,0,35,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.13968,100);

   // Adding 40-ZR-80-0
   nuc = new Nucleus("ZR",80,40,0,79.9406,0,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,0,5.8221,0);

   // Adding 30-ZN-81-0
   nuc = new Nucleus("ZN",81,30,0,80.9505,0,0.29,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.86781,7.5);
   nuc->AddDecay(0,1,0,11.8541,92.5);

   // Adding 31-GA-81-0
   nuc = new Nucleus("GA",81,31,0,80.9378,0,1.221,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.3937,11.4);
   nuc->AddDecay(0,1,0,8.31999,47.1792);
   nuc->AddDecay(0,1,1,7.64099,41.4208);

   // Adding 32-GE-81-0
   nuc = new Nucleus("GE",81,32,0,80.9288,0,7.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.23004,100);

   // Adding 32-GE-81-1
   nuc = new Nucleus("GE",81,32,1,80.9296,0.679,7.6,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.90904,100);

   // Adding 33-AS-81-0
   nuc = new Nucleus("AS",81,33,0,80.9221,0,33.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.85637,98.7);
   nuc->AddDecay(0,1,1,3.75337,1.3);

   // Adding 34-SE-81-0
   nuc = new Nucleus("SE",81,34,0,80.918,0,1107,0,2.7e-11,2.4e-11,0,0);
   nuc->AddDecay(0,1,0,1.58511,100);

   // Adding 34-SE-81-1
   nuc = new Nucleus("SE",81,34,1,80.9181,0.103,3436.8,0,5.9e-11,6.8e-11,0,0);
   nuc->AddDecay(0,0,-1,0.103,99.95);
   nuc->AddDecay(0,1,-1,1.6881,0.05);

   // Adding 35-BR-81-0
   nuc = new Nucleus("BR",81,35,0,80.9163,0,0,49.31,0,0,0,0);

   // Adding 36-KR-81-0
   nuc = new Nucleus("KR",81,36,0,80.9166,0,7.22174e+12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.280701,100);

   // Adding 36-KR-81-1
   nuc = new Nucleus("KR",81,36,1,80.9168,0.191,13.1,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.191,100);
   nuc->AddDecay(0,-1,-1,0.471703,0.0025);

   // Adding 37-RB-81-0
   nuc = new Nucleus("RB",81,37,0,80.919,0,16473.6,0,5.4e-11,6.8e-11,0,0);
   nuc->AddDecay(0,-1,0,2.23797,100);

   // Adding 37-RB-81-1
   nuc = new Nucleus("RB",81,37,1,80.9191,0.086,1830,0,9.7e-12,1.3e-11,0,0);
   nuc->AddDecay(0,-1,-1,2.32397,2.4);
   nuc->AddDecay(0,0,-1,0.086,97.6);

   // Adding 38-SR-81-0
   nuc = new Nucleus("SR",81,38,0,80.9232,0,1338,0,7.8e-11,6.1e-11,0,0);
   nuc->AddDecay(0,-1,0,3.93158,100);

   // Adding 39-Y-81-0
   nuc = new Nucleus("Y",81,39,0,80.9291,0,72.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.51042,100);

   // Adding 40-ZR-81-0
   nuc = new Nucleus("ZR",81,40,0,80.9368,0,15,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.16003,100);
   nuc->AddDecay(-1,-2,0,4.15975,0);

   // Adding 31-GA-82-0
   nuc = new Nucleus("GA",82,31,0,81.9432,0,0.602,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.5923,80.2);
   nuc->AddDecay(-1,1,0,5.28492,19.8);

   // Adding 32-GE-82-0
   nuc = new Nucleus("GE",82,32,0,81.9296,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.70005,100);

   // Adding 33-AS-82-0
   nuc = new Nucleus("AS",82,33,0,81.9246,0,19.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.35456,100);

   // Adding 33-AS-82-1
   nuc = new Nucleus("AS",82,33,1,81.9246,0,13.6,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.35456,100);

   // Adding 34-SE-82-0
   nuc = new Nucleus("SE",82,34,0,81.9167,0,4.41504e+27,9.2,0,0,0,0);
   nuc->AddDecay(0,2,0,2.995,100);

   // Adding 35-BR-82-0
   nuc = new Nucleus("BR",82,35,0,81.9168,0,127080,0,5.4e-10,8.9e-10,0,0);
   nuc->AddDecay(0,1,0,3.0926,100);

   // Adding 35-BR-82-1
   nuc = new Nucleus("BR",82,35,1,81.9169,0.046,367.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.046,97.6);
   nuc->AddDecay(0,1,-1,3.1386,2.4);

   // Adding 36-KR-82-0
   nuc = new Nucleus("KR",82,36,0,81.9135,0,0,11.6,0,0,0,0);

   // Adding 37-RB-82-0
   nuc = new Nucleus("RB",82,37,0,81.9182,0,76.38,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.40136,100);

   // Adding 37-RB-82-1
   nuc = new Nucleus("RB",82,37,1,81.9183,0.08,23299.2,0,1.3e-10,2.2e-10,0,0);
   nuc->AddDecay(0,-1,-1,4.48136,99.67);
   nuc->AddDecay(0,0,-1,0.08,0.33);

   // Adding 38-SR-82-0
   nuc = new Nucleus("SR",82,38,0,81.9184,0,2.20752e+06,0,6.1e-09,1e-08,0,0);
   nuc->AddDecay(0,-1,0,0.180305,100);

   // Adding 39-Y-82-0
   nuc = new Nucleus("Y",82,39,0,81.9268,0,9.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.81605,100);

   // Adding 40-ZR-82-0
   nuc = new Nucleus("ZR",82,40,0,81.9311,0,32,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.99996,100);

   // Adding 31-GA-83-0
   nuc = new Nucleus("GA",83,31,0,82.9469,0,0.31,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.5138,46);
   nuc->AddDecay(-1,1,0,7.97681,54);

   // Adding 32-GE-83-0
   nuc = new Nucleus("GE",83,32,0,82.9345,0,1.85,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.87552,100);

   // Adding 33-AS-83-0
   nuc = new Nucleus("AS",83,33,0,82.925,0,13.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.46001,30);
   nuc->AddDecay(0,1,1,5.23202,70);

   // Adding 34-SE-83-0
   nuc = new Nucleus("SE",83,34,0,82.9191,0,1338,0,5.2e-11,5.3e-11,0,0);
   nuc->AddDecay(0,1,0,3.66851,100);

   // Adding 34-SE-83-1
   nuc = new Nucleus("SE",83,34,1,82.9194,0.228,70.1,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.89651,100);

   // Adding 35-BR-83-0
   nuc = new Nucleus("BR",83,35,0,82.9152,0,8640,0,4.3e-11,6.7e-11,0,0);
   nuc->AddDecay(0,1,0,0.97229,0.025);
   nuc->AddDecay(0,1,1,0.93029,99.975);

   // Adding 36-KR-83-0
   nuc = new Nucleus("KR",83,36,0,82.9141,0,0,11.5,0,0,0,0);

   // Adding 36-KR-83-1
   nuc = new Nucleus("KR",83,36,1,82.9142,0.042,6588,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.042,100);

   // Adding 37-RB-83-0
   nuc = new Nucleus("RB",83,37,0,82.9151,0,7.44768e+06,0,1.9e-09,1e-09,0,0);
   nuc->AddDecay(0,-1,0,0.909035,100);

   // Adding 37-RB-83-1
   nuc = new Nucleus("RB",83,37,1,82.9156,0.426,0.0078,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.426,100);

   // Adding 38-SR-83-0
   nuc = new Nucleus("SR",83,38,0,82.9176,0,116676,0,5.8e-10,4.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.27621,100);

   // Adding 38-SR-83-1
   nuc = new Nucleus("SR",83,38,1,82.9178,0.259,4.95,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.259,100);

   // Adding 39-Y-83-0
   nuc = new Nucleus("Y",83,39,0,82.9223,0,424.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.46605,100);

   // Adding 39-Y-83-1
   nuc = new Nucleus("Y",83,39,1,82.9224,0.062,171,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.062,40);
   nuc->AddDecay(0,-1,-1,4.52805,60);

   // Adding 40-ZR-83-0
   nuc = new Nucleus("ZR",83,40,0,82.9286,0,44,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.86804,94);
   nuc->AddDecay(-1,-2,0,2.25627,6);

   // Adding 41-NB-83-0
   nuc = new Nucleus("NB",83,41,0,82.9367,0,4.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.50006,100);

   // Adding 31-GA-84-0
   nuc = new Nucleus("GA",84,31,0,83.9523,0,0.085,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.9954,30);
   nuc->AddDecay(-1,1,0,8.533,70);

   // Adding 32-GE-84-0
   nuc = new Nucleus("GE",84,32,0,83.9373,0,1.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.6849,100);

   // Adding 33-AS-84-0
   nuc = new Nucleus("AS",84,33,0,83.9291,0,5.5,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.18824,0.08);
   nuc->AddDecay(0,1,0,9.86922,99.92);

   // Adding 33-AS-84-1
   nuc = new Nucleus("AS",84,33,1,83.9291,0,0.65,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,9.86922,100);

   // Adding 34-SE-84-0
   nuc = new Nucleus("SE",84,34,0,83.9185,0,186,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.82581,100);

   // Adding 35-BR-84-0
   nuc = new Nucleus("BR",84,35,0,83.9165,0,1908,0,8.8e-11,6.2e-11,0,0);
   nuc->AddDecay(0,1,0,4.65441,100);

   // Adding 35-BR-84-1
   nuc = new Nucleus("BR",84,35,1,83.9168,0.32,360,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.97441,100);

   // Adding 36-KR-84-0
   nuc = new Nucleus("KR",84,36,0,83.9115,0,0,57,0,0,0,0);

   // Adding 37-RB-84-0
   nuc = new Nucleus("RB",84,37,0,83.9144,0,2.83133e+06,0,2.8e-09,1.6e-09,0,0);
   nuc->AddDecay(0,-1,0,2.6813,96.2);
   nuc->AddDecay(0,1,0,0.894302,3.8);

   // Adding 37-RB-84-1
   nuc = new Nucleus("RB",84,37,1,83.9149,0.464,1215.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.464,100);

   // Adding 38-SR-84-0
   nuc = new Nucleus("SR",84,38,0,83.9134,0,0,0.56,0,0,0,0);

   // Adding 39-Y-84-0
   nuc = new Nucleus("Y",84,39,0,83.9203,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.41001,100);

   // Adding 39-Y-84-1
   nuc = new Nucleus("Y",84,39,1,83.9208,0.5,2400,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.91001,100);

   // Adding 40-ZR-84-0
   nuc = new Nucleus("ZR",84,40,0,83.9232,0,1554,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.74043,100);

   // Adding 41-NB-84-0
   nuc = new Nucleus("NB",84,41,0,83.9336,0,12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.61289,100);
   nuc->AddDecay(-1,-2,0,3.16094,0);

   // Adding 32-GE-85-0
   nuc = new Nucleus("GE",85,32,0,84.9427,0,0.58,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.1394,86);
   nuc->AddDecay(-1,1,0,4.629,14);

   // Adding 33-AS-85-0
   nuc = new Nucleus("AS",85,33,0,84.9318,0,2.028,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.90643,77);
   nuc->AddDecay(-1,1,0,4.35882,23);

   // Adding 34-SE-85-0
   nuc = new Nucleus("SE",85,34,0,84.9222,0,31.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.18199,100);

   // Adding 35-BR-85-0
   nuc = new Nucleus("BR",85,35,0,84.9156,0,174,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.87001,0.270081);
   nuc->AddDecay(0,1,1,2.56501,99.7299);

   // Adding 36-KR-85-0
   nuc = new Nucleus("KR",85,36,0,84.9125,0,3.39201e+08,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.686996,100);

   // Adding 36-KR-85-1
   nuc = new Nucleus("KR",85,36,1,84.9129,0.305,16128,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,0.991997,78.6);
   nuc->AddDecay(0,0,-1,0.305,21.4);

   // Adding 37-RB-85-0
   nuc = new Nucleus("RB",85,37,0,84.9118,0,0,72.165,0,0,0,0);

   // Adding 38-SR-85-0
   nuc = new Nucleus("SR",85,38,0,84.9129,0,5.60218e+06,0,5.6e-10,7.7e-10,0,0);
   nuc->AddDecay(0,-1,0,1.06479,100);

   // Adding 38-SR-85-1
   nuc = new Nucleus("SR",85,38,1,84.9132,0.239,4057.8,0,6.1e-12,7.4e-12,0,0);
   nuc->AddDecay(0,-1,-1,1.30379,13.4);
   nuc->AddDecay(0,0,-1,0.239,86.6);

   // Adding 39-Y-85-0
   nuc = new Nucleus("Y",85,39,0,84.9164,0,9648,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.25461,100);

   // Adding 39-Y-85-1
   nuc = new Nucleus("Y",85,39,1,84.9165,0.02,17496,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.02,0.002);
   nuc->AddDecay(0,-1,-1,3.2746,100);

   // Adding 40-ZR-85-0
   nuc = new Nucleus("ZR",85,40,0,84.9215,0,471.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.69302,100);

   // Adding 40-ZR-85-1
   nuc = new Nucleus("ZR",85,40,1,84.9218,0.292,10.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.98502,8);
   nuc->AddDecay(0,0,-1,0.292,92);

   // Adding 41-NB-85-0
   nuc = new Nucleus("NB",85,41,0,84.9279,0,20.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.99999,100);

   // Adding 33-AS-86-0
   nuc = new Nucleus("AS",86,33,0,85.9362,0,0.9,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.95302,12);
   nuc->AddDecay(0,1,0,11.135,88);

   // Adding 34-SE-86-0
   nuc = new Nucleus("SE",86,34,0,85.9243,0,15.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.099,100);

   // Adding 35-BR-86-0
   nuc = new Nucleus("BR",86,35,0,85.9188,0,55.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.62604,100);

   // Adding 36-KR-86-0
   nuc = new Nucleus("KR",86,36,0,85.9106,0,0,17.3,0,0,0,0);

   // Adding 37-RB-86-0
   nuc = new Nucleus("RB",86,37,0,85.9112,0,1.60972e+06,0,2.8e-09,1.3e-09,0,0);
   nuc->AddDecay(0,-1,0,0.516823,0.0052);
   nuc->AddDecay(0,1,0,1.7747,99.99);

   // Adding 37-RB-86-1
   nuc = new Nucleus("RB",86,37,1,85.9118,0.556,61.02,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.556,100);

   // Adding 38-SR-86-0
   nuc = new Nucleus("SR",86,38,0,85.9093,0,0,9.86,0,0,0,0);

   // Adding 39-Y-86-0
   nuc = new Nucleus("Y",86,39,0,85.9149,0,53064,0,9.6e-10,8.2e-10,0,0);
   nuc->AddDecay(0,-1,0,5.24001,100);

   // Adding 39-Y-86-1
   nuc = new Nucleus("Y",86,39,1,85.9151,0.218,2880,0,5.7e-11,4.9e-11,0,0);
   nuc->AddDecay(0,0,-1,0.218,99.31);
   nuc->AddDecay(0,-1,-1,5.45802,0.69);

   // Adding 40-ZR-86-0
   nuc = new Nucleus("ZR",86,40,0,85.9165,0,59400,0,8.6e-10,7e-10,0,0);
   nuc->AddDecay(0,-1,0,1.47298,100);

   // Adding 41-NB-86-0
   nuc = new Nucleus("NB",86,41,0,85.925,0,88,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.97794,100);

   // Adding 42-MO-86-0
   nuc = new Nucleus("MO",86,42,0,85.9302,0,19.6,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,4.81019,100);

   // Adding 33-AS-87-0
   nuc = new Nucleus("AS",87,33,0,86.9396,0,0.73,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.2965,56);
   nuc->AddDecay(-1,1,0,6.18362,44);

   // Adding 34-SE-87-0
   nuc = new Nucleus("SE",87,34,0,86.9285,0,5.85,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.986073,0.18);
   nuc->AddDecay(0,1,0,7.27498,99.82);

   // Adding 35-BR-87-0
   nuc = new Nucleus("BR",87,35,0,86.9207,0,55.6,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.33713,2.57);
   nuc->AddDecay(0,1,0,6.85253,97.43);

   // Adding 36-KR-87-0
   nuc = new Nucleus("KR",87,36,0,86.9134,0,4578,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.88728,100);

   // Adding 37-RB-87-0
   nuc = new Nucleus("RB",87,37,0,86.9092,0,1.49796e+18,27.835,1.5e-09,7.6e-10,0,0);
   nuc->AddDecay(0,1,0,0.283295,100);

   // Adding 38-SR-87-0
   nuc = new Nucleus("SR",87,38,0,86.9089,0,0,7,0,0,0,0);

   // Adding 38-SR-87-1
   nuc = new Nucleus("SR",87,38,1,86.9093,0.389,10090.8,0,3.3e-11,3.5e-11,0,0);
   nuc->AddDecay(0,0,-1,0.389,99.7);
   nuc->AddDecay(0,-1,-1,0.105705,0.3);

   // Adding 39-Y-87-0
   nuc = new Nucleus("Y",87,39,0,86.9109,0,287280,0,5.5e-10,5.3e-10,0,0);
   nuc->AddDecay(0,-1,0,1.8616,100);

   // Adding 39-Y-87-1
   nuc = new Nucleus("Y",87,39,1,86.9113,0.381,48132,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.24259,1.57);
   nuc->AddDecay(0,0,-1,0.381,98.43);

   // Adding 40-ZR-87-0
   nuc = new Nucleus("ZR",87,40,0,86.9148,0,6048,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.66534,100);

   // Adding 40-ZR-87-1
   nuc = new Nucleus("ZR",87,40,1,86.9152,0.336,14,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.336,100);

   // Adding 41-NB-87-0
   nuc = new Nucleus("NB",87,41,0,86.9204,0,222,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.16902,100);

   // Adding 41-NB-87-1
   nuc = new Nucleus("NB",87,41,1,86.9204,0,156,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.16902,100);

   // Adding 42-MO-87-0
   nuc = new Nucleus("MO",87,42,0,86.9273,0,13.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.48634,100);
   nuc->AddDecay(-1,-2,0,2.8235,0);

   // Adding 34-SE-88-0
   nuc = new Nucleus("SE",88,34,0,87.9314,0,1.52,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.90807,0.94);
   nuc->AddDecay(0,1,0,6.85399,99.06);

   // Adding 35-BR-88-0
   nuc = new Nucleus("BR",88,35,0,87.9241,0,16.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.95998,93.6);
   nuc->AddDecay(-1,1,0,1.90661,6.4);

   // Adding 36-KR-88-0
   nuc = new Nucleus("KR",88,36,0,87.9145,0,10224,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.91393,100);

   // Adding 37-RB-88-0
   nuc = new Nucleus("RB",88,37,0,87.9113,0,1066.8,0,9e-11,2.8e-11,0,0);
   nuc->AddDecay(0,1,0,5.31589,100);

   // Adding 38-SR-88-0
   nuc = new Nucleus("SR",88,38,0,87.9056,0,0,82.58,0,0,0,0);

   // Adding 39-Y-88-0
   nuc = new Nucleus("Y",88,39,0,87.9095,0,9.21456e+06,0,1.3e-09,4.2e-09,0,0);
   nuc->AddDecay(0,-1,0,3.6226,100);

   // Adding 40-ZR-88-0
   nuc = new Nucleus("ZR",88,40,0,87.9102,0,7.20576e+06,0,3.3e-10,4.1e-09,0,0);
   nuc->AddDecay(0,-1,0,0.669823,100);

   // Adding 41-NB-88-0
   nuc = new Nucleus("NB",88,41,0,87.918,0,870,0,6.3e-11,5e-11,0,0);
   nuc->AddDecay(0,-1,0,7.19991,100);

   // Adding 41-NB-88-1
   nuc = new Nucleus("NB",88,41,1,87.918,0,468,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.19991,100);

   // Adding 42-MO-88-0
   nuc = new Nucleus("MO",88,42,0,87.922,0,480,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.72378,100);

   // Adding 43-TC-88-0
   nuc = new Nucleus("TC",88,43,0,87.9328,0,5.8,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.1331,100);

   // Adding 43-TC-88-1
   nuc = new Nucleus("TC",88,43,1,87.9328,0,6.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,10.1331,100);

   // Adding 34-SE-89-0
   nuc = new Nucleus("SE",89,34,0,88.936,0,0.41,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.05904,5);
   nuc->AddDecay(0,1,0,8.96516,95);

   // Adding 35-BR-89-0
   nuc = new Nucleus("BR",89,35,0,88.9264,0,4.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.15499,87);
   nuc->AddDecay(-1,1,0,3.05385,13);

   // Adding 36-KR-89-0
   nuc = new Nucleus("KR",89,36,0,88.9176,0,189,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.98592,100);

   // Adding 37-RB-89-0
   nuc = new Nucleus("RB",89,37,0,88.9123,0,909,0,4.7e-11,2.5e-11,0,0);
   nuc->AddDecay(0,1,0,4.50146,100);

   // Adding 38-SR-89-0
   nuc = new Nucleus("SR",89,38,0,88.9075,0,4.36579e+06,0,2.6e-09,7.5e-09,0,0);
   nuc->AddDecay(0,1,0,1.4966,99.9907);
   nuc->AddDecay(0,1,1,0.587601,0.00929997);

   // Adding 39-Y-89-0
   nuc = new Nucleus("Y",89,39,0,88.9058,0,0,100,0,0,0,0);

   // Adding 39-Y-89-1
   nuc = new Nucleus("Y",89,39,1,88.9068,0.909,16.06,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.909,100);

   // Adding 40-ZR-89-0
   nuc = new Nucleus("ZR",89,40,0,88.9089,0,282276,0,7.9e-10,7.5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.83229,0.13);
   nuc->AddDecay(0,-1,1,1.92329,99.87);

   // Adding 40-ZR-89-1
   nuc = new Nucleus("ZR",89,40,1,88.9095,0.588,250.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.42029,6.23);
   nuc->AddDecay(0,0,-1,0.588,93.77);

   // Adding 41-NB-89-0
   nuc = new Nucleus("NB",89,41,0,88.9135,0,6840,0,3e-10,1.9e-10,0,0);
   nuc->AddDecay(0,-1,0,4.28989,100);

   // Adding 41-NB-89-1
   nuc = new Nucleus("NB",89,41,1,88.9135,0,4248,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.28989,100);

   // Adding 42-MO-89-0
   nuc = new Nucleus("MO",89,42,0,88.9195,0,122.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.57512,100);

   // Adding 42-MO-89-1
   nuc = new Nucleus("MO",89,42,1,88.9199,0.387,0.19,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.387,100);

   // Adding 43-TC-89-0
   nuc = new Nucleus("TC",89,43,0,88.9275,0,12.8,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.50999,100);

   // Adding 43-TC-89-1
   nuc = new Nucleus("TC",89,43,1,88.9275,0,12.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.51,100);

   // Adding 35-BR-90-0
   nuc = new Nucleus("BR",90,35,0,89.9306,0,1.92,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.3499,75.4);
   nuc->AddDecay(-1,1,0,4.03667,24.6);

   // Adding 36-KR-90-0
   nuc = new Nucleus("KR",90,36,0,89.9195,0,32.32,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.39186,87.8);
   nuc->AddDecay(0,1,1,4.28486,12.2);

   // Adding 37-RB-90-0
   nuc = new Nucleus("RB",90,37,0,89.9148,0,158,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.58975,100);

   // Adding 37-RB-90-1
   nuc = new Nucleus("RB",90,37,1,89.9149,0.107,258,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.107,2.6);
   nuc->AddDecay(0,1,-1,6.69675,97.4);

   // Adding 38-SR-90-0
   nuc = new Nucleus("SR",90,38,0,89.9077,0,9.07606e+08,0,2.8e-08,1.5e-07,0,0);
   nuc->AddDecay(0,1,0,0.546196,100);

   // Adding 39-Y-90-0
   nuc = new Nucleus("Y",90,39,0,89.9072,0,230760,0,2.7e-09,1.7e-09,0,0);
   nuc->AddDecay(0,1,0,2.282,100);

   // Adding 39-Y-90-1
   nuc = new Nucleus("Y",90,39,1,89.9079,0.682,11484,0,1.7e-10,1.3e-10,0,0);
   nuc->AddDecay(0,0,-1,0.682,100);
   nuc->AddDecay(0,1,0,0.644997,0.0018);

   // Adding 40-ZR-90-0
   nuc = new Nucleus("ZR",90,40,0,89.9047,0,0,51.45,0,0,0,0);

   // Adding 40-ZR-90-1
   nuc = new Nucleus("ZR",90,40,1,89.9072,2.319,0.8092,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.319,100);

   // Adding 41-NB-90-0
   nuc = new Nucleus("NB",90,41,0,89.9113,0,52560,0,1.2e-09,1.1e-09,0,0);
   nuc->AddDecay(0,-1,0,6.11098,100);

   // Adding 41-NB-90-1
   nuc = new Nucleus("NB",90,41,1,89.9114,0.125,18.81,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.125,100);

   // Adding 42-MO-90-0
   nuc = new Nucleus("MO",90,42,0,89.9139,0,20412,0,6.2e-10,5.6e-10,0,0);
   nuc->AddDecay(0,-1,0,2.48899,100);

   // Adding 43-TC-90-0
   nuc = new Nucleus("TC",90,43,0,89.9237,0,8.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.14006,100);

   // Adding 43-TC-90-1
   nuc = new Nucleus("TC",90,43,1,89.9243,0.5,49.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,9.64006,100);

   // Adding 44-RU-90-0
   nuc = new Nucleus("RU",90,44,0,89.9298,0,13,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.6198,100);

   // Adding 34-SE-91-0
   nuc = new Nucleus("SE",91,34,0,90.9454,0,0.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.6625,79);
   nuc->AddDecay(-1,1,0,5.64958,21);

   // Adding 35-BR-91-0
   nuc = new Nucleus("BR",91,35,0,90.9339,0,0.541,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.80198,81.7);
   nuc->AddDecay(-1,1,0,5.33704,18.3);

   // Adding 36-KR-91-0
   nuc = new Nucleus("KR",91,36,0,90.9234,0,8.57,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.43502,100);

   // Adding 37-RB-91-0
   nuc = new Nucleus("RB",91,37,0,90.9165,0,58.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.86109,100);

   // Adding 38-SR-91-0
   nuc = new Nucleus("SR",91,38,0,90.9102,0,34668,0,7.6e-10,5.7e-10,0,0);
   nuc->AddDecay(0,1,0,2.69926,50);
   nuc->AddDecay(0,1,1,2.14326,50);

   // Adding 39-Y-91-0
   nuc = new Nucleus("Y",91,39,0,90.9073,0,5.05526e+06,0,2.4e-09,8.4e-09,0,0);
   nuc->AddDecay(0,1,0,1.5441,100);

   // Adding 39-Y-91-1
   nuc = new Nucleus("Y",91,39,1,90.9079,0.556,2982.6,0,1.2e-11,1.5e-11,0,0);
   nuc->AddDecay(0,1,-1,2.1001,1.5);
   nuc->AddDecay(0,0,-1,0.556,98.5);

   // Adding 40-ZR-91-0
   nuc = new Nucleus("ZR",91,40,0,90.9056,0,0,11.22,0,0,0,0);

   // Adding 41-NB-91-0
   nuc = new Nucleus("NB",91,41,0,90.907,0,2.14445e+10,0,6.4e-11,4.1e-09,0,0);
   nuc->AddDecay(0,-1,0,1.2534,100);

   // Adding 41-NB-91-1
   nuc = new Nucleus("NB",91,41,1,90.9071,0.104,5.2583e+06,0,6.3e-10,2.3e-09,0,0);
   nuc->AddDecay(0,0,-1,0.104,93);
   nuc->AddDecay(0,-1,-1,1.35739,7);

   // Adding 42-MO-91-0
   nuc = new Nucleus("MO",91,42,0,90.9118,0,929.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.43382,100);

   // Adding 42-MO-91-1
   nuc = new Nucleus("MO",91,42,1,90.9125,0.653,65,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.653,50.1);
   nuc->AddDecay(0,-1,-1,5.08681,49.9);

   // Adding 43-TC-91-0
   nuc = new Nucleus("TC",91,43,0,90.9184,0,188.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.21999,100);

   // Adding 43-TC-91-1
   nuc = new Nucleus("TC",91,43,1,90.9188,0.35,198,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.56999,99);
   nuc->AddDecay(0,0,-1,0.35,1);

   // Adding 44-RU-91-0
   nuc = new Nucleus("RU",91,44,0,90.9264,0,9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.40507,100);

   // Adding 44-RU-91-1
   nuc = new Nucleus("RU",91,44,1,90.9264,0,7.6,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0,33.3333);
   nuc->AddDecay(0,-1,-1,7.40507,33.3333);
   nuc->AddDecay(-1,-2,-1,4.3,33.3333);

   // Adding 35-BR-92-0
   nuc = new Nucleus("BR",92,35,0,91.9392,0,0.343,0,0,0,0,0);
   nuc->AddDecay(0,1,0,12.205,67);
   nuc->AddDecay(-1,1,0,6.65931,33);

   // Adding 36-KR-92-0
   nuc = new Nucleus("KR",92,36,0,91.9261,0,1.84,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.98699,99.97);
   nuc->AddDecay(-1,1,0,0.889363,0.03);

   // Adding 37-RB-92-0
   nuc = new Nucleus("RB",92,37,0,91.9197,0,4.51,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.763463,0.0099);
   nuc->AddDecay(0,1,0,8.1052,99.9901);

   // Adding 38-SR-92-0
   nuc = new Nucleus("SR",92,38,0,91.911,0,9756,0,4.9e-10,3.4e-10,0,0);
   nuc->AddDecay(0,1,0,1.91109,100);

   // Adding 39-Y-92-0
   nuc = new Nucleus("Y",92,39,0,91.9089,0,12744,0,4.9e-10,2.8e-10,0,0);
   nuc->AddDecay(0,1,0,3.62523,100);

   // Adding 40-ZR-92-0
   nuc = new Nucleus("ZR",92,40,0,91.905,0,0,17.15,0,0,0,0);

   // Adding 41-NB-92-0
   nuc = new Nucleus("NB",92,41,0,91.9072,0,1.0943e+15,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.35611,0.05);
   nuc->AddDecay(0,-1,0,2.0057,99.95);

   // Adding 41-NB-92-1
   nuc = new Nucleus("NB",92,41,1,91.9073,0.135,876960,0,6e-10,5.9e-10,0,0);
   nuc->AddDecay(0,-1,-1,2.1407,100);

   // Adding 42-MO-92-0
   nuc = new Nucleus("MO",92,42,0,91.9068,0,0,14.84,0,0,0,0);

   // Adding 43-TC-92-0
   nuc = new Nucleus("TC",92,43,0,91.9153,0,253.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.87018,100);

   // Adding 44-RU-92-0
   nuc = new Nucleus("RU",92,44,0,91.9201,0,219,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.52796,100);

   // Adding 45-RH-92-0
   nuc = new Nucleus("RH",92,45,0,91.932,0,0,0,0,0,0,-2);

   // Adding 35-BR-93-0
   nuc = new Nucleus("BR",93,35,0,92.9431,0,0.102,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.0997,23);
   nuc->AddDecay(-1,1,0,7.75392,77);

   // Adding 36-KR-93-0
   nuc = new Nucleus("KR",93,36,0,92.9312,0,1.286,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.6,97.99);
   nuc->AddDecay(-1,1,0,2.64119,2.01);

   // Adding 37-RB-93-0
   nuc = new Nucleus("RB",93,37,0,92.922,0,5.84,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.1464,1.35);
   nuc->AddDecay(0,1,0,7.4599,98.65);

   // Adding 38-SR-93-0
   nuc = new Nucleus("SR",93,38,0,92.9139,0,445.38,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.083,64.4);
   nuc->AddDecay(0,1,1,3.324,35.6);

   // Adding 39-Y-93-0
   nuc = new Nucleus("Y",93,39,0,92.9096,0,36648,0,1.2e-09,6e-10,0,0);
   nuc->AddDecay(0,1,0,2.87391,100);

   // Adding 39-Y-93-1
   nuc = new Nucleus("Y",93,39,1,92.9104,0.759,0.82,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.759,100);

   // Adding 40-ZR-93-0
   nuc = new Nucleus("ZR",93,40,0,92.9065,0,4.82501e+13,0,2.9e-10,2.9e-08,0,0);
   nuc->AddDecay(0,1,1,0.0602036,100);

   // Adding 41-NB-93-0
   nuc = new Nucleus("NB",93,41,0,92.9064,0,0,100,0,0,0,0);

   // Adding 41-NB-93-1
   nuc = new Nucleus("NB",93,41,1,92.9064,0.031,5.08676e+08,0,1.2e-10,1.6e-09,0,0);
   nuc->AddDecay(0,0,-1,0.031,100);

   // Adding 42-MO-93-0
   nuc = new Nucleus("MO",93,42,0,92.9068,0,1.26144e+11,0,2.6e-09,2.2e-09,0,0);
   nuc->AddDecay(0,-1,1,0.37429,96);
   nuc->AddDecay(0,-1,0,0.405289,4);

   // Adding 42-MO-93-1
   nuc = new Nucleus("MO",93,42,1,92.9094,2.425,24660,0,2.8e-10,3e-10,0,0);
   nuc->AddDecay(0,0,-1,2.425,99.88);
   nuc->AddDecay(0,-1,-1,2.83029,0.12);

   // Adding 43-TC-93-0
   nuc = new Nucleus("TC",93,43,0,92.9102,0,9900,0,4.9e-11,6.6e-11,0,0);
   nuc->AddDecay(0,-1,0,3.2008,100);

   // Adding 43-TC-93-1
   nuc = new Nucleus("TC",93,43,1,92.9107,0.392,2610,0,2.4e-11,3.1e-11,0,0);
   nuc->AddDecay(0,-1,-1,3.5928,23.3);
   nuc->AddDecay(0,0,-1,0.392,76.7);

   // Adding 44-RU-93-0
   nuc = new Nucleus("RU",93,44,0,92.9171,0,59.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.33695,100);

   // Adding 44-RU-93-1
   nuc = new Nucleus("RU",93,44,1,92.9178,0.734,10.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.07095,77.99);
   nuc->AddDecay(0,0,-1,0.734,22);
   nuc->AddDecay(-1,-2,-1,2.98445,0.01);

   // Adding 45-RH-93-0
   nuc = new Nucleus("RH",93,45,0,92.9257,0,0,0,0,0,0,-2);

   // Adding 46-PD-93-0
   nuc = new Nucleus("PD",93,46,0,92.9366,0,60,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.1231,100);

   // Adding 35-BR-94-0
   nuc = new Nucleus("BR",94,35,0,93.9476,0,0.07,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,7.20701,30);
   nuc->AddDecay(0,1,0,12.3993,70);

   // Adding 36-KR-94-0
   nuc = new Nucleus("KR",94,36,0,93.9343,0,0.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.30731,94.3);
   nuc->AddDecay(-1,1,0,3.40771,5.7);

   // Adding 37-RB-94-0
   nuc = new Nucleus("RB",94,37,0,93.9264,0,2.702,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.3068,89.6);
   nuc->AddDecay(-1,1,0,3.5603,10.4);

   // Adding 38-SR-94-0
   nuc = new Nucleus("SR",94,38,0,93.9154,0,75.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.51128,100);

   // Adding 39-Y-94-0
   nuc = new Nucleus("Y",94,39,0,93.9116,0,1122,0,8.1e-11,4.6e-11,0,0);
   nuc->AddDecay(0,1,0,4.91927,100);

   // Adding 40-ZR-94-0
   nuc = new Nucleus("ZR",94,40,0,93.9063,0,0,17.38,0,0,0,0);

   // Adding 41-NB-94-0
   nuc = new Nucleus("NB",94,41,0,93.9073,0,6.40181e+11,0,1.7e-09,4.5e-08,0,0);
   nuc->AddDecay(0,1,0,2.0451,100);

   // Adding 41-NB-94-1
   nuc = new Nucleus("NB",94,41,1,93.9073,0.041,375.78,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.041,99.5);
   nuc->AddDecay(0,1,-1,2.0861,0.5);

   // Adding 42-MO-94-0
   nuc = new Nucleus("MO",94,42,0,93.9051,0,0,9.25,0,0,0,0);

   // Adding 43-TC-94-0
   nuc = new Nucleus("TC",94,43,0,93.9097,0,17580,0,1.8e-10,2.2e-10,0,0);
   nuc->AddDecay(0,-1,0,4.25578,100);

   // Adding 43-TC-94-1
   nuc = new Nucleus("TC",94,43,1,93.9097,0.075,3120,0,1.1e-10,8e-11,0,0);
   nuc->AddDecay(0,-1,-1,4.33077,99.9);
   nuc->AddDecay(0,0,-1,0.075,0.1);

   // Adding 44-RU-94-0
   nuc = new Nucleus("RU",94,44,0,93.9114,0,3108,0,9.4e-11,7.4e-11,0,0);
   nuc->AddDecay(0,-1,0,1.59293,100);

   // Adding 45-RH-94-0
   nuc = new Nucleus("RH",94,45,0,93.9217,0,70.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.62907,100);

   // Adding 45-RH-94-1
   nuc = new Nucleus("RH",94,45,1,93.9217,0,25.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,9.62907,100);

   // Adding 46-PD-94-0
   nuc = new Nucleus("PD",94,46,0,93.9288,0,9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.58295,100);

   // Adding 36-KR-95-0
   nuc = new Nucleus("KR",95,36,0,94.9397,0,0.78,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.72152,100);

   // Adding 37-RB-95-0
   nuc = new Nucleus("RB",95,37,0,94.9293,0,0.377,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.90286,8.62);
   nuc->AddDecay(0,1,0,9.296,91.38);

   // Adding 38-SR-95-0
   nuc = new Nucleus("SR",95,38,0,94.9193,0,23.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.08028,100);

   // Adding 39-Y-95-0
   nuc = new Nucleus("Y",95,39,0,94.9128,0,618,0,4.6e-11,2.6e-11,0,0);
   nuc->AddDecay(0,1,0,4.41972,100);

   // Adding 40-ZR-95-0
   nuc = new Nucleus("ZR",95,40,0,94.908,0,5.53133e+06,0,8.8e-10,5.5e-09,0,0);
   nuc->AddDecay(0,1,0,1.1244,98.89);
   nuc->AddDecay(0,1,1,0.888397,1.11);

   // Adding 41-NB-95-0
   nuc = new Nucleus("NB",95,41,0,94.9068,0,3.02184e+06,0,5.9e-10,1.6e-09,0,0);
   nuc->AddDecay(0,1,0,0.925598,100);

   // Adding 41-NB-95-1
   nuc = new Nucleus("NB",95,41,1,94.9071,0.236,311760,0,5.7e-10,8.5e-10,0,0);
   nuc->AddDecay(0,1,-1,1.1616,5.6);
   nuc->AddDecay(0,0,-1,0.236,94.4);

   // Adding 42-MO-95-0
   nuc = new Nucleus("MO",95,42,0,94.9058,0,0,15.92,0,0,0,0);

   // Adding 43-TC-95-0
   nuc = new Nucleus("TC",95,43,0,94.9077,0,72000,0,1.6e-10,1.8e-10,0,0);
   nuc->AddDecay(0,-1,0,1.69136,100);

   // Adding 43-TC-95-1
   nuc = new Nucleus("TC",95,43,1,94.9077,0.039,5.2704e+06,0,6.2e-10,8.8e-10,0,0);
   nuc->AddDecay(0,0,-1,0.039,3.88);
   nuc->AddDecay(0,-1,-1,1.73036,96.12);

   // Adding 44-RU-95-0
   nuc = new Nucleus("RU",95,44,0,94.9104,0,5914.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.57244,100);

   // Adding 45-RH-95-0
   nuc = new Nucleus("RH",95,45,0,94.9159,0,301.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.11,100);

   // Adding 45-RH-95-1
   nuc = new Nucleus("RH",95,45,1,94.9165,0.543,117.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.543,88);
   nuc->AddDecay(0,-1,-1,5.653,12);

   // Adding 46-PD-95-0
   nuc = new Nucleus("PD",95,46,0,94.9247,0,0,0,0,0,0,-2);

   // Adding 46-PD-95-1
   nuc = new Nucleus("PD",95,46,1,94.9268,2,13.3,0,0,0,0,0);
   nuc->AddDecay(-1,-2,-1,7.12212,0.815094);
   nuc->AddDecay(0,-1,-1,10.1837,90.4);
   nuc->AddDecay(0,0,-1,2,7.87924);

   // Adding 37-RB-96-0
   nuc = new Nucleus("RB",96,37,0,95.9343,0,0.2028,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,5.86069,14);
   nuc->AddDecay(0,1,0,11.7565,86);

   // Adding 38-SR-96-0
   nuc = new Nucleus("SR",96,38,0,95.9216,0,1.07,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.3712,100);

   // Adding 39-Y-96-0
   nuc = new Nucleus("Y",96,39,0,95.9159,0,5.34,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.08691,100);

   // Adding 39-Y-96-1
   nuc = new Nucleus("Y",96,39,1,95.9159,0,9.6,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.08691,100);

   // Adding 40-ZR-96-0
   nuc = new Nucleus("ZR",96,40,0,95.9083,0,0,2.8,0,0,0,0);

   // Adding 41-NB-96-0
   nuc = new Nucleus("NB",96,41,0,95.9081,0,84060,0,1.1e-09,1e-09,0,0);
   nuc->AddDecay(0,1,0,3.18678,100);

   // Adding 42-MO-96-0
   nuc = new Nucleus("MO",96,42,0,95.9047,0,0,16.68,0,0,0,0);

   // Adding 43-TC-96-0
   nuc = new Nucleus("TC",96,43,0,95.9079,0,369792,0,1.1e-09,1.1e-09,0,0);
   nuc->AddDecay(0,-1,0,2.97326,100);

   // Adding 43-TC-96-1
   nuc = new Nucleus("TC",96,43,1,95.9079,0.034,3090,0,1.3e-11,1.1e-11,0,0);
   nuc->AddDecay(0,0,-1,0.034,98);
   nuc->AddDecay(0,-1,-1,3.00726,2);

   // Adding 44-RU-96-0
   nuc = new Nucleus("RU",96,44,0,95.9076,0,0,5.52,0,0,0,0);

   // Adding 45-RH-96-0
   nuc = new Nucleus("RH",96,45,0,95.9145,0,594,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.44646,100);

   // Adding 45-RH-96-1
   nuc = new Nucleus("RH",96,45,1,95.9146,0.052,90.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.052,60);
   nuc->AddDecay(0,-1,-1,6.49847,40);

   // Adding 46-PD-96-0
   nuc = new Nucleus("PD",96,46,0,95.9182,0,122,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.45,100);

   // Adding 47-AG-96-0
   nuc = new Nucleus("AG",96,47,0,95.9307,0,5.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.5988,92);
   nuc->AddDecay(-1,-2,0,6.47461,8);

   // Adding 36-KR-97-0
   nuc = new Nucleus("KR",97,36,0,97,0,0.1,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 37-RB-97-0
   nuc = new Nucleus("RB",97,37,0,96.9373,0,0.1699,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.42,74.9);
   nuc->AddDecay(-1,1,0,6.53689,25.1);

   // Adding 38-SR-97-0
   nuc = new Nucleus("SR",97,38,0,96.9261,0,0.429,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.46679,84.95);
   nuc->AddDecay(-1,1,0,1.4881,0.05);
   nuc->AddDecay(0,1,1,6.79879,15);

   // Adding 39-Y-97-0
   nuc = new Nucleus("Y",97,39,0,96.9181,0,3.75,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.10822,0.06);
   nuc->AddDecay(0,1,0,6.68782,99.94);

   // Adding 39-Y-97-1
   nuc = new Nucleus("Y",97,39,1,96.9188,0.668,1.17,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.668,0.7);
   nuc->AddDecay(-1,1,-1,1.77622,0.08);
   nuc->AddDecay(0,1,-1,7.35582,99.22);

   // Adding 39-Y-97-2
   nuc = new Nucleus("Y",97,39,2,96.9219,3.523,0.142,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,3.523,80);
   nuc->AddDecay(0,1,-2,10.2108,20);

   // Adding 40-ZR-97-0
   nuc = new Nucleus("ZR",97,40,0,96.9109,0,60840,0,2.1e-09,1.4e-09,0,0);
   nuc->AddDecay(0,1,0,2.658,5.32);
   nuc->AddDecay(0,1,1,1.915,94.68);

   // Adding 41-NB-97-0
   nuc = new Nucleus("NB",97,41,0,96.9081,0,4326,0,6.9e-11,7.2e-11,0,0);
   nuc->AddDecay(0,1,0,1.93389,100);

   // Adding 41-NB-97-1
   nuc = new Nucleus("NB",97,41,1,96.9089,0.743,52.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.743,100);

   // Adding 42-MO-97-0
   nuc = new Nucleus("MO",97,42,0,96.906,0,0,9.55,0,0,0,0);

   // Adding 43-TC-97-0
   nuc = new Nucleus("TC",97,43,0,96.9064,0,8.19936e+13,0,8.3e-11,2.1e-10,0,0);
   nuc->AddDecay(0,-1,0,0.320274,100);

   // Adding 43-TC-97-1
   nuc = new Nucleus("TC",97,43,1,96.9065,0.097,7.78464e+06,0,6.6e-10,3.1e-09,0,0);
   nuc->AddDecay(0,0,-1,0.097,99.66);
   nuc->AddDecay(0,-1,-1,0.417274,0.34);

   // Adding 44-RU-97-0
   nuc = new Nucleus("RU",97,44,0,96.9076,0,250560,0,1.5e-10,1.6e-10,0,0);
   nuc->AddDecay(0,-1,0,1.11446,99.962);
   nuc->AddDecay(0,-1,1,1.01746,0.038);

   // Adding 45-RH-97-0
   nuc = new Nucleus("RH",97,45,0,96.9113,0,1842,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.52305,100);

   // Adding 45-RH-97-1
   nuc = new Nucleus("RH",97,45,1,96.9116,0.259,2772,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.78205,94.4);
   nuc->AddDecay(0,0,-1,0.259,5.6);

   // Adding 46-PD-97-0
   nuc = new Nucleus("PD",97,46,0,96.9165,0,186,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.79001,100);

   // Adding 47-AG-97-0
   nuc = new Nucleus("AG",97,47,0,96.924,0,19,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.99953,100);

   // Adding 48-CD-97-0
   nuc = new Nucleus("CD",97,48,0,96.9352,0,3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.408,100);
   nuc->AddDecay(-1,-2,0,8.49492,0);

   // Adding 37-RB-98-0
   nuc = new Nucleus("RB",98,37,0,97.9417,0,0.114,0,0,0,0,0);
   nuc->AddDecay(-2,1,0,2.57498,0.05);
   nuc->AddDecay(0,1,0,12.344,86.35);
   nuc->AddDecay(-1,1,0,6.45809,13.6);

   // Adding 37-RB-98-1
   nuc = new Nucleus("RB",98,37,1,97.942,0.27,0.096,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,12.614,100);
   nuc->AddDecay(-1,1,-1,6.72809,0);

   // Adding 38-SR-98-0
   nuc = new Nucleus("SR",98,38,0,97.9285,0,0.653,0,0,0,0,0);
   nuc->AddDecay(0,1,1,5.826,49.865);
   nuc->AddDecay(-1,1,0,1.58088,0.23);
   nuc->AddDecay(0,1,0,5.826,49.905);

   // Adding 39-Y-98-0
   nuc = new Nucleus("Y",98,39,0,97.9222,0,0.548,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.8303,99.76);
   nuc->AddDecay(-1,1,0,2.44271,0.24);

   // Adding 39-Y-98-1
   nuc = new Nucleus("Y",98,39,1,97.9222,0,2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,10);
   nuc->AddDecay(-1,1,-1,2.44271,3.4);
   nuc->AddDecay(0,1,-1,8.8303,86.6);

   // Adding 40-ZR-98-0
   nuc = new Nucleus("ZR",98,40,0,97.9128,0,30.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.26124,100);

   // Adding 41-NB-98-0
   nuc = new Nucleus("NB",98,41,0,97.9103,0,2.86,0,1.2e-10,9.9e-11,0,0);
   nuc->AddDecay(0,1,0,4.58556,100);

   // Adding 41-NB-98-1
   nuc = new Nucleus("NB",98,41,1,97.9104,0.084,3078,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.66956,99.9);
   nuc->AddDecay(0,0,-1,0.084,0.100002);

   // Adding 42-MO-98-0
   nuc = new Nucleus("MO",98,42,0,97.9054,0,0,24.13,0,0,0,0);

   // Adding 43-TC-98-0
   nuc = new Nucleus("TC",98,43,0,97.9072,0,1.32451e+14,0,2.3e-09,8.1e-09,0,0);
   nuc->AddDecay(0,1,0,1.79582,100);

   // Adding 44-RU-98-0
   nuc = new Nucleus("RU",98,44,0,97.9053,0,0,1.88,0,0,0,0);

   // Adding 45-RH-98-0
   nuc = new Nucleus("RH",98,45,0,97.9107,0,522,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.05745,100);

   // Adding 45-RH-98-1
   nuc = new Nucleus("RH",98,45,1,97.9107,0,210,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0,50);
   nuc->AddDecay(0,-1,-1,5.05745,50);

   // Adding 46-PD-98-0
   nuc = new Nucleus("PD",98,46,0,97.9127,0,1062,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.87259,100);

   // Adding 47-AG-98-0
   nuc = new Nucleus("AG",98,47,0,97.9218,0,46.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.42001,100);

   // Adding 48-CD-98-0
   nuc = new Nucleus("CD",98,48,0,97.9276,0,9.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.41941,100);

   // Adding 37-RB-99-0
   nuc = new Nucleus("RB",99,37,0,98.9453,0,0.059,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.2471,85);
   nuc->AddDecay(-1,1,0,7.61231,15);

   // Adding 38-SR-99-0
   nuc = new Nucleus("SR",99,38,0,98.9333,0,0.271,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.03071,99.68);
   nuc->AddDecay(-1,1,0,2.19121,0.32);

   // Adding 39-Y-99-0
   nuc = new Nucleus("Y",99,39,0,98.9246,0,1.47,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.9908,0.96);
   nuc->AddDecay(0,1,0,7.5668,99.04);

   // Adding 40-ZR-99-0
   nuc = new Nucleus("ZR",99,40,0,98.9165,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.5575,62.5);
   nuc->AddDecay(0,1,1,4.1925,37.5);

   // Adding 41-NB-99-0
   nuc = new Nucleus("NB",99,41,0,98.9116,0,15,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.6386,100);

   // Adding 41-NB-99-1
   nuc = new Nucleus("NB",99,41,1,98.912,0.365,156,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.365,2.5);
   nuc->AddDecay(0,1,-1,4.0036,97.5);

   // Adding 42-MO-99-0
   nuc = new Nucleus("MO",99,42,0,98.9077,0,237384,0,1.2e-09,1.1e-09,0,0);
   nuc->AddDecay(0,1,0,1.3573,12.5);
   nuc->AddDecay(0,1,1,1.2143,87.5);

   // Adding 43-TC-99-0
   nuc = new Nucleus("TC",99,43,0,98.9062,0,6.65725e+12,0,7.8e-10,3.9e-09,0,0);
   nuc->AddDecay(0,1,0,0.293503,100);

   // Adding 43-TC-99-1
   nuc = new Nucleus("TC",99,43,1,98.9064,0.143,21636,0,2.2e-11,2.9e-11,0,0);
   nuc->AddDecay(0,1,-1,0.436501,0.004);
   nuc->AddDecay(0,0,-1,0.143,100);

   // Adding 44-RU-99-0
   nuc = new Nucleus("RU",99,44,0,98.9059,0,0,12.7,0,0,0,0);

   // Adding 45-RH-99-0
   nuc = new Nucleus("RH",99,45,0,98.9082,0,1.39104e+06,0,5.1e-10,8.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.10301,100);

   // Adding 45-RH-99-1
   nuc = new Nucleus("RH",99,45,1,98.9083,0.064,16920,0,6.6e-11,7.3e-11,0,0);
   nuc->AddDecay(0,-1,-1,2.16702,99.84);
   nuc->AddDecay(0,0,-1,0.064,0.16);

   // Adding 46-PD-99-0
   nuc = new Nucleus("PD",99,46,0,98.9118,0,1284,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.3653,100);

   // Adding 47-AG-99-0
   nuc = new Nucleus("AG",99,47,0,98.9176,0,124,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.43,100);

   // Adding 47-AG-99-1
   nuc = new Nucleus("AG",99,47,1,98.9182,0.506,10.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.506,100);

   // Adding 48-CD-99-0
   nuc = new Nucleus("CD",99,48,0,98.925,0,16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.86621,99.83);
   nuc->AddDecay(-1,-2,0,4.15242,0.17);
   nuc->AddDecay(-4,-3,0,6.05701,1e-06);

   // Adding 37-RB-100-0
   nuc = new Nucleus("RB",100,37,0,99.9499,0,0.051,0,0,0,0,0);
   nuc->AddDecay(0,1,0,13.5238,94);
   nuc->AddDecay(-1,1,0,7.40572,6);

   // Adding 38-SR-100-0
   nuc = new Nucleus("SR",100,38,0,99.9353,0,0.202,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.91261,0.73);
   nuc->AddDecay(0,1,0,7.07507,99.27);

   // Adding 39-Y-100-0
   nuc = new Nucleus("Y",100,39,0,99.9278,0,0.735,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.30995,99.19);
   nuc->AddDecay(-1,1,0,2.40434,0.81);

   // Adding 39-Y-100-1
   nuc = new Nucleus("Y",100,39,1,99.9278,0,0.94,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,9.30995,100);

   // Adding 40-ZR-100-0
   nuc = new Nucleus("ZR",100,40,0,99.9178,0,7.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.33499,100);

   // Adding 41-NB-100-0
   nuc = new Nucleus("NB",100,41,0,99.9142,0,1.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.24503,100);

   // Adding 41-NB-100-1
   nuc = new Nucleus("NB",100,41,1,99.9147,0.48,2.99,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.72504,100);

   // Adding 42-MO-100-0
   nuc = new Nucleus("MO",100,42,0,99.9075,0,0,9.63,0,0,0,0);

   // Adding 43-TC-100-0
   nuc = new Nucleus("TC",100,43,0,99.9077,0,15.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.2023,100);

   // Adding 44-RU-100-0
   nuc = new Nucleus("RU",100,44,0,99.9042,0,0,12.6,0,0,0,0);

   // Adding 45-RH-100-0
   nuc = new Nucleus("RH",100,45,0,99.9081,0,74880,0,7.1e-10,6.3e-10,0,0);
   nuc->AddDecay(0,-1,0,3.63,100);

   // Adding 45-RH-100-1
   nuc = new Nucleus("RH",100,45,1,99.9081,0,276,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.63,1.7);
   nuc->AddDecay(0,0,-1,0,98.3);

   // Adding 46-PD-100-0
   nuc = new Nucleus("PD",100,46,0,99.9085,0,313632,0,9.4e-10,9.8e-10,0,0);
   nuc->AddDecay(0,-1,0,0.362915,100);

   // Adding 47-AG-100-0
   nuc = new Nucleus("AG",100,47,0,99.9161,0,120.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.07343,100);

   // Adding 47-AG-100-1
   nuc = new Nucleus("AG",100,47,1,99.9161,0.016,134.4,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,7.08942,50);
   nuc->AddDecay(0,0,-1,0.016,50);

   // Adding 48-CD-100-0
   nuc = new Nucleus("CD",100,48,0,99.9203,0,49.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.89008,100);

   // Adding 49-IN-100-0
   nuc = new Nucleus("IN",100,49,0,99.9316,0,6.1,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.5297,50);
   nuc->AddDecay(-1,-2,0,5.69701,50);

   // Adding 37-RB-101-0
   nuc = new Nucleus("RB",101,37,0,100.953,0,0.032,0,0,0,0,0);
   nuc->AddDecay(0,1,0,11.81,69);
   nuc->AddDecay(-1,1,0,8.5505,31);

   // Adding 38-SR-101-0
   nuc = new Nucleus("SR",101,38,0,100.941,0,0.118,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.81557,2.37);
   nuc->AddDecay(0,1,0,9.50508,97.63);

   // Adding 39-Y-101-0
   nuc = new Nucleus("Y",101,39,0,100.93,0,0.448,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.62043,1.94);
   nuc->AddDecay(0,1,0,8.54493,98.06);

   // Adding 40-ZR-101-0
   nuc = new Nucleus("ZR",101,40,0,100.921,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.48499,100);

   // Adding 41-NB-101-0
   nuc = new Nucleus("NB",101,41,0,100.915,0,7.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.56904,100);

   // Adding 42-MO-101-0
   nuc = new Nucleus("MO",101,42,0,100.91,0,876.6,0,4.3e-11,4.5e-11,0,0);
   nuc->AddDecay(0,1,0,2.82436,100);

   // Adding 43-TC-101-0
   nuc = new Nucleus("TC",101,43,0,100.907,0,853.2,0,1.9e-11,2.1e-11,0,0);
   nuc->AddDecay(0,1,0,1.61359,100);

   // Adding 44-RU-101-0
   nuc = new Nucleus("RU",101,44,0,100.906,0,0,17,0,0,0,0);

   // Adding 44-RU-101-1
   nuc = new Nucleus("RU",101,44,1,100.906,0.527,1.75e-05,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.527,100);

   // Adding 45-RH-101-0
   nuc = new Nucleus("RH",101,45,0,100.906,0,1.04069e+08,0,5.5e-10,5.1e-09,0,0);
   nuc->AddDecay(0,-1,0,0.541603,100);

   // Adding 45-RH-101-1
   nuc = new Nucleus("RH",101,45,1,100.906,0.157,374976,0,2.2e-10,2.7e-10,0,0);
   nuc->AddDecay(0,-1,-1,0.698601,93.6);
   nuc->AddDecay(0,0,-1,0.157,6.4);

   // Adding 46-PD-101-0
   nuc = new Nucleus("PD",101,46,0,100.908,0,30492,0,9.4e-11,1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.98,100);

   // Adding 47-AG-101-0
   nuc = new Nucleus("AG",101,47,0,100.913,0,666,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.20381,100);

   // Adding 47-AG-101-1
   nuc = new Nucleus("AG",101,47,1,100.913,0.274,3.1,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.274,100);

   // Adding 48-CD-101-0
   nuc = new Nucleus("CD",101,48,0,100.919,0,72,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.47659,100);

   // Adding 49-IN-101-0
   nuc = new Nucleus("IN",101,49,0,100.927,0,16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.33901,100);
   nuc->AddDecay(-1,-2,0,2.45488,0);

   // Adding 37-RB-102-0
   nuc = new Nucleus("RB",102,37,0,101.959,0,0.037,0,0,0,0,0);
   nuc->AddDecay(0,1,0,14.4783,82);
   nuc->AddDecay(-1,1,0,8.73702,18);

   // Adding 38-SR-102-0
   nuc = new Nucleus("SR",102,38,0,101.943,0,0.069,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.76379,4.8);
   nuc->AddDecay(0,1,0,8.81169,95.2);

   // Adding 39-Y-102-0
   nuc = new Nucleus("Y",102,39,0,101.934,0,0.36,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.85335,94);
   nuc->AddDecay(-1,1,0,3.49703,6);

   // Adding 39-Y-102-1
   nuc = new Nucleus("Y",102,39,1,101.934,0.2,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,10.0533,94);
   nuc->AddDecay(-1,1,-1,3.69703,6);

   // Adding 40-ZR-102-0
   nuc = new Nucleus("ZR",102,40,0,101.923,0,2.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.605,100);

   // Adding 41-NB-102-0
   nuc = new Nucleus("NB",102,41,0,101.918,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.20998,100);

   // Adding 41-NB-102-1
   nuc = new Nucleus("NB",102,41,1,101.918,0,4.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.20998,100);

   // Adding 42-MO-102-0
   nuc = new Nucleus("MO",102,42,0,101.91,0,678,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.00998,100);

   // Adding 43-TC-102-0
   nuc = new Nucleus("TC",102,43,0,101.909,0,5.28,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.53022,100);

   // Adding 43-TC-102-1
   nuc = new Nucleus("TC",102,43,1,101.909,0,261,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,2);
   nuc->AddDecay(0,1,-1,4.53022,98);

   // Adding 44-RU-102-0
   nuc = new Nucleus("RU",102,44,0,101.904,0,0,31.6,0,0,0,0);

   // Adding 45-RH-102-0
   nuc = new Nucleus("RH",102,45,0,101.907,0,1.78848e+07,0,2.6e-09,1.6e-08,0,0);
   nuc->AddDecay(0,-1,0,2.32257,80);
   nuc->AddDecay(0,1,0,1.15048,20);

   // Adding 45-RH-102-1
   nuc = new Nucleus("RH",102,45,1,101.907,0.141,9.14544e+07,0,1.2e-09,6.7e-09,0,0);
   nuc->AddDecay(0,-1,-1,2.46357,99.73);
   nuc->AddDecay(0,0,-1,0.141,0.23);

   // Adding 46-PD-102-0
   nuc = new Nucleus("PD",102,46,0,101.906,0,0,1.02,0,0,0,0);

   // Adding 47-AG-102-0
   nuc = new Nucleus("AG",102,47,0,101.912,0,774,0,4e-11,3.2e-11,0,0);
   nuc->AddDecay(0,-1,0,5.92317,100);

   // Adding 47-AG-102-1
   nuc = new Nucleus("AG",102,47,1,101.912,0.009,462,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.009,49);
   nuc->AddDecay(0,-1,-1,5.93217,51);

   // Adding 48-CD-102-0
   nuc = new Nucleus("CD",102,48,0,101.915,0,330,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.5871,100);

   // Adding 49-IN-102-0
   nuc = new Nucleus("IN",102,49,0,101.924,0,24,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.90022,100);

   // Adding 50-SN-102-0
   nuc = new Nucleus("SN",102,50,0,101.93,0,0,0,0,0,0,-2);

   // Adding 40-ZR-103-0
   nuc = new Nucleus("ZR",103,40,0,102.927,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.94506,100);

   // Adding 41-NB-103-0
   nuc = new Nucleus("NB",103,41,0,102.919,0,1.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.52999,100);

   // Adding 42-MO-103-0
   nuc = new Nucleus("MO",103,42,0,102.913,0,67.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.74995,100);

   // Adding 43-TC-103-0
   nuc = new Nucleus("TC",103,43,0,102.909,0,54.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.65961,100);

   // Adding 44-RU-103-0
   nuc = new Nucleus("RU",103,44,0,102.906,0,3.39206e+06,0,7.3e-10,2.8e-09,0,0);
   nuc->AddDecay(0,1,0,0.763306,0.25);
   nuc->AddDecay(0,1,1,0.723305,99.75);

   // Adding 44-RU-103-1
   nuc = new Nucleus("RU",103,44,1,102.907,0.238,0.00169,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.238,100);

   // Adding 45-RH-103-0
   nuc = new Nucleus("RH",103,45,0,102.906,0,0,100,0,0,0,0);

   // Adding 45-RH-103-1
   nuc = new Nucleus("RH",103,45,1,102.906,0.04,3366.84,0,3.8e-12,2.5e-12,0,0);
   nuc->AddDecay(0,0,-1,0.04,100);

   // Adding 46-PD-103-0
   nuc = new Nucleus("PD",103,46,0,102.906,0,1.46802e+06,0,1.9e-10,4e-10,0,0);
   nuc->AddDecay(0,-1,0,0.543098,0.1);
   nuc->AddDecay(0,-1,1,0.503098,99.9);

   // Adding 47-AG-103-0
   nuc = new Nucleus("AG",103,47,0,102.909,0,3942,0,4.3e-11,4.5e-11,0,0);
   nuc->AddDecay(0,-1,0,2.68801,100);

   // Adding 47-AG-103-1
   nuc = new Nucleus("AG",103,47,1,102.909,0.134,5.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.134,100);

   // Adding 48-CD-103-0
   nuc = new Nucleus("CD",103,48,0,102.913,0,438,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.1419,100);

   // Adding 49-IN-103-0
   nuc = new Nucleus("IN",103,49,0,102.92,0,65,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.05,100);

   // Adding 50-SN-103-0
   nuc = new Nucleus("SN",103,50,0,102.928,0,7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.65363,100);

   // Adding 40-ZR-104-0
   nuc = new Nucleus("ZR",104,40,0,103.929,0,1.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.88771,100);

   // Adding 41-NB-104-0
   nuc = new Nucleus("NB",104,41,0,103.922,0,4.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.10505,99.29);
   nuc->AddDecay(-1,1,0,0.549649,0.71);

   // Adding 41-NB-104-1
   nuc = new Nucleus("NB",104,41,1,103.923,0.215,0.92,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.32005,100);

   // Adding 42-MO-104-0
   nuc = new Nucleus("MO",104,42,0,103.914,0,60,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.15498,100);

   // Adding 43-TC-104-0
   nuc = new Nucleus("TC",104,43,0,103.911,0,1098,0,8.1e-11,4.8e-11,0,0);
   nuc->AddDecay(0,1,0,5.60269,100);

   // Adding 44-RU-104-0
   nuc = new Nucleus("RU",104,44,0,103.905,0,0,18.7,0,0,0,0);

   // Adding 45-RH-104-0
   nuc = new Nucleus("RH",104,45,0,103.907,0,42.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.44102,99.55);
   nuc->AddDecay(0,-1,0,1.1412,0.45);

   // Adding 45-RH-104-1
   nuc = new Nucleus("RH",104,45,1,103.907,0.129,260.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.129,99.87);
   nuc->AddDecay(0,1,-1,2.57001,0.13);

   // Adding 46-PD-104-0
   nuc = new Nucleus("PD",104,46,0,103.904,0,0,11.14,0,0,0,0);

   // Adding 47-AG-104-0
   nuc = new Nucleus("AG",104,47,0,103.909,0,4152,0,6.1e-11,7.1e-11,0,0);
   nuc->AddDecay(0,-1,0,4.27869,100);

   // Adding 47-AG-104-1
   nuc = new Nucleus("AG",104,47,1,103.909,0.007,2010,0,5.4e-11,4.5e-11,0,0);
   nuc->AddDecay(0,0,-1,0.007,33);
   nuc->AddDecay(0,-1,-1,4.28569,67);

   // Adding 48-CD-104-0
   nuc = new Nucleus("CD",104,48,0,103.91,0,3462,0,5.8e-11,6.4e-11,0,0);
   nuc->AddDecay(0,-1,0,1.13686,100);

   // Adding 49-IN-104-0
   nuc = new Nucleus("IN",104,49,0,103.918,0,108,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.90878,100);

   // Adding 49-IN-104-1
   nuc = new Nucleus("IN",104,49,1,103.918,0.094,15.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.094,80);
   nuc->AddDecay(0,-1,-1,8.00278,20);

   // Adding 50-SN-104-0
   nuc = new Nucleus("SN",104,50,0,103.923,0,20.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.515,100);

   // Adding 51-SB-104-0
   nuc = new Nucleus("SB",104,51,0,103.937,0,0,0,0,0,0,-2);

   // Adding 41-NB-105-0
   nuc = new Nucleus("NB",105,41,0,104.924,0,2.95,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.48497,100);

   // Adding 42-MO-105-0
   nuc = new Nucleus("MO",105,42,0,104.917,0,35.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.94999,100);

   // Adding 43-TC-105-0
   nuc = new Nucleus("TC",105,43,0,104.912,0,456,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.63998,100);

   // Adding 44-RU-105-0
   nuc = new Nucleus("RU",105,44,0,104.908,0,15984,0,2.6e-10,2.5e-10,0,0);
   nuc->AddDecay(0,1,0,1.91702,72);
   nuc->AddDecay(0,1,1,1.78702,28);

   // Adding 45-RH-105-0
   nuc = new Nucleus("RH",105,45,0,104.906,0,127296,0,3.7e-10,4.4e-10,0,0);
   nuc->AddDecay(0,1,0,0.566696,100);

   // Adding 45-RH-105-1
   nuc = new Nucleus("RH",105,45,1,104.906,0.13,40,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.13,100);

   // Adding 46-PD-105-0
   nuc = new Nucleus("PD",105,46,0,104.905,0,0,22.33,0,0,0,0);

   // Adding 47-AG-105-0
   nuc = new Nucleus("AG",105,47,0,104.907,0,3.56746e+06,0,4.7e-10,8e-10,0,0);
   nuc->AddDecay(0,-1,0,1.34563,100);

   // Adding 47-AG-105-1
   nuc = new Nucleus("AG",105,47,1,104.907,0.025,433.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.025,99.66);
   nuc->AddDecay(0,-1,-1,1.37064,0.34);

   // Adding 48-CD-105-0
   nuc = new Nucleus("CD",105,48,0,104.909,0,3330,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.7387,100);

   // Adding 49-IN-105-0
   nuc = new Nucleus("IN",105,49,0,104.915,0,304.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.84899,100);

   // Adding 49-IN-105-1
   nuc = new Nucleus("IN",105,49,1,104.915,0.674,48,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.674,100);

   // Adding 50-SN-105-0
   nuc = new Nucleus("SN",105,50,0,104.921,0,31,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.24804,100);
   nuc->AddDecay(-1,-2,0,3.45412,0);

   // Adding 51-SB-105-0
   nuc = new Nucleus("SB",105,51,0,104.931,0,0,0,0,0,0,-2);

   // Adding 41-NB-106-0
   nuc = new Nucleus("NB",106,41,0,105.928,0,1.02,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.27272,100);

   // Adding 42-MO-106-0
   nuc = new Nucleus("MO",106,42,0,105.918,0,8.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.52,100);

   // Adding 43-TC-106-0
   nuc = new Nucleus("TC",106,43,0,105.914,0,36,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.54707,100);

   // Adding 44-RU-106-0
   nuc = new Nucleus("RU",106,44,0,105.907,0,3.22782e+07,0,7e-09,6.2e-08,0,0);
   nuc->AddDecay(0,1,0,0.0393982,100);

   // Adding 45-RH-106-0
   nuc = new Nucleus("RH",106,45,0,105.907,0,29.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.54107,100);

   // Adding 45-RH-106-1
   nuc = new Nucleus("RH",106,45,1,105.907,0.137,7800,0,1.6e-10,1.9e-10,0,0);
   nuc->AddDecay(0,1,-1,3.67807,100);

   // Adding 46-PD-106-0
   nuc = new Nucleus("PD",106,46,0,105.903,0,0,27.33,0,0,0,0);

   // Adding 47-AG-106-0
   nuc = new Nucleus("AG",106,47,0,105.907,0,1437.6,0,3.2e-11,2.7e-11,0,0);
   nuc->AddDecay(0,-1,0,2.96529,99.5);
   nuc->AddDecay(0,1,0,0.194603,0.5);

   // Adding 47-AG-106-1
   nuc = new Nucleus("AG",106,47,1,105.907,0.09,730944,0,1.5e-09,1.6e-09,0,0);
   nuc->AddDecay(0,-1,-1,3.05529,100);

   // Adding 48-CD-106-0
   nuc = new Nucleus("CD",106,48,0,105.906,0,0,1.25,0,0,0,0);

   // Adding 49-IN-106-0
   nuc = new Nucleus("IN",106,49,0,105.913,0,372,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.52135,100);

   // Adding 49-IN-106-1
   nuc = new Nucleus("IN",106,49,1,105.913,0.029,312,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.55035,100);

   // Adding 50-SN-106-0
   nuc = new Nucleus("SN",106,50,0,105.917,0,126,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.18436,100);

   // Adding 51-SB-106-0
   nuc = new Nucleus("SB",106,51,0,105.929,0,0,0,0,0,0,-2);

   // Adding 52-TE-106-0
   nuc = new Nucleus("TE",106,52,0,105.938,0,7e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.3231,100);

   // Adding 41-NB-107-0
   nuc = new Nucleus("NB",107,41,0,106.93,0,0.33,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.90321,100);

   // Adding 42-MO-107-0
   nuc = new Nucleus("MO",107,42,0,106.922,0,3.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.16,100);

   // Adding 43-TC-107-0
   nuc = new Nucleus("TC",107,43,0,106.915,0,21.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.81999,100);

   // Adding 44-RU-107-0
   nuc = new Nucleus("RU",107,44,0,106.91,0,225,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.94051,100);

   // Adding 45-RH-107-0
   nuc = new Nucleus("RH",107,45,0,106.907,0,1302,0,2.4e-11,2.8e-11,0,0);
   nuc->AddDecay(0,1,0,1.51114,100);

   // Adding 46-PD-107-0
   nuc = new Nucleus("PD",107,46,0,106.905,0,2.04984e+14,0,3.7e-11,5.5e-10,0,0);
   nuc->AddDecay(0,1,0,0.0329971,100);

   // Adding 46-PD-107-1
   nuc = new Nucleus("PD",107,46,1,106.905,0.215,21.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.215,100);

   // Adding 47-AG-107-0
   nuc = new Nucleus("AG",107,47,0,106.905,0,0,51.839,0,0,0,0);

   // Adding 47-AG-107-1
   nuc = new Nucleus("AG",107,47,1,106.905,0.093,44.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.093,100);

   // Adding 48-CD-107-0
   nuc = new Nucleus("CD",107,48,0,106.907,0,23400,0,6.2e-11,1.1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.41699,0.059998);
   nuc->AddDecay(0,-1,1,1.32399,99.94);

   // Adding 49-IN-107-0
   nuc = new Nucleus("IN",107,49,0,106.91,0,1944,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.42606,100);

   // Adding 49-IN-107-1
   nuc = new Nucleus("IN",107,49,1,106.911,0.678,50.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.678,100);

   // Adding 50-SN-107-0
   nuc = new Nucleus("SN",107,50,0,106.916,0,174,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.99963,100);

   // Adding 51-SB-107-0
   nuc = new Nucleus("SB",107,51,0,106.924,0,0,0,0,0,0,-2);

   // Adding 52-TE-107-0
   nuc = new Nucleus("TE",107,52,0,106.935,0,0.0036,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.135,30);
   nuc->AddDecay(-4,-2,0,4.00209,70);

   // Adding 42-MO-108-0
   nuc = new Nucleus("MO",108,42,0,107.923,0,1.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.63497,100);

   // Adding 43-TC-108-0
   nuc = new Nucleus("TC",108,43,0,107.918,0,5.17,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.72,100);

   // Adding 44-RU-108-0
   nuc = new Nucleus("RU",108,44,0,107.91,0,273,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.3613,100);

   // Adding 45-RH-108-0
   nuc = new Nucleus("RH",108,45,0,107.909,0,16.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.50503,100);

   // Adding 45-RH-108-1
   nuc = new Nucleus("RH",108,45,1,107.909,0,360,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.50503,100);

   // Adding 46-PD-108-0
   nuc = new Nucleus("PD",108,46,0,107.904,0,0,26.46,0,0,0,0);

   // Adding 47-AG-108-0
   nuc = new Nucleus("AG",108,47,0,107.906,0,142.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.64951,97.15);
   nuc->AddDecay(0,-1,0,1.91798,2.85);

   // Adding 47-AG-108-1
   nuc = new Nucleus("AG",108,47,1,107.906,0.109,1.3182e+10,0,2.3e-09,3.5e-08,0,0);
   nuc->AddDecay(0,-1,-1,2.02699,91.3);
   nuc->AddDecay(0,0,-1,0.109,8.7);

   // Adding 48-CD-108-0
   nuc = new Nucleus("CD",108,48,0,107.904,0,0,0.89,0,0,0,0);

   // Adding 49-IN-108-0
   nuc = new Nucleus("IN",108,49,0,107.91,0,3480,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.14765,100);

   // Adding 49-IN-108-1
   nuc = new Nucleus("IN",108,49,1,107.91,0.03,2376,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.17765,100);

   // Adding 50-SN-108-0
   nuc = new Nucleus("SN",108,50,0,107.912,0,618,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.09208,100);

   // Adding 51-SB-108-0
   nuc = new Nucleus("SB",108,51,0,107.922,0,7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.50603,100);

   // Adding 52-TE-108-0
   nuc = new Nucleus("TE",108,52,0,107.929,0,2.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.4417,68);
   nuc->AddDecay(0,-1,0,6.82149,32);

   // Adding 53-I-108-0
   nuc = new Nucleus("I",108,53,0,107.944,0,0.036,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,4.0351,100);

   // Adding 42-MO-109-0
   nuc = new Nucleus("MO",109,42,0,108.928,0,0.53,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.51166,100);

   // Adding 43-TC-109-0
   nuc = new Nucleus("TC",109,43,0,108.92,0,0.87,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.98411,100);

   // Adding 44-RU-109-0
   nuc = new Nucleus("RU",109,44,0,108.913,0,34.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.16006,50);
   nuc->AddDecay(0,1,1,3.93407,50);

   // Adding 45-RH-109-0
   nuc = new Nucleus("RH",109,45,0,108.909,0,80,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.59132,100);

   // Adding 45-RH-109-1
   nuc = new Nucleus("RH",109,45,1,108.909,0.22598,1.6e-06,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.22598,100);

   // Adding 46-PD-109-0
   nuc = new Nucleus("PD",109,46,0,108.906,0,49324.3,0,5.5e-10,5e-10,0,0);
   nuc->AddDecay(0,1,0,1.11589,0.05);
   nuc->AddDecay(0,1,1,1.02789,99.95);

   // Adding 46-PD-109-1
   nuc = new Nucleus("PD",109,46,1,108.906,0.189,281.76,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.189,100);

   // Adding 47-AG-109-0
   nuc = new Nucleus("AG",109,47,0,108.905,0,0,48.161,0,0,0,0);

   // Adding 47-AG-109-1
   nuc = new Nucleus("AG",109,47,1,108.905,0.088,39.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.088,100);

   // Adding 48-CD-109-0
   nuc = new Nucleus("CD",109,48,0,108.905,0,3.99686e+07,0,2e-09,9.6e-09,0,0);
   nuc->AddDecay(0,-1,1,0.125793,100);

   // Adding 48-CD-109-1
   nuc = new Nucleus("CD",109,48,1,108.905,0.06,1.2e-05,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.06,100);

   // Adding 48-CD-109-2
   nuc = new Nucleus("CD",109,48,2,108.905,0.463,1.09e-05,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.463,100);

   // Adding 49-IN-109-0
   nuc = new Nucleus("IN",109,49,0,108.907,0,15120,0,6.6e-11,7.3e-11,0,0);
   nuc->AddDecay(0,-1,0,2.02028,100);

   // Adding 49-IN-109-1
   nuc = new Nucleus("IN",109,49,1,108.908,0.65,80.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.65,100);

   // Adding 49-IN-109-2
   nuc = new Nucleus("IN",109,49,2,108.909,2.102,0.21,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.102,100);

   // Adding 50-SN-109-0
   nuc = new Nucleus("SN",109,50,0,108.911,0,1080,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.85036,100);

   // Adding 51-SB-109-0
   nuc = new Nucleus("SB",109,51,0,108.918,0,17,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.38007,100);

   // Adding 52-TE-109-0
   nuc = new Nucleus("TE",109,52,0,108.927,0,4.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.22561,4);
   nuc->AddDecay(0,-1,0,8.67236,96);

   // Adding 53-I-109-0
   nuc = new Nucleus("I",109,53,0,108.938,0,0.0001,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,0.820299,100);

   // Adding 42-MO-110-0
   nuc = new Nucleus("MO",110,42,0,109.929,0,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.69241,100);

   // Adding 43-TC-110-0
   nuc = new Nucleus("TC",110,43,0,109.923,0,0.92,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.77732,100);

   // Adding 44-RU-110-0
   nuc = new Nucleus("RU",110,44,0,109.914,0,14.6,0,0,0,0,0);
   nuc->AddDecay(0,1,1,2.81,100);

   // Adding 45-RH-110-0
   nuc = new Nucleus("RH",110,45,0,109.911,0,3.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.39999,100);

   // Adding 45-RH-110-1
   nuc = new Nucleus("RH",110,45,1,109.911,0,28.5,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.39999,100);

   // Adding 46-PD-110-0
   nuc = new Nucleus("PD",110,46,0,109.905,0,0,11.72,0,0,0,0);

   // Adding 47-AG-110-0
   nuc = new Nucleus("AG",110,47,0,109.906,0,24.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.8921,99.7);
   nuc->AddDecay(0,-1,0,0.892479,0.3);

   // Adding 47-AG-110-1
   nuc = new Nucleus("AG",110,47,1,109.906,0.118,2.15819e+07,0,2.8e-09,1.2e-08,0,0);
   nuc->AddDecay(0,0,-1,0.118,1.36);
   nuc->AddDecay(0,1,-1,3.01009,98.64);

   // Adding 48-CD-110-0
   nuc = new Nucleus("CD",110,48,0,109.903,0,0,12.49,0,0,0,0);

   // Adding 49-IN-110-0
   nuc = new Nucleus("IN",110,49,0,109.907,0,17640,0,2.4e-10,2.5e-10,0,0);
   nuc->AddDecay(0,-1,0,3.87791,100);

   // Adding 49-IN-110-1
   nuc = new Nucleus("IN",110,49,1,109.907,0.062,4146,0,1e-10,8.1e-11,0,0);
   nuc->AddDecay(0,-1,-1,3.93991,100);

   // Adding 50-SN-110-0
   nuc = new Nucleus("SN",110,50,0,109.908,0,14796,0,3.5e-10,2.6e-10,0,0);
   nuc->AddDecay(0,-1,0,0.637695,100);

   // Adding 51-SB-110-0
   nuc = new Nucleus("SB",110,51,0,109.917,0,23,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.29951,100);

   // Adding 52-TE-110-0
   nuc = new Nucleus("TE",110,52,0,109.922,0,18.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.25415,100);
   nuc->AddDecay(-4,-2,0,2.7232,0.003);

   // Adding 53-I-110-0
   nuc = new Nucleus("I",110,53,0,109.935,0,0.65,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.5811,17);
   nuc->AddDecay(-1,-2,0,8.61851,11);
   nuc->AddDecay(-4,-3,0,14.6559,1.1);
   nuc->AddDecay(0,-1,0,11.9327,70.9);

   // Adding 54-XE-110-0
   nuc = new Nucleus("XE",110,54,0,109.945,0,0.944,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,3.8861,50);
   nuc->AddDecay(0,-1,0,8.65791,50);

   // Adding 43-TC-111-0
   nuc = new Nucleus("TC",111,43,0,110.925,0,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.9769,100);

   // Adding 44-RU-111-0
   nuc = new Nucleus("RU",111,44,0,110.918,0,2.12,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.49591,100);

   // Adding 45-RH-111-0
   nuc = new Nucleus("RH",111,45,0,110.912,0,11,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.74083,100);

   // Adding 46-PD-111-0
   nuc = new Nucleus("PD",111,46,0,110.908,0,1404,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.188,0.75);
   nuc->AddDecay(0,1,1,2.128,99.25);

   // Adding 46-PD-111-1
   nuc = new Nucleus("PD",111,46,1,110.908,0.172,19800,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.172,73);
   nuc->AddDecay(0,1,-1,2.35999,7.5);
   nuc->AddDecay(0,1,0,2.3,19.5);

   // Adding 47-AG-111-0
   nuc = new Nucleus("AG",111,47,0,110.905,0,643680,0,1.3e-09,1.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.0368,100);

   // Adding 47-AG-111-1
   nuc = new Nucleus("AG",111,47,1,110.905,0.06,64.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.06,99.3);
   nuc->AddDecay(0,1,-1,1.09679,0.7);

   // Adding 48-CD-111-0
   nuc = new Nucleus("CD",111,48,0,110.904,0,0,12.8,0,0,0,0);

   // Adding 48-CD-111-1
   nuc = new Nucleus("CD",111,48,1,110.905,0.396,2912.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.396,100);

   // Adding 49-IN-111-0
   nuc = new Nucleus("IN",111,49,0,110.905,0,242343,0,2.9e-10,3.1e-10,0,0);
   nuc->AddDecay(0,-1,0,0.865875,50);
   nuc->AddDecay(0,-1,1,0.469872,50);

   // Adding 49-IN-111-1
   nuc = new Nucleus("IN",111,49,1,110.906,0.537,462,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.537,100);

   // Adding 50-SN-111-0
   nuc = new Nucleus("SN",111,50,0,110.908,0,2118,0,2.3e-11,2.2e-11,0,0);
   nuc->AddDecay(0,-1,0,2.44489,100);

   // Adding 51-SB-111-0
   nuc = new Nucleus("SB",111,51,0,110.913,0,75,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.09987,100);

   // Adding 52-TE-111-0
   nuc = new Nucleus("TE",111,52,0,110.921,0,19.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.36833,100);
   nuc->AddDecay(-1,-2,0,5.06985,0);

   // Adding 53-I-111-0
   nuc = new Nucleus("I",111,53,0,110.93,0,2.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.52357,99.9);
   nuc->AddDecay(-4,-2,0,3.2781,0.1);

   // Adding 54-XE-111-0
   nuc = new Nucleus("XE",111,54,0,110.942,0,0.74,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.57,99);
   nuc->AddDecay(-4,-2,0,3.71309,1);

   // Adding 43-TC-112-0
   nuc = new Nucleus("TC",112,43,0,111.929,0,0.28,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.9575,100);

   // Adding 44-RU-112-0
   nuc = new Nucleus("RU",112,44,0,111.919,0,1.75,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.66999,100);

   // Adding 45-RH-112-0
   nuc = new Nucleus("RH",112,45,0,111.915,0,3.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.79932,100);

   // Adding 45-RH-112-1
   nuc = new Nucleus("RH",112,45,1,111.915,0,6.8,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.79932,100);

   // Adding 46-PD-112-0
   nuc = new Nucleus("PD",112,46,0,111.907,0,75708,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.287903,100);

   // Adding 47-AG-112-0
   nuc = new Nucleus("AG",112,47,0,111.907,0,11268,0,4.3e-10,2.6e-10,0,0);
   nuc->AddDecay(0,1,0,3.95591,100);

   // Adding 48-CD-112-0
   nuc = new Nucleus("CD",112,48,0,111.903,0,0,24.13,0,0,0,0);

   // Adding 49-IN-112-0
   nuc = new Nucleus("IN",112,49,0,111.906,0,898.2,0,1e-11,1.3e-11,0,0);
   nuc->AddDecay(0,-1,0,2.58617,56);
   nuc->AddDecay(0,1,0,0.66349,44);

   // Adding 49-IN-112-1
   nuc = new Nucleus("IN",112,49,1,111.906,0.157,1233.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.157,100);

   // Adding 50-SN-112-0
   nuc = new Nucleus("SN",112,50,0,111.905,0,0,0.97,0,0,0,0);

   // Adding 51-SB-112-0
   nuc = new Nucleus("SB",112,51,0,111.912,0,51.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.05502,100);

   // Adding 52-TE-112-0
   nuc = new Nucleus("TE",112,52,0,111.917,0,120,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.3438,100);

   // Adding 53-I-112-0
   nuc = new Nucleus("I",112,53,0,111.928,0,3.42,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,2.98609,0.0012);
   nuc->AddDecay(-1,-2,0,6.45799,0);
   nuc->AddDecay(-4,-3,0,12.4921,0);
   nuc->AddDecay(0,-1,0,10.1629,100);

   // Adding 54-XE-112-0
   nuc = new Nucleus("XE",112,54,0,111.936,0,2.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.1525,99.16);
   nuc->AddDecay(-4,-2,0,3.3171,0.84);

   // Adding 43-TC-113-0
   nuc = new Nucleus("TC",113,43,0,112.931,0,0.13,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.18791,100);

   // Adding 44-RU-113-0
   nuc = new Nucleus("RU",113,44,0,112.923,0,0.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.6319,100);

   // Adding 45-RH-113-0
   nuc = new Nucleus("RH",113,45,0,112.915,0,2.72,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.9036,100);

   // Adding 46-PD-113-0
   nuc = new Nucleus("PD",113,46,0,112.91,0,93,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.34312,81.5);
   nuc->AddDecay(0,1,1,3.30012,18.5);

   // Adding 46-PD-113-1
   nuc = new Nucleus("PD",113,46,1,112.91,0.04,0,0,0,0,0,-2);
   nuc->AddDecay(0,0,-1,0,0);

   // Adding 46-PD-113-2
   nuc = new Nucleus("PD",113,46,2,112.91,0.08,0.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.08,100);

   // Adding 47-AG-113-0
   nuc = new Nucleus("AG",113,47,0,112.907,0,19332,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.01641,98.3);
   nuc->AddDecay(0,1,1,1.75241,1.7);

   // Adding 47-AG-113-1
   nuc = new Nucleus("AG",113,47,1,112.907,0.043,68.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.043,64);
   nuc->AddDecay(0,1,-1,2.05941,36);

   // Adding 48-CD-113-0
   nuc = new Nucleus("CD",113,48,0,112.904,0,2.93285e+23,12.22,2.5e-08,1.4e-07,0,0);
   nuc->AddDecay(0,1,0,0.315903,100);

   // Adding 48-CD-113-1
   nuc = new Nucleus("CD",113,48,1,112.905,0.264,4.44658e+08,0,2.3e-08,1.3e-07,0,0);
   nuc->AddDecay(0,0,-1,0.264,0.14);
   nuc->AddDecay(0,1,-1,0.579903,99.86);

   // Adding 49-IN-113-0
   nuc = new Nucleus("IN",113,49,0,112.904,0,0,4.3,0,0,0,0);

   // Adding 49-IN-113-1
   nuc = new Nucleus("IN",113,49,1,112.904,0.392,5969.52,0,2.8e-11,3.2e-11,0,0);
   nuc->AddDecay(0,0,-1,0.392,100);

   // Adding 50-SN-113-0
   nuc = new Nucleus("SN",113,50,0,112.905,0,9.94378e+06,0,7.4e-10,2.5e-09,0,0);
   nuc->AddDecay(0,-1,0,1.0359,4e-06);
   nuc->AddDecay(0,-1,1,0.643898,100);

   // Adding 50-SN-113-1
   nuc = new Nucleus("SN",113,50,1,112.905,0.077,1284,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.077,91.1);
   nuc->AddDecay(0,-1,-1,1.1129,8.9);

   // Adding 51-SB-113-0
   nuc = new Nucleus("SB",113,51,0,112.909,0,400.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.90551,100);

   // Adding 52-TE-113-0
   nuc = new Nucleus("TE",113,52,0,112.916,0,102,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.09982,100);

   // Adding 53-I-113-0
   nuc = new Nucleus("I",113,53,0,112.924,0,6.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.20015,100);
   nuc->AddDecay(-4,-2,0,2.70587,3.3e-07);

   // Adding 54-XE-113-0
   nuc = new Nucleus("XE",113,54,0,112.933,0,2.74,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.06257,95.77);
   nuc->AddDecay(-4,-2,0,3.09608,0.04);
   nuc->AddDecay(-1,-2,0,7.90864,4.2);

   // Adding 55-CS-113-0
   nuc = new Nucleus("CS",113,55,0,112.945,0,3.3e-05,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,0.977499,100);

   // Adding 44-RU-114-0
   nuc = new Nucleus("RU",114,44,0,113.924,0,0.57,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.79994,100);

   // Adding 45-RH-114-0
   nuc = new Nucleus("RH",114,45,0,113.919,0,1.85,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.89953,100);

   // Adding 45-RH-114-1
   nuc = new Nucleus("RH",114,45,1,113.919,0,1.85,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.89953,100);

   // Adding 46-PD-114-0
   nuc = new Nucleus("PD",114,46,0,113.91,0,145.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.4507,100);

   // Adding 47-AG-114-0
   nuc = new Nucleus("AG",114,47,0,113.909,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.0764,100);

   // Adding 48-CD-114-0
   nuc = new Nucleus("CD",114,48,0,113.903,0,0,28.73,0,0,0,0);

   // Adding 49-IN-114-0
   nuc = new Nucleus("IN",114,49,0,113.905,0,71.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.98869,99.5);
   nuc->AddDecay(0,-1,0,1.45249,0.5);

   // Adding 49-IN-114-1
   nuc = new Nucleus("IN",114,49,1,113.905,0.19,4.27766e+06,0,4.1e-09,1.1e-08,0,0);
   nuc->AddDecay(0,0,-1,0.19,95.6);
   nuc->AddDecay(0,-1,-1,1.64249,4.4);

   // Adding 49-IN-114-2
   nuc = new Nucleus("IN",114,49,2,113.905,0.502,0.0431,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.502,100);

   // Adding 50-SN-114-0
   nuc = new Nucleus("SN",114,50,0,113.903,0,0,0.65,0,0,0,0);

   // Adding 51-SB-114-0
   nuc = new Nucleus("SB",114,51,0,113.909,0,209.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.88151,100);

   // Adding 52-TE-114-0
   nuc = new Nucleus("TE",114,52,0,113.912,0,912,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.74242,100);

   // Adding 53-I-114-0
   nuc = new Nucleus("I",114,53,0,113.922,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.13691,100);
   nuc->AddDecay(-1,-2,0,4.33873,0);

   // Adding 54-XE-114-0
   nuc = new Nucleus("XE",114,54,0,113.928,0,10,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.86109,100);

   // Adding 55-CS-114-0
   nuc = new Nucleus("CS",114,55,0,113.941,0,0.57,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.3699,92.82);
   nuc->AddDecay(-4,-2,0,3.35711,0.02);
   nuc->AddDecay(-1,-2,0,9.26975,7);
   nuc->AddDecay(-4,-3,0,15.2898,0.16);

   // Adding 44-RU-115-0
   nuc = new Nucleus("RU",115,44,0,114.928,0,0.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.62335,100);
   nuc->AddDecay(-1,1,0,0.743299,0);

   // Adding 45-RH-115-0
   nuc = new Nucleus("RH",115,45,0,114.92,0,0.99,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.00056,100);

   // Adding 46-PD-115-0
   nuc = new Nucleus("PD",115,46,0,114.914,0,25,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.58343,73);
   nuc->AddDecay(0,1,1,4.54243,27);

   // Adding 46-PD-115-1
   nuc = new Nucleus("PD",115,46,1,114.914,0.089,50,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.67242,92);
   nuc->AddDecay(0,0,-1,0.089,8);

   // Adding 47-AG-115-0
   nuc = new Nucleus("AG",115,47,0,114.909,0,1200,0,6e-11,4.4e-11,0,0);
   nuc->AddDecay(0,1,0,3.10349,94.3);
   nuc->AddDecay(0,1,1,2.92249,5.7);

   // Adding 47-AG-115-1
   nuc = new Nucleus("AG",115,47,1,114.909,0.041,18,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.041,21);
   nuc->AddDecay(0,1,-1,3.14449,79);

   // Adding 48-CD-115-0
   nuc = new Nucleus("CD",115,48,0,114.905,0,192456,0,1.4e-09,1.3e-09,0,0);
   nuc->AddDecay(0,1,0,1.44561,7e-05);
   nuc->AddDecay(0,1,1,1.10961,99.9999);

   // Adding 48-CD-115-1
   nuc = new Nucleus("CD",115,48,1,114.906,0.181,3.85344e+06,0,3.3e-09,7.3e-09,0,0);
   nuc->AddDecay(0,1,-1,1.62661,99.989);
   nuc->AddDecay(0,1,0,1.29061,0.011);

   // Adding 49-IN-115-0
   nuc = new Nucleus("IN",115,49,0,114.904,0,1.39074e+22,95.7,3.2e-08,4.5e-07,0,0);
   nuc->AddDecay(0,1,0,0.495293,100);

   // Adding 49-IN-115-1
   nuc = new Nucleus("IN",115,49,1,114.904,0.336,16149.6,0,8.6e-11,8.7e-11,0,0);
   nuc->AddDecay(0,0,-1,0.336,95);
   nuc->AddDecay(0,1,-1,0.831291,5);

   // Adding 50-SN-115-0
   nuc = new Nucleus("SN",115,50,0,114.903,0,0,0.36,0,0,0,0);

   // Adding 51-SB-115-0
   nuc = new Nucleus("SB",115,51,0,114.907,0,1926,0,2.4e-11,2.4e-11,0,0);
   nuc->AddDecay(0,-1,0,3.03001,100);

   // Adding 52-TE-115-0
   nuc = new Nucleus("TE",115,52,0,114.912,0,348,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.63871,100);

   // Adding 52-TE-115-1
   nuc = new Nucleus("TE",115,52,1,114.912,0.02,402,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.65871,100);
   nuc->AddDecay(0,0,-1,0.02,0);

   // Adding 53-I-115-0
   nuc = new Nucleus("I",115,53,0,114.918,0,78,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.95921,100);

   // Adding 54-XE-115-0
   nuc = new Nucleus("XE",115,54,0,114.927,0,18,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.95926,100);
   nuc->AddDecay(-1,-2,0,6.19997,0);

   // Adding 55-CS-115-0
   nuc = new Nucleus("CS",115,55,0,114.936,0,1.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.76681,99.93);
   nuc->AddDecay(-1,-2,0,5.82987,0.07);

   // Adding 45-RH-116-0
   nuc = new Nucleus("RH",116,45,0,115.924,0,0.68,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.89915,100);

   // Adding 45-RH-116-1
   nuc = new Nucleus("RH",116,45,1,115.924,0,0.9,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.89915,100);

   // Adding 46-PD-116-0
   nuc = new Nucleus("PD",116,46,0,115.914,0,11.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.60699,100);

   // Adding 47-AG-116-0
   nuc = new Nucleus("AG",116,47,0,115.911,0,160.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.15978,100);

   // Adding 47-AG-116-1
   nuc = new Nucleus("AG",116,47,1,115.911,0.082,8.6,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.24178,94);
   nuc->AddDecay(0,0,-1,0.082,6);

   // Adding 48-CD-116-0
   nuc = new Nucleus("CD",116,48,0,115.905,0,0,7.49,0,0,0,0);

   // Adding 49-IN-116-0
   nuc = new Nucleus("IN",116,49,0,115.905,0,14.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.470284,0.06);
   nuc->AddDecay(0,1,0,3.27438,99.94);

   // Adding 49-IN-116-1
   nuc = new Nucleus("IN",116,49,1,115.905,0.127,3264.6,0,6.4e-11,8e-11,0,0);
   nuc->AddDecay(0,1,-1,3.40138,100);

   // Adding 49-IN-116-2
   nuc = new Nucleus("IN",116,49,2,115.906,0.29,2.18,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.29,100);

   // Adding 50-SN-116-0
   nuc = new Nucleus("SN",116,50,0,115.902,0,0,14.53,0,0,0,0);

   // Adding 51-SB-116-0
   nuc = new Nucleus("SB",116,51,0,115.907,0,948,0,2.6e-11,2.3e-11,0,0);
   nuc->AddDecay(0,-1,0,4.70737,100);

   // Adding 51-SB-116-1
   nuc = new Nucleus("SB",116,51,1,115.907,0.383,3618,0,6.7e-11,8.5e-11,0,0);
   nuc->AddDecay(0,-1,-1,5.09037,100);

   // Adding 52-TE-116-0
   nuc = new Nucleus("TE",116,52,0,115.908,0,8964,0,1.7e-10,1.7e-10,0,0);
   nuc->AddDecay(0,-1,0,1.49987,100);

   // Adding 53-I-116-0
   nuc = new Nucleus("I",116,53,0,115.917,0,2.91,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.74528,100);

   // Adding 53-I-116-1
   nuc = new Nucleus("I",116,53,1,115.917,0,3.27e-06,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 54-XE-116-0
   nuc = new Nucleus("XE",116,54,0,115.922,0,56,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.65977,100);

   // Adding 55-CS-116-0
   nuc = new Nucleus("CS",116,55,0,115.933,0,0.7,0,0,0,1,0);
   nuc->AddDecay(-4,-3,0,12.4,0);
   nuc->AddDecay(-1,-2,0,6.68023,0);
   nuc->AddDecay(0,-1,0,10.477,100);

   // Adding 55-CS-116-1
   nuc = new Nucleus("CS",116,55,1,115.933,0,3.84,0,0,0,1,0);
   nuc->AddDecay(-4,-3,-1,12.4,0);
   nuc->AddDecay(-1,-2,-1,6.68023,0);
   nuc->AddDecay(0,-1,-1,10.477,100);

   // Adding 45-RH-117-0
   nuc = new Nucleus("RH",117,45,0,116.925,0,0.44,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.9956,100);

   // Adding 46-PD-117-0
   nuc = new Nucleus("PD",117,46,0,116.918,0,4.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.71105,50);
   nuc->AddDecay(0,1,1,5.68205,50);

   // Adding 46-PD-117-1
   nuc = new Nucleus("PD",117,46,1,116.918,0.203,0.0191,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.203,100);

   // Adding 47-AG-117-0
   nuc = new Nucleus("AG",117,47,0,116.912,0,72.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.18198,86);
   nuc->AddDecay(0,1,1,4.04598,14);

   // Adding 47-AG-117-1
   nuc = new Nucleus("AG",117,47,1,116.912,0.029,5.34,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.21098,20.21);
   nuc->AddDecay(0,0,-1,0.029,6);
   nuc->AddDecay(0,1,0,4.07498,73.79);

   // Adding 48-CD-117-0
   nuc = new Nucleus("CD",117,48,0,116.907,0,8964,0,2.8e-10,2.5e-10,0,0);
   nuc->AddDecay(0,1,0,2.51612,8.4);
   nuc->AddDecay(0,1,1,2.20112,91.6);

   // Adding 48-CD-117-1
   nuc = new Nucleus("CD",117,48,1,116.907,0.136,12096,0,2.8e-10,3.2e-10,0,0);
   nuc->AddDecay(0,1,-1,2.65212,98.6);
   nuc->AddDecay(0,1,0,2.33712,1.4);

   // Adding 49-IN-117-0
   nuc = new Nucleus("IN",117,49,0,116.905,0,2592,0,3.1e-11,4.8e-11,0,0);
   nuc->AddDecay(0,1,0,1.45528,99.68);
   nuc->AddDecay(0,1,1,1.14027,0.32);

   // Adding 49-IN-117-1
   nuc = new Nucleus("IN",117,49,1,116.905,0.315,6972,0,1.2e-10,1.1e-10,0,0);
   nuc->AddDecay(0,1,-1,1.77028,52.9);
   nuc->AddDecay(0,0,-1,0.315,47.1);

   // Adding 50-SN-117-0
   nuc = new Nucleus("SN",117,50,0,116.903,0,0,7.68,0,0,0,0);

   // Adding 50-SN-117-1
   nuc = new Nucleus("SN",117,50,1,116.903,0.315,1.17504e+06,0,7.1e-10,2.3e-09,0,0);
   nuc->AddDecay(0,0,-1,0.315,100);

   // Adding 51-SB-117-0
   nuc = new Nucleus("SB",117,51,0,116.905,0,10080,0,1.8e-11,2.7e-11,0,0);
   nuc->AddDecay(0,-1,0,1.75653,100);

   // Adding 52-TE-117-0
   nuc = new Nucleus("TE",117,52,0,116.909,0,3720,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.53488,100);

   // Adding 52-TE-117-1
   nuc = new Nucleus("TE",117,52,1,116.909,0.296,0.103,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.296,100);

   // Adding 53-I-117-0
   nuc = new Nucleus("I",117,53,0,116.914,0,133.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.65365,100);

   // Adding 54-XE-117-0
   nuc = new Nucleus("XE",117,54,0,116.921,0,61,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,4.02178,0.0029);
   nuc->AddDecay(0,-1,0,6.44615,99.9971);

   // Adding 55-CS-117-0
   nuc = new Nucleus("CS",117,55,0,116.929,0,8.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.52362,100);

   // Adding 55-CS-117-1
   nuc = new Nucleus("CS",117,55,1,116.929,0,6.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.52362,100);

   // Adding 56-BA-117-0
   nuc = new Nucleus("BA",117,56,0,116.939,0,1.75,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,8.6596,0);
   nuc->AddDecay(0,-1,0,9.51925,100);
   nuc->AddDecay(-4,-3,0,11.7365,0);

   // Adding 46-PD-118-0
   nuc = new Nucleus("PD",118,46,0,117.919,0,2.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.09999,50);
   nuc->AddDecay(0,1,1,3.97199,50);

   // Adding 47-AG-118-0
   nuc = new Nucleus("AG",118,47,0,117.914,0,3.76,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.06461,100);

   // Adding 47-AG-118-1
   nuc = new Nucleus("AG",118,47,1,117.915,0.128,2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.128,41);
   nuc->AddDecay(0,1,-1,7.1926,59);

   // Adding 48-CD-118-0
   nuc = new Nucleus("CD",118,48,0,117.907,0,3018,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.519958,100);

   // Adding 49-IN-118-0
   nuc = new Nucleus("IN",118,49,0,117.906,0,5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.42325,100);

   // Adding 49-IN-118-1
   nuc = new Nucleus("IN",118,49,1,117.906,0.06,267,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.48325,100);

   // Adding 49-IN-118-2
   nuc = new Nucleus("IN",118,49,2,117.907,0.2,8.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.2,98.6);
   nuc->AddDecay(0,1,-2,4.62325,1.4);

   // Adding 50-SN-118-0
   nuc = new Nucleus("SN",118,50,0,117.902,0,0,24.22,0,0,0,0);

   // Adding 51-SB-118-0
   nuc = new Nucleus("SB",118,51,0,117.906,0,216,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.65668,100);

   // Adding 51-SB-118-1
   nuc = new Nucleus("SB",118,51,1,117.906,0.212,18000,0,2.1e-10,2.3e-10,0,0);
   nuc->AddDecay(0,-1,-1,3.86868,100);

   // Adding 52-TE-118-0
   nuc = new Nucleus("TE",118,52,0,117.906,0,518400,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.277725,100);

   // Adding 53-I-118-0
   nuc = new Nucleus("I",118,53,0,117.913,0,822,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.04443,100);

   // Adding 53-I-118-1
   nuc = new Nucleus("I",118,53,1,117.913,0,510,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.04443,100);

   // Adding 54-XE-118-0
   nuc = new Nucleus("XE",118,54,0,117.917,0,228,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.94477,100);

   // Adding 55-CS-118-0
   nuc = new Nucleus("CS",118,55,0,117.927,0,14,0,0,0,1,0);
   nuc->AddDecay(-4,-3,0,11.0802,0);
   nuc->AddDecay(0,-1,0,9.29999,100);
   nuc->AddDecay(-1,-2,0,4.73454,0);

   // Adding 55-CS-118-1
   nuc = new Nucleus("CS",118,55,1,117.927,0,17,0,0,0,1,0);
   nuc->AddDecay(-4,-3,-1,11.0802,0);
   nuc->AddDecay(0,-1,-1,9.29999,100);
   nuc->AddDecay(-1,-2,-1,4.73454,0);

   // Adding 56-BA-118-0
   nuc = new Nucleus("BA",118,56,0,117.933,0,5.5,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,6.42762,100);

   // Adding 46-PD-119-0
   nuc = new Nucleus("PD",119,46,0,118.923,0,0.92,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.53177,100);

   // Adding 47-AG-119-0
   nuc = new Nucleus("AG",119,47,0,118.916,0,6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.34999,79);
   nuc->AddDecay(0,1,1,5.20299,21);

   // Adding 47-AG-119-1
   nuc = new Nucleus("AG",119,47,1,118.916,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.34999,100);

   // Adding 48-CD-119-0
   nuc = new Nucleus("CD",119,48,0,118.91,0,161.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.79662,6.8);
   nuc->AddDecay(0,1,1,3.48562,93.2);

   // Adding 48-CD-119-1
   nuc = new Nucleus("CD",119,48,1,118.91,0.147,132,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.94362,98.6);
   nuc->AddDecay(0,1,0,3.63262,1.4);

   // Adding 49-IN-119-0
   nuc = new Nucleus("IN",119,49,0,118.906,0,144,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.36395,99.07);
   nuc->AddDecay(0,1,1,2.27396,0.93);

   // Adding 49-IN-119-1
   nuc = new Nucleus("IN",119,49,1,118.906,0.311,1080,0,4.7e-11,2.9e-11,0,0);
   nuc->AddDecay(0,1,-1,2.67495,94.4);
   nuc->AddDecay(0,0,-1,0.311,5.6);

   // Adding 50-SN-119-0
   nuc = new Nucleus("SN",119,50,0,118.903,0,0,8.58,0,0,0,0);

   // Adding 50-SN-119-1
   nuc = new Nucleus("SN",119,50,1,118.903,0.09,2.53238e+07,0,3.4e-10,2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.09,100);

   // Adding 51-SB-119-0
   nuc = new Nucleus("SB",119,51,0,118.904,0,137484,0,8.1e-11,5.9e-11,0,0);
   nuc->AddDecay(0,-1,0,0.593742,100);

   // Adding 52-TE-119-0
   nuc = new Nucleus("TE",119,52,0,118.906,0,57708,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.293,100);

   // Adding 52-TE-119-1
   nuc = new Nucleus("TE",119,52,1,118.907,0.261,406080,0,8.3e-10,6.3e-10,0,0);
   nuc->AddDecay(0,-1,-1,2.554,100);
   nuc->AddDecay(0,0,-1,0.261,0.008);

   // Adding 53-I-119-0
   nuc = new Nucleus("I",119,53,0,118.91,0,1146,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.51363,100);

   // Adding 54-XE-119-0
   nuc = new Nucleus("XE",119,54,0,118.916,0,348,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.00165,100);

   // Adding 55-CS-119-0
   nuc = new Nucleus("CS",119,55,0,118.922,0,43,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.32789,100);

   // Adding 55-CS-119-1
   nuc = new Nucleus("CS",119,55,1,118.922,0,30.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.32789,100);

   // Adding 56-BA-119-0
   nuc = new Nucleus("BA",119,56,0,118.931,0,5.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.09661,100);
   nuc->AddDecay(-1,-2,0,6.2,0);

   // Adding 46-PD-120-0
   nuc = new Nucleus("PD",120,46,0,119.924,0,0.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.00301,100);

   // Adding 47-AG-120-0
   nuc = new Nucleus("AG",120,47,0,119.919,0,1.23,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.20001,100);

   // Adding 47-AG-120-1
   nuc = new Nucleus("AG",120,47,1,119.919,0.203,0.32,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.203,37);
   nuc->AddDecay(0,1,-1,8.40302,63);

   // Adding 48-CD-120-0
   nuc = new Nucleus("CD",120,48,0,119.91,0,50.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.75851,100);

   // Adding 49-IN-120-0
   nuc = new Nucleus("IN",120,49,0,119.908,0,3.08,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.36999,100);

   // Adding 49-IN-120-1
   nuc = new Nucleus("IN",120,49,1,119.908,0,46.2,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.36999,100);

   // Adding 49-IN-120-2
   nuc = new Nucleus("IN",120,49,2,119.908,0,47.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-2,5.36999,100);

   // Adding 50-SN-120-0
   nuc = new Nucleus("SN",120,50,0,119.902,0,0,32.59,0,0,0,0);

   // Adding 51-SB-120-0
   nuc = new Nucleus("SB",120,51,0,119.905,0,953.4,0,1.4e-11,1.2e-11,0,0);
   nuc->AddDecay(0,-1,0,2.68055,100);

   // Adding 51-SB-120-1
   nuc = new Nucleus("SB",120,51,1,119.905,0,497664,0,1.2e-09,1.3e-09,0,0);
   nuc->AddDecay(0,-1,-1,2.68055,100);

   // Adding 52-TE-120-0
   nuc = new Nucleus("TE",120,52,0,119.904,0,0,0.096,0,0,0,0);

   // Adding 53-I-120-0
   nuc = new Nucleus("I",120,53,0,119.91,0,4860,0,3.4e-10,3e-10,0,0);
   nuc->AddDecay(0,-1,0,5.61499,100);

   // Adding 53-I-120-1
   nuc = new Nucleus("I",120,53,1,119.91,0,3180,0,2.2e-10,1.8e-10,0,0);
   nuc->AddDecay(0,-1,-1,5.61499,100);

   // Adding 54-XE-120-0
   nuc = new Nucleus("XE",120,54,0,119.912,0,2400,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.95998,100);

   // Adding 55-CS-120-0
   nuc = new Nucleus("CS",120,55,0,119.921,0,64,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.92192,100);

   // Adding 55-CS-120-1
   nuc = new Nucleus("CS",120,55,1,119.921,0,57,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.92192,100);
   nuc->AddDecay(-1,-2,-1,2.47424,1e-05);

   // Adding 56-BA-120-0
   nuc = new Nucleus("BA",120,56,0,119.926,0,32,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.99999,100);

   // Adding 57-LA-120-0
   nuc = new Nucleus("LA",120,57,0,119.938,0,2.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.2142,0);
   nuc->AddDecay(-1,-2,0,7.35892,100);

   // Adding 47-AG-121-0
   nuc = new Nucleus("AG",121,47,0,120.92,0,0.78,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.39999,99.92);
   nuc->AddDecay(-1,1,0,1.35441,0.08);

   // Adding 48-CD-121-0
   nuc = new Nucleus("CD",121,48,0,120.913,0,13.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.89001,34);
   nuc->AddDecay(0,1,1,4.57601,66);

   // Adding 48-CD-121-1
   nuc = new Nucleus("CD",121,48,1,120.913,0.215,8.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.10501,100);

   // Adding 49-IN-121-0
   nuc = new Nucleus("IN",121,49,0,120.908,0,23.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.3636,88);
   nuc->AddDecay(0,1,1,3.3576,12);

   // Adding 49-IN-121-1
   nuc = new Nucleus("IN",121,49,1,120.908,0.314,232.8,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.6776,98.8);
   nuc->AddDecay(0,0,-1,0.314,1.2);

   // Adding 50-SN-121-0
   nuc = new Nucleus("SN",121,50,0,120.904,0,97416,0,2.3e-10,2.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.3881,100);

   // Adding 50-SN-121-1
   nuc = new Nucleus("SN",121,50,1,120.904,0.006,1.73448e+09,0,3.8e-10,4.2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.006,77.6);
   nuc->AddDecay(0,1,-1,0.394096,22.4);

   // Adding 51-SB-121-0
   nuc = new Nucleus("SB",121,51,0,120.904,0,0,57.3,0,0,0,0);

   // Adding 52-TE-121-0
   nuc = new Nucleus("TE",121,52,0,120.905,0,1.44979e+06,0,4.3e-10,4.4e-10,0,0);
   nuc->AddDecay(0,-1,0,1.0363,100);

   // Adding 52-TE-121-1
   nuc = new Nucleus("TE",121,52,1,120.905,0.294,1.33056e+07,0,2.3e-09,4.3e-09,0,0);
   nuc->AddDecay(0,0,-1,0.294,88.6);
   nuc->AddDecay(0,-1,-1,1.3303,11.4);

   // Adding 53-I-121-0
   nuc = new Nucleus("I",121,53,0,120.907,0,7632,0,8.2e-11,8.6e-11,0,0);
   nuc->AddDecay(0,-1,0,2.27081,100);

   // Adding 54-XE-121-0
   nuc = new Nucleus("XE",121,54,0,120.911,0,2406,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.73229,100);

   // Adding 55-CS-121-0
   nuc = new Nucleus("CS",121,55,0,120.917,0,155,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.40001,100);

   // Adding 55-CS-121-1
   nuc = new Nucleus("CS",121,55,1,120.917,0.068,122,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.068,17);
   nuc->AddDecay(0,-1,-1,5.46801,83);

   // Adding 56-BA-121-0
   nuc = new Nucleus("BA",121,56,0,120.924,0,29.7,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,4.20001,0.02);
   nuc->AddDecay(0,-1,0,6.81468,99.98);

   // Adding 57-LA-121-0
   nuc = new Nucleus("LA",121,57,0,120.933,0,5.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.93323,100);

   // Adding 58-CE-121-0
   nuc = new Nucleus("CE",121,58,0,120.944,0,0,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.9317,100);

   // Adding 59-PR-121-0
   nuc = new Nucleus("PR",121,59,0,120.954,0,1.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.6231,100);

   // Adding 47-AG-122-0
   nuc = new Nucleus("AG",122,47,0,121.923,0,0.56,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.147,99.814);
   nuc->AddDecay(-1,1,0,1.4488,0.186);

   // Adding 47-AG-122-1
   nuc = new Nucleus("AG",122,47,1,121.923,0,1.5,0,0,0,0,0);
   nuc->AddDecay(-1,1,-1,1.4488,0);
   nuc->AddDecay(0,1,-1,9.147,100);

   // Adding 48-CD-122-0
   nuc = new Nucleus("CD",122,48,0,121.913,0,5.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.00124,100);

   // Adding 49-IN-122-0
   nuc = new Nucleus("IN",122,49,0,121.91,0,1.501,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.36858,100);

   // Adding 49-IN-122-1
   nuc = new Nucleus("IN",122,49,1,121.91,0.11,10.3,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.47858,100);

   // Adding 49-IN-122-2
   nuc = new Nucleus("IN",122,49,2,121.911,0.22,10.8,0,0,0,0,0);
   nuc->AddDecay(0,1,-2,6.58858,100);

   // Adding 50-SN-122-0
   nuc = new Nucleus("SN",122,50,0,121.903,0,0,4.63,0,0,0,0);

   // Adding 51-SB-122-0
   nuc = new Nucleus("SB",122,51,0,121.905,0,233280,0,1.7e-09,1.2e-09,0,0);
   nuc->AddDecay(0,1,0,1.9786,97.6);
   nuc->AddDecay(0,-1,0,1.61971,2.4);

   // Adding 51-SB-122-1
   nuc = new Nucleus("SB",122,51,1,121.905,0.164,252.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.164,100);

   // Adding 52-TE-122-0
   nuc = new Nucleus("TE",122,52,0,121.903,0,0,2.6,0,0,0,0);

   // Adding 53-I-122-0
   nuc = new Nucleus("I",122,53,0,121.908,0,217.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.23397,100);

   // Adding 54-XE-122-0
   nuc = new Nucleus("XE",122,54,0,121.909,0,72360,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.89447,100);

   // Adding 55-CS-122-0
   nuc = new Nucleus("CS",122,55,0,121.916,0,21,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.05477,100);

   // Adding 55-CS-122-1
   nuc = new Nucleus("CS",122,55,1,121.916,0,270,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.05477,100);

   // Adding 55-CS-122-2
   nuc = new Nucleus("CS",122,55,2,121.916,0,0.36,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0,100);

   // Adding 56-BA-122-0
   nuc = new Nucleus("BA",122,56,0,121.92,0,117,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.84241,100);

   // Adding 57-LA-122-0
   nuc = new Nucleus("LA",122,57,0,121.931,0,8.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.7337,100);
   nuc->AddDecay(-1,-2,0,5.31702,0);

   // Adding 58-CE-122-0
   nuc = new Nucleus("CE",122,58,0,121.938,0,8.701,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.80359,100);

   // Adding 47-AG-123-0
   nuc = new Nucleus("AG",123,47,0,122.925,0,0.309,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.35734,99.45);
   nuc->AddDecay(-1,1,0,2.5476,0.55);

   // Adding 48-CD-123-0
   nuc = new Nucleus("CD",123,48,0,122.917,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.11498,77);
   nuc->AddDecay(0,1,1,5.78798,23);

   // Adding 48-CD-123-1
   nuc = new Nucleus("CD",123,48,1,122.917,0.317,1.82,0,0,0,0,-8);
   nuc->AddDecay(0,1,-1,6.43198,50);
   nuc->AddDecay(0,0,-1,0.317,50);

   // Adding 49-IN-123-0
   nuc = new Nucleus("IN",123,49,0,122.91,0,5.98,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.39101,3.5);
   nuc->AddDecay(0,1,1,4.366,96.5);

   // Adding 49-IN-123-1
   nuc = new Nucleus("IN",123,49,1,122.911,0.327,47.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.69301,100);

   // Adding 50-SN-123-0
   nuc = new Nucleus("SN",123,50,0,122.906,0,1.11629e+07,0,2.1e-09,7.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.40359,100);

   // Adding 50-SN-123-1
   nuc = new Nucleus("SN",123,50,1,122.906,0.025,2403.6,0,3.8e-11,4.4e-11,0,0);
   nuc->AddDecay(0,1,-1,1.4286,100);

   // Adding 51-SB-123-0
   nuc = new Nucleus("SB",123,51,0,122.904,0,0,42.7,0,0,0,0);

   // Adding 52-TE-123-0
   nuc = new Nucleus("TE",123,52,0,122.904,0,3.1536e+20,0.908,4.4e-09,5e-09,0,-5);
   nuc->AddDecay(0,-1,0,0.0513,100);

   // Adding 52-TE-123-1
   nuc = new Nucleus("TE",123,52,1,122.905,0.248,1.03421e+07,0,1.4e-09,3.9e-09,0,0);
   nuc->AddDecay(0,0,-1,0.248,100);

   // Adding 53-I-123-0
   nuc = new Nucleus("I",123,53,0,122.906,0,47772,0,2.2e-10,2.1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.24208,100);

   // Adding 54-XE-123-0
   nuc = new Nucleus("XE",123,54,0,122.908,0,7488,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.67603,100);

   // Adding 55-CS-123-0
   nuc = new Nucleus("CS",123,55,0,122.913,0,356.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.2003,100);

   // Adding 55-CS-123-1
   nuc = new Nucleus("CS",123,55,1,122.913,0.157,1.64,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.157,100);

   // Adding 56-BA-123-0
   nuc = new Nucleus("BA",123,56,0,122.919,0,162,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.46121,100);

   // Adding 57-LA-123-0
   nuc = new Nucleus("LA",123,57,0,122.926,0,17,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.8838,100);

   // Adding 58-CE-123-0
   nuc = new Nucleus("CE",123,58,0,122.936,0,3.8,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,6.9155,0);
   nuc->AddDecay(0,-1,0,8.6347,100);

   // Adding 47-AG-124-0
   nuc = new Nucleus("AG",124,47,0,123.929,0,0.22,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.66694,0.1);
   nuc->AddDecay(0,1,0,10.1359,99.9);

   // Adding 48-CD-124-0
   nuc = new Nucleus("CD",124,48,0,123.918,0,1.24,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.16599,100);

   // Adding 49-IN-124-0
   nuc = new Nucleus("IN",124,49,0,123.913,0,3.17,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.35996,100);

   // Adding 49-IN-124-1
   nuc = new Nucleus("IN",124,49,1,123.913,0.19,3.4,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.54996,100);

   // Adding 50-SN-124-0
   nuc = new Nucleus("SN",124,50,0,123.905,0,0,5.79,0,0,0,0);

   // Adding 51-SB-124-0
   nuc = new Nucleus("SB",124,51,0,123.906,0,5.20128e+06,0,2.5e-09,6.1e-09,0,0);
   nuc->AddDecay(0,1,0,2.90529,100);

   // Adding 51-SB-124-1
   nuc = new Nucleus("SB",124,51,1,123.906,0.011,93,0,8e-12,8.3e-12,0,0);
   nuc->AddDecay(0,0,-1,0.011,75);
   nuc->AddDecay(0,1,-1,2.91629,25);

   // Adding 51-SB-124-2
   nuc = new Nucleus("SB",124,51,2,123.906,0.037,1212,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.037,100);

   // Adding 52-TE-124-0
   nuc = new Nucleus("TE",124,52,0,123.903,0,0,4.816,0,0,0,0);

   // Adding 53-I-124-0
   nuc = new Nucleus("I",124,53,0,123.906,0,361152,0,1.3e-08,1.2e-08,0,0);
   nuc->AddDecay(0,-1,0,3.15948,100);

   // Adding 54-XE-124-0
   nuc = new Nucleus("XE",124,54,0,123.906,0,0,0.1,0,0,0,0);

   // Adding 55-CS-124-0
   nuc = new Nucleus("CS",124,55,0,123.912,0,30.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.91701,100);

   // Adding 55-CS-124-1
   nuc = new Nucleus("CS",124,55,1,123.913,0.463,6.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.463,100);

   // Adding 56-BA-124-0
   nuc = new Nucleus("BA",124,56,0,123.915,0,714,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.6465,100);

   // Adding 57-LA-124-0
   nuc = new Nucleus("LA",124,57,0,123.925,0,29,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.79391,100);

   // Adding 58-CE-124-0
   nuc = new Nucleus("CE",124,58,0,123.931,0,6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.5797,100);

   // Adding 59-PR-124-0
   nuc = new Nucleus("PR",124,59,0,123.943,0,1.2,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,8.3966,0);
   nuc->AddDecay(0,-1,0,11.6988,100);

   // Adding 48-CD-125-0
   nuc = new Nucleus("CD",125,48,0,124.921,0,0.65,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.15897,70);
   nuc->AddDecay(0,1,1,6.79897,30);

   // Adding 48-CD-125-1
   nuc = new Nucleus("CD",125,48,1,124.921,0.049,0.57,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.20798,100);

   // Adding 49-IN-125-0
   nuc = new Nucleus("IN",125,49,0,124.914,0,2.36,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.41799,88);
   nuc->AddDecay(0,1,1,5.38999,12);

   // Adding 49-IN-125-1
   nuc = new Nucleus("IN",125,49,1,124.914,0.36,12.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.74999,100);

   // Adding 50-SN-125-0
   nuc = new Nucleus("SN",125,50,0,124.908,0,832896,0,3.1e-09,3e-09,0,0);
   nuc->AddDecay(0,1,0,2.36381,100);

   // Adding 50-SN-125-1
   nuc = new Nucleus("SN",125,50,1,124.908,0.028,571.2,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.39181,100);

   // Adding 51-SB-125-0
   nuc = new Nucleus("SB",125,51,0,124.905,0,8.69826e+07,0,1.1e-09,4.5e-09,0,0);
   nuc->AddDecay(0,1,0,0.766693,77);
   nuc->AddDecay(0,1,1,0.621696,23);

   // Adding 52-TE-125-0
   nuc = new Nucleus("TE",125,52,0,124.904,0,0,7.14,0,0,0,0);

   // Adding 52-TE-125-1
   nuc = new Nucleus("TE",125,52,1,124.905,0.145,4.95936e+06,0,8.7e-10,3.3e-09,0,0);
   nuc->AddDecay(0,0,-1,0.145,100);

   // Adding 53-I-125-0
   nuc = new Nucleus("I",125,53,0,124.905,0,5.13285e+06,0,1.5e-08,1.4e-08,0,0);
   nuc->AddDecay(0,-1,0,0.186203,100);

   // Adding 54-XE-125-0
   nuc = new Nucleus("XE",125,54,0,124.906,0,60840,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.6524,100);

   // Adding 54-XE-125-1
   nuc = new Nucleus("XE",125,54,1,124.907,0.253,57,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.253,100);

   // Adding 55-CS-125-0
   nuc = new Nucleus("CS",125,55,0,124.91,0,2700,0,3.5e-11,2.3e-11,0,0);
   nuc->AddDecay(0,-1,0,3.09213,100);

   // Adding 56-BA-125-0
   nuc = new Nucleus("BA",125,56,0,124.915,0,210,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.55996,100);

   // Adding 57-LA-125-0
   nuc = new Nucleus("LA",125,57,0,124.921,0,76,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.64243,100);

   // Adding 58-CE-125-0
   nuc = new Nucleus("CE",125,58,0,124.929,0,9,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,5.23972,0);
   nuc->AddDecay(0,-1,0,7.3297,100);

   // Adding 48-CD-126-0
   nuc = new Nucleus("CD",126,48,0,125.922,0,0.506,0,0,0,0,0);
   nuc->AddDecay(0,1,1,5.38399,100);

   // Adding 49-IN-126-0
   nuc = new Nucleus("IN",126,49,0,125.916,0,1.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.20697,100);

   // Adding 49-IN-126-1
   nuc = new Nucleus("IN",126,49,1,125.917,0.102,1.64,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.30897,100);

   // Adding 50-SN-126-0
   nuc = new Nucleus("SN",126,50,0,125.908,0,3.1536e+12,0,4.8e-09,2.7e-08,0,0);
   nuc->AddDecay(0,1,1,0.360023,33.5);
   nuc->AddDecay(0,1,2,0.33802,66.5);

   // Adding 51-SB-126-0
   nuc = new Nucleus("SB",126,51,0,125.907,0,1.07654e+06,0,2.4e-09,3.2e-09,0,0);
   nuc->AddDecay(0,1,0,3.67298,100);

   // Adding 51-SB-126-1
   nuc = new Nucleus("SB",126,51,1,125.907,0.018,1149,0,3.6e-11,3.3e-11,0,0);
   nuc->AddDecay(0,1,-1,3.69098,86);
   nuc->AddDecay(0,0,-1,0.018,14);

   // Adding 51-SB-126-2
   nuc = new Nucleus("SB",126,51,2,125.907,0.04,11,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.04,100);

   // Adding 52-TE-126-0
   nuc = new Nucleus("TE",126,52,0,125.903,0,0,18.95,0,0,0,0);

   // Adding 53-I-126-0
   nuc = new Nucleus("I",126,53,0,125.906,0,1.1327e+06,0,2.9e-08,2.6e-08,0,0);
   nuc->AddDecay(0,-1,0,2.15527,56.3);
   nuc->AddDecay(0,1,0,1.25802,43.7);

   // Adding 54-XE-126-0
   nuc = new Nucleus("XE",126,54,0,125.904,0,0,0.09,0,0,0,0);

   // Adding 55-CS-126-0
   nuc = new Nucleus("CS",126,55,0,125.909,0,98.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.82555,100);

   // Adding 56-BA-126-0
   nuc = new Nucleus("BA",126,56,0,125.911,0,6000,0,2.6e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.67279,100);

   // Adding 57-LA-126-0
   nuc = new Nucleus("LA",126,57,0,125.919,0,54,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.56902,100);

   // Adding 58-CE-126-0
   nuc = new Nucleus("CE",126,58,0,125.924,0,50,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,4.4058,100);

   // Adding 59-PR-126-0
   nuc = new Nucleus("PR",126,59,0,125.935,0,3.1,0,0,0,0,-8);
   nuc->AddDecay(-1,-2,0,6.3476,50);
   nuc->AddDecay(0,-1,0,10.4418,50);

   // Adding 48-CD-127-0
   nuc = new Nucleus("CD",127,48,0,126.926,0,0.4,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,8.46796,66.6667);
   nuc->AddDecay(0,1,1,8.30796,33.3333);

   // Adding 49-IN-127-0
   nuc = new Nucleus("IN",127,49,0,126.917,0,1.15,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.51398,15.4078);
   nuc->AddDecay(0,1,1,6.50899,84.5922);

   // Adding 49-IN-127-1
   nuc = new Nucleus("IN",127,49,1,126.918,0.16,3.76,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.66899,50.1556);
   nuc->AddDecay(-1,1,-1,1.11448,0);
   nuc->AddDecay(0,1,-1,6.67399,49.8444);

   // Adding 50-SN-127-0
   nuc = new Nucleus("SN",127,50,0,126.91,0,7560,0,2e-10,2e-10,0,0);
   nuc->AddDecay(0,1,0,3.20103,100);

   // Adding 50-SN-127-1
   nuc = new Nucleus("SN",127,50,1,126.91,0.005,247.8,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.20603,100);

   // Adding 51-SB-127-0
   nuc = new Nucleus("SB",127,51,0,126.907,0,332640,0,1.7e-09,1.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.58097,82.5);
   nuc->AddDecay(0,1,1,1.49297,17.5);

   // Adding 52-TE-127-0
   nuc = new Nucleus("TE",127,52,0,126.905,0,33660,0,1.7e-10,1.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.697609,100);

   // Adding 52-TE-127-1
   nuc = new Nucleus("TE",127,52,1,126.905,0.088,9.4176e+06,0,2.3e-09,7.2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.088,97.6);
   nuc->AddDecay(0,1,-1,0.785606,2.4);

   // Adding 53-I-127-0
   nuc = new Nucleus("I",127,53,0,126.904,0,0,100,0,0,0,0);

   // Adding 54-XE-127-0
   nuc = new Nucleus("XE",127,54,0,126.905,0,3.14496e+06,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.6623,100);

   // Adding 54-XE-127-1
   nuc = new Nucleus("XE",127,54,1,126.905,0.297,69.2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.297,100);

   // Adding 55-CS-127-0
   nuc = new Nucleus("CS",127,55,0,126.907,0,22500,0,2.4e-11,4e-11,0,0);
   nuc->AddDecay(0,-1,0,2.08066,100);

   // Adding 56-BA-127-0
   nuc = new Nucleus("BA",127,56,0,126.911,0,762,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.45007,100);

   // Adding 57-LA-127-0
   nuc = new Nucleus("LA",127,57,0,126.916,0,228,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.69849,100);

   // Adding 57-LA-127-1
   nuc = new Nucleus("LA",127,57,1,126.916,0,300,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 58-CE-127-0
   nuc = new Nucleus("CE",127,58,0,126.923,0,32,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.13782,100);

   // Adding 59-PR-127-0
   nuc = new Nucleus("PR",127,59,0,126.931,0,0,0,0,0,0,-6);

   // Adding 60-ND-127-0
   nuc = new Nucleus("ND",127,60,0,126.94,0,1.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.0051,100);
   nuc->AddDecay(-1,-2,0,7.9866,0);

   // Adding 48-CD-128-0
   nuc = new Nucleus("CD",128,48,0,127.928,0,0.28,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.07002,100);

   // Adding 49-IN-128-0
   nuc = new Nucleus("IN",128,49,0,127.92,0,0.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.97558,99.962);
   nuc->AddDecay(-1,1,0,1.07638,0.038);

   // Adding 49-IN-128-1
   nuc = new Nucleus("IN",128,49,1,127.92,0.08,0.7,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,9.05558,0);
   nuc->AddDecay(-1,1,-1,1.15638,100);
   nuc->AddDecay(0,1,0,6.96458,0);

   // Adding 50-SN-128-0
   nuc = new Nucleus("SN",128,50,0,127.911,0,3546,0,1.5e-10,1.5e-10,0,0);
   nuc->AddDecay(0,1,1,1.254,100);

   // Adding 50-SN-128-1
   nuc = new Nucleus("SN",128,50,1,127.913,2.091,6.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.091,100);

   // Adding 51-SB-128-0
   nuc = new Nucleus("SB",128,51,0,127.909,0,32436,0,7.6e-10,6.7e-10,0,0);
   nuc->AddDecay(0,1,0,4.38349,100);

   // Adding 51-SB-128-1
   nuc = new Nucleus("SB",128,51,1,127.909,0.02,624,0,3.3e-11,2.6e-11,0,0);
   nuc->AddDecay(0,1,-1,4.40349,96.4);
   nuc->AddDecay(0,0,-1,0.02,3.6);

   // Adding 52-TE-128-0
   nuc = new Nucleus("TE",128,52,0,127.904,0,0,31.69,0,0,0,0);

   // Adding 53-I-128-0
   nuc = new Nucleus("I",128,53,0,127.906,0,1499.4,0,4.6e-11,6.5e-11,0,0);
   nuc->AddDecay(0,-1,0,1.25098,6.9);
   nuc->AddDecay(0,1,0,2.11818,93.1);

   // Adding 54-XE-128-0
   nuc = new Nucleus("XE",128,54,0,127.904,0,0,1.91,0,0,0,0);

   // Adding 55-CS-128-0
   nuc = new Nucleus("CS",128,55,0,127.908,0,217.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.93006,100);

   // Adding 56-BA-128-0
   nuc = new Nucleus("BA",128,56,0,127.908,0,209952,0,2.7e-09,1.3e-09,0,0);
   nuc->AddDecay(0,-1,0,0.521248,100);

   // Adding 57-LA-128-0
   nuc = new Nucleus("LA",128,57,0,127.915,0,300,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.64997,100);

   // Adding 58-CE-128-0
   nuc = new Nucleus("CE",128,58,0,127.919,0,360,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.18714,100);

   // Adding 59-PR-128-0
   nuc = new Nucleus("PR",128,59,0,127.929,0,3.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.24979,100);
   nuc->AddDecay(-1,-2,0,4.48472,0);

   // Adding 60-ND-128-0
   nuc = new Nucleus("ND",128,60,0,127.935,0,4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.1377,100);
   nuc->AddDecay(-1,-2,0,4.48461,0);

   // Adding 48-CD-129-0
   nuc = new Nucleus("CD",129,48,0,129,0,0.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 49-IN-129-0
   nuc = new Nucleus("IN",129,49,0,128.922,0,0.63,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.655,89.17);
   nuc->AddDecay(-1,1,0,2.28981,0.23);
   nuc->AddDecay(0,1,1,7.62,10.6);

   // Adding 49-IN-129-1
   nuc = new Nucleus("IN",129,49,1,128.922,0.2,1.23,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.855,96.4);
   nuc->AddDecay(-1,1,-1,2.48981,3.6);

   // Adding 50-SN-129-0
   nuc = new Nucleus("SN",129,50,0,128.913,0,144,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.99601,100);

   // Adding 50-SN-129-1
   nuc = new Nucleus("SN",129,50,1,128.913,0.035,414,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.03101,100);
   nuc->AddDecay(0,0,-1,0.035,0.0002);

   // Adding 51-SB-129-0
   nuc = new Nucleus("SB",129,51,0,128.909,0,15840,0,4.2e-10,3.5e-10,0,0);
   nuc->AddDecay(0,1,0,2.37951,82);
   nuc->AddDecay(0,1,1,2.27451,18);

   // Adding 51-SB-129-1
   nuc = new Nucleus("SB",129,51,1,128.909,0,1062,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.37951,100);

   // Adding 52-TE-129-0
   nuc = new Nucleus("TE",129,52,0,128.907,0,4176,0,6.3e-11,5.7e-11,0,0);
   nuc->AddDecay(0,1,0,1.4979,100);

   // Adding 52-TE-129-1
   nuc = new Nucleus("TE",129,52,1,128.907,0.105,2.90304e+06,0,3e-09,6.3e-09,0,0);
   nuc->AddDecay(0,1,-1,1.60291,36);
   nuc->AddDecay(0,0,-1,0.105,64);

   // Adding 53-I-129-0
   nuc = new Nucleus("I",129,53,0,128.905,0,4.95115e+14,0,1.1e-07,9.6e-08,0,0);
   nuc->AddDecay(0,1,0,0.193748,100);

   // Adding 54-XE-129-0
   nuc = new Nucleus("XE",129,54,0,128.905,0,0,26.4,0,0,0,0);

   // Adding 54-XE-129-1
   nuc = new Nucleus("XE",129,54,1,128.905,0.236,768096,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.236,100);

   // Adding 55-CS-129-0
   nuc = new Nucleus("CS",129,55,0,128.906,0,115416,0,6e-11,8.1e-11,0,0);
   nuc->AddDecay(0,-1,0,1.19553,100);

   // Adding 56-BA-129-0
   nuc = new Nucleus("BA",129,56,0,128.909,0,8028,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.43333,100);

   // Adding 56-BA-129-1
   nuc = new Nucleus("BA",129,56,1,128.909,0.008,7812,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,2.44134,100);

   // Adding 57-LA-129-0
   nuc = new Nucleus("LA",129,57,0,128.913,0,696,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.71996,100);

   // Adding 57-LA-129-1
   nuc = new Nucleus("LA",129,57,1,128.913,0.172,0.56,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.172,100);

   // Adding 58-CE-129-0
   nuc = new Nucleus("CE",129,58,0,128.918,0,210,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,5.05005,100);

   // Adding 59-PR-129-0
   nuc = new Nucleus("PR",129,59,0,128.925,0,24,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,6.3058,100);

   // Adding 60-ND-129-0
   nuc = new Nucleus("ND",129,60,0,128.933,0,4.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.8236,100);
   nuc->AddDecay(-1,-2,0,6.1145,0);

   // Adding 48-CD-130-0
   nuc = new Nucleus("CD",130,48,0,129.931,0,0.2,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.544515,4);
   nuc->AddDecay(0,1,0,5.63435,96);

   // Adding 49-IN-130-0
   nuc = new Nucleus("IN",130,49,0,129.925,0,0.32,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.56517,0.9);
   nuc->AddDecay(0,1,0,10.249,70.0678);
   nuc->AddDecay(0,1,1,8.30199,29.0322);

   // Adding 49-IN-130-1
   nuc = new Nucleus("IN",130,49,1,129.925,0.05,0.55,0,0,0,0,0);
   nuc->AddDecay(-1,1,-1,2.61517,1.67);
   nuc->AddDecay(0,1,0,8.35199,98.33);

   // Adding 49-IN-130-2
   nuc = new Nucleus("IN",130,49,2,129.925,0.4,0.55,0,0,0,0,0);
   nuc->AddDecay(0,1,-2,10.649,82.5972);
   nuc->AddDecay(-1,1,-2,2.96517,1.67);
   nuc->AddDecay(0,1,-1,8.70199,15.7328);

   // Adding 50-SN-130-0
   nuc = new Nucleus("SN",130,50,0,129.914,0,223.2,0,0,0,0,0);
   nuc->AddDecay(0,1,1,2.15039,100);

   // Adding 50-SN-130-1
   nuc = new Nucleus("SN",130,50,1,129.916,1.947,102,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.09739,100);

   // Adding 51-SB-130-0
   nuc = new Nucleus("SB",130,51,0,129.912,0,2370,0,9.1e-11,9.1e-11,0,0);
   nuc->AddDecay(0,1,0,4.96,100);

   // Adding 51-SB-130-1
   nuc = new Nucleus("SB",130,51,1,129.912,0,378,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.96,100);

   // Adding 52-TE-130-0
   nuc = new Nucleus("TE",130,52,0,129.906,0,3.942e+28,33.8,0,0,0,-4);
   nuc->AddDecay(0,1,0,-0.420486,100);

   // Adding 53-I-130-0
   nuc = new Nucleus("I",130,53,0,129.907,0,44496,0,2e-09,1.9e-09,0,0);
   nuc->AddDecay(0,1,0,2.94858,100);

   // Adding 53-I-130-1
   nuc = new Nucleus("I",130,53,1,129.907,0.04,540,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.98858,16);
   nuc->AddDecay(0,0,-1,0.04,84);

   // Adding 54-XE-130-0
   nuc = new Nucleus("XE",130,54,0,129.904,0,0,4.1,0,0,0,0);

   // Adding 55-CS-130-0
   nuc = new Nucleus("CS",130,55,0,129.907,0,1752.6,0,2.8e-11,1.5e-11,0,0);
   nuc->AddDecay(0,1,0,0.37278,1.6);
   nuc->AddDecay(0,-1,0,2.98283,98.4);

   // Adding 55-CS-130-1
   nuc = new Nucleus("CS",130,55,1,129.907,0.163,207.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.163,99.84);
   nuc->AddDecay(0,-1,-1,3.14583,0.16);

   // Adding 56-BA-130-0
   nuc = new Nucleus("BA",130,56,0,129.906,0,0,0.106,0,0,0,0);

   // Adding 56-BA-130-1
   nuc = new Nucleus("BA",130,56,1,129.909,2.475,0.011,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.475,100);

   // Adding 57-LA-130-0
   nuc = new Nucleus("LA",130,57,0,129.912,0,522,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.59766,100);

   // Adding 58-CE-130-0
   nuc = new Nucleus("CE",130,58,0,129.915,0,1500,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.207,100);

   // Adding 59-PR-130-0
   nuc = new Nucleus("PR",130,59,0,129.923,0,40,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.0949,100);

   // Adding 60-ND-130-0
   nuc = new Nucleus("ND",130,60,0,129.929,0,28,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.0296,100);

   // Adding 61-PM-130-0
   nuc = new Nucleus("PM",130,61,0,129.94,0,2.2,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,7.2326,0);
   nuc->AddDecay(0,-1,0,10.8709,100);

   // Adding 49-IN-131-0
   nuc = new Nucleus("IN",131,49,0,130.927,0,0.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.18079,92.8919);
   nuc->AddDecay(-1,1,0,3.97185,2.2);
   nuc->AddDecay(0,1,1,8.9408,4.9081);

   // Adding 49-IN-131-1
   nuc = new Nucleus("IN",131,49,1,130.927,0,0.35,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,9.18079,99.982);

   // Adding 49-IN-131-2
   nuc = new Nucleus("IN",131,49,2,130.927,0,0.32,0,0,0,0,0);
   nuc->AddDecay(0,1,-2,9.18079,99);

   // Adding 50-SN-131-0
   nuc = new Nucleus("SN",131,50,0,130.917,0,61,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.6411,100);

   // Adding 50-SN-131-1
   nuc = new Nucleus("SN",131,50,1,130.917,0.24,39,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.8811,100);

   // Adding 51-SB-131-0
   nuc = new Nucleus("SB",131,51,0,130.912,0,1380,0,1e-10,8.3e-11,0,0);
   nuc->AddDecay(0,1,0,3.19005,93.2);
   nuc->AddDecay(0,1,1,3.00805,6.8);

   // Adding 52-TE-131-0
   nuc = new Nucleus("TE",131,52,0,130.909,0,1500,0,8.7e-11,6.1e-11,0,0);
   nuc->AddDecay(0,1,0,2.2327,100);

   // Adding 52-TE-131-1
   nuc = new Nucleus("TE",131,52,1,130.909,0.182,108000,0,1.9e-09,1.6e-09,0,0);
   nuc->AddDecay(0,1,-1,2.4147,77.8);
   nuc->AddDecay(0,0,-1,0.182,22.2);

   // Adding 53-I-131-0
   nuc = new Nucleus("I",131,53,0,130.906,0,694656,0,2.2e-08,2e-08,0,0);
   nuc->AddDecay(0,1,0,0.970901,98.914);
   nuc->AddDecay(0,1,1,0.8069,1.086);

   // Adding 54-XE-131-0
   nuc = new Nucleus("XE",131,54,0,130.905,0,0,21.2,0,0,0,0);

   // Adding 54-XE-131-1
   nuc = new Nucleus("XE",131,54,1,130.905,0.164,1.02816e+06,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.164,100);

   // Adding 55-CS-131-0
   nuc = new Nucleus("CS",131,55,0,130.905,0,837216,0,5.8e-11,4.6e-11,0,0);
   nuc->AddDecay(0,-1,0,0.351662,100);

   // Adding 56-BA-131-0
   nuc = new Nucleus("BA",131,56,0,130.907,0,1.01952e+06,0,4.5e-10,3.5e-10,0,0);
   nuc->AddDecay(0,-1,0,1.37009,100);

   // Adding 56-BA-131-1
   nuc = new Nucleus("BA",131,56,1,130.907,0.188,876,0,4.9e-12,6.4e-12,0,0);
   nuc->AddDecay(0,0,-1,0.188,100);

   // Adding 57-LA-131-0
   nuc = new Nucleus("LA",131,57,0,130.91,0,3540,0,3.5e-11,3.6e-11,0,0);
   nuc->AddDecay(0,-1,0,2.96006,100);

   // Adding 58-CE-131-0
   nuc = new Nucleus("CE",131,58,0,130.914,0,600,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.01997,100);

   // Adding 58-CE-131-1
   nuc = new Nucleus("CE",131,58,1,130.914,0,300,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,4.01997,100);

   // Adding 59-PR-131-0
   nuc = new Nucleus("PR",131,59,0,130.92,0,102,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.25,100);

   // Adding 60-ND-131-0
   nuc = new Nucleus("ND",131,60,0,130.927,0,24,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,4.27416,0);
   nuc->AddDecay(0,-1,0,6.56,100);

   // Adding 61-PM-131-0
   nuc = new Nucleus("PM",131,61,0,130.936,0,0,0,0,0,0,-6);

   // Adding 62-SM-131-0
   nuc = new Nucleus("SM",131,62,0,130.945,0,1.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.5174,100);
   nuc->AddDecay(-1,-2,0,7.7701,0);

   // Adding 49-IN-132-0
   nuc = new Nucleus("IN",132,49,0,131.932,0,0.201,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,6.28833,6.2);
   nuc->AddDecay(0,1,0,13.6,93.8);

   // Adding 50-SN-132-0
   nuc = new Nucleus("SN",132,50,0,131.918,0,39.7,0,0,0,0,0);
   nuc->AddDecay(0,1,1,3.10303,100);

   // Adding 51-SB-132-0
   nuc = new Nucleus("SB",132,51,0,131.914,0,167.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.28596,100);

   // Adding 51-SB-132-1
   nuc = new Nucleus("SB",132,51,1,131.914,0.2,246,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.48595,100);

   // Adding 52-TE-132-0
   nuc = new Nucleus("TE",132,52,0,131.909,0,276826,0,3.7e-09,3e-09,0,0);
   nuc->AddDecay(0,1,0,0.492996,100);

   // Adding 53-I-132-0
   nuc = new Nucleus("I",132,53,0,131.908,0,8262,0,2.9e-10,3.1e-10,0,0);
   nuc->AddDecay(0,1,0,3.577,100);

   // Adding 53-I-132-1
   nuc = new Nucleus("I",132,53,1,131.908,0.12,4993.2,0,2.2e-10,2.7e-10,0,0);
   nuc->AddDecay(0,1,-1,3.69701,14);
   nuc->AddDecay(0,0,-1,0.12,86);

   // Adding 54-XE-132-0
   nuc = new Nucleus("XE",132,54,0,131.904,0,0,26.9,0,0,0,0);

   // Adding 54-XE-132-1
   nuc = new Nucleus("XE",132,54,1,131.907,2.752,0.00839,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.752,100);

   // Adding 55-CS-132-0
   nuc = new Nucleus("CS",132,55,0,131.906,0,559786,0,5e-10,3.8e-10,0,0);
   nuc->AddDecay(0,-1,0,2.11948,98.13);
   nuc->AddDecay(0,1,0,1.2794,1.87);

   // Adding 56-BA-132-0
   nuc = new Nucleus("BA",132,56,0,131.905,0,0,0.101,0,0,0,0);

   // Adding 57-LA-132-0
   nuc = new Nucleus("LA",132,57,0,131.91,0,17280,0,3.9e-10,2.8e-10,0,0);
   nuc->AddDecay(0,-1,0,4.70798,100);

   // Adding 57-LA-132-1
   nuc = new Nucleus("LA",132,57,1,131.91,0.188,1458,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.188,76);
   nuc->AddDecay(0,-1,-1,4.89599,24);

   // Adding 58-CE-132-0
   nuc = new Nucleus("CE",132,58,0,131.911,0,12636,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.28415,100);

   // Adding 59-PR-132-0
   nuc = new Nucleus("PR",132,59,0,131.919,0,96,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.10799,100);

   // Adding 60-ND-132-0
   nuc = new Nucleus("ND",132,60,0,131.923,0,105,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.7257,100);

   // Adding 61-PM-132-0
   nuc = new Nucleus("PM",132,61,0,131.934,0,6.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.9018,99.9999);
   nuc->AddDecay(-1,-2,0,5.46234,5e-05);

   // Adding 62-SM-132-0
   nuc = new Nucleus("SM",132,62,0,131.94,0,4,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,4.64095,0);
   nuc->AddDecay(0,-1,0,5.7386,100);

   // Adding 49-IN-133-0
   nuc = new Nucleus("IN",133,49,0,132.935,0,0.18,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,8.22132,0);
   nuc->AddDecay(0,1,0,10.7986,100);

   // Adding 50-SN-133-0
   nuc = new Nucleus("SN",133,50,0,132.924,0,1.44,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.83006,99.92);
   nuc->AddDecay(-1,1,0,0.725743,0.08);

   // Adding 51-SB-133-0
   nuc = new Nucleus("SB",133,51,0,132.915,0,150,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.003,71);
   nuc->AddDecay(0,1,1,3.669,29);

   // Adding 52-TE-133-0
   nuc = new Nucleus("TE",133,52,0,132.911,0,750,0,7.2e-11,4.4e-11,0,0);
   nuc->AddDecay(0,1,0,2.91795,100);

   // Adding 52-TE-133-1
   nuc = new Nucleus("TE",133,52,1,132.911,0.334,3324,0,2.8e-10,1.9e-10,0,0);
   nuc->AddDecay(0,1,-1,3.25195,72.3);
   nuc->AddDecay(0,0,-1,0.334,17.5);
   nuc->AddDecay(0,1,0,1.61795,10.2);

   // Adding 53-I-133-0
   nuc = new Nucleus("I",133,53,0,132.908,0,74880,0,4.3e-09,4e-09,0,0);
   nuc->AddDecay(0,1,0,1.77061,97.12);
   nuc->AddDecay(0,1,1,1.53761,2.88);

   // Adding 53-I-133-1
   nuc = new Nucleus("I",133,53,1,132.91,1.634,9,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.634,100);

   // Adding 54-XE-133-0
   nuc = new Nucleus("XE",133,54,0,132.906,0,452995,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.427391,100);

   // Adding 54-XE-133-1
   nuc = new Nucleus("XE",133,54,1,132.906,0.233,189216,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.233,100);

   // Adding 55-CS-133-0
   nuc = new Nucleus("CS",133,55,0,132.905,0,0,100,0,0,0,0);

   // Adding 56-BA-133-0
   nuc = new Nucleus("BA",133,56,0,132.906,0,3.31759e+08,0,1e-09,1.8e-09,0,0);
   nuc->AddDecay(0,-1,0,0.517502,100);

   // Adding 56-BA-133-1
   nuc = new Nucleus("BA",133,56,1,132.906,0.288,140040,0,5.5e-10,2.8e-10,0,0);
   nuc->AddDecay(0,-1,-1,0.805504,0.01);
   nuc->AddDecay(0,0,-1,0.288,99.99);

   // Adding 57-LA-133-0
   nuc = new Nucleus("LA",133,57,0,132.908,0,14083.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.23001,100);

   // Adding 58-CE-133-0
   nuc = new Nucleus("CE",133,58,0,132.912,0,17640,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.93672,100);

   // Adding 58-CE-133-1
   nuc = new Nucleus("CE",133,58,1,132.912,0,5820,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.93672,100);

   // Adding 59-PR-133-0
   nuc = new Nucleus("PR",133,59,0,132.916,0,390,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.33199,100);

   // Adding 60-ND-133-0
   nuc = new Nucleus("ND",133,60,0,132.922,0,70,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.5978,100);

   // Adding 60-ND-133-1
   nuc = new Nucleus("ND",133,60,1,132.922,0,120,0,0,0,0,-4);
   nuc->AddDecay(0,-1,-1,5.5978,100);

   // Adding 61-PM-133-0
   nuc = new Nucleus("PM",133,61,0,132.93,0,12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.9958,100);

   // Adding 62-SM-133-0
   nuc = new Nucleus("SM",133,62,0,132.939,0,2.9,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,7.2507,100);
   nuc->AddDecay(0,-1,0,8.3918,0);

   // Adding 50-SN-134-0
   nuc = new Nucleus("SN",134,50,0,133.928,0,1.04,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.75372,83);
   nuc->AddDecay(-1,1,0,3.65888,17);

   // Adding 51-SB-134-0
   nuc = new Nucleus("SB",134,51,0,133.921,0,0.85,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,8.41439,100);

   // Adding 51-SB-134-1
   nuc = new Nucleus("SB",134,51,1,133.921,0.06,10.43,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,8.47439,99.9);
   nuc->AddDecay(-1,1,-1,0.968152,0.1);

   // Adding 52-TE-134-0
   nuc = new Nucleus("TE",134,52,0,133.912,0,2508,0,1.1e-10,1.1e-10,0,0);
   nuc->AddDecay(0,1,0,1.56005,89.8);
   nuc->AddDecay(0,1,1,1.24405,10.2);

   // Adding 53-I-134-0
   nuc = new Nucleus("I",134,53,0,133.91,0,3156,0,1.1e-10,1.5e-10,0,0);
   nuc->AddDecay(0,1,0,4.17002,100);

   // Adding 53-I-134-1
   nuc = new Nucleus("I",134,53,1,133.91,0.316,221.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.316,97.7);
   nuc->AddDecay(0,1,0,2.52702,2.3);

   // Adding 54-XE-134-0
   nuc = new Nucleus("XE",134,54,0,133.905,0,0,10.4,0,0,0,0);

   // Adding 54-XE-134-1
   nuc = new Nucleus("XE",134,54,1,133.908,1.959,0.29,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.959,100);

   // Adding 55-CS-134-0
   nuc = new Nucleus("CS",134,55,0,133.907,0,6.50272e+07,0,1.9e-08,9.6e-09,0,0);
   nuc->AddDecay(0,1,0,2.0587,100);
   nuc->AddDecay(0,-1,0,1.22885,0.0003);

   // Adding 55-CS-134-1
   nuc = new Nucleus("CS",134,55,1,133.907,0.139,10476,0,2e-11,2.6e-11,0,0);
   nuc->AddDecay(0,0,-1,0.139,100);

   // Adding 56-BA-134-0
   nuc = new Nucleus("BA",134,56,0,133.905,0,0,2.417,0,0,0,0);

   // Adding 57-LA-134-0
   nuc = new Nucleus("LA",134,57,0,133.908,0,387,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.7132,100);

   // Adding 58-CE-134-0
   nuc = new Nucleus("CE",134,58,0,133.909,0,273240,0,2.5e-09,1.6e-09,0,0);
   nuc->AddDecay(0,-1,0,0.500008,100);

   // Adding 59-PR-134-0
   nuc = new Nucleus("PR",134,59,0,133.916,0,1020,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.20692,100);

   // Adding 59-PR-134-1
   nuc = new Nucleus("PR",134,59,1,133.916,0,660,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,6.20692,100);

   // Adding 60-ND-134-0
   nuc = new Nucleus("ND",134,60,0,133.919,0,510,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.76995,100);

   // Adding 61-PM-134-0
   nuc = new Nucleus("PM",134,61,0,133.928,0,24,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.88274,100);

   // Adding 62-SM-134-0
   nuc = new Nucleus("SM",134,62,0,133.934,0,11,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.4208,100);

   // Adding 63-EU-134-0
   nuc = new Nucleus("EU",134,63,0,134,0,0.5,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,0,100);
   nuc->AddDecay(0,-1,0,0,0);

   // Adding 51-SB-135-0
   nuc = new Nucleus("SB",135,51,0,134.925,0,1.71,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.12009,83.6);
   nuc->AddDecay(-1,1,0,4.6178,16.4);

   // Adding 52-TE-135-0
   nuc = new Nucleus("TE",135,52,0,134.916,0,19,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.96193,100);

   // Adding 53-I-135-0
   nuc = new Nucleus("I",135,53,0,134.91,0,23652,0,9.3e-10,9.2e-10,0,0);
   nuc->AddDecay(0,1,0,2.64809,84.5);
   nuc->AddDecay(0,1,1,2.12109,15.5);

   // Adding 54-XE-135-0
   nuc = new Nucleus("XE",135,54,0,134.907,0,32904,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.15092,100);

   // Adding 54-XE-135-1
   nuc = new Nucleus("XE",135,54,1,134.908,0.527,917.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.527,100);
   nuc->AddDecay(0,1,-1,1.67793,0.004);

   // Adding 55-CS-135-0
   nuc = new Nucleus("CS",135,55,0,134.906,0,7.25328e+13,0,2e-09,9.9e-10,0,0);
   nuc->AddDecay(0,1,0,0.2686,100);

   // Adding 55-CS-135-1
   nuc = new Nucleus("CS",135,55,1,134.908,1.633,3180,0,1.9e-11,2.4e-11,0,0);
   nuc->AddDecay(0,0,-1,1.633,100);

   // Adding 56-BA-135-0
   nuc = new Nucleus("BA",135,56,0,134.906,0,0,6.592,0,0,0,0);

   // Adding 56-BA-135-1
   nuc = new Nucleus("BA",135,56,1,134.906,0.268,103320,0,4.5e-10,2.3e-10,0,0);
   nuc->AddDecay(0,0,-1,0.268,100);

   // Adding 57-LA-135-0
   nuc = new Nucleus("LA",135,57,0,134.907,0,70200,0,3e-11,2.5e-11,0,0);
   nuc->AddDecay(0,-1,0,1.20003,100);

   // Adding 58-CE-135-0
   nuc = new Nucleus("CE",135,58,0,134.909,0,63720,0,8e-10,7.6e-10,0,0);
   nuc->AddDecay(0,-1,0,2.0256,100);

   // Adding 58-CE-135-1
   nuc = new Nucleus("CE",135,58,1,134.91,0.446,20,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.446,100);

   // Adding 59-PR-135-0
   nuc = new Nucleus("PR",135,59,0,134.913,0,1440,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.71989,100);

   // Adding 60-ND-135-0
   nuc = new Nucleus("ND",135,60,0,134.918,0,744,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.75021,100);

   // Adding 60-ND-135-1
   nuc = new Nucleus("ND",135,60,1,134.918,0,330,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.75021,100);

   // Adding 61-PM-135-0
   nuc = new Nucleus("PM",135,61,0,134.925,0,49,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.0178,100);

   // Adding 62-SM-135-0
   nuc = new Nucleus("SM",135,62,0,134.932,0,10,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.12481,0);
   nuc->AddDecay(-1,-2,0,5.45865,100);

   // Adding 63-EU-135-0
   nuc = new Nucleus("EU",135,63,0,134.942,0,1.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.7288,100);

   // Adding 51-SB-136-0
   nuc = new Nucleus("SB",136,51,0,135.93,0,0.82,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.6708,24);
   nuc->AddDecay(0,1,0,9.33985,76);

   // Adding 52-TE-136-0
   nuc = new Nucleus("TE",136,52,0,135.92,0,17.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.0748,98.9);
   nuc->AddDecay(-1,1,0,1.29287,1.1);

   // Adding 53-I-136-0
   nuc = new Nucleus("I",136,53,0,135.915,0,83.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.92622,100);

   // Adding 53-I-136-1
   nuc = new Nucleus("I",136,53,1,135.915,0.64,46.9,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,7.56622,100);

   // Adding 54-XE-136-0
   nuc = new Nucleus("XE",136,54,0,135.907,0,0,8.9,0,0,0,0);

   // Adding 55-CS-136-0
   nuc = new Nucleus("CS",136,55,0,135.907,0,1.13702e+06,0,3.1e-09,1.9e-09,0,0);
   nuc->AddDecay(0,1,0,2.54819,88.8);
   nuc->AddDecay(0,1,1,0.517197,11.2);

   // Adding 55-CS-136-1
   nuc = new Nucleus("CS",136,55,1,135.907,0,19,0,0,0,0,-8);
   nuc->AddDecay(0,1,-1,2.54819,50);
   nuc->AddDecay(0,0,-1,0,50);

   // Adding 56-BA-136-0
   nuc = new Nucleus("BA",136,56,0,135.905,0,0,7.854,0,0,0,0);

   // Adding 56-BA-136-1
   nuc = new Nucleus("BA",136,56,1,135.907,2.031,0.3084,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.031,100);

   // Adding 57-LA-136-0
   nuc = new Nucleus("LA",136,57,0,135.908,0,592.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.86987,100);

   // Adding 57-LA-136-1
   nuc = new Nucleus("LA",136,57,1,135.908,0.23,0.114,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.23,100);

   // Adding 58-CE-136-0
   nuc = new Nucleus("CE",136,58,0,135.907,0,0,0.19,0,0,0,0);

   // Adding 59-PR-136-0
   nuc = new Nucleus("PR",136,59,0,135.913,0,786,0,3.3e-11,2.5e-11,0,0);
   nuc->AddDecay(0,-1,0,5.1263,100);

   // Adding 60-ND-136-0
   nuc = new Nucleus("ND",136,60,0,135.915,0,3039,0,9.9e-11,8.9e-11,0,0);
   nuc->AddDecay(0,-1,0,2.21099,100);

   // Adding 61-PM-136-0
   nuc = new Nucleus("PM",136,61,0,135.923,0,107,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.85004,100);

   // Adding 61-PM-136-1
   nuc = new Nucleus("PM",136,61,1,135.923,0,107,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.85004,100);

   // Adding 62-SM-136-0
   nuc = new Nucleus("SM",136,62,0,135.928,0,43,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.51871,100);

   // Adding 63-EU-136-0
   nuc = new Nucleus("EU",136,63,0,135.939,0,3.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.4328,99.91);
   nuc->AddDecay(-1,-2,0,6.49671,0.09);

   // Adding 63-EU-136-1
   nuc = new Nucleus("EU",136,63,1,135.939,0,3.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,10.4328,99.91);
   nuc->AddDecay(-1,-2,-1,6.4967,0.09);

   // Adding 52-TE-137-0
   nuc = new Nucleus("TE",137,52,0,136.925,0,2.49,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.94162,97.3);
   nuc->AddDecay(-1,1,0,1.86744,2.7);

   // Adding 53-I-137-0
   nuc = new Nucleus("I",137,53,0,136.918,0,24.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.87744,92.9);
   nuc->AddDecay(-1,1,0,1.85204,7.1);

   // Adding 54-XE-137-0
   nuc = new Nucleus("XE",137,54,0,136.912,0,229.08,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.17196,100);

   // Adding 55-CS-137-0
   nuc = new Nucleus("CS",137,55,0,136.907,0,9.49234e+08,0,1.3e-08,6.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.1756,5.4);
   nuc->AddDecay(0,1,1,0.513596,94.6);

   // Adding 56-BA-137-0
   nuc = new Nucleus("BA",137,56,0,136.906,0,0,11.23,0,0,0,0);

   // Adding 56-BA-137-1
   nuc = new Nucleus("BA",137,56,1,136.907,0.662,153.12,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.662,100);

   // Adding 57-LA-137-0
   nuc = new Nucleus("LA",137,57,0,136.906,0,1.89216e+12,0,8.1e-11,1e-08,0,0);
   nuc->AddDecay(0,-1,0,0.599785,100);

   // Adding 58-CE-137-0
   nuc = new Nucleus("CE",137,58,0,136.908,0,32400,0,2.5e-11,1.9e-11,0,0);
   nuc->AddDecay(0,-1,0,1.2221,100);

   // Adding 58-CE-137-1
   nuc = new Nucleus("CE",137,58,1,136.908,0.254,123840,0,5.4e-10,5.9e-10,0,0);
   nuc->AddDecay(0,0,-1,0.254,99.22);
   nuc->AddDecay(0,-1,-1,1.4761,0.78);

   // Adding 59-PR-137-0
   nuc = new Nucleus("PR",137,59,0,136.911,0,4608,0,4e-11,3.5e-11,0,0);
   nuc->AddDecay(0,-1,0,2.702,100);

   // Adding 60-ND-137-0
   nuc = new Nucleus("ND",137,60,0,136.915,0,2310,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.68998,100);

   // Adding 60-ND-137-1
   nuc = new Nucleus("ND",137,60,1,136.915,0.52,1.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.52,100);

   // Adding 61-PM-137-0
   nuc = new Nucleus("PM",137,61,0,136.921,0,144,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.5798,100);

   // Adding 62-SM-137-0
   nuc = new Nucleus("SM",137,62,0,136.927,0,45,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.05376,100);

   // Adding 63-EU-137-0
   nuc = new Nucleus("EU",137,63,0,136.935,0,11,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.52681,100);

   // Adding 64-GD-137-0
   nuc = new Nucleus("GD",137,64,0,136.945,0,7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.7929,0);
   nuc->AddDecay(-1,-2,0,7.9408,100);

   // Adding 52-TE-138-0
   nuc = new Nucleus("TE",138,52,0,137.929,0,1.4,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.49833,6.3);
   nuc->AddDecay(0,1,0,6.36807,93.7);

   // Adding 53-I-138-0
   nuc = new Nucleus("I",138,53,0,137.922,0,6.49,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.0077,5.5);
   nuc->AddDecay(0,1,0,7.81996,94.5);

   // Adding 54-XE-138-0
   nuc = new Nucleus("XE",138,54,0,137.914,0,844.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.77406,100);

   // Adding 55-CS-138-0
   nuc = new Nucleus("CS",138,55,0,137.911,0,2004.6,0,9.2e-11,4.6e-11,0,0);
   nuc->AddDecay(0,1,0,5.37293,100);

   // Adding 55-CS-138-1
   nuc = new Nucleus("CS",138,55,1,137.911,0.08,174.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.08,81);
   nuc->AddDecay(0,1,-1,5.45293,19);

   // Adding 56-BA-138-0
   nuc = new Nucleus("BA",138,56,0,137.905,0,0,71.7,0,0,0,0);

   // Adding 57-LA-138-0
   nuc = new Nucleus("LA",138,57,0,137.907,0,3.31128e+18,0.09,1.1e-09,1.8e-07,0,0);
   nuc->AddDecay(0,-1,0,1.73749,66.4);
   nuc->AddDecay(0,1,0,1.04447,33.6);

   // Adding 58-CE-138-0
   nuc = new Nucleus("CE",138,58,0,137.906,0,0,0.25,0,0,0,0);

   // Adding 58-CE-138-1
   nuc = new Nucleus("CE",138,58,1,137.908,2.129,0.00865,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.129,100);

   // Adding 59-PR-138-0
   nuc = new Nucleus("PR",138,59,0,137.911,0,87,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.437,100);

   // Adding 59-PR-138-1
   nuc = new Nucleus("PR",138,59,1,137.911,0.364,7632,0,1.3e-10,1.4e-10,0,0);
   nuc->AddDecay(0,-1,-1,4.80099,100);

   // Adding 60-ND-138-0
   nuc = new Nucleus("ND",138,60,0,137.912,0,18144,0,6.4e-10,3.8e-10,0,0);
   nuc->AddDecay(0,-1,0,1.10001,100);

   // Adding 61-PM-138-0
   nuc = new Nucleus("PM",138,61,0,137.919,0,194.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.89976,100);

   // Adding 61-PM-138-1
   nuc = new Nucleus("PM",138,61,1,137.919,0,10,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.89976,100);

   // Adding 61-PM-138-2
   nuc = new Nucleus("PM",138,61,2,137.919,0,194.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,6.89976,100);

   // Adding 62-SM-138-0
   nuc = new Nucleus("SM",138,62,0,137.924,0,186,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.91415,100);

   // Adding 63-EU-138-0
   nuc = new Nucleus("EU",138,63,0,137.933,0,12.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.23079,100);

   // Adding 64-GD-138-0
   nuc = new Nucleus("GD",138,64,0,137.94,0,0,0,0,0,0,-2);

   // Adding 53-I-139-0
   nuc = new Nucleus("I",139,53,0,138.926,0,2.29,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.20531,9.9);
   nuc->AddDecay(0,1,0,6.80599,90.1);

   // Adding 54-XE-139-0
   nuc = new Nucleus("XE",139,54,0,138.919,0,39.68,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.05702,100);

   // Adding 55-CS-139-0
   nuc = new Nucleus("CS",139,55,0,138.913,0,556.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.21268,100);

   // Adding 56-BA-139-0
   nuc = new Nucleus("BA",139,56,0,138.909,0,4983.6,0,1.2e-10,5.5e-11,0,0);
   nuc->AddDecay(0,1,0,2.31711,100);

   // Adding 57-LA-139-0
   nuc = new Nucleus("LA",139,57,0,138.906,0,0,99.91,0,0,0,0);

   // Adding 58-CE-139-0
   nuc = new Nucleus("CE",139,58,0,138.907,0,1.18921e+07,0,2.6e-10,1.8e-09,0,0);
   nuc->AddDecay(0,-1,0,0.277962,100);

   // Adding 58-CE-139-1
   nuc = new Nucleus("CE",139,58,1,138.907,0.754,54.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.754,100);

   // Adding 59-PR-139-0
   nuc = new Nucleus("PR",139,59,0,138.909,0,15876,0,3.1e-11,3e-11,0,0);
   nuc->AddDecay(0,-1,0,2.12899,100);

   // Adding 60-ND-139-0
   nuc = new Nucleus("ND",139,60,0,138.912,0,1782,0,2e-11,1.7e-11,0,0);
   nuc->AddDecay(0,-1,0,2.78654,100);

   // Adding 60-ND-139-1
   nuc = new Nucleus("ND",139,60,1,138.912,0.231,19800,0,2.5e-10,2.5e-10,0,0);
   nuc->AddDecay(0,0,-1,0.231,11.8);
   nuc->AddDecay(0,-1,-1,3.01755,88.2);

   // Adding 61-PM-139-0
   nuc = new Nucleus("PM",139,61,0,138.917,0,249,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.52199,100);

   // Adding 61-PM-139-1
   nuc = new Nucleus("PM",139,61,1,138.917,0.189,0.18,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.189,99.94);
   nuc->AddDecay(0,-1,-1,4.71099,0.06);

   // Adding 62-SM-139-0
   nuc = new Nucleus("SM",139,62,0,138.923,0,154.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.46004,100);

   // Adding 62-SM-139-1
   nuc = new Nucleus("SM",139,62,1,138.923,0.458,10.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.458,93.7);
   nuc->AddDecay(0,-1,-1,5.91804,6.3);

   // Adding 63-EU-139-0
   nuc = new Nucleus("EU",139,63,0,138.93,0,17.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.67751,100);

   // Adding 64-GD-139-0
   nuc = new Nucleus("GD",139,64,0,138.938,0,4.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.7039,100);
   nuc->AddDecay(-1,-2,0,6.2548,0);

   // Adding 53-I-140-0
   nuc = new Nucleus("I",140,53,0,139.931,0,0.86,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.76295,90.6);
   nuc->AddDecay(-1,1,0,3.34111,9.4);

   // Adding 54-XE-140-0
   nuc = new Nucleus("XE",140,54,0,139.922,0,13.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.05995,100);

   // Adding 55-CS-140-0
   nuc = new Nucleus("CS",140,55,0,139.917,0,63.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.21877,100);

   // Adding 56-BA-140-0
   nuc = new Nucleus("BA",140,56,0,139.911,0,1.10177e+06,0,2.5e-09,1.6e-09,0,0);
   nuc->AddDecay(0,1,0,1.04715,100);

   // Adding 57-LA-140-0
   nuc = new Nucleus("LA",140,57,0,139.909,0,144988,0,2e-09,1.5e-09,0,0);
   nuc->AddDecay(0,1,0,3.76189,100);

   // Adding 58-CE-140-0
   nuc = new Nucleus("CE",140,58,0,139.905,0,0,88.48,0,0,0,0);

   // Adding 59-PR-140-0
   nuc = new Nucleus("PR",140,59,0,139.909,0,203.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.38796,100);

   // Adding 60-ND-140-0
   nuc = new Nucleus("ND",140,60,0,139.909,0,291168,0,2.8e-09,2e-09,0,0);
   nuc->AddDecay(0,-1,0,0.222145,100);

   // Adding 61-PM-140-0
   nuc = new Nucleus("PM",140,61,0,139.916,0,9.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.08868,100);

   // Adding 61-PM-140-1
   nuc = new Nucleus("PM",140,61,1,139.916,0,357,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.08868,100);

   // Adding 62-SM-140-0
   nuc = new Nucleus("SM",140,62,0,139.919,0,889.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.01994,100);

   // Adding 63-EU-140-0
   nuc = new Nucleus("EU",140,63,0,139.928,0,1.54,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.39975,100);

   // Adding 63-EU-140-1
   nuc = new Nucleus("EU",140,63,1,139.928,0,0.125,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.39975,100);

   // Adding 64-GD-140-0
   nuc = new Nucleus("GD",140,64,0,139.934,0,16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.45985,100);

   // Adding 65-TB-140-0
   nuc = new Nucleus("TB",140,65,0,139.946,0,2.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.8005,100);
   nuc->AddDecay(-1,-2,0,7.3853,0);

   // Adding 53-I-141-0
   nuc = new Nucleus("I",141,53,0,140.935,0,0.43,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.83889,78.8);
   nuc->AddDecay(-1,1,0,4.44556,21.2);

   // Adding 54-XE-141-0
   nuc = new Nucleus("XE",141,54,0,140.927,0,1.73,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.666623,0.04);
   nuc->AddDecay(0,1,0,6.14993,99.96);

   // Adding 55-CS-141-0
   nuc = new Nucleus("CS",141,55,0,140.92,0,24.94,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,0.735471,0.03);
   nuc->AddDecay(0,1,0,5.2551,99.97);

   // Adding 56-BA-141-0
   nuc = new Nucleus("BA",141,56,0,140.914,0,1096.2,0,7e-11,3.5e-11,0,0);
   nuc->AddDecay(0,1,0,3.21604,100);

   // Adding 57-LA-141-0
   nuc = new Nucleus("LA",141,57,0,140.911,0,14112,0,3.6e-10,2.2e-10,0,0);
   nuc->AddDecay(0,1,0,2.50198,100);

   // Adding 58-CE-141-0
   nuc = new Nucleus("CE",141,58,0,140.908,0,2.80809e+06,0,7.1e-10,3.6e-09,0,0);
   nuc->AddDecay(0,1,0,0.580704,100);

   // Adding 59-PR-141-0
   nuc = new Nucleus("PR",141,59,0,140.908,0,0,100,0,0,0,0);

   // Adding 60-ND-141-0
   nuc = new Nucleus("ND",141,60,0,140.91,0,8964,0,8.3e-12,8.8e-12,0,0);
   nuc->AddDecay(0,-1,0,1.82289,100);

   // Adding 60-ND-141-1
   nuc = new Nucleus("ND",141,60,1,140.91,0.757,62,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.757,99.95);
   nuc->AddDecay(0,-1,-1,2.5799,0.05);

   // Adding 61-PM-141-0
   nuc = new Nucleus("PM",141,61,0,140.914,0,1254,0,3.6e-11,2.5e-11,0,0);
   nuc->AddDecay(0,-1,0,3.71502,100);

   // Adding 62-SM-141-0
   nuc = new Nucleus("SM",141,62,0,140.918,0,612,0,3.9e-11,2.7e-11,0,0);
   nuc->AddDecay(0,-1,0,4.54311,100);

   // Adding 62-SM-141-1
   nuc = new Nucleus("SM",141,62,1,140.919,0.176,1356,0,6.5e-11,5.6e-11,0,0);
   nuc->AddDecay(0,-1,-1,4.71912,99.69);
   nuc->AddDecay(0,0,-1,0.176,0.31);

   // Adding 63-EU-141-0
   nuc = new Nucleus("EU",141,63,0,140.924,0,40,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.55,100);

   // Adding 63-EU-141-1
   nuc = new Nucleus("EU",141,63,1,140.925,0.096,2.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.096,87);
   nuc->AddDecay(0,-1,-1,5.646,13);

   // Adding 64-GD-141-0
   nuc = new Nucleus("GD",141,64,0,140.932,0,14,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.24762,99.97);
   nuc->AddDecay(-1,-2,0,4.9329,0.03);

   // Adding 64-GD-141-1
   nuc = new Nucleus("GD",141,64,1,140.933,0.378,24.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.62561,89);
   nuc->AddDecay(0,0,-1,0.378,11);

   // Adding 65-TB-141-0
   nuc = new Nucleus("TB",141,65,0,140.941,0,3.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.3367,100);

   // Adding 65-TB-141-1
   nuc = new Nucleus("TB",141,65,1,140.941,0,7.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.3367,100);

   // Adding 66-DY-141-0
   nuc = new Nucleus("DY",141,66,0,140.951,0,0.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.3429,100);
   nuc->AddDecay(-1,-2,0,8.7529,0);

   // Adding 53-I-142-0
   nuc = new Nucleus("I",142,53,0,141.94,0,0.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.87831,45.5);
   nuc->AddDecay(-1,1,0,4.64989,54.5);

   // Adding 54-XE-142-0
   nuc = new Nucleus("XE",142,54,0,141.93,0,1.22,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.04,100);

   // Adding 55-CS-142-0
   nuc = new Nucleus("CS",142,55,0,141.924,0,1.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.30647,99.72);
   nuc->AddDecay(-1,1,0,1.1366,0.28);

   // Adding 56-BA-142-0
   nuc = new Nucleus("BA",142,56,0,141.916,0,636,0,3.5e-11,2.7e-11,0,0);
   nuc->AddDecay(0,1,0,2.21209,100);

   // Adding 57-LA-142-0
   nuc = new Nucleus("LA",142,57,0,141.914,0,5466,0,1.8e-10,1.5e-10,0,0);
   nuc->AddDecay(0,1,0,4.50507,100);

   // Adding 58-CE-142-0
   nuc = new Nucleus("CE",142,58,0,141.909,0,0,11.08,0,0,0,0);

   // Adding 59-PR-142-0
   nuc = new Nucleus("PR",142,59,0,141.91,0,68832,0,1.3e-09,7.4e-10,0,0);
   nuc->AddDecay(0,1,0,2.1622,99.98);
   nuc->AddDecay(0,-1,0,0.745209,0.02);

   // Adding 59-PR-142-1
   nuc = new Nucleus("PR",142,59,1,141.91,0.004,876,0,1.7e-11,9.4e-12,0,0);
   nuc->AddDecay(0,0,-1,0.004,100);

   // Adding 60-ND-142-0
   nuc = new Nucleus("ND",142,60,0,141.908,0,0,27.13,0,0,0,0);

   // Adding 61-PM-142-0
   nuc = new Nucleus("PM",142,61,0,141.913,0,40.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.87359,100);

   // Adding 62-SM-142-0
   nuc = new Nucleus("SM",142,62,0,141.915,0,4349.4,0,1.9e-10,1.1e-10,0,0);
   nuc->AddDecay(0,-1,0,2.09813,100);

   // Adding 63-EU-142-0
   nuc = new Nucleus("EU",142,63,0,141.923,0,2.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.35992,100);

   // Adding 63-EU-142-1
   nuc = new Nucleus("EU",142,63,1,141.923,0,73.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.35992,100);

   // Adding 64-GD-142-0
   nuc = new Nucleus("GD",142,64,0,141.928,0,70.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.49998,100);

   // Adding 65-TB-142-0
   nuc = new Nucleus("TB",142,65,0,141.939,0,0.597,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.0595,100);
   nuc->AddDecay(-1,-2,0,6.03715,3e-07);

   // Adding 65-TB-142-1
   nuc = new Nucleus("TB",142,65,1,141.939,0.28,0.303,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.28,33.3333);
   nuc->AddDecay(0,-1,-1,10.3395,33.3333);
   nuc->AddDecay(-1,-2,-1,6.31714,33.3333);

   // Adding 66-DY-142-0
   nuc = new Nucleus("DY",142,66,0,141.946,0,2.3,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,5.6895,8e-05);
   nuc->AddDecay(0,-1,0,6.89997,99.9999);

   // Adding 54-XE-143-0
   nuc = new Nucleus("XE",143,54,0,142.935,0,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.30699,100);

   // Adding 55-CS-143-0
   nuc = new Nucleus("CS",143,55,0,142.927,0,1.78,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,2.04826,1.62);
   nuc->AddDecay(0,1,0,6.2426,98.38);

   // Adding 56-BA-143-0
   nuc = new Nucleus("BA",143,56,0,142.921,0,14.33,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.2433,100);

   // Adding 57-LA-143-0
   nuc = new Nucleus("LA",143,57,0,142.916,0,852,0,5.6e-11,3.3e-11,0,0);
   nuc->AddDecay(0,1,0,3.42452,100);

   // Adding 58-CE-143-0
   nuc = new Nucleus("CE",143,58,0,142.912,0,118940,0,1.1e-09,1e-09,0,0);
   nuc->AddDecay(0,1,0,1.46159,100);

   // Adding 59-PR-143-0
   nuc = new Nucleus("PR",143,59,0,142.911,0,1.17245e+06,0,1.2e-09,2.3e-09,0,0);
   nuc->AddDecay(0,1,0,0.933998,100);

   // Adding 60-ND-143-0
   nuc = new Nucleus("ND",143,60,0,142.91,0,0,12.18,0,0,0,0);

   // Adding 61-PM-143-0
   nuc = new Nucleus("PM",143,61,0,142.911,0,2.2896e+07,0,2.3e-10,1.4e-09,0,0);
   nuc->AddDecay(0,-1,0,1.0414,100);

   // Adding 62-SM-143-0
   nuc = new Nucleus("SM",143,62,0,142.915,0,529.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.44299,100);

   // Adding 62-SM-143-1
   nuc = new Nucleus("SM",143,62,1,142.915,0.754,66,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.754,99.76);
   nuc->AddDecay(0,-1,-1,4.19699,0.24);

   // Adding 62-SM-143-2
   nuc = new Nucleus("SM",143,62,2,142.918,2.795,0.03,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.795,100);

   // Adding 63-EU-143-0
   nuc = new Nucleus("EU",143,63,0,142.92,0,157.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.1684,100);

   // Adding 64-GD-143-0
   nuc = new Nucleus("GD",143,64,0,142.927,0,39,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.00742,100);

   // Adding 64-GD-143-1
   nuc = new Nucleus("GD",143,64,1,142.927,0.153,112,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.16042,100);

   // Adding 65-TB-143-0
   nuc = new Nucleus("TB",143,65,0,142.935,0,12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.39362,100);

   // Adding 65-TB-143-1
   nuc = new Nucleus("TB",143,65,1,142.935,0,21,0,0,0,0,-3);
   nuc->AddDecay(0,-1,-1,7.39362,100);

   // Adding 66-DY-143-0
   nuc = new Nucleus("DY",143,66,0,142.944,0,3.9,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,7.64581,0);
   nuc->AddDecay(0,-1,0,8.7649,100);

   // Adding 54-XE-144-0
   nuc = new Nucleus("XE",144,54,0,143.939,0,1.15,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.06531,100);

   // Adding 55-CS-144-0
   nuc = new Nucleus("CS",144,55,0,143.932,0,1.01,0,0,0,0,0);
   nuc->AddDecay(0,1,0,8.46459,96.83);
   nuc->AddDecay(-1,1,0,2.56089,3.17);

   // Adding 55-CS-144-1
   nuc = new Nucleus("CS",144,55,1,143.932,0,1,0,0,0,0,-4);
   nuc->AddDecay(0,1,-1,8.46459,100);

   // Adding 56-BA-144-0
   nuc = new Nucleus("BA",144,56,0,143.923,0,11.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.11935,96.4);
   nuc->AddDecay(-1,1,0,-1.6604,3.6);

   // Adding 57-LA-144-0
   nuc = new Nucleus("LA",144,57,0,143.92,0,40.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.54118,100);

   // Adding 58-CE-144-0
   nuc = new Nucleus("CE",144,58,0,143.914,0,2.46148e+07,0,5.2e-09,4.9e-08,0,0);
   nuc->AddDecay(0,1,0,0.318703,98.5);
   nuc->AddDecay(0,1,1,0.259705,1.5);

   // Adding 59-PR-144-0
   nuc = new Nucleus("PR",144,59,0,143.913,0,1036.8,0,5.1e-11,3e-11,0,0);
   nuc->AddDecay(0,1,0,2.99749,100);

   // Adding 59-PR-144-1
   nuc = new Nucleus("PR",144,59,1,143.913,0.059,432,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.059,99.93);
   nuc->AddDecay(0,1,-1,3.05649,0.07);

   // Adding 60-ND-144-0
   nuc = new Nucleus("ND",144,60,0,143.91,0,7.22174e+22,23.8,0,0,1,0);
   nuc->AddDecay(-4,-2,0,1.9051,100);

   // Adding 61-PM-144-0
   nuc = new Nucleus("PM",144,61,0,143.913,0,3.13632e+07,0,9.7e-10,7.8e-09,0,0);
   nuc->AddDecay(0,-1,0,2.33179,100);

   // Adding 62-SM-144-0
   nuc = new Nucleus("SM",144,62,0,143.912,0,0,3.1,0,0,0,0);

   // Adding 63-EU-144-0
   nuc = new Nucleus("EU",144,63,0,143.919,0,10.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.32881,100);

   // Adding 64-GD-144-0
   nuc = new Nucleus("GD",144,64,0,143.923,0,270,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.73932,100);

   // Adding 65-TB-144-0
   nuc = new Nucleus("TB",144,65,0,143.932,0,1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.91891,100);

   // Adding 65-TB-144-1
   nuc = new Nucleus("TB",144,65,1,143.933,0.397,4.25,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.397,66);
   nuc->AddDecay(0,-1,-1,9.31591,34);

   // Adding 66-DY-144-0
   nuc = new Nucleus("DY",144,66,0,143.939,0,9.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.2319,100);
   nuc->AddDecay(-1,-2,0,4.30562,0);

   // Adding 67-HO-144-0
   nuc = new Nucleus("HO",144,67,0,143.952,0,0.7,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,11.7558,50);
   nuc->AddDecay(-1,-2,0,8.66781,50);

   // Adding 54-XE-145-0
   nuc = new Nucleus("XE",145,54,0,144.942,0,0.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.29984,97.62);
   nuc->AddDecay(-1,1,0,1.38033,1.19);

   // Adding 55-CS-145-0
   nuc = new Nucleus("CS",145,55,0,144.935,0,0.594,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,3.54507,13.8);
   nuc->AddDecay(0,1,0,7.88681,86.2);

   // Adding 56-BA-145-0
   nuc = new Nucleus("BA",145,56,0,144.927,0,4.31,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.93172,100);

   // Adding 57-LA-145-0
   nuc = new Nucleus("LA",145,57,0,144.922,0,24.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.11667,100);

   // Adding 58-CE-145-0
   nuc = new Nucleus("CE",145,58,0,144.917,0,180.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.53674,100);

   // Adding 59-PR-145-0
   nuc = new Nucleus("PR",145,59,0,144.915,0,21542.4,0,3.9e-10,2.6e-10,0,0);
   nuc->AddDecay(0,1,0,1.80526,100);

   // Adding 60-ND-145-0
   nuc = new Nucleus("ND",145,60,0,144.913,0,0,8.3,0,0,0,0);

   // Adding 61-PM-145-0
   nuc = new Nucleus("PM",145,61,0,144.913,0,5.58187e+08,0,1.1e-10,3.4e-09,0,0);
   nuc->AddDecay(0,-1,0,0.163193,100);
   nuc->AddDecay(-4,-2,0,2.32219,3e-09);

   // Adding 62-SM-145-0
   nuc = new Nucleus("SM",145,62,0,144.913,0,2.9376e+07,0,2.1e-10,1.5e-09,0,0);
   nuc->AddDecay(0,-1,0,0.616707,100);

   // Adding 63-EU-145-0
   nuc = new Nucleus("EU",145,63,0,144.916,0,512352,0,7.5e-10,7.4e-10,0,0);
   nuc->AddDecay(0,-1,0,2.66019,100);

   // Adding 64-GD-145-0
   nuc = new Nucleus("GD",145,64,0,144.922,0,1380,0,4.4e-11,3.5e-11,0,0);
   nuc->AddDecay(0,-1,0,5.05441,100);

   // Adding 64-GD-145-1
   nuc = new Nucleus("GD",145,64,1,144.922,0.749,85,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.749,94.3);
   nuc->AddDecay(0,-1,-1,5.80341,5.7);

   // Adding 65-TB-145-0
   nuc = new Nucleus("TB",145,65,0,144.929,0,0,0,0,0,0,-2);

   // Adding 65-TB-145-1
   nuc = new Nucleus("TB",145,65,1,144.929,0,29.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.50802,100);

   // Adding 66-DY-145-0
   nuc = new Nucleus("DY",145,66,0,144.937,0,10,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.72028,100);

   // Adding 66-DY-145-1
   nuc = new Nucleus("DY",145,66,1,144.937,0,13.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.72028,100);

   // Adding 67-HO-145-0
   nuc = new Nucleus("HO",145,67,0,144.947,0,2.4,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,9.10664,100);

   // Adding 68-ER-145-0
   nuc = new Nucleus("ER",145,68,0,144.958,0,0.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.3516,100);

   // Adding 54-XE-146-0
   nuc = new Nucleus("XE",146,54,0,145.944,0,1.5e-07,0,0,0,0,-5);
   nuc->AddDecay(0,1,0,3.64086,100);

   // Adding 55-CS-146-0
   nuc = new Nucleus("CS",146,55,0,145.94,0,0.343,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,4.31679,13.2);
   nuc->AddDecay(0,1,0,9.37651,86.8);

   // Adding 56-BA-146-0
   nuc = new Nucleus("BA",146,56,0,145.93,0,2.22,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.118,100);

   // Adding 57-LA-146-0
   nuc = new Nucleus("LA",146,57,0,145.926,0,6.27,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.54669,100);

   // Adding 57-LA-146-1
   nuc = new Nucleus("LA",146,57,1,145.926,0,10,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,6.54669,100);

   // Adding 58-CE-146-0
   nuc = new Nucleus("CE",146,58,0,145.919,0,811.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.0351,100);

   // Adding 59-PR-146-0
   nuc = new Nucleus("PR",146,59,0,145.918,0,1449,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.19617,100);

   // Adding 60-ND-146-0
   nuc = new Nucleus("ND",146,60,0,145.913,0,0,17.19,0,0,0,0);

   // Adding 61-PM-146-0
   nuc = new Nucleus("PM",146,61,0,145.915,0,1.74394e+08,0,9e-10,1.9e-08,0,0);
   nuc->AddDecay(0,1,0,1.5419,34);
   nuc->AddDecay(0,-1,0,1.47153,66);

   // Adding 62-SM-146-0
   nuc = new Nucleus("SM",146,62,0,145.913,0,3.24821e+15,0,5.4e-08,9.9e-06,1,0);
   nuc->AddDecay(-4,-2,0,2.52863,100);

   // Adding 63-EU-146-0
   nuc = new Nucleus("EU",146,63,0,145.917,0,396576,0,1.3e-09,1.2e-09,0,0);
   nuc->AddDecay(0,-1,0,3.87813,100);

   // Adding 64-GD-146-0
   nuc = new Nucleus("GD",146,64,0,145.918,0,4.17053e+06,0,9.6e-10,6e-09,0,0);
   nuc->AddDecay(0,-1,0,1.03022,100);

   // Adding 65-TB-146-0
   nuc = new Nucleus("TB",146,65,0,145.927,0,8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.07504,100);

   // Adding 65-TB-146-1
   nuc = new Nucleus("TB",146,65,1,145.927,0,23,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.07504,100);

   // Adding 66-DY-146-0
   nuc = new Nucleus("DY",146,66,0,145.933,0,29,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.16,100);

   // Adding 66-DY-146-1
   nuc = new Nucleus("DY",146,66,1,145.933,0,0.15,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 67-HO-146-0
   nuc = new Nucleus("HO",146,67,0,145.944,0,3.6,0,0,0,0,0);
   nuc->AddDecay(-1,-2,0,6.96701,100);
   nuc->AddDecay(0,-1,0,10.6795,0);

   // Adding 68-ER-146-0
   nuc = new Nucleus("ER",146,68,0,145.952,0,1.7,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.42391,100);

   // Adding 55-CS-147-0
   nuc = new Nucleus("CS",147,55,0,146.944,0,0.225,0,0,0,0,0);
   nuc->AddDecay(0,1,0,9.25328,57);
   nuc->AddDecay(-1,1,0,4.73596,43);

   // Adding 56-BA-147-0
   nuc = new Nucleus("BA",147,56,0,146.934,0,0.893,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.74998,99.98);
   nuc->AddDecay(-1,1,0,-0.399325,0.02);

   // Adding 57-LA-147-0
   nuc = new Nucleus("LA",147,57,0,146.928,0,4.015,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.94498,99.96);
   nuc->AddDecay(-1,1,0,0.397382,0.04);

   // Adding 58-CE-147-0
   nuc = new Nucleus("CE",147,58,0,146.923,0,56.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.28999,100);

   // Adding 59-PR-147-0
   nuc = new Nucleus("PR",147,59,0,146.919,0,804,0,3.3e-11,3e-11,0,0);
   nuc->AddDecay(0,1,0,2.68569,100);

   // Adding 60-ND-147-0
   nuc = new Nucleus("ND",147,60,0,146.916,0,948672,0,1.1e-09,2.3e-09,0,0);
   nuc->AddDecay(0,1,0,0.896095,100);

   // Adding 61-PM-147-0
   nuc = new Nucleus("PM",147,61,0,146.915,0,8.27315e+07,0,2.6e-10,4.7e-09,0,0);
   nuc->AddDecay(0,1,0,0.224106,100);

   // Adding 62-SM-147-0
   nuc = new Nucleus("SM",147,62,0,146.915,0,3.34282e+18,15,4.9e-08,9e-06,1,0);
   nuc->AddDecay(-4,-2,0,2.3104,100);

   // Adding 63-EU-147-0
   nuc = new Nucleus("EU",147,63,0,146.917,0,2.08224e+06,0,4.4e-10,1e-09,0,0);
   nuc->AddDecay(0,-1,0,1.7214,100);
   nuc->AddDecay(-4,-2,0,2.9904,0.0022);

   // Adding 64-GD-147-0
   nuc = new Nucleus("GD",147,64,0,146.919,0,137016,0,6.1e-10,5.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.1878,100);

   // Adding 65-TB-147-0
   nuc = new Nucleus("TB",147,65,0,146.924,0,6120,0,1.6e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,4.61143,100);

   // Adding 65-TB-147-1
   nuc = new Nucleus("TB",147,65,1,146.924,0.051,109.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.66243,100);

   // Adding 66-DY-147-0
   nuc = new Nucleus("DY",147,66,0,146.931,0,40,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.37256,100);
   nuc->AddDecay(-1,-2,0,4.4253,0);

   // Adding 66-DY-147-1
   nuc = new Nucleus("DY",147,66,1,146.932,0.751,55,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.12356,65);
   nuc->AddDecay(0,0,-1,0.751,35);

   // Adding 67-HO-147-0
   nuc = new Nucleus("HO",147,67,0,146.94,0,5.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.14835,100);
   nuc->AddDecay(-1,-2,0,4.49861,0);

   // Adding 68-ER-147-0
   nuc = new Nucleus("ER",147,68,0,146.949,0,2.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.0999,100);
   nuc->AddDecay(-1,-2,0,8.43851,0);

   // Adding 68-ER-147-1
   nuc = new Nucleus("ER",147,68,1,146.949,0,2.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,9.0999,100);
   nuc->AddDecay(-1,-2,-1,8.43851,0);

   // Adding 69-TM-147-0
   nuc = new Nucleus("TM",147,69,0,146.961,0,0.56,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.7259,90);
   nuc->AddDecay(-1,-1,0,1.061,10);

   // Adding 55-CS-148-0
   nuc = new Nucleus("CS",148,55,0,147.949,0,0.158,0,0,0,0,0);
   nuc->AddDecay(0,1,0,10.5249,100);

   // Adding 56-BA-148-0
   nuc = new Nucleus("BA",148,56,0,147.938,0,0.607,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.115,99.98);
   nuc->AddDecay(-1,1,0,1.11576,0.02);

   // Adding 57-LA-148-0
   nuc = new Nucleus("LA",148,57,0,147.932,0,1.05,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.26229,99.89);
   nuc->AddDecay(-1,1,0,0.945741,0.11);

   // Adding 58-CE-148-0
   nuc = new Nucleus("CE",148,58,0,147.924,0,56,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.06001,100);

   // Adding 59-PR-148-0
   nuc = new Nucleus("PR",148,59,0,147.922,0,136.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.93201,100);

   // Adding 59-PR-148-1
   nuc = new Nucleus("PR",148,59,1,147.922,0.09,120,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,5.02201,100);

   // Adding 60-ND-148-0
   nuc = new Nucleus("ND",148,60,0,147.917,0,0,5.76,0,0,0,0);

   // Adding 61-PM-148-0
   nuc = new Nucleus("PM",148,61,0,147.917,0,463968,0,2.7e-09,2.2e-09,0,0);
   nuc->AddDecay(0,1,0,2.46826,100);

   // Adding 61-PM-148-1
   nuc = new Nucleus("PM",148,61,1,147.918,0.138,3.56746e+06,0,1.8e-09,5.4e-09,0,0);
   nuc->AddDecay(0,1,-1,2.60626,95);
   nuc->AddDecay(0,0,-1,0.138,5);

   // Adding 62-SM-148-0
   nuc = new Nucleus("SM",148,62,0,147.915,0,2.20752e+23,11.3,0,0,1,0);
   nuc->AddDecay(-4,-2,0,1.9858,100);

   // Adding 63-EU-148-0
   nuc = new Nucleus("EU",148,63,0,147.918,0,4.7088e+06,0,1.3e-09,2.7e-09,0,0);
   nuc->AddDecay(0,-1,0,3.10741,100);
   nuc->AddDecay(-4,-2,0,2.76142,9.4e-07);

   // Adding 64-GD-148-0
   nuc = new Nucleus("GD",148,64,0,147.918,0,2.35259e+09,0,5.5e-08,3e-05,1,0);
   nuc->AddDecay(-4,-2,0,3.2712,100);

   // Adding 65-TB-148-0
   nuc = new Nucleus("TB",148,65,0,147.924,0,3600,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.69301,100);

   // Adding 65-TB-148-1
   nuc = new Nucleus("TB",148,65,1,147.924,0.09,132,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.783,100);

   // Adding 66-DY-148-0
   nuc = new Nucleus("DY",148,66,0,147.927,0,186,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.67809,100);

   // Adding 67-HO-148-0
   nuc = new Nucleus("HO",148,67,0,147.937,0,2.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.39986,100);

   // Adding 67-HO-148-1
   nuc = new Nucleus("HO",148,67,1,147.937,0,9.59,0,0,0,0,0);
   nuc->AddDecay(-1,-2,-1,4.95804,0.08);
   nuc->AddDecay(0,-1,-1,9.39986,99.92);

   // Adding 68-ER-148-0
   nuc = new Nucleus("ER",148,68,0,147.944,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.75387,100);

   // Adding 69-TM-148-0
   nuc = new Nucleus("TM",148,69,0,147.957,0,0.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.9977,100);

   // Adding 56-BA-149-0
   nuc = new Nucleus("BA",149,56,0,148.942,0,0.356,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.13031,0.43);
   nuc->AddDecay(0,1,0,7.33089,99.57);

   // Adding 57-LA-149-0
   nuc = new Nucleus("LA",149,57,0,148.934,0,1.2,0,0,0,0,0);
   nuc->AddDecay(-1,1,0,1.06171,1.4);
   nuc->AddDecay(0,1,0,5.50558,98.6);

   // Adding 58-CE-149-0
   nuc = new Nucleus("CE",149,58,0,148.928,0,5.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.18993,100);

   // Adding 59-PR-149-0
   nuc = new Nucleus("PR",149,59,0,148.924,0,135.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.39693,100);

   // Adding 60-ND-149-0
   nuc = new Nucleus("ND",149,60,0,148.92,0,6192,0,1.3e-10,1.3e-10,0,0);
   nuc->AddDecay(0,1,0,1.69071,100);

   // Adding 61-PM-149-0
   nuc = new Nucleus("PM",149,61,0,148.918,0,191088,0,9.9e-10,8.2e-10,0,0);
   nuc->AddDecay(0,1,0,1.07108,100);

   // Adding 62-SM-149-0
   nuc = new Nucleus("SM",149,62,0,148.917,0,0,13.8,0,0,0,0);

   // Adding 63-EU-149-0
   nuc = new Nucleus("EU",149,63,0,148.918,0,8.04384e+06,0,1e-10,2.7e-10,0,0);
   nuc->AddDecay(0,-1,0,0.692177,100);

   // Adding 64-GD-149-0
   nuc = new Nucleus("GD",149,64,0,148.919,0,812160,0,4.5e-10,7.9e-10,0,0);
   nuc->AddDecay(0,-1,0,1.3195,99.9996);
   nuc->AddDecay(-4,-2,0,3.10138,0.00043);

   // Adding 65-TB-149-0
   nuc = new Nucleus("TB",149,65,0,148.923,0,14868,0,2.5e-10,4.3e-09,1,0);
   nuc->AddDecay(-4,-2,0,4.0773,15.8);
   nuc->AddDecay(0,-1,0,3.6361,84.2);

   // Adding 65-TB-149-1
   nuc = new Nucleus("TB",149,65,1,148.923,0.036,249.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.6721,99.978);
   nuc->AddDecay(-4,-2,-1,4.1133,0.022);

   // Adding 66-DY-149-0
   nuc = new Nucleus("DY",149,66,0,148.927,0,253.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.81203,100);

   // Adding 67-HO-149-0
   nuc = new Nucleus("HO",149,67,0,148.934,0,21.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.01369,100);

   // Adding 67-HO-149-1
   nuc = new Nucleus("HO",149,67,1,148.934,0.049,58,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.06269,100);

   // Adding 68-ER-149-0
   nuc = new Nucleus("ER",149,68,0,148.942,0,10.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.73355,93);
   nuc->AddDecay(-1,-2,0,6.67966,7);

   // Adding 68-ER-149-1
   nuc = new Nucleus("ER",149,68,1,148.943,0.742,10.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.47555,96.32);
   nuc->AddDecay(-1,-2,-1,7.42166,0.18);
   nuc->AddDecay(0,0,-1,0.742,3.5);

   // Adding 69-TM-149-0
   nuc = new Nucleus("TM",149,69,0,148.952,0,0.9,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,9.57188,100);

   // Adding 70-YB-149-0
   nuc = new Nucleus("YB",149,70,0,148.964,0,0,0,0,0,0,-6);

   // Adding 56-BA-150-0
   nuc = new Nucleus("BA",150,56,0,149.946,0,0.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.4464,100);

   // Adding 57-LA-150-0
   nuc = new Nucleus("LA",150,57,0,149.939,0,0.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.83691,97.3);
   nuc->AddDecay(-1,1,0,1.57018,2.7);

   // Adding 58-CE-150-0
   nuc = new Nucleus("CE",150,58,0,149.93,0,4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.01007,100);

   // Adding 59-PR-150-0
   nuc = new Nucleus("PR",150,59,0,149.927,0,6.19,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.68996,100);

   // Adding 60-ND-150-0
   nuc = new Nucleus("ND",150,60,0,149.921,0,0,5.64,0,0,0,0);

   // Adding 61-PM-150-0
   nuc = new Nucleus("PM",150,61,0,149.921,0,9648,0,2.6e-10,2.1e-10,0,0);
   nuc->AddDecay(0,1,0,3.45401,100);

   // Adding 62-SM-150-0
   nuc = new Nucleus("SM",150,62,0,149.917,0,0,7.4,0,0,0,0);

   // Adding 63-EU-150-0
   nuc = new Nucleus("EU",150,63,0,149.92,0,1.12899e+09,0,1.3e-09,5e-08,0,0);
   nuc->AddDecay(0,-1,0,2.26066,100);

   // Adding 63-EU-150-1
   nuc = new Nucleus("EU",150,63,1,149.92,0.042,46080,0,3.8e-10,2.8e-10,0,0);
   nuc->AddDecay(0,1,-1,1.0132,89);
   nuc->AddDecay(0,-1,-1,2.30266,11);

   // Adding 64-GD-150-0
   nuc = new Nucleus("GD",150,64,0,149.919,0,5.64494e+13,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,2.80913,100);

   // Adding 65-TB-150-0
   nuc = new Nucleus("TB",150,65,0,149.924,0,12528,0,2.5e-10,1.8e-10,1,0);
   nuc->AddDecay(-4,-2,0,3.58739,0.05);
   nuc->AddDecay(0,-1,0,4.65639,99.95);

   // Adding 65-TB-150-1
   nuc = new Nucleus("TB",150,65,1,149.924,0,348,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.65639,100);

   // Adding 66-DY-150-0
   nuc = new Nucleus("DY",150,66,0,149.926,0,430.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.3511,36);
   nuc->AddDecay(0,-1,0,1.79393,64);

   // Adding 67-HO-150-0
   nuc = new Nucleus("HO",150,67,0,149.933,0,72,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.23995,100);

   // Adding 67-HO-150-1
   nuc = new Nucleus("HO",150,67,1,149.933,0,26,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.23995,100);

   // Adding 68-ER-150-0
   nuc = new Nucleus("ER",150,68,0,149.938,0,18.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.108,100);

   // Adding 69-TM-150-0
   nuc = new Nucleus("TM",150,69,0,149.949,0,2.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.8296,100);

   // Adding 70-YB-150-0
   nuc = new Nucleus("YB",150,70,0,149.958,0,0,0,0,0,0,-6);

   // Adding 71-LU-150-0
   nuc = new Nucleus("LU",150,71,0,149.973,0,0.035,0,0,0,0,-8);
   nuc->AddDecay(-1,-1,0,1.2663,80);
   nuc->AddDecay(0,-1,0,13.8853,20);

   // Adding 58-CE-151-0
   nuc = new Nucleus("CE",151,58,0,150.934,0,1.02,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.32618,100);

   // Adding 59-PR-151-0
   nuc = new Nucleus("PR",151,59,0,150.928,0,18.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.16996,100);

   // Adding 60-ND-151-0
   nuc = new Nucleus("ND",151,60,0,150.924,0,746.4,0,3e-11,2.9e-11,0,0);
   nuc->AddDecay(0,1,0,2.44242,100);

   // Adding 61-PM-151-0
   nuc = new Nucleus("PM",151,61,0,150.921,0,102240,0,7.3e-10,6.4e-10,0,0);
   nuc->AddDecay(0,1,0,1.18707,100);

   // Adding 62-SM-151-0
   nuc = new Nucleus("SM",151,62,0,150.92,0,2.83824e+09,0,9.8e-11,3.7e-09,0,0);
   nuc->AddDecay(0,1,0,0.0767975,100);

   // Adding 63-EU-151-0
   nuc = new Nucleus("EU",151,63,0,150.92,0,0,47.8,0,0,0,0);

   // Adding 64-GD-151-0
   nuc = new Nucleus("GD",151,64,0,150.92,0,1.07136e+07,0,2e-10,9.3e-10,0,0);
   nuc->AddDecay(0,-1,0,0.464188,100);
   nuc->AddDecay(-4,-2,0,2.65249,1e-06);

   // Adding 65-TB-151-0
   nuc = new Nucleus("TB",151,65,0,150.923,0,63392.4,0,3.4e-10,3.3e-10,1,0);
   nuc->AddDecay(-4,-2,0,3.49649,0.0095);
   nuc->AddDecay(0,-1,0,2.56539,100);

   // Adding 65-TB-151-1
   nuc = new Nucleus("TB",151,65,1,150.923,0.1,25,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.66539,6.2);
   nuc->AddDecay(0,0,-1,0.1,93.8);

   // Adding 66-DY-151-0
   nuc = new Nucleus("DY",151,66,0,150.926,0,1074,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.17949,5.6);
   nuc->AddDecay(0,-1,0,2.8708,94.4);

   // Adding 67-HO-151-0
   nuc = new Nucleus("HO",151,67,0,150.932,0,35.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.6958,22);
   nuc->AddDecay(0,-1,0,5.12773,78);

   // Adding 67-HO-151-1
   nuc = new Nucleus("HO",151,67,1,150.932,0.041,47.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,4.7368,100);

   // Adding 68-ER-151-0
   nuc = new Nucleus("ER",151,68,0,150.937,0,23.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.22031,100);

   // Adding 68-ER-151-1
   nuc = new Nucleus("ER",151,68,1,150.94,2.585,0.58,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.80531,4.7);
   nuc->AddDecay(0,0,-1,2.585,95.3);

   // Adding 69-TM-151-0
   nuc = new Nucleus("TM",151,69,0,150.945,0,4.13,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.52994,100);

   // Adding 69-TM-151-1
   nuc = new Nucleus("TM",151,69,1,150.945,0,5.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.52994,100);

   // Adding 70-YB-151-0
   nuc = new Nucleus("YB",151,70,0,150.955,0,1.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.20004,100);
   nuc->AddDecay(-1,-2,0,8.99978,0);

   // Adding 70-YB-151-1
   nuc = new Nucleus("YB",151,70,1,150.955,0,1.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,9.20004,100);
   nuc->AddDecay(-1,-2,-1,8.99978,0);

   // Adding 71-LU-151-0
   nuc = new Nucleus("LU",151,71,0,150.967,0,0.085,0,0,0,0,0);
   nuc->AddDecay(-1,-1,0,1.03739,70);

   // Adding 58-CE-152-0
   nuc = new Nucleus("CE",152,58,0,151.937,0,3.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.4159,100);

   // Adding 59-PR-152-0
   nuc = new Nucleus("PR",152,59,0,151.932,0,3.24,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.69413,100);

   // Adding 60-ND-152-0
   nuc = new Nucleus("ND",152,60,0,151.925,0,684,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.11024,100);

   // Adding 61-PM-152-0
   nuc = new Nucleus("PM",152,61,0,151.923,0,246,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.50456,100);

   // Adding 61-PM-152-1
   nuc = new Nucleus("PM",152,61,1,151.924,0.14,451.2,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.64456,100);

   // Adding 61-PM-152-2
   nuc = new Nucleus("PM",152,61,2,151.924,0.17,828,0,0,0,0,0);
   nuc->AddDecay(0,1,-2,3.67456,100);
   nuc->AddDecay(0,0,-2,0.17,0);

   // Adding 62-SM-152-0
   nuc = new Nucleus("SM",152,62,0,151.92,0,0,26.7,0,0,0,0);

   // Adding 63-EU-152-0
   nuc = new Nucleus("EU",152,63,0,151.922,0,4.27061e+08,0,1.4e-09,3.9e-08,0,0);
   nuc->AddDecay(0,1,0,1.8181,27.92);
   nuc->AddDecay(0,-1,0,1.8741,72.08);

   // Adding 63-EU-152-1
   nuc = new Nucleus("EU",152,63,1,151.922,0.046,33386.4,0,5e-10,3.2e-10,0,0);
   nuc->AddDecay(0,1,-1,1.8641,72);
   nuc->AddDecay(0,-1,-1,1.9201,28);

   // Adding 63-EU-152-2
   nuc = new Nucleus("EU",152,63,2,151.922,0.148,5760,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.148,100);

   // Adding 64-GD-152-0
   nuc = new Nucleus("GD",152,64,0,151.92,0,3.40589e+21,0.2,4.1e-08,2.2e-05,1,0);
   nuc->AddDecay(-4,-2,0,2.2051,100);

   // Adding 65-TB-152-0
   nuc = new Nucleus("TB",152,65,0,151.924,0,63000,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.98999,100);
   nuc->AddDecay(-4,-2,0,3.08767,7e-07);

   // Adding 65-TB-152-1
   nuc = new Nucleus("TB",152,65,1,151.925,0.502,252,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.49199,21.1);
   nuc->AddDecay(0,0,-1,0.502,78.9);

   // Adding 66-DY-152-0
   nuc = new Nucleus("DY",152,66,0,151.925,0,8568,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.598686,99.9);
   nuc->AddDecay(-4,-2,0,3.72678,0.1);

   // Adding 67-HO-152-0
   nuc = new Nucleus("HO",152,67,0,151.932,0,161.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.47372,88);
   nuc->AddDecay(-4,-2,0,4.5075,12);

   // Adding 67-HO-152-1
   nuc = new Nucleus("HO",152,67,1,151.932,0.16,49.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.63372,89.2);
   nuc->AddDecay(-4,-2,-1,4.6675,10.8);

   // Adding 68-ER-152-0
   nuc = new Nucleus("ER",152,68,0,151.935,0,10.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.9344,90);
   nuc->AddDecay(0,-1,0,3.105,10);

   // Adding 69-TM-152-0
   nuc = new Nucleus("TM",152,69,0,151.944,0,8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.66454,100);

   // Adding 69-TM-152-1
   nuc = new Nucleus("TM",152,69,1,151.944,0,5.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.66454,100);

   // Adding 70-YB-152-0
   nuc = new Nucleus("YB",152,70,0,151.95,0,3.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.46494,100);

   // Adding 71-LU-152-0
   nuc = new Nucleus("LU",152,71,0,151.963,0,0.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,12.3447,100);

   // Adding 59-PR-153-0
   nuc = new Nucleus("PR",153,59,0,152.934,0,4.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.52391,100);

   // Adding 60-ND-153-0
   nuc = new Nucleus("ND",153,60,0,152.928,0,28.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.60001,100);

   // Adding 61-PM-153-0
   nuc = new Nucleus("PM",153,61,0,152.924,0,324,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.90041,100);

   // Adding 62-SM-153-0
   nuc = new Nucleus("SM",153,62,0,152.922,0,166572,0,7.4e-10,6.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.80867,100);

   // Adding 63-EU-153-0
   nuc = new Nucleus("EU",153,63,0,152.921,0,0,52.2,0,0,0,0);

   // Adding 64-GD-153-0
   nuc = new Nucleus("GD",153,64,0,152.922,0,2.08742e+07,0,2.7e-10,2.5e-09,0,0);
   nuc->AddDecay(0,-1,0,0.485069,100);

   // Adding 65-TB-153-0
   nuc = new Nucleus("TB",153,65,0,152.923,0,202176,0,2.5e-10,2.4e-10,0,0);
   nuc->AddDecay(0,-1,0,1.57028,100);

   // Adding 66-DY-153-0
   nuc = new Nucleus("DY",153,66,0,152.926,0,23040,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.5585,0.0094);
   nuc->AddDecay(0,-1,0,2.17049,99.99);

   // Adding 67-HO-153-0
   nuc = new Nucleus("HO",153,67,0,152.93,0,120,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.12868,99.95);
   nuc->AddDecay(-4,-2,0,4.05107,0.05);

   // Adding 67-HO-153-1
   nuc = new Nucleus("HO",153,67,1,152.93,0.068,558,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.19668,99.82);
   nuc->AddDecay(-4,-2,-1,4.11908,0.18);

   // Adding 68-ER-153-0
   nuc = new Nucleus("ER",153,68,0,152.935,0,37.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.8027,53);
   nuc->AddDecay(0,-1,0,4.56366,47);

   // Adding 69-TM-153-0
   nuc = new Nucleus("TM",153,69,0,152.942,0,1.48,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.45909,9);
   nuc->AddDecay(-4,-2,0,5.2481,91);

   // Adding 69-TM-153-1
   nuc = new Nucleus("TM",153,69,1,152.942,0.043,2.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.50209,5);
   nuc->AddDecay(-4,-2,-1,5.2911,95);

   // Adding 70-YB-153-0
   nuc = new Nucleus("YB",153,70,0,152.949,0,4.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.20327,50);
   nuc->AddDecay(0,-1,0,6.68872,50);

   // Adding 71-LU-153-0
   nuc = new Nucleus("LU",153,71,0,152.959,0,0,0,0,0,0,-2);

   // Adding 59-PR-154-0
   nuc = new Nucleus("PR",154,59,0,153.938,0,2.3,0,0,0,0,0);
   nuc->AddDecay(0,1,0,7.9169,100);

   // Adding 60-ND-154-0
   nuc = new Nucleus("ND",154,60,0,153.93,0,25.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.79652,100);

   // Adding 61-PM-154-0
   nuc = new Nucleus("PM",154,61,0,153.927,0,103.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.05402,100);

   // Adding 61-PM-154-1
   nuc = new Nucleus("PM",154,61,1,153.927,0,160.8,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,4.05402,100);

   // Adding 62-SM-154-0
   nuc = new Nucleus("SM",154,62,0,153.922,0,0,22.7,0,0,0,0);

   // Adding 63-EU-154-0
   nuc = new Nucleus("EU",154,63,0,153.923,0,2.70989e+08,0,2e-09,5e-08,0,0);
   nuc->AddDecay(0,-1,0,0.717102,0.02);
   nuc->AddDecay(0,1,0,1.96851,99.98);

   // Adding 63-EU-154-1
   nuc = new Nucleus("EU",154,63,1,153.923,0.145,2778,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.145,100);

   // Adding 64-GD-154-0
   nuc = new Nucleus("GD",154,64,0,153.921,0,0,2.18,0,0,0,0);

   // Adding 65-TB-154-0
   nuc = new Nucleus("TB",154,65,0,153.925,0,77400,0,6.5e-10,6e-10,0,0);
   nuc->AddDecay(0,-1,0,3.56198,99.9);
   nuc->AddDecay(0,1,0,0.245636,0.1);

   // Adding 65-TB-154-1
   nuc = new Nucleus("TB",154,65,1,153.925,0,33840,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,0.245636,0.1);
   nuc->AddDecay(0,0,-1,0,21.7782);
   nuc->AddDecay(0,-1,-1,3.56198,78.1218);

   // Adding 65-TB-154-2
   nuc = new Nucleus("TB",154,65,2,153.925,0,81720,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,3.56198,98.2);
   nuc->AddDecay(0,0,-2,0,1.8);

   // Adding 66-DY-154-0
   nuc = new Nucleus("DY",154,66,0,153.924,0,9.4608e+13,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,2.94658,100);

   // Adding 67-HO-154-0
   nuc = new Nucleus("HO",154,67,0,153.931,0,705.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.7514,99.98);
   nuc->AddDecay(-4,-2,0,4.04159,0.02);

   // Adding 67-HO-154-1
   nuc = new Nucleus("HO",154,67,1,153.931,0.32,186,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,4.36159,0.001);
   nuc->AddDecay(0,0,-1,0.32,0);
   nuc->AddDecay(0,-1,-1,6.0714,100);

   // Adding 68-ER-154-0
   nuc = new Nucleus("ER",154,68,0,153.933,0,223.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.03193,99.53);
   nuc->AddDecay(-4,-2,0,4.27959,0.47);

   // Adding 69-TM-154-0
   nuc = new Nucleus("TM",154,69,0,153.941,0,8.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.05345,56);
   nuc->AddDecay(-4,-2,0,5.09309,44);

   // Adding 69-TM-154-1
   nuc = new Nucleus("TM",154,69,1,153.941,0,3.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.05345,10);
   nuc->AddDecay(-4,-2,-1,5.09309,90);
   nuc->AddDecay(0,0,-1,0,0);

   // Adding 70-YB-154-0
   nuc = new Nucleus("YB",154,70,0,153.946,0,0.404,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.4741,92.8);
   nuc->AddDecay(0,-1,0,4.48901,7.2);

   // Adding 71-LU-154-0
   nuc = new Nucleus("LU",154,71,0,153.957,0,1.12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,10.1126,100);

   // Adding 72-HF-154-0
   nuc = new Nucleus("HF",154,72,0,153.964,0,2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.6598,100);

   // Adding 60-ND-155-0
   nuc = new Nucleus("ND",155,60,0,154.933,0,8.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.0198,100);

   // Adding 61-PM-155-0
   nuc = new Nucleus("PM",155,61,0,154.928,0,48,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.17052,100);

   // Adding 62-SM-155-0
   nuc = new Nucleus("SM",155,62,0,154.925,0,1338,0,2.9e-11,2.8e-11,0,0);
   nuc->AddDecay(0,1,0,1.6271,100);

   // Adding 63-EU-155-0
   nuc = new Nucleus("EU",155,63,0,154.923,0,1.47588e+08,0,3.2e-10,6.5e-09,0,0);
   nuc->AddDecay(0,1,0,0.252472,100);

   // Adding 64-GD-155-0
   nuc = new Nucleus("GD",155,64,0,154.923,0,0,14.8,0,0,0,0);

   // Adding 65-TB-155-0
   nuc = new Nucleus("TB",155,65,0,154.924,0,459648,0,2.1e-10,2.5e-10,0,0);
   nuc->AddDecay(0,-1,0,0.821487,100);

   // Adding 66-DY-155-0
   nuc = new Nucleus("DY",155,66,0,154.926,0,36000,0,1.3e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,2.0945,100);

   // Adding 67-HO-155-0
   nuc = new Nucleus("HO",155,67,0,154.929,0,2880,0,3.7e-11,3.2e-11,1,-8);
   nuc->AddDecay(-4,-2,0,3.14592,50);
   nuc->AddDecay(0,-1,0,3.10199,50);

   // Adding 68-ER-155-0
   nuc = new Nucleus("ER",155,68,0,154.933,0,318,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.84348,99.98);
   nuc->AddDecay(-4,-2,0,4.1186,0.02);

   // Adding 69-TM-155-0
   nuc = new Nucleus("TM",155,69,0,154.939,0,32,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.57843,94);
   nuc->AddDecay(-4,-2,0,4.5693,6);

   // Adding 70-YB-155-0
   nuc = new Nucleus("YB",155,70,0,154.946,0,1.72,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.98812,16);
   nuc->AddDecay(-4,-2,0,5.3371,84);

   // Adding 71-LU-155-0
   nuc = new Nucleus("LU",155,71,0,154.954,0,0.07,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.96394,21);
   nuc->AddDecay(-4,-2,0,5.7711,79);

   // Adding 71-LU-155-1
   nuc = new Nucleus("LU",155,71,1,154.956,1.798,0.0026,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,7.5691,100);

   // Adding 72-HF-155-0
   nuc = new Nucleus("HF",155,72,0,154.963,0,0.89,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.56982,0);
   nuc->AddDecay(0,-1,0,7.99877,100);

   // Adding 60-ND-156-0
   nuc = new Nucleus("ND",156,60,0,155.935,0,5.47,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.10723,100);

   // Adding 61-PM-156-0
   nuc = new Nucleus("PM",156,61,0,155.931,0,26.7,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.15506,100);

   // Adding 62-SM-156-0
   nuc = new Nucleus("SM",156,62,0,155.926,0,33840,0,2.5e-10,2.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.722267,100);

   // Adding 63-EU-156-0
   nuc = new Nucleus("EU",156,63,0,155.925,0,1.31242e+06,0,2.2e-09,3.3e-09,0,0);
   nuc->AddDecay(0,1,0,2.45107,100);

   // Adding 64-GD-156-0
   nuc = new Nucleus("GD",156,64,0,155.922,0,0,20.47,0,0,0,0);

   // Adding 65-TB-156-0
   nuc = new Nucleus("TB",156,65,0,155.925,0,462240,0,1.2e-09,1.4e-09,0,0);
   nuc->AddDecay(0,-1,0,2.44438,100);
   nuc->AddDecay(0,1,0,0.433624,0);

   // Adding 65-TB-156-1
   nuc = new Nucleus("TB",156,65,1,155.925,0.05,87840,0,1.7e-10,2.3e-10,0,0);
   nuc->AddDecay(0,0,-1,0.05,100);

   // Adding 65-TB-156-2
   nuc = new Nucleus("TB",156,65,2,155.925,0.088,19080,0,8.1e-11,1.3e-10,0,-8);
   nuc->AddDecay(0,-1,-2,2.53238,50);
   nuc->AddDecay(0,0,-2,0.088,50);

   // Adding 66-DY-156-0
   nuc = new Nucleus("DY",156,66,0,155.924,0,0,0.06,0,0,0,0);

   // Adding 67-HO-156-0
   nuc = new Nucleus("HO",156,67,0,155.93,0,3360,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.06017,100);

   // Adding 68-ER-156-0
   nuc = new Nucleus("ER",156,68,0,155.931,0,1170,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.36995,100);

   // Adding 69-TM-156-0
   nuc = new Nucleus("TM",156,69,0,155.939,0,83.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.21889,99.94);
   nuc->AddDecay(-4,-2,0,4.34357,0.06);

   // Adding 69-TM-156-1
   nuc = new Nucleus("TM",156,69,1,155.939,0,19,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,4.34357,100);

   // Adding 70-YB-156-0
   nuc = new Nucleus("YB",156,70,0,155.943,0,26.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.57293,90);
   nuc->AddDecay(-4,-2,0,4.8115,10);

   // Adding 71-LU-156-0
   nuc = new Nucleus("LU",156,71,0,155.953,0,0.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.59309,75);
   nuc->AddDecay(0,-1,0,9.44613,25);

   // Adding 71-LU-156-1
   nuc = new Nucleus("LU",156,71,1,155.953,0.32,0.18,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,5.91309,95);
   nuc->AddDecay(0,-1,-1,9.76613,5);

   // Adding 72-HF-156-0
   nuc = new Nucleus("HF",156,72,0,155.959,0,0.025,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.0331,100);

   // Adding 73-TA-156-0
   nuc = new Nucleus("TA",156,73,0,155.972,0,0.22,0,0,0,0,-8);
   nuc->AddDecay(-1,-1,0,1.029,100);

   // Adding 73-TA-156-1
   nuc = new Nucleus("TA",156,73,1,155.972,0.0816,0.32,0,0,0,0,0);
   nuc->AddDecay(-1,-1,-1,1.11059,100);

   // Adding 61-PM-157-0
   nuc = new Nucleus("PM",157,61,0,156.933,0,10.9,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.54692,100);

   // Adding 62-SM-157-0
   nuc = new Nucleus("SM",157,62,0,156.928,0,484.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.70004,100);

   // Adding 63-EU-157-0
   nuc = new Nucleus("EU",157,63,0,156.925,0,54648,0,6e-10,4.4e-10,0,0);
   nuc->AddDecay(0,1,0,1.36257,100);

   // Adding 64-GD-157-0
   nuc = new Nucleus("GD",157,64,0,156.924,0,0,15.65,0,0,0,0);

   // Adding 65-TB-157-0
   nuc = new Nucleus("TB",157,65,0,156.924,0,3.12206e+09,0,3.4e-11,1.2e-09,0,0);
   nuc->AddDecay(0,-1,0,0.0601044,100);

   // Adding 66-DY-157-0
   nuc = new Nucleus("DY",157,66,0,156.925,0,29304,0,6.1e-11,5.5e-11,0,0);
   nuc->AddDecay(0,-1,0,1.34136,100);

   // Adding 66-DY-157-1
   nuc = new Nucleus("DY",157,66,1,156.926,0.199,0.0202,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.199,100);

   // Adding 67-HO-157-0
   nuc = new Nucleus("HO",157,67,0,156.928,0,756,0,6.5e-12,7.6e-12,0,0);
   nuc->AddDecay(0,-1,0,2.54002,100);

   // Adding 68-ER-157-0
   nuc = new Nucleus("ER",157,68,0,156.932,0,1119,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.30416,0);
   nuc->AddDecay(0,-1,0,3.46996,100);

   // Adding 69-TM-157-0
   nuc = new Nucleus("TM",157,69,0,156.937,0,210,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.48008,100);

   // Adding 70-YB-157-0
   nuc = new Nucleus("YB",157,70,0,156.943,0,38.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.53036,99.5);
   nuc->AddDecay(-4,-2,0,4.62226,0.5);

   // Adding 71-LU-157-0
   nuc = new Nucleus("LU",157,71,0,156.95,0,5.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.0959,6);
   nuc->AddDecay(0,-1,0,6.93273,94);

   // Adding 72-HF-157-0
   nuc = new Nucleus("HF",157,72,0,156.958,0,0.11,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.8821,91);
   nuc->AddDecay(0,-1,0,7.47492,9);

   // Adding 73-TA-157-0
   nuc = new Nucleus("TA",157,73,0,156.968,0,0.0053,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.3821,100);

   // Adding 61-PM-158-0
   nuc = new Nucleus("PM",158,61,0,157.937,0,4.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,6.2968,100);

   // Adding 62-SM-158-0
   nuc = new Nucleus("SM",158,62,0,157.93,0,330.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.94457,100);

   // Adding 63-EU-158-0
   nuc = new Nucleus("EU",158,63,0,157.928,0,2754,0,9.4e-11,7.5e-11,0,0);
   nuc->AddDecay(0,1,0,3.48505,100);

   // Adding 64-GD-158-0
   nuc = new Nucleus("GD",158,64,0,157.924,0,0,24.84,0,0,0,0);

   // Adding 65-TB-158-0
   nuc = new Nucleus("TB",158,65,0,157.925,0,5.67648e+09,0,1.1e-09,4.4e-08,0,0);
   nuc->AddDecay(0,1,0,0.936806,16.6);
   nuc->AddDecay(0,-1,0,1.22,83.4);

   // Adding 65-TB-158-1
   nuc = new Nucleus("TB",158,65,1,157.926,0.11,10.5,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,1.04681,0.6);
   nuc->AddDecay(0,-1,-1,1.33,0.01);
   nuc->AddDecay(0,0,-1,0.11,99.39);

   // Adding 66-DY-158-0
   nuc = new Nucleus("DY",158,66,0,157.924,0,0,0.1,0,0,0,0);

   // Adding 67-HO-158-0
   nuc = new Nucleus("HO",158,67,0,157.929,0,678,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.23661,100);

   // Adding 67-HO-158-1
   nuc = new Nucleus("HO",158,67,1,157.929,0.067,1620,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.067,81);
   nuc->AddDecay(0,-1,-1,4.30361,19);

   // Adding 67-HO-158-2
   nuc = new Nucleus("HO",158,67,2,157.929,0.18,1278,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,4.41661,100);

   // Adding 68-ER-158-0
   nuc = new Nucleus("ER",158,68,0,157.93,0,8064,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.899918,100);

   // Adding 69-TM-158-0
   nuc = new Nucleus("TM",158,69,0,157.937,0,241.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.52996,100);

   // Adding 69-TM-158-1
   nuc = new Nucleus("TM",158,69,1,157.937,0,20,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 70-YB-158-0
   nuc = new Nucleus("YB",158,70,0,157.94,0,94.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.17056,0.003);
   nuc->AddDecay(0,-1,0,2.72905,100);

   // Adding 71-LU-158-0
   nuc = new Nucleus("LU",158,71,0,157.949,0,10.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.67298,98.5);
   nuc->AddDecay(-4,-2,0,4.79009,1.5);

   // Adding 72-HF-158-0
   nuc = new Nucleus("HF",158,72,0,157.955,0,2.9,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.4041,46);
   nuc->AddDecay(0,-1,0,5.10302,54);

   // Adding 73-TA-158-0
   nuc = new Nucleus("TA",158,73,0,157.966,0,0.0368,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.2081,93);
   nuc->AddDecay(0,-1,0,10.9166,7);

   // Adding 74-W-158-0
   nuc = new Nucleus("W",158,74,0,157.974,0,0.0014,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6001,100);

   // Adding 62-SM-159-0
   nuc = new Nucleus("SM",159,62,0,158.933,0,11.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.83308,100);

   // Adding 63-EU-159-0
   nuc = new Nucleus("EU",159,63,0,158.929,0,1086,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.51435,100);

   // Adding 64-GD-159-0
   nuc = new Nucleus("GD",159,64,0,158.926,0,66816,0,5e-10,3.9e-10,0,0);
   nuc->AddDecay(0,1,0,0.970596,100);

   // Adding 65-TB-159-0
   nuc = new Nucleus("TB",159,65,0,158.925,0,0,100,0,0,0,0);

   // Adding 66-DY-159-0
   nuc = new Nucleus("DY",159,66,0,158.926,0,1.24762e+07,0,1e-10,3.5e-10,0,0);
   nuc->AddDecay(0,-1,0,0.365593,100);

   // Adding 67-HO-159-0
   nuc = new Nucleus("HO",159,67,0,158.928,0,1983,0,7.9e-12,1e-11,0,0);
   nuc->AddDecay(0,-1,0,1.83759,100);

   // Adding 67-HO-159-1
   nuc = new Nucleus("HO",159,67,1,158.928,0.206,8.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.206,100);

   // Adding 68-ER-159-0
   nuc = new Nucleus("ER",159,68,0,158.931,0,2160,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.76849,100);

   // Adding 69-TM-159-0
   nuc = new Nucleus("TM",159,69,0,158.935,0,549,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.85004,100);

   // Adding 70-YB-159-0
   nuc = new Nucleus("YB",159,70,0,158.94,0,84,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.04999,100);

   // Adding 71-LU-159-0
   nuc = new Nucleus("LU",159,71,0,158.947,0,12.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.98787,99.96);
   nuc->AddDecay(-4,-2,0,4.53276,0.04);

   // Adding 72-HF-159-0
   nuc = new Nucleus("HF",159,72,0,158.954,0,5.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.67845,88);
   nuc->AddDecay(-4,-2,0,5.22309,12);

   // Adding 73-TA-159-0
   nuc = new Nucleus("TA",159,73,0,158.963,0,0.57,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.7461,80);
   nuc->AddDecay(0,-1,0,8.48694,20);

   // Adding 74-W-159-0
   nuc = new Nucleus("W",159,74,0,158.972,0,0.0073,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4431,100);

   // Adding 62-SM-160-0
   nuc = new Nucleus("SM",160,62,0,159.935,0,9.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.0858,100);

   // Adding 63-EU-160-0
   nuc = new Nucleus("EU",160,63,0,159.932,0,38,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.57963,100);

   // Adding 64-GD-160-0
   nuc = new Nucleus("GD",160,64,0,159.927,0,0,21.86,0,0,0,0);

   // Adding 65-TB-160-0
   nuc = new Nucleus("TB",160,65,0,159.927,0,6.24672e+06,0,1.6e-09,6.6e-09,0,0);
   nuc->AddDecay(0,1,0,1.8353,100);

   // Adding 66-DY-160-0
   nuc = new Nucleus("DY",160,66,0,159.925,0,0,2.34,0,0,0,0);

   // Adding 67-HO-160-0
   nuc = new Nucleus("HO",160,67,0,159.929,0,1536,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.29143,100);

   // Adding 67-HO-160-1
   nuc = new Nucleus("HO",160,67,1,159.929,0.06,18072,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.35143,35);
   nuc->AddDecay(0,0,-1,0.06,65);

   // Adding 67-HO-160-2
   nuc = new Nucleus("HO",160,67,2,159.929,0.169,3,0,0,0,0,-8);
   nuc->AddDecay(0,0,-2,0.169,100);

   // Adding 68-ER-160-0
   nuc = new Nucleus("ER",160,68,0,159.929,0,102888,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.327652,100);

   // Adding 69-TM-160-0
   nuc = new Nucleus("TM",160,69,0,159.935,0,564,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.89004,100);

   // Adding 69-TM-160-1
   nuc = new Nucleus("TM",160,69,1,159.935,0.07,74.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.07,85);
   nuc->AddDecay(0,-1,-1,5.96004,15);

   // Adding 70-YB-160-0
   nuc = new Nucleus("YB",160,70,0,159.938,0,288,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.01021,100);

   // Adding 71-LU-160-0
   nuc = new Nucleus("LU",160,71,0,159.946,0,36.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.87997,100);
   nuc->AddDecay(-4,-2,0,4.17823,0.0001);

   // Adding 71-LU-160-1
   nuc = new Nucleus("LU",160,71,1,159.946,0,40,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.87997,100);
   nuc->AddDecay(-4,-2,-1,4.17823,0);

   // Adding 72-HF-160-0
   nuc = new Nucleus("HF",160,72,0,159.951,0,13,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.2974,97.7);
   nuc->AddDecay(-4,-2,0,4.9027,2.3);

   // Adding 73-TA-160-0
   nuc = new Nucleus("TA",160,73,0,159.961,0,1.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.5451,34);
   nuc->AddDecay(0,-1,0,10.0885,66);

   // Adding 74-W-160-0
   nuc = new Nucleus("W",160,74,0,159.968,0,0.081,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.0721,100);

   // Adding 75-RE-160-0
   nuc = new Nucleus("RE",160,75,0,159.981,0,0.00079,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6991,9);
   nuc->AddDecay(-1,-1,0,1.285,91);

   // Adding 63-EU-161-0
   nuc = new Nucleus("EU",161,63,0,160.934,0,26,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.73863,100);

   // Adding 64-GD-161-0
   nuc = new Nucleus("GD",161,64,0,160.93,0,219.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.9556,100);

   // Adding 65-TB-161-0
   nuc = new Nucleus("TB",161,65,0,160.928,0,594432,0,7.2e-10,1.2e-09,0,0);
   nuc->AddDecay(0,1,0,0.593094,100);

   // Adding 66-DY-161-0
   nuc = new Nucleus("DY",161,66,0,160.927,0,0,18.9,0,0,0,0);

   // Adding 67-HO-161-0
   nuc = new Nucleus("HO",161,67,0,160.928,0,8928,0,1.3e-11,1e-11,0,0);
   nuc->AddDecay(0,-1,0,0.858795,100);

   // Adding 67-HO-161-1
   nuc = new Nucleus("HO",161,67,1,160.928,0.211,6.76,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.211,100);

   // Adding 68-ER-161-0
   nuc = new Nucleus("ER",161,68,0,160.93,0,11556,0,8e-11,8.5e-11,0,0);
   nuc->AddDecay(0,-1,0,2.00264,100);

   // Adding 69-TM-161-0
   nuc = new Nucleus("TM",161,69,0,160.933,0,1980,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.16401,100);

   // Adding 70-YB-161-0
   nuc = new Nucleus("YB",161,70,0,160.938,0,252,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.14997,100);

   // Adding 71-LU-161-0
   nuc = new Nucleus("LU",161,71,0,160.944,0,72,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.29998,100);

   // Adding 71-LU-161-1
   nuc = new Nucleus("LU",161,71,1,160.944,0.136,0.0073,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.136,100);

   // Adding 72-HF-161-0
   nuc = new Nucleus("HF",161,72,0,160.95,0,17,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,4.72168,50);
   nuc->AddDecay(0,-1,0,6.32387,50);

   // Adding 73-TA-161-0
   nuc = new Nucleus("TA",161,73,0,160.958,0,2.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.49092,95);
   nuc->AddDecay(-4,-2,0,5.27987,5);

   // Adding 74-W-161-0
   nuc = new Nucleus("W",161,74,0,160.967,0,0.41,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.11815,18);
   nuc->AddDecay(-4,-2,0,5.9231,82);

   // Adding 75-RE-161-0
   nuc = new Nucleus("RE",161,75,0,160.978,0,0.01,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4391,100);

   // Adding 63-EU-162-0
   nuc = new Nucleus("EU",162,63,0,161.937,0,10.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.64315,100);

   // Adding 64-GD-162-0
   nuc = new Nucleus("GD",162,64,0,161.931,0,504,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.39388,100);

   // Adding 65-TB-162-0
   nuc = new Nucleus("TB",162,65,0,161.929,0,456,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.5058,100);

   // Adding 66-DY-162-0
   nuc = new Nucleus("DY",162,66,0,161.927,0,0,25.5,0,0,0,0);

   // Adding 67-HO-162-0
   nuc = new Nucleus("HO",162,67,0,161.929,0,900,0,3.3e-12,4.5e-12,0,0);
   nuc->AddDecay(0,-1,0,2.14019,100);

   // Adding 67-HO-162-1
   nuc = new Nucleus("HO",162,67,1,161.929,0.106,4020,0,2.6e-11,3.3e-11,0,0);
   nuc->AddDecay(0,0,-1,0.106,62);
   nuc->AddDecay(0,-1,-1,2.24619,38);

   // Adding 68-ER-162-0
   nuc = new Nucleus("ER",162,68,0,161.929,0,0,0.14,0,0,0,0);

   // Adding 69-TM-162-0
   nuc = new Nucleus("TM",162,69,0,161.934,0,1302,0,2.9e-11,2.8e-11,0,0);
   nuc->AddDecay(0,-1,0,4.8094,100);

   // Adding 69-TM-162-1
   nuc = new Nucleus("TM",162,69,1,161.934,0.067,24.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.067,82);
   nuc->AddDecay(0,-1,-1,4.8764,18);

   // Adding 70-YB-162-0
   nuc = new Nucleus("YB",162,70,0,161.936,0,1132.2,0,2.3e-11,2.3e-11,0,0);
   nuc->AddDecay(0,-1,0,1.68793,100);

   // Adding 71-LU-162-0
   nuc = new Nucleus("LU",162,71,0,161.943,0,82.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.21998,100);

   // Adding 71-LU-162-1
   nuc = new Nucleus("LU",162,71,1,161.943,0,90,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.21998,100);

   // Adding 71-LU-162-2
   nuc = new Nucleus("LU",162,71,2,161.943,0,114,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,7.21998,100);

   // Adding 72-HF-162-0
   nuc = new Nucleus("HF",162,72,0,161.947,0,37.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.44912,99.99);
   nuc->AddDecay(-4,-2,0,4.41709,0.0087);

   // Adding 73-TA-162-0
   nuc = new Nucleus("TA",162,73,0,161.957,0,3.52,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.26298,99.93);
   nuc->AddDecay(-4,-2,0,5.00709,0.07);

   // Adding 74-W-162-0
   nuc = new Nucleus("W",162,74,0,161.963,0,1.39,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.77003,53);
   nuc->AddDecay(-4,-2,0,5.6741,47);

   // Adding 75-RE-162-0
   nuc = new Nucleus("RE",162,75,0,161.976,0,0.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.5166,97);
   nuc->AddDecay(-4,-2,0,6.2741,3);

   // Adding 76-OS-162-0
   nuc = new Nucleus("OS",162,76,0,161.984,0,0.0019,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7791,100);

   // Adding 64-GD-163-0
   nuc = new Nucleus("GD",163,64,0,162.934,0,68,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.11646,100);

   // Adding 65-TB-163-0
   nuc = new Nucleus("TB",163,65,0,162.931,0,1170,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.78508,100);

   // Adding 66-DY-163-0
   nuc = new Nucleus("DY",163,66,0,162.929,0,0,24.9,0,0,0,0);

   // Adding 67-HO-163-0
   nuc = new Nucleus("HO",163,67,0,162.929,0,1.4412e+11,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.00260162,100);

   // Adding 67-HO-163-1
   nuc = new Nucleus("HO",163,67,1,162.929,0.298,1.09,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.298,100);

   // Adding 68-ER-163-0
   nuc = new Nucleus("ER",163,68,0,162.93,0,4500,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.20998,100);

   // Adding 69-TM-163-0
   nuc = new Nucleus("TM",163,69,0,162.933,0,6516,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.43899,100);

   // Adding 70-YB-163-0
   nuc = new Nucleus("YB",163,70,0,162.936,0,663,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.37005,100);

   // Adding 71-LU-163-0
   nuc = new Nucleus("LU",163,71,0,162.941,0,238,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.59999,100);

   // Adding 72-HF-163-0
   nuc = new Nucleus("HF",163,72,0,162.947,0,40,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.44991,100);

   // Adding 73-TA-163-0
   nuc = new Nucleus("TA",163,73,0,162.954,0,11,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.74898,0.2);
   nuc->AddDecay(0,-1,0,6.80944,99.8);

   // Adding 74-W-163-0
   nuc = new Nucleus("W",163,74,0,162.962,0,2.75,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.5201,41);
   nuc->AddDecay(0,-1,0,7.44957,59);

   // Adding 75-RE-163-0
   nuc = new Nucleus("RE",163,75,0,162.972,0,0.26,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.06709,64);
   nuc->AddDecay(0,-1,0,9.03394,36);

   // Adding 76-OS-163-0
   nuc = new Nucleus("OS",163,76,0,162.982,0,0.833,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.6741,50);
   nuc->AddDecay(0,-1,0,9.30277,50);

   // Adding 64-GD-164-0
   nuc = new Nucleus("GD",164,64,0,163.936,0,45,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.34021,100);

   // Adding 65-TB-164-0
   nuc = new Nucleus("TB",164,65,0,163.933,0,180,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.89002,100);

   // Adding 66-DY-164-0
   nuc = new Nucleus("DY",164,66,0,163.929,0,0,28.2,0,0,0,0);

   // Adding 67-HO-164-0
   nuc = new Nucleus("HO",164,67,0,163.93,0,1740,0,9.5e-12,1.3e-11,0,0);
   nuc->AddDecay(0,-1,0,0.986702,60);
   nuc->AddDecay(0,1,0,0.962402,40);

   // Adding 67-HO-164-1
   nuc = new Nucleus("HO",164,67,1,163.93,0.14,2250,0,1.6e-11,1.6e-11,0,0);
   nuc->AddDecay(0,0,-1,0.14,100);

   // Adding 68-ER-164-0
   nuc = new Nucleus("ER",164,68,0,163.929,0,0,1.61,0,0,0,0);

   // Adding 69-TM-164-0
   nuc = new Nucleus("TM",164,69,0,163.933,0,120,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.96201,100);

   // Adding 69-TM-164-1
   nuc = new Nucleus("TM",164,69,1,163.933,0,306,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,80);
   nuc->AddDecay(0,-1,-1,3.96201,20);

   // Adding 70-YB-164-0
   nuc = new Nucleus("YB",164,70,0,163.935,0,4548,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.99622,100);

   // Adding 71-LU-164-0
   nuc = new Nucleus("LU",164,71,0,163.941,0,188.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.24997,100);

   // Adding 72-HF-164-0
   nuc = new Nucleus("HF",164,72,0,163.944,0,111,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.97394,100);

   // Adding 73-TA-164-0
   nuc = new Nucleus("TA",164,73,0,163.954,0,14.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.52079,100);

   // Adding 74-W-164-0
   nuc = new Nucleus("W",164,74,0,163.959,0,6.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.96827,97.4);
   nuc->AddDecay(-4,-2,0,5.2788,2.6);

   // Adding 75-RE-164-0
   nuc = new Nucleus("RE",164,75,0,163.97,0,0.88,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.9231,58);
   nuc->AddDecay(0,-1,0,10.7328,42);

   // Adding 76-OS-164-0
   nuc = new Nucleus("OS",164,76,0,163.978,0,0.041,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4781,98);
   nuc->AddDecay(0,-1,0,6.98695,2);

   // Adding 65-TB-165-0
   nuc = new Nucleus("TB",165,65,0,164.935,0,126.6,0,0,0,0,0);
   nuc->AddDecay(0,1,1,2.85383,86);
   nuc->AddDecay(0,1,0,2.96183,14);

   // Adding 66-DY-165-0
   nuc = new Nucleus("DY",165,66,0,164.932,0,8402.4,0,1.1e-10,8.7e-11,0,0);
   nuc->AddDecay(0,1,0,1.2862,100);

   // Adding 66-DY-165-1
   nuc = new Nucleus("DY",165,66,1,164.932,0.108,75.42,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.108,97.76);
   nuc->AddDecay(0,1,-1,1.3942,2.24);

   // Adding 67-HO-165-0
   nuc = new Nucleus("HO",165,67,0,164.93,0,0,100,0,0,0,0);

   // Adding 68-ER-165-0
   nuc = new Nucleus("ER",165,68,0,164.931,0,37296,0,1.9e-11,1.4e-11,0,0);
   nuc->AddDecay(0,-1,0,0.376297,100);

   // Adding 69-TM-165-0
   nuc = new Nucleus("TM",165,69,0,164.932,0,108216,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.5925,100);

   // Adding 70-YB-165-0
   nuc = new Nucleus("YB",165,70,0,164.935,0,594,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.76202,100);

   // Adding 71-LU-165-0
   nuc = new Nucleus("LU",165,71,0,164.94,0,644.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.91994,100);

   // Adding 71-LU-165-1
   nuc = new Nucleus("LU",165,71,1,164.94,0,720,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,3.91994,100);

   // Adding 72-HF-165-0
   nuc = new Nucleus("HF",165,72,0,164.945,0,76,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.59521,100);

   // Adding 73-TA-165-0
   nuc = new Nucleus("TA",165,73,0,164.951,0,31,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.84815,100);

   // Adding 74-W-165-0
   nuc = new Nucleus("W",165,74,0,164.958,0,5.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.03178,0.2);
   nuc->AddDecay(0,-1,0,7.00453,99.8);

   // Adding 75-RE-165-0
   nuc = new Nucleus("RE",165,75,0,164.967,0,2.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.65788,13);
   nuc->AddDecay(0,-1,0,8.11702,87);

   // Adding 76-OS-165-0
   nuc = new Nucleus("OS",165,76,0,164.976,0,0.065,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.77736,40);
   nuc->AddDecay(-4,-2,0,6.31709,60);

   // Adding 66-DY-166-0
   nuc = new Nucleus("DY",166,66,0,165.933,0,293760,0,1.6e-09,1.8e-09,0,0);
   nuc->AddDecay(0,1,0,0.486301,100);

   // Adding 67-HO-166-0
   nuc = new Nucleus("HO",166,67,0,165.932,0,96588,0,1.4e-09,8.3e-10,0,0);
   nuc->AddDecay(0,1,0,1.8545,100);

   // Adding 67-HO-166-1
   nuc = new Nucleus("HO",166,67,1,165.932,0.006,3.78432e+10,0,2e-09,1.1e-07,0,0);
   nuc->AddDecay(0,1,-1,1.8605,100);

   // Adding 68-ER-166-0
   nuc = new Nucleus("ER",166,68,0,165.93,0,0,33.6,0,0,0,0);

   // Adding 69-TM-166-0
   nuc = new Nucleus("TM",166,69,0,165.934,0,27720,0,2.8e-10,2.9e-10,0,0);
   nuc->AddDecay(0,-1,0,3.04002,100);

   // Adding 70-YB-166-0
   nuc = new Nucleus("YB",166,70,0,165.934,0,204120,0,9.5e-10,9.5e-10,0,0);
   nuc->AddDecay(0,-1,0,0.303528,100);

   // Adding 71-LU-166-0
   nuc = new Nucleus("LU",166,71,0,165.94,0,159,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.48007,100);

   // Adding 71-LU-166-1
   nuc = new Nucleus("LU",166,71,1,165.94,0.034,84.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.034,42);
   nuc->AddDecay(0,-1,-1,5.51407,58);

   // Adding 71-LU-166-2
   nuc = new Nucleus("LU",166,71,2,165.94,0.043,127.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,5.52307,80);
   nuc->AddDecay(0,0,-2,0.043,20);

   // Adding 72-HF-166-0
   nuc = new Nucleus("HF",166,72,0,165.942,0,406.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.31622,100);

   // Adding 73-TA-166-0
   nuc = new Nucleus("TA",166,73,0,165.95,0,34.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.657,100);

   // Adding 74-W-166-0
   nuc = new Nucleus("W",166,74,0,165.955,0,18.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.23959,99.97);
   nuc->AddDecay(-4,-2,0,4.8565,0.04);

   // Adding 75-RE-166-0
   nuc = new Nucleus("RE",166,75,0,165.966,0,2.8,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,5.63709,100);

   // Adding 76-OS-166-0
   nuc = new Nucleus("OS",166,76,0,165.973,0,0.181,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.1301,72);
   nuc->AddDecay(0,-1,0,6.26304,18);

   // Adding 77-IR-166-0
   nuc = new Nucleus("IR",166,77,0,165.986,0,0.005,0,0,0,1,-5);
   nuc->AddDecay(-4,-2,0,6.7031,99);

   // Adding 66-DY-167-0
   nuc = new Nucleus("DY",167,66,0,166.936,0,372,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.35,100);

   // Adding 67-HO-167-0
   nuc = new Nucleus("HO",167,67,0,166.933,0,11160,0,8.3e-11,1e-10,0,0);
   nuc->AddDecay(0,1,0,1.00667,88.5);
   nuc->AddDecay(0,1,1,0.798668,11.5);

   // Adding 68-ER-167-0
   nuc = new Nucleus("ER",167,68,0,166.932,0,0,22.95,0,0,0,0);

   // Adding 68-ER-167-1
   nuc = new Nucleus("ER",167,68,1,166.932,0.208,2.269,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.208,100);

   // Adding 69-TM-167-0
   nuc = new Nucleus("TM",167,69,0,166.933,0,799200,0,5.6e-10,1.1e-09,0,0);
   nuc->AddDecay(0,-1,0,0.748295,100);

   // Adding 70-YB-167-0
   nuc = new Nucleus("YB",167,70,0,166.935,0,1050,0,6.7e-12,9.5e-12,0,0);
   nuc->AddDecay(0,-1,0,1.95428,100);

   // Adding 71-LU-167-0
   nuc = new Nucleus("LU",167,71,0,166.938,0,3090,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.13004,100);

   // Adding 72-HF-167-0
   nuc = new Nucleus("HF",167,72,0,166.943,0,123,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.9981,100);

   // Adding 73-TA-167-0
   nuc = new Nucleus("TA",167,73,0,166.948,0,84,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.00377,100);

   // Adding 74-W-167-0
   nuc = new Nucleus("W",167,74,0,166.955,0,19.9,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,6.24012,50);
   nuc->AddDecay(-4,-2,0,4.6691,50);

   // Adding 75-RE-167-0
   nuc = new Nucleus("RE",167,75,0,166.963,0,6.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.24284,0.7);
   nuc->AddDecay(0,-1,0,7.38318,99.3);

   // Adding 76-OS-167-0
   nuc = new Nucleus("OS",167,76,0,166.971,0,0.83,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.9791,67);
   nuc->AddDecay(0,-1,0,8.18582,33);

   // Adding 77-IR-167-0
   nuc = new Nucleus("IR",167,77,0,166.982,0,0.005,0,0,0,1,-5);
   nuc->AddDecay(-4,-2,0,6.5431,100);

   // Adding 66-DY-168-0
   nuc = new Nucleus("DY",168,66,0,167.937,0,510,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.61413,100);

   // Adding 67-HO-168-0
   nuc = new Nucleus("HO",168,67,0,167.936,0,179.4,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.9143,100);

   // Adding 68-ER-168-0
   nuc = new Nucleus("ER",168,68,0,167.932,0,0,26.8,0,0,0,0);

   // Adding 69-TM-168-0
   nuc = new Nucleus("TM",168,69,0,167.934,0,8.04384e+06,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.257011,0.01);
   nuc->AddDecay(0,-1,0,1.6791,99.99);

   // Adding 70-YB-168-0
   nuc = new Nucleus("YB",168,70,0,167.934,0,0,0.13,0,0,0,0);

   // Adding 71-LU-168-0
   nuc = new Nucleus("LU",168,71,0,167.939,0,330,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.47486,100);

   // Adding 71-LU-168-1
   nuc = new Nucleus("LU",168,71,1,167.939,0.22,402,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.69487,95);
   nuc->AddDecay(0,0,-1,0.22,5);

   // Adding 72-HF-168-0
   nuc = new Nucleus("HF",168,72,0,167.941,0,1557,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.79868,100);

   // Adding 73-TA-168-0
   nuc = new Nucleus("TA",168,73,0,167.948,0,146.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.66973,100);

   // Adding 74-W-168-0
   nuc = new Nucleus("W",168,74,0,167.952,0,53,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.79417,100);

   // Adding 75-RE-168-0
   nuc = new Nucleus("RE",168,75,0,167.962,0,6.9,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.0631,0.005);
   nuc->AddDecay(0,-1,0,9.07779,99.995);

   // Adding 75-RE-168-1
   nuc = new Nucleus("RE",168,75,1,167.962,0,6.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,5.0631,100);
   nuc->AddDecay(0,-1,-1,9.07779,0);

   // Adding 76-OS-168-0
   nuc = new Nucleus("OS",168,76,0,167.968,0,2.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.8187,49);
   nuc->AddDecay(0,-1,0,5.72387,51);

   // Adding 77-IR-168-0
   nuc = new Nucleus("IR",168,77,0,167.98,0,0.715,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.41109,100);

   // Adding 78-PT-168-0
   nuc = new Nucleus("PT",168,78,0,167.988,0,1.17,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.9901,100);

   // Adding 66-DY-169-0
   nuc = new Nucleus("DY",169,66,0,168.94,0,39,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.19999,100);

   // Adding 67-HO-169-0
   nuc = new Nucleus("HO",169,67,0,168.937,0,282,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.12401,100);

   // Adding 68-ER-169-0
   nuc = new Nucleus("ER",169,68,0,168.935,0,812160,0,3.7e-10,9.8e-10,0,0);
   nuc->AddDecay(0,1,0,0.3512,100);

   // Adding 69-TM-169-0
   nuc = new Nucleus("TM",169,69,0,168.934,0,0,100,0,0,0,0);

   // Adding 70-YB-169-0
   nuc = new Nucleus("YB",169,70,0,168.935,0,2.76705e+06,0,7.1e-10,2.8e-09,0,0);
   nuc->AddDecay(0,-1,0,0.909187,100);

   // Adding 70-YB-169-1
   nuc = new Nucleus("YB",169,70,1,168.935,0.024,46,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.024,100);

   // Adding 71-LU-169-0
   nuc = new Nucleus("LU",169,71,0,168.938,0,122616,0,4.6e-10,4.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.29299,100);

   // Adding 71-LU-169-1
   nuc = new Nucleus("LU",169,71,1,168.938,0.029,160,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.029,100);

   // Adding 72-HF-169-0
   nuc = new Nucleus("HF",169,72,0,168.941,0,194.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.26927,100);

   // Adding 73-TA-169-0
   nuc = new Nucleus("TA",169,73,0,168.946,0,294,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.43508,100);

   // Adding 74-W-169-0
   nuc = new Nucleus("W",169,74,0,168.952,0,76,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.43889,100);

   // Adding 75-RE-169-0
   nuc = new Nucleus("RE",169,75,0,168.959,0,8.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.58611,100);

   // Adding 75-RE-169-1
   nuc = new Nucleus("RE",169,75,1,168.959,0.15,16.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.73611,100);

   // Adding 76-OS-169-0
   nuc = new Nucleus("OS",169,76,0,168.967,0,3.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.68419,89);
   nuc->AddDecay(-4,-2,0,5.71778,11);

   // Adding 77-IR-169-0
   nuc = new Nucleus("IR",169,77,0,168.976,0,0.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.27578,100);
   nuc->AddDecay(0,-1,0,8.67502,0);
   nuc->AddDecay(-1,-1,0,0.757543,0);

   // Adding 78-PT-169-0
   nuc = new Nucleus("PT",169,78,0,168.986,0,0.0025,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.8401,100);

   // Adding 67-HO-170-0
   nuc = new Nucleus("HO",170,67,0,169.94,0,165.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.86998,100);

   // Adding 67-HO-170-1
   nuc = new Nucleus("HO",170,67,1,169.94,0.12,43,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.98998,100);

   // Adding 68-ER-170-0
   nuc = new Nucleus("ER",170,68,0,169.935,0,0,14.9,0,0,0,0);

   // Adding 69-TM-170-0
   nuc = new Nucleus("TM",170,69,0,169.936,0,1.1111e+07,0,1.3e-09,6.6e-09,0,0);
   nuc->AddDecay(0,-1,0,0.3144,0.15);
   nuc->AddDecay(0,1,0,0.968098,99.85);

   // Adding 70-YB-170-0
   nuc = new Nucleus("YB",170,70,0,169.935,0,0,3.05,0,0,0,0);

   // Adding 71-LU-170-0
   nuc = new Nucleus("LU",170,71,0,169.938,0,172800,0,9.9e-10,9.6e-10,0,0);
   nuc->AddDecay(0,-1,0,3.45921,100);

   // Adding 71-LU-170-1
   nuc = new Nucleus("LU",170,71,1,169.939,0.093,0.67,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.093,100);

   // Adding 72-HF-170-0
   nuc = new Nucleus("HF",170,72,0,169.94,0,57636,0,4.8e-10,4.3e-10,0,0);
   nuc->AddDecay(0,-1,0,1.09632,100);

   // Adding 73-TA-170-0
   nuc = new Nucleus("TA",170,73,0,169.946,0,405.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.999,100);

   // Adding 74-W-170-0
   nuc = new Nucleus("W",170,74,0,169.949,0,240,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.97873,100);

   // Adding 75-RE-170-0
   nuc = new Nucleus("RE",170,75,0,169.958,0,8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.26706,100);

   // Adding 76-OS-170-0
   nuc = new Nucleus("OS",170,76,0,169.964,0,7.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.03909,88);
   nuc->AddDecay(-4,-2,0,5.5405,12);

   // Adding 77-IR-170-0
   nuc = new Nucleus("IR",170,77,0,169.975,0,1.05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.17309,75);
   nuc->AddDecay(0,-1,0,10.6762,25);

   // Adding 78-PT-170-0
   nuc = new Nucleus("PT",170,78,0,169.982,0,0.006,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7041,100);

   // Adding 67-HO-171-0
   nuc = new Nucleus("HO",171,67,0,170.941,0,53,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.19997,100);

   // Adding 68-ER-171-0
   nuc = new Nucleus("ER",171,68,0,170.938,0,27057.6,0,3.6e-10,3e-10,0,0);
   nuc->AddDecay(0,1,0,1.4905,100);

   // Adding 69-TM-171-0
   nuc = new Nucleus("TM",171,69,0,170.936,0,6.05491e+07,0,1.1e-10,1.3e-09,0,0);
   nuc->AddDecay(0,1,0,0.0963974,100);

   // Adding 70-YB-171-0
   nuc = new Nucleus("YB",171,70,0,170.936,0,0,14.3,0,0,0,0);

   // Adding 71-LU-171-0
   nuc = new Nucleus("LU",171,71,0,170.938,0,711936,0,6.7e-10,9.4e-10,0,0);
   nuc->AddDecay(0,-1,0,1.4788,100);

   // Adding 71-LU-171-1
   nuc = new Nucleus("LU",171,71,1,170.938,0.071,79,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.071,100);

   // Adding 72-HF-171-0
   nuc = new Nucleus("HF",171,72,0,170.94,0,43560,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.40313,100);

   // Adding 73-TA-171-0
   nuc = new Nucleus("TA",171,73,0,170.944,0,1398,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.69799,100);

   // Adding 74-W-171-0
   nuc = new Nucleus("W",171,74,0,170.949,0,142.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.57293,100);

   // Adding 75-RE-171-0
   nuc = new Nucleus("RE",171,75,0,170.955,0,15.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.66993,100);

   // Adding 76-OS-171-0
   nuc = new Nucleus("OS",171,76,0,170.963,0,8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.06303,98.3);
   nuc->AddDecay(-4,-2,0,5.37011,1.7);

   // Adding 77-IR-171-0
   nuc = new Nucleus("IR",171,77,0,170.972,0,1.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.1591,100);
   nuc->AddDecay(0,-1,0,8.17218,0);
   nuc->AddDecay(-1,-1,0,0.386178,0);

   // Adding 78-PT-171-0
   nuc = new Nucleus("PT",171,78,0,170.981,0,0.025,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.63382,1);
   nuc->AddDecay(-4,-2,0,6.6071,99);

   // Adding 67-HO-172-0
   nuc = new Nucleus("HO",172,67,0,171.945,0,25,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.79285,100);

   // Adding 68-ER-172-0
   nuc = new Nucleus("ER",172,68,0,171.939,0,177480,0,1e-09,1.2e-09,0,0);
   nuc->AddDecay(0,1,0,0.890514,100);

   // Adding 69-TM-172-0
   nuc = new Nucleus("TM",172,69,0,171.938,0,228960,0,1.7e-09,1.4e-09,0,0);
   nuc->AddDecay(0,1,0,1.88017,100);

   // Adding 70-YB-172-0
   nuc = new Nucleus("YB",172,70,0,171.936,0,0,21.9,0,0,0,0);

   // Adding 71-LU-172-0
   nuc = new Nucleus("LU",172,71,0,171.939,0,578880,0,1.3e-09,1.9e-09,0,0);
   nuc->AddDecay(0,-1,0,2.5192,100);

   // Adding 71-LU-172-1
   nuc = new Nucleus("LU",172,71,1,171.939,0.042,222,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.042,100);

   // Adding 72-HF-172-0
   nuc = new Nucleus("HF",172,72,0,171.939,0,5.89723e+07,0,1e-09,3.7e-08,0,0);
   nuc->AddDecay(0,-1,0,0.349979,100);

   // Adding 73-TA-172-0
   nuc = new Nucleus("TA",172,73,0,171.945,0,2208,0,5.3e-11,5.8e-11,0,0);
   nuc->AddDecay(0,-1,0,4.92003,100);

   // Adding 74-W-172-0
   nuc = new Nucleus("W",172,74,0,171.947,0,402,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.50005,100);

   // Adding 75-RE-172-0
   nuc = new Nucleus("RE",172,75,0,171.955,0,15,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.56016,0);
   nuc->AddDecay(0,-1,0,7.32597,100);

   // Adding 75-RE-172-1
   nuc = new Nucleus("RE",172,75,1,171.955,0,55,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,0);
   nuc->AddDecay(-4,-2,-1,4.56016,0);
   nuc->AddDecay(0,-1,-1,7.32597,100);

   // Adding 76-OS-172-0
   nuc = new Nucleus("OS",172,76,0,171.96,0,19,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.2271,0.2);
   nuc->AddDecay(0,-1,0,4.46111,99.8);

   // Adding 77-IR-172-0
   nuc = new Nucleus("IR",172,77,0,171.971,0,2.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.84079,97);
   nuc->AddDecay(-4,-2,0,5.9901,3);

   // Adding 78-PT-172-0
   nuc = new Nucleus("PT",172,78,0,171.977,0,0.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.19837,2);
   nuc->AddDecay(-4,-2,0,6.4646,98);

   // Adding 79-AU-172-0
   nuc = new Nucleus("AU",172,79,0,171.99,0,0.004,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.09741,98);
   nuc->AddDecay(-1,-1,0,1.14431,2);

   // Adding 68-ER-173-0
   nuc = new Nucleus("ER",173,68,0,172.942,0,84,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.59856,100);

   // Adding 69-TM-173-0
   nuc = new Nucleus("TM",173,69,0,172.94,0,29664,0,3.1e-10,2.6e-10,0,0);
   nuc->AddDecay(0,1,0,1.29808,100);

   // Adding 70-YB-173-0
   nuc = new Nucleus("YB",173,70,0,172.938,0,0,16.12,0,0,0,0);

   // Adding 71-LU-173-0
   nuc = new Nucleus("LU",173,71,0,172.939,0,4.32043e+07,0,2.6e-10,2.3e-09,0,0);
   nuc->AddDecay(0,-1,0,0.670803,100);

   // Adding 72-HF-173-0
   nuc = new Nucleus("HF",173,72,0,172.941,0,84960,0,2.3e-10,2.2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.60492,100);

   // Adding 73-TA-173-0
   nuc = new Nucleus("TA",173,73,0,172.944,0,11304,0,1.9e-10,1.6e-10,0,0);
   nuc->AddDecay(0,-1,0,2.78988,100);

   // Adding 74-W-173-0
   nuc = new Nucleus("W",173,74,0,172.948,0,478.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.99985,100);

   // Adding 75-RE-173-0
   nuc = new Nucleus("RE",173,75,0,172.953,0,118.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.77192,100);

   // Adding 76-OS-173-0
   nuc = new Nucleus("OS",173,76,0,172.96,0,16,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.26814,99.98);
   nuc->AddDecay(-4,-2,0,5.0571,0.02);

   // Adding 77-IR-173-0
   nuc = new Nucleus("IR",173,77,0,172.968,0,3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.84508,2.02);
   nuc->AddDecay(0,-1,0,7.37408,97.98);

   // Adding 78-PT-173-0
   nuc = new Nucleus("PT",173,78,0,172.977,0,0.342,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.3529,84);
   nuc->AddDecay(0,-1,0,8.19202,16);

   // Adding 79-AU-173-0
   nuc = new Nucleus("AU",173,79,0,172.986,0,0.059,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.89688,100);

   // Adding 68-ER-174-0
   nuc = new Nucleus("ER",174,68,0,173.944,0,198,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.75585,100);

   // Adding 69-TM-174-0
   nuc = new Nucleus("TM",174,69,0,173.942,0,324,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.07998,100);

   // Adding 70-YB-174-0
   nuc = new Nucleus("YB",174,70,0,173.939,0,0,31.8,0,0,0,0);

   // Adding 71-LU-174-0
   nuc = new Nucleus("LU",174,71,0,173.94,0,1.04384e+08,0,2.7e-10,4e-09,0,0);
   nuc->AddDecay(0,-1,0,1.3744,100);

   // Adding 71-LU-174-1
   nuc = new Nucleus("LU",174,71,1,173.941,0.171,1.22688e+07,0,5.3e-10,3.8e-09,0,0);
   nuc->AddDecay(0,0,-1,0.171,99.38);
   nuc->AddDecay(0,-1,-1,1.5454,0.62);

   // Adding 72-HF-174-0
   nuc = new Nucleus("HF",174,72,0,173.94,0,6.3072e+22,0.162,0,0,1,0);
   nuc->AddDecay(-4,-2,0,2.4958,100);

   // Adding 73-TA-174-0
   nuc = new Nucleus("TA",174,73,0,173.944,0,3780,0,5.7e-11,6.6e-11,0,0);
   nuc->AddDecay(0,-1,0,3.84495,100);

   // Adding 74-W-174-0
   nuc = new Nucleus("W",174,74,0,173.946,0,1860,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.85378,100);

   // Adding 75-RE-174-0
   nuc = new Nucleus("RE",174,75,0,173.953,0,144,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.47689,100);

   // Adding 76-OS-174-0
   nuc = new Nucleus("OS",174,76,0,173.957,0,44,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.8721,0.02);
   nuc->AddDecay(0,-1,0,3.73394,99.98);

   // Adding 77-IR-174-0
   nuc = new Nucleus("IR",174,77,0,173.967,0,4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.6241,0.47);
   nuc->AddDecay(0,-1,0,9.01907,99.53);

   // Adding 78-PT-174-0
   nuc = new Nucleus("PT",174,78,0,173.973,0,0.9,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.1836,83);
   nuc->AddDecay(0,-1,0,5.59859,17);

   // Adding 79-AU-174-0
   nuc = new Nucleus("AU",174,79,0,173.985,0,0.12,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.7821,100);

   // Adding 69-TM-175-0
   nuc = new Nucleus("TM",175,69,0,174.944,0,912,0,2.7e-11,3.1e-11,0,0);
   nuc->AddDecay(0,1,0,2.38498,100);

   // Adding 70-YB-175-0
   nuc = new Nucleus("YB",175,70,0,174.941,0,361584,0,4.4e-10,7e-10,0,0);
   nuc->AddDecay(0,1,0,0.469997,100);

   // Adding 71-LU-175-0
   nuc = new Nucleus("LU",175,71,0,174.941,0,0,97.41,0,0,0,0);

   // Adding 72-HF-175-0
   nuc = new Nucleus("HF",175,72,0,174.941,0,6.048e+06,0,4.1e-10,1.1e-09,0,0);
   nuc->AddDecay(0,-1,0,0.685791,100);

   // Adding 73-TA-175-0
   nuc = new Nucleus("TA",175,73,0,174.944,0,37800,0,2.1e-10,2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.99823,100);

   // Adding 74-W-175-0
   nuc = new Nucleus("W",175,74,0,174.947,0,2112,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.90691,100);

   // Adding 75-RE-175-0
   nuc = new Nucleus("RE",175,75,0,174.951,0,353.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.30574,100);

   // Adding 76-OS-175-0
   nuc = new Nucleus("OS",175,76,0,174.957,0,84,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.25315,100);

   // Adding 77-IR-175-0
   nuc = new Nucleus("IR",175,77,0,174.964,0,9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.57695,99.15);
   nuc->AddDecay(-4,-2,0,5.62009,0.85);

   // Adding 78-PT-175-0
   nuc = new Nucleus("PT",175,78,0,174.972,0,2.52,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.1791,64);
   nuc->AddDecay(0,-1,0,7.62204,36);

   // Adding 79-AU-175-0
   nuc = new Nucleus("AU",175,79,0,174.982,0,0.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7781,94);
   nuc->AddDecay(0,-1,0,8.77118,6);

   // Adding 80-HG-175-0
   nuc = new Nucleus("HG",175,80,0,174.991,0,0.02,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.0391,100);

   // Adding 69-TM-176-0
   nuc = new Nucleus("TM",176,69,0,175.947,0,114,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.87973,100);

   // Adding 70-YB-176-0
   nuc = new Nucleus("YB",176,70,0,175.943,0,0,12.7,0,0,0,0);

   // Adding 70-YB-176-1
   nuc = new Nucleus("YB",176,70,1,175.944,1.05,11.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.05,90);
   nuc->AddDecay(0,1,-1,0.943897,10);

   // Adding 71-LU-176-0
   nuc = new Nucleus("LU",176,71,0,175.943,0,1.19206e+18,2.59,1.8e-09,6.6e-08,0,0);
   nuc->AddDecay(0,1,0,1.1916,100);

   // Adding 71-LU-176-1
   nuc = new Nucleus("LU",176,71,1,175.943,0.123,13086,0,1.7e-10,1.6e-10,0,0);
   nuc->AddDecay(0,1,-1,1.3146,99.91);
   nuc->AddDecay(0,-1,-1,0.229103,0.1);

   // Adding 72-HF-176-0
   nuc = new Nucleus("HF",176,72,0,175.941,0,0,5.206,0,0,0,0);

   // Adding 73-TA-176-0
   nuc = new Nucleus("TA",176,73,0,175.945,0,29124,0,3.2e-10,3.3e-10,0,0);
   nuc->AddDecay(0,-1,0,3.11002,100);

   // Adding 74-W-176-0
   nuc = new Nucleus("W",176,74,0,175.946,0,9000,0,1.1e-10,7.6e-11,0,0);
   nuc->AddDecay(0,-1,0,0.789211,100);

   // Adding 75-RE-176-0
   nuc = new Nucleus("RE",176,75,0,175.952,0,318,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.571,100);

   // Adding 76-OS-176-0
   nuc = new Nucleus("OS",176,76,0,175.955,0,216,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.164,100);

   // Adding 77-IR-176-0
   nuc = new Nucleus("IR",176,77,0,175.964,0,8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.96189,97.9);
   nuc->AddDecay(-4,-2,0,5.2371,2.1);

   // Adding 78-PT-176-0
   nuc = new Nucleus("PT",176,78,0,175.969,0,6.33,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.11011,62);
   nuc->AddDecay(-4,-2,0,5.8861,38);

   // Adding 79-AU-176-0
   nuc = new Nucleus("AU",176,79,0,175.98,0,1.25,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,10.4968,50);
   nuc->AddDecay(-4,-2,0,6.5421,50);

   // Adding 80-HG-176-0
   nuc = new Nucleus("HG",176,80,0,175.987,0,0.034,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.9246,100);

   // Adding 69-TM-177-0
   nuc = new Nucleus("TM",177,69,0,176.949,0,85,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.18813,100);

   // Adding 70-YB-177-0
   nuc = new Nucleus("YB",177,70,0,176.945,0,6879.6,0,9.7e-11,9.4e-11,0,0);
   nuc->AddDecay(0,1,0,1.3993,100);

   // Adding 70-YB-177-1
   nuc = new Nucleus("YB",177,70,1,176.946,0.331,6.41,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.331,100);

   // Adding 71-LU-177-0
   nuc = new Nucleus("LU",177,71,0,176.944,0,581818,0,5.3e-10,1.2e-09,0,0);
   nuc->AddDecay(0,1,0,0.498196,100);

   // Adding 71-LU-177-1
   nuc = new Nucleus("LU",177,71,1,176.945,0.97,1.38586e+07,0,1.7e-09,1.5e-08,0,0);
   nuc->AddDecay(0,1,-1,1.4682,78.3);
   nuc->AddDecay(0,0,-1,0.97,21.7);

   // Adding 72-HF-177-0
   nuc = new Nucleus("HF",177,72,0,176.943,0,0,18.606,0,0,0,0);

   // Adding 72-HF-177-1
   nuc = new Nucleus("HF",177,72,1,176.945,1.315,1.08,0,8.1e-11,1.5e-10,0,0);
   nuc->AddDecay(0,0,-1,1.315,100);

   // Adding 72-HF-177-2
   nuc = new Nucleus("HF",177,72,2,176.946,2.74,3084,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.74,100);

   // Adding 73-TA-177-0
   nuc = new Nucleus("TA",177,73,0,176.944,0,203616,0,1.1e-10,1.3e-10,0,0);
   nuc->AddDecay(0,-1,0,1.16599,100);

   // Adding 74-W-177-0
   nuc = new Nucleus("W",177,74,0,176.947,0,8100,0,6.1e-11,4.6e-11,0,0);
   nuc->AddDecay(0,-1,0,2.00064,100);

   // Adding 75-RE-177-0
   nuc = new Nucleus("RE",177,75,0,176.95,0,840,0,2.2e-11,2.2e-11,0,0);
   nuc->AddDecay(0,-1,0,3.4001,100);

   // Adding 76-OS-177-0
   nuc = new Nucleus("OS",177,76,0,176.955,0,168,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.47292,100);

   // Adding 77-IR-177-0
   nuc = new Nucleus("IR",177,77,0,176.961,0,30,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.67983,99.94);
   nuc->AddDecay(-4,-2,0,5.1271,0.06);

   // Adding 78-PT-177-0
   nuc = new Nucleus("PT",177,78,0,176.968,0,11,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.78414,94.4);
   nuc->AddDecay(-4,-2,0,5.6431,5.6);

   // Adding 79-AU-177-0
   nuc = new Nucleus("AU",177,79,0,176.977,0,1.18,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.42533,40);
   nuc->AddDecay(0,-1,0,8.15631,60);

   // Adding 79-AU-177-1
   nuc = new Nucleus("AU",177,79,1,176.978,0.49,0,0,0,0,0,-2);
   nuc->AddDecay(0,-1,-1,8.64631,0);

   // Adding 80-HG-177-0
   nuc = new Nucleus("HG",177,80,0,176.986,0,0.13,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7385,85);
   nuc->AddDecay(0,-1,0,8.50519,15);

   // Adding 70-YB-178-0
   nuc = new Nucleus("YB",178,70,0,177.947,0,4440,0,1.2e-10,1.1e-10,0,0);
   nuc->AddDecay(0,1,0,0.644623,100);

   // Adding 71-LU-178-0
   nuc = new Nucleus("LU",178,71,0,177.946,0,1704,0,4.7e-11,4.1e-11,0,0);
   nuc->AddDecay(0,1,0,2.09919,100);

   // Adding 71-LU-178-1
   nuc = new Nucleus("LU",178,71,1,177.946,0.22,1386,0,3.8e-11,5.6e-11,0,0);
   nuc->AddDecay(0,1,-1,2.31919,100);

   // Adding 72-HF-178-0
   nuc = new Nucleus("HF",178,72,0,177.944,0,0,27.297,0,0,0,0);

   // Adding 72-HF-178-1
   nuc = new Nucleus("HF",178,72,1,177.945,1.147,4,0,4.7e-09,3.1e-07,0,0);
   nuc->AddDecay(0,0,-1,1.147,100);

   // Adding 72-HF-178-2
   nuc = new Nucleus("HF",178,72,2,177.946,2.446,9.77616e+08,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.446,100);

   // Adding 73-TA-178-0
   nuc = new Nucleus("TA",178,73,0,177.946,0,558.6,0,7.8e-11,1.1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.91201,100);

   // Adding 73-TA-178-1
   nuc = new Nucleus("TA",178,73,1,177.946,0,8496,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,1.91201,100);

   // Adding 74-W-178-0
   nuc = new Nucleus("W",178,74,0,177.946,0,1.86624e+06,0,2.5e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,0.091301,100);

   // Adding 75-RE-178-0
   nuc = new Nucleus("RE",178,75,0,177.951,0,792,0,2.5e-11,2.4e-11,0,0);
   nuc->AddDecay(0,-1,0,4.65999,100);

   // Adding 76-OS-178-0
   nuc = new Nucleus("OS",178,76,0,177.953,0,300,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.3348,100);

   // Adding 77-IR-178-0
   nuc = new Nucleus("IR",178,77,0,177.961,0,12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.19646,100);

   // Adding 78-PT-178-0
   nuc = new Nucleus("PT",178,78,0,177.966,0,21,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.30789,92.3);
   nuc->AddDecay(-4,-2,0,5.5741,7.7);

   // Adding 79-AU-178-0
   nuc = new Nucleus("AU",178,79,0,177.976,0,2.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.56307,60);
   nuc->AddDecay(-4,-2,0,6.1181,40);

   // Adding 80-HG-178-0
   nuc = new Nucleus("HG",178,80,0,177.982,0,0.26,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.578,50);
   nuc->AddDecay(0,-1,0,6.05849,50);

   // Adding 70-YB-179-0
   nuc = new Nucleus("YB",179,70,0,178.95,0,486,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.35306,100);

   // Adding 71-LU-179-0
   nuc = new Nucleus("LU",179,71,0,178.947,0,16524,0,2.1e-10,1.7e-10,0,0);
   nuc->AddDecay(0,1,0,1.40527,100);

   // Adding 72-HF-179-0
   nuc = new Nucleus("HF",179,72,0,178.946,0,0,13.629,0,0,0,0);

   // Adding 72-HF-179-1
   nuc = new Nucleus("HF",179,72,1,178.946,0.375,18.67,0,1.3e-09,3.6e-09,0,0);
   nuc->AddDecay(0,0,-1,0.375,100);

   // Adding 72-HF-179-2
   nuc = new Nucleus("HF",179,72,2,178.947,1.106,2.16864e+06,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,1.106,100);

   // Adding 73-TA-179-0
   nuc = new Nucleus("TA",179,73,0,178.946,0,5.64494e+07,0,6.5e-11,5.2e-10,0,0);
   nuc->AddDecay(0,-1,0,0.110371,100);

   // Adding 74-W-179-0
   nuc = new Nucleus("W",179,74,0,178.947,0,2250,0,3.3e-12,1.8e-12,0,0);
   nuc->AddDecay(0,-1,0,1.05974,100);

   // Adding 74-W-179-1
   nuc = new Nucleus("W",179,74,1,178.947,0.222,384,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,1.28174,0.28);
   nuc->AddDecay(0,0,-1,0.222,99.72);

   // Adding 75-RE-179-0
   nuc = new Nucleus("RE",179,75,0,178.95,0,1170,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.70996,100);

   // Adding 76-OS-179-0
   nuc = new Nucleus("OS",179,76,0,178.954,0,390.001,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.67831,100);

   // Adding 77-IR-179-0
   nuc = new Nucleus("IR",179,77,0,178.959,0,240,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.86184,100);

   // Adding 78-PT-179-0
   nuc = new Nucleus("PT",179,78,0,178.965,0,43,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.28111,0.24);
   nuc->AddDecay(0,-1,0,5.7341,99.76);

   // Adding 79-AU-179-0
   nuc = new Nucleus("AU",179,79,0,178.973,0,7.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.37695,78);
   nuc->AddDecay(-4,-2,0,6.0811,22);

   // Adding 80-HG-179-0
   nuc = new Nucleus("HG",179,80,0,178.982,0,1.09,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.97204,46.85);
   nuc->AddDecay(-1,-2,0,7.68416,0.15);
   nuc->AddDecay(-4,-2,0,6.4311,53);

   // Adding 81-TL-179-0
   nuc = new Nucleus("TL",179,81,0,178.992,0,0.16,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.86006,100);

   // Adding 81-TL-179-1
   nuc = new Nucleus("TL",179,81,1,178.992,0,0.0014,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.86006,100);

   // Adding 70-YB-180-0
   nuc = new Nucleus("YB",180,70,0,180,0,144,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 71-LU-180-0
   nuc = new Nucleus("LU",180,71,0,179.95,0,342,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.10296,100);

   // Adding 72-HF-180-0
   nuc = new Nucleus("HF",180,72,0,179.947,0,0,35.1,0,0,0,0);

   // Adding 72-HF-180-1
   nuc = new Nucleus("HF",180,72,1,179.948,1.142,19800,0,1.7e-10,2e-10,0,0);
   nuc->AddDecay(0,0,-1,1.142,98.6);
   nuc->AddDecay(0,1,-1,0.288002,1.4);

   // Adding 73-TA-180-0
   nuc = new Nucleus("TA",180,73,0,179.947,0,29347.2,0,8.4e-10,2.4e-08,0,0);
   nuc->AddDecay(0,-1,0,0.853996,86);
   nuc->AddDecay(0,1,0,0.708015,14);

   // Adding 73-TA-180-1
   nuc = new Nucleus("TA",180,73,1,179.948,0.075,0,0.0122,5.4e-11,6.2e-11,0,0);
   nuc->AddDecay(0,0,-1,0,0);

   // Adding 74-W-180-0
   nuc = new Nucleus("W",180,74,0,179.947,0,0,0.13,0,0,0,0);

   // Adding 75-RE-180-0
   nuc = new Nucleus("RE",180,75,0,179.951,0,146.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.80231,100);

   // Adding 76-OS-180-0
   nuc = new Nucleus("OS",180,76,0,179.952,0,1290,0,1.7e-11,2.5e-11,0,0);
   nuc->AddDecay(0,-1,0,1.46605,100);

   // Adding 77-IR-180-0
   nuc = new Nucleus("IR",180,77,0,179.959,0,90,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.41688,100);

   // Adding 78-PT-180-0
   nuc = new Nucleus("PT",180,78,0,179.963,0,52,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.6921,99.7);
   nuc->AddDecay(-4,-2,0,5.2571,0.3);

   // Adding 79-AU-180-0
   nuc = new Nucleus("AU",180,79,0,179.972,0,8.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.55289,98.2);
   nuc->AddDecay(-4,-2,0,5.8481,1.8);

   // Adding 80-HG-180-0
   nuc = new Nucleus("HG",180,80,0,179.978,0,3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.52011,51);
   nuc->AddDecay(-4,-2,0,6.2581,49);

   // Adding 81-TL-180-0
   nuc = new Nucleus("TL",180,81,0,179.99,0,0.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.0577,99.999);
   nuc->AddDecay(0,1000,0,0,0.001);

   // Adding 71-LU-181-0
   nuc = new Nucleus("LU",181,71,0,180.952,0,210,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.48753,100);

   // Adding 72-HF-181-0
   nuc = new Nucleus("HF",181,72,0,180.949,0,3.6625e+06,0,1.1e-09,4.7e-09,0,0);
   nuc->AddDecay(0,1,0,1.0274,100);

   // Adding 73-TA-181-0
   nuc = new Nucleus("TA",181,73,0,180.948,0,0,99.988,0,0,0,0);

   // Adding 74-W-181-0
   nuc = new Nucleus("W",181,74,0,180.948,0,1.04717e+07,0,8.2e-11,4.3e-11,0,0);
   nuc->AddDecay(0,-1,0,0.187878,100);

   // Adding 75-RE-181-0
   nuc = new Nucleus("RE",181,75,0,180.95,0,71640,0,4.2e-10,3.7e-10,0,0);
   nuc->AddDecay(0,-1,0,1.73854,100);

   // Adding 76-OS-181-0
   nuc = new Nucleus("OS",181,76,0,180.953,0,6300,0,8.9e-11,1e-10,0,0);
   nuc->AddDecay(0,-1,0,2.92959,100);

   // Adding 76-OS-181-1
   nuc = new Nucleus("OS",181,76,1,180.953,0.049,162,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.97859,100);

   // Adding 77-IR-181-0
   nuc = new Nucleus("IR",181,77,0,180.958,0,294,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.06898,100);

   // Adding 78-PT-181-0
   nuc = new Nucleus("PT",181,78,0,180.963,0,51,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.13311,0.06);
   nuc->AddDecay(0,-1,0,5.22396,99.94);

   // Adding 79-AU-181-0
   nuc = new Nucleus("AU",181,79,0,180.97,0,11.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.29883,98.5);
   nuc->AddDecay(-4,-2,0,5.7521,1.5);

   // Adding 80-HG-181-0
   nuc = new Nucleus("HG",181,80,0,180.978,0,3.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.31914,64);
   nuc->AddDecay(-4,-2,0,6.2871,36);

   // Adding 81-TL-181-0
   nuc = new Nucleus("TL",181,81,0,180.987,0,3.4,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,8.46993,100);

   // Adding 82-PB-181-0
   nuc = new Nucleus("PB",181,82,0,180.997,0,0,0,0,0,0,-2);

   // Adding 82-PB-181-1
   nuc = new Nucleus("PB",181,82,1,180.997,0.09,0.055,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,7.45991,98);
   nuc->AddDecay(0,-1,-1,9.36438,2);

   // Adding 71-LU-182-0
   nuc = new Nucleus("LU",182,71,0,181.955,0,120,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.45657,100);

   // Adding 72-HF-182-0
   nuc = new Nucleus("HF",182,72,0,181.951,0,2.83824e+14,0,3e-09,3.6e-07,0,0);
   nuc->AddDecay(0,1,0,0.373066,100);

   // Adding 72-HF-182-1
   nuc = new Nucleus("HF",182,72,1,181.952,1.173,3690,0,4.2e-11,7.1e-11,0,0);
   nuc->AddDecay(0,0,-1,1.173,42);
   nuc->AddDecay(0,1,-1,1.54607,58);

   // Adding 73-TA-182-0
   nuc = new Nucleus("TA",182,73,0,181.95,0,9.88675e+06,0,1.5e-09,9.8e-09,0,0);
   nuc->AddDecay(0,1,0,1.8136,100);

   // Adding 73-TA-182-1
   nuc = new Nucleus("TA",182,73,1,181.95,0.016,0.283,0,1.2e-11,3.6e-11,0,0);
   nuc->AddDecay(0,0,-1,0.016,100);

   // Adding 73-TA-182-2
   nuc = new Nucleus("TA",182,73,2,181.951,0.52,950.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.52,100);

   // Adding 74-W-182-0
   nuc = new Nucleus("W",182,74,0,181.948,0,0,26.3,0,0,0,0);

   // Adding 75-RE-182-0
   nuc = new Nucleus("RE",182,75,0,181.951,0,230400,0,1.4e-09,1.7e-09,0,0);
   nuc->AddDecay(0,-1,0,2.80002,100);

   // Adding 75-RE-182-1
   nuc = new Nucleus("RE",182,75,1,181.951,0,45720,0,2.7e-10,3e-10,0,0);
   nuc->AddDecay(0,-1,-1,2.80002,100);

   // Adding 76-OS-182-0
   nuc = new Nucleus("OS",182,76,0,181.952,0,79560,0,5.6e-10,5.3e-10,0,0);
   nuc->AddDecay(0,-1,0,0.907986,100);

   // Adding 77-IR-182-0
   nuc = new Nucleus("IR",182,77,0,181.958,0,900,0,4.8e-11,4e-11,0,0);
   nuc->AddDecay(0,-1,0,5.60941,100);

   // Adding 78-PT-182-0
   nuc = new Nucleus("PT",182,78,0,181.961,0,132,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.9426,0.02);
   nuc->AddDecay(0,-1,0,2.8497,99.98);

   // Adding 79-AU-182-0
   nuc = new Nucleus("AU",182,79,0,181.97,0,21,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.5261,0.038);
   nuc->AddDecay(0,-1,0,7.77996,99.962);

   // Adding 80-HG-182-0
   nuc = new Nucleus("HG",182,80,0,181.975,0,11.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.77989,84.8);
   nuc->AddDecay(-4,-2,0,5.9981,15.2);

   // Adding 81-TL-182-0
   nuc = new Nucleus("TL",182,81,0,181.986,0,2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.5501,100);

   // Adding 82-PB-182-0
   nuc = new Nucleus("PB",182,82,0,181.993,0,0.055,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.0761,100);

   // Adding 71-LU-183-0
   nuc = new Nucleus("LU",183,71,0,182.957,0,58,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.53033,100);

   // Adding 72-HF-183-0
   nuc = new Nucleus("HF",183,72,0,182.954,0,3841.2,0,7.3e-11,8.3e-11,0,0);
   nuc->AddDecay(0,1,0,2.01,100);

   // Adding 73-TA-183-0
   nuc = new Nucleus("TA",183,73,0,182.951,0,440640,0,1.3e-09,2e-09,0,0);
   nuc->AddDecay(0,1,0,1.0701,95);
   nuc->AddDecay(0,1,1,0.761101,5);

   // Adding 74-W-183-0
   nuc = new Nucleus("W",183,74,0,182.95,0,0,14.3,0,0,0,0);

   // Adding 74-W-183-1
   nuc = new Nucleus("W",183,74,1,182.951,0.309,5.2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.309,100);

   // Adding 75-RE-183-0
   nuc = new Nucleus("RE",183,75,0,182.951,0,6.048e+06,0,7.6e-10,1.8e-09,0,0);
   nuc->AddDecay(0,-1,0,0.555946,100);

   // Adding 75-RE-183-1
   nuc = new Nucleus("RE",183,75,1,182.953,1.907,0.00104,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.907,100);

   // Adding 76-OS-183-0
   nuc = new Nucleus("OS",183,76,0,182.953,0,46800,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.13178,100);

   // Adding 76-OS-183-1
   nuc = new Nucleus("OS",183,76,1,182.953,0.171,35640,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.171,15);
   nuc->AddDecay(0,-1,-1,2.30278,85);

   // Adding 77-IR-183-0
   nuc = new Nucleus("IR",183,77,0,182.957,0,3420,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.44996,100);

   // Adding 78-PT-183-0
   nuc = new Nucleus("PT",183,78,0,182.962,0,390,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.8391,0.0013);
   nuc->AddDecay(0,-1,0,4.57791,100);

   // Adding 78-PT-183-1
   nuc = new Nucleus("PT",183,78,1,182.962,0.035,43,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.61291,100);
   nuc->AddDecay(0,0,-1,0.035,0);
   nuc->AddDecay(-4,-2,-1,4.8741,0.0004);

   // Adding 79-AU-183-0
   nuc = new Nucleus("AU",183,79,0,182.968,0,42,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.48883,99.64);
   nuc->AddDecay(-4,-2,0,5.4661,0.36);

   // Adding 80-HG-183-0
   nuc = new Nucleus("HG",183,80,0,182.974,0,8.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.3071,74.44);
   nuc->AddDecay(-4,-2,0,6.0391,25.5);
   nuc->AddDecay(-1,-2,0,4.93602,0.06);

   // Adding 81-TL-183-0
   nuc = new Nucleus("TL",183,81,0,182.983,0,6.9,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.6459,100);

   // Adding 81-TL-183-1
   nuc = new Nucleus("TL",183,81,1,182.983,0.55,0.06,0,0,0,1,-2);
   nuc->AddDecay(-4,-2,-1,6.85805,100);

   // Adding 82-PB-183-0
   nuc = new Nucleus("PB",183,82,0,182.992,0,0.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.0271,94);
   nuc->AddDecay(0,-1,0,8.69109,6);

   // Adding 71-LU-184-0
   nuc = new Nucleus("LU",184,71,0,183.961,0,20,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.41194,100);

   // Adding 72-HF-184-0
   nuc = new Nucleus("HF",184,72,0,183.955,0,14832,0,5.3e-10,4.5e-10,0,0);
   nuc->AddDecay(0,1,0,1.33998,100);

   // Adding 73-TA-184-0
   nuc = new Nucleus("TA",184,73,0,183.954,0,31320,0,6.8e-10,6.3e-10,0,0);
   nuc->AddDecay(0,1,0,2.866,100);

   // Adding 74-W-184-0
   nuc = new Nucleus("W",184,74,0,183.951,0,0,30.67,0,0,0,0);

   // Adding 75-RE-184-0
   nuc = new Nucleus("RE",184,75,0,183.953,0,3.2832e+06,0,1e-09,1.8e-09,0,0);
   nuc->AddDecay(0,-1,0,1.48267,100);

   // Adding 75-RE-184-1
   nuc = new Nucleus("RE",184,75,1,183.953,0.188,1.46016e+07,0,1.5e-09,6.1e-09,0,0);
   nuc->AddDecay(0,0,-1,0.188,75.4);
   nuc->AddDecay(0,-1,-1,1.67067,24.6);

   // Adding 76-OS-184-0
   nuc = new Nucleus("OS",184,76,0,183.952,0,0,0.02,0,0,0,0);

   // Adding 77-IR-184-0
   nuc = new Nucleus("IR",184,77,0,183.957,0,11124,0,1.7e-10,1.9e-10,0,0);
   nuc->AddDecay(0,-1,0,4.562,100);

   // Adding 78-PT-184-0
   nuc = new Nucleus("PT",184,78,0,183.96,0,1038,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.33265,100);
   nuc->AddDecay(-4,-2,0,4.5901,0.001);

   // Adding 78-PT-184-1
   nuc = new Nucleus("PT",184,78,1,183.962,1.839,0.00101,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.839,100);

   // Adding 79-AU-184-0
   nuc = new Nucleus("AU",184,79,0,183.968,0,53,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.12688,99.98);
   nuc->AddDecay(-4,-2,0,5.3001,0.02);

   // Adding 80-HG-184-0
   nuc = new Nucleus("HG",184,80,0,183.972,0,30.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.05411,98.89);
   nuc->AddDecay(-4,-2,0,5.6621,1.11);

   // Adding 81-TL-184-0
   nuc = new Nucleus("TL",184,81,0,183.982,0,11,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,9.1889,97.9);
   nuc->AddDecay(-4,-2,0,6.29811,2.1);

   // Adding 82-PB-184-0
   nuc = new Nucleus("PB",184,82,0,183.988,0,0.55,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.7751,100);

   // Adding 72-HF-185-0
   nuc = new Nucleus("HF",185,72,0,184.959,0,210,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.99661,100);

   // Adding 73-TA-185-0
   nuc = new Nucleus("TA",185,73,0,184.956,0,2940,0,6.8e-11,7.2e-11,0,0);
   nuc->AddDecay(0,1,0,1.99201,100);

   // Adding 74-W-185-0
   nuc = new Nucleus("W",185,74,0,184.953,0,6.48864e+06,0,5e-10,2.2e-10,0,0);
   nuc->AddDecay(0,1,0,0.433102,100);

   // Adding 74-W-185-1
   nuc = new Nucleus("W",185,74,1,184.954,0.197,100.2,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.197,100);

   // Adding 75-RE-185-0
   nuc = new Nucleus("RE",185,75,0,184.953,0,0,37.4,0,0,0,0);

   // Adding 76-OS-185-0
   nuc = new Nucleus("OS",185,76,0,184.954,0,8.08704e+06,0,5.1e-10,1.5e-09,0,0);
   nuc->AddDecay(0,-1,0,1.0128,100);

   // Adding 77-IR-185-0
   nuc = new Nucleus("IR",185,77,0,184.957,0,51840,0,2.6e-10,2.6e-10,0,0);
   nuc->AddDecay(0,-1,0,2.37273,100);

   // Adding 78-PT-185-0
   nuc = new Nucleus("PT",185,78,0,184.961,0,4254,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.81797,100);

   // Adding 78-PT-185-1
   nuc = new Nucleus("PT",185,78,1,184.961,0.103,1980,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.92097,99);
   nuc->AddDecay(0,0,-1,0.103,1);

   // Adding 79-AU-185-0
   nuc = new Nucleus("AU",185,79,0,184.966,0,258,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.18011,0.1);
   nuc->AddDecay(0,-1,0,4.707,99.9);

   // Adding 79-AU-185-1
   nuc = new Nucleus("AU",185,79,1,184.966,0,408,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.707,100);
   nuc->AddDecay(0,0,-1,0,0);

   // Adding 80-HG-185-0
   nuc = new Nucleus("HG",185,80,0,184.972,0,49,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.82196,94);
   nuc->AddDecay(-4,-2,0,5.7781,6);

   // Adding 80-HG-185-1
   nuc = new Nucleus("HG",185,80,1,184.972,0.099,21,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.099,53.97);
   nuc->AddDecay(0,-1,-1,5.92096,46);
   nuc->AddDecay(-4,-2,-1,5.8771,0.03);

   // Adding 81-TL-185-0
   nuc = new Nucleus("TL",185,81,0,184.979,0,19.5,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,6.62087,100);

   // Adding 81-TL-185-1
   nuc = new Nucleus("TL",185,81,1,184.98,0.454,1.8,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.454,50);
   nuc->AddDecay(-4,-2,-1,6.55415,50);

   // Adding 82-PB-185-0
   nuc = new Nucleus("PB",185,82,0,184.988,0,4.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6801,100);

   // Adding 73-TA-186-0
   nuc = new Nucleus("TA",186,73,0,185.959,0,630,0,3.3e-11,3.1e-11,0,0);
   nuc->AddDecay(0,1,0,3.90097,100);

   // Adding 74-W-186-0
   nuc = new Nucleus("W",186,74,0,185.954,0,0,28.6,0,0,0,0);

   // Adding 75-RE-186-0
   nuc = new Nucleus("RE",186,75,0,185.955,0,326304,0,1.5e-09,1.2e-09,0,0);
   nuc->AddDecay(0,1,0,1.0695,93.1);
   nuc->AddDecay(0,-1,0,0.5816,6.9);

   // Adding 75-RE-186-1
   nuc = new Nucleus("RE",186,75,1,185.955,0.149,6.3072e+12,0,2.2e-09,1.1e-08,0,0);
   nuc->AddDecay(0,0,-1,0.149,90);
   nuc->AddDecay(0,1,-1,1.2185,10);

   // Adding 76-OS-186-0
   nuc = new Nucleus("OS",186,76,0,185.954,0,6.3072e+22,1.58,0,0,1,0);
   nuc->AddDecay(-4,-2,0,2.822,100);

   // Adding 77-IR-186-0
   nuc = new Nucleus("IR",186,77,0,185.958,0,59904,0,4.9e-10,5e-10,0,0);
   nuc->AddDecay(0,-1,0,3.83101,100);

   // Adding 77-IR-186-1
   nuc = new Nucleus("IR",186,77,1,185.958,0,7200,0,6.1e-11,7.1e-11,0,0);
   nuc->AddDecay(0,-1,-1,3.83101,100);
   nuc->AddDecay(0,0,-1,0,0);

   // Adding 78-PT-186-0
   nuc = new Nucleus("PT",186,78,0,185.959,0,7200,0,9.3e-11,6.6e-11,0,0);
   nuc->AddDecay(0,-1,0,1.37899,100);
   nuc->AddDecay(-4,-2,0,4.32399,0.00014);

   // Adding 79-AU-186-0
   nuc = new Nucleus("AU",186,79,0,185.966,0,642,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.04062,100);

   // Adding 80-HG-186-0
   nuc = new Nucleus("HG",186,80,0,185.969,0,82.8,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.2058,0.02);
   nuc->AddDecay(0,-1,0,3.30029,99.98);

   // Adding 81-TL-186-0
   nuc = new Nucleus("TL",186,81,0,185.979,0,27.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.46526,100);
   nuc->AddDecay(-4,-2,0,5.8911,0.0006);

   // Adding 81-TL-186-1
   nuc = new Nucleus("TL",186,81,1,185.979,0.374,2.9,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.374,100);

   // Adding 82-PB-186-0
   nuc = new Nucleus("PB",186,82,0,185.984,0,4.79,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4701,100);

   // Adding 83-BI-186-0
   nuc = new Nucleus("BI",186,83,0,185.996,0,1.18,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,11.345,100);

   // Adding 74-W-187-0
   nuc = new Nucleus("W",187,74,0,186.957,0,85392,0,7.1e-10,3.3e-10,0,0);
   nuc->AddDecay(0,1,0,1.3112,100);

   // Adding 75-RE-187-0
   nuc = new Nucleus("RE",187,75,0,186.956,0,1.37182e+18,62.6,5.1e-12,6e-12,1,0);
   nuc->AddDecay(-4,-2,0,1.6526,0.0001);
   nuc->AddDecay(0,1,0,0.00260162,100);

   // Adding 76-OS-187-0
   nuc = new Nucleus("OS",187,76,0,186.956,0,0,1.6,0,0,0,0);

   // Adding 77-IR-187-0
   nuc = new Nucleus("IR",187,77,0,186.957,0,37800,0,1.2e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.50236,100);

   // Adding 78-PT-187-0
   nuc = new Nucleus("PT",187,78,0,186.961,0,8460,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.10831,100);

   // Adding 79-AU-187-0
   nuc = new Nucleus("AU",187,79,0,186.965,0,504,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.60001,100);
   nuc->AddDecay(-4,-2,0,4.79309,0.003);

   // Adding 79-AU-187-1
   nuc = new Nucleus("AU",187,79,1,186.965,0.121,2.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.121,100);

   // Adding 80-HG-187-0
   nuc = new Nucleus("HG",187,80,0,186.97,0,114,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.0801,0.00025);
   nuc->AddDecay(0,-1,0,4.86491,100);

   // Adding 80-HG-187-1
   nuc = new Nucleus("HG",187,80,1,186.97,0,144,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,5.0801,0.00012);
   nuc->AddDecay(0,-1,-1,4.86491,100);

   // Adding 81-TL-187-0
   nuc = new Nucleus("TL",187,81,0,186.976,0,51,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.94784,100);
   nuc->AddDecay(-4,-2,0,5.5391,0);

   // Adding 81-TL-187-1
   nuc = new Nucleus("TL",187,81,1,186.977,0.335,15.6,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,5.8741,33.3332);
   nuc->AddDecay(0,-1,-1,6.28284,33.3334);
   nuc->AddDecay(0,0,-1,0.335,33.3334);

   // Adding 82-PB-187-0
   nuc = new Nucleus("PB",187,82,0,186.984,0,15.2,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.3951,50);
   nuc->AddDecay(0,-1,0,7.1631,50);

   // Adding 82-PB-187-1
   nuc = new Nucleus("PB",187,82,1,186.984,0,18.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.1631,98);
   nuc->AddDecay(-4,-2,-1,6.3951,2);

   // Adding 83-BI-187-0
   nuc = new Nucleus("BI",187,83,0,186.993,0,0.035,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.6891,100);

   // Adding 83-BI-187-1
   nuc = new Nucleus("BI",187,83,1,186.994,0.06,0.008,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,7.7491,100);

   // Adding 74-W-188-0
   nuc = new Nucleus("W",188,74,0,187.958,0,5.99616e+06,0,2.3e-09,8.4e-10,0,0);
   nuc->AddDecay(0,1,0,0.348988,100);

   // Adding 75-RE-188-0
   nuc = new Nucleus("RE",188,75,0,187.958,0,61128,0,1.4e-09,7.5e-10,0,0);
   nuc->AddDecay(0,1,0,2.1204,100);

   // Adding 75-RE-188-1
   nuc = new Nucleus("RE",188,75,1,187.958,0.172,1116,0,3e-11,2e-11,0,0);
   nuc->AddDecay(0,0,-1,0.172,100);

   // Adding 76-OS-188-0
   nuc = new Nucleus("OS",188,76,0,187.956,0,0,13.3,0,0,0,0);

   // Adding 77-IR-188-0
   nuc = new Nucleus("IR",188,77,0,187.959,0,149400,0,6.3e-10,6.2e-10,0,0);
   nuc->AddDecay(0,-1,0,2.80935,100);

   // Adding 78-PT-188-0
   nuc = new Nucleus("PT",188,78,0,187.959,0,881280,0,7.6e-10,6.3e-10,0,0);
   nuc->AddDecay(0,-1,0,0.506516,100);
   nuc->AddDecay(-4,-2,0,4.00697,2.6e-05);

   // Adding 79-AU-188-0
   nuc = new Nucleus("AU",188,79,0,187.965,0,530.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.29986,100);

   // Adding 80-HG-188-0
   nuc = new Nucleus("HG",188,80,0,187.968,0,195,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.29792,100);
   nuc->AddDecay(-4,-2,0,4.7101,3.7e-05);

   // Adding 81-TL-188-0
   nuc = new Nucleus("TL",188,81,0,187.976,0,71,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.79495,100);

   // Adding 81-TL-188-1
   nuc = new Nucleus("TL",188,81,1,187.976,0,71,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.79495,100);

   // Adding 82-PB-188-0
   nuc = new Nucleus("PB",188,82,0,187.981,0,24.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.78803,78);
   nuc->AddDecay(-4,-2,0,6.1121,22);

   // Adding 83-BI-188-0
   nuc = new Nucleus("BI",188,83,0,187.992,0,0.044,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,7.2741,50);
   nuc->AddDecay(0,-1,0,10.3509,50);

   // Adding 83-BI-188-1
   nuc = new Nucleus("BI",188,83,1,187.992,0,0.21,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,10.3509,50);
   nuc->AddDecay(-4,-2,-1,7.2741,50);

   // Adding 74-W-189-0
   nuc = new Nucleus("W",189,74,0,188.962,0,690,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.50007,100);

   // Adding 75-RE-189-0
   nuc = new Nucleus("RE",189,75,0,188.959,0,87480,0,7.8e-10,6e-10,0,0);
   nuc->AddDecay(0,1,0,1.00914,100);

   // Adding 76-OS-189-0
   nuc = new Nucleus("OS",189,76,0,188.958,0,0,16.1,0,0,0,0);

   // Adding 76-OS-189-1
   nuc = new Nucleus("OS",189,76,1,188.958,0.031,20880,0,1.8e-11,7.9e-12,0,0);
   nuc->AddDecay(0,0,-1,0.031,100);

   // Adding 77-IR-189-0
   nuc = new Nucleus("IR",189,77,0,188.959,0,1.14048e+06,0,2.4e-10,5.5e-10,0,0);
   nuc->AddDecay(0,-1,0,0.532417,100);

   // Adding 77-IR-189-1
   nuc = new Nucleus("IR",189,77,1,188.959,0.372,0.0133,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.372,100);

   // Adding 77-IR-189-2
   nuc = new Nucleus("IR",189,77,2,188.961,2.333,0.0037,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.333,100);

   // Adding 78-PT-189-0
   nuc = new Nucleus("PT",189,78,0,188.961,0,39132,0,1.2e-10,7.3e-11,0,0);
   nuc->AddDecay(0,-1,0,1.9705,100);

   // Adding 79-AU-189-0
   nuc = new Nucleus("AU",189,79,0,188.964,0,1722,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.85001,100);
   nuc->AddDecay(-4,-2,0,4.37609,3e-05);

   // Adding 79-AU-189-1
   nuc = new Nucleus("AU",189,79,1,188.964,0.247,275.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.247,0);
   nuc->AddDecay(0,-1,-1,3.09701,100);

   // Adding 80-HG-189-0
   nuc = new Nucleus("HG",189,80,0,188.968,0,456,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.50805,3e-05);
   nuc->AddDecay(0,-1,0,3.94992,100);

   // Adding 80-HG-189-1
   nuc = new Nucleus("HG",189,80,1,188.968,0,516,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.94992,100);
   nuc->AddDecay(-4,-2,-1,4.50805,3e-05);

   // Adding 81-TL-189-0
   nuc = new Nucleus("TL",189,81,0,188.974,0,138,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.17594,100);

   // Adding 81-TL-189-1
   nuc = new Nucleus("TL",189,81,1,188.974,0.281,84,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.45694,96);
   nuc->AddDecay(0,0,-1,0.281,4);

   // Adding 82-PB-189-0
   nuc = new Nucleus("PB",189,82,0,188.981,0,51.0001,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.69665,99.6);
   nuc->AddDecay(-4,-2,0,5.85168,0.4);

   // Adding 83-BI-189-0
   nuc = new Nucleus("BI",189,83,0,188.99,0,0.68,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.2691,50);
   nuc->AddDecay(0,-1,0,8.0383,50);

   // Adding 83-BI-189-1
   nuc = new Nucleus("BI",189,83,1,188.99,0.092,0.005,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.1303,50);
   nuc->AddDecay(-4,-2,-1,7.3611,50);

   // Adding 74-W-190-0
   nuc = new Nucleus("W",190,74,0,189.963,0,1800,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.27,100);

   // Adding 75-RE-190-0
   nuc = new Nucleus("RE",190,75,0,189.962,0,186,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.15061,100);

   // Adding 75-RE-190-1
   nuc = new Nucleus("RE",190,75,1,189.962,0.119,11520,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.26961,54.4);
   nuc->AddDecay(0,0,-1,0.119,45.6);

   // Adding 76-OS-190-0
   nuc = new Nucleus("OS",190,76,0,189.958,0,0,26.4,0,0,0,0);

   // Adding 76-OS-190-1
   nuc = new Nucleus("OS",190,76,1,189.96,1.705,594,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.705,100);

   // Adding 77-IR-190-0
   nuc = new Nucleus("IR",190,77,0,189.961,0,1.01779e+06,0,1.2e-09,2.5e-09,0,0);
   nuc->AddDecay(0,-1,0,2.00001,100);

   // Adding 77-IR-190-1
   nuc = new Nucleus("IR",190,77,1,189.961,0.026,4320,0,1.2e-10,1.4e-10,0,0);
   nuc->AddDecay(0,0,-1,0.026,100);

   // Adding 77-IR-190-2
   nuc = new Nucleus("IR",190,77,2,189.961,0.175,11700,0,8e-12,1.1e-11,0,0);
   nuc->AddDecay(0,-1,-2,2.17501,94.4);
   nuc->AddDecay(0,0,-2,0.175,5.6);

   // Adding 78-PT-190-0
   nuc = new Nucleus("PT",190,78,0,189.96,0,2.04984e+19,0.01,8.2e-09,2.3e-07,1,0);
   nuc->AddDecay(-4,-2,0,3.24937,100);

   // Adding 79-AU-190-0
   nuc = new Nucleus("AU",190,79,0,189.965,0,2568,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.8604,1e-06);
   nuc->AddDecay(0,-1,0,4.44204,100);

   // Adding 79-AU-190-1
   nuc = new Nucleus("AU",190,79,1,189.965,0.202999,0.125,0,0,0,0,-2);
   nuc->AddDecay(0,0,-1,0.202999,100);

   // Adding 80-HG-190-0
   nuc = new Nucleus("HG",190,80,0,189.966,0,1200,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,3.95548,5e-05);
   nuc->AddDecay(0,-1,0,1.47407,100);

   // Adding 81-TL-190-0
   nuc = new Nucleus("TL",190,81,0,189.974,0,156,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.99972,100);

   // Adding 81-TL-190-1
   nuc = new Nucleus("TL",190,81,1,189.974,0,222,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.99972,100);

   // Adding 82-PB-190-0
   nuc = new Nucleus("PB",190,82,0,189.978,0,72,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.69752,0.9);
   nuc->AddDecay(0,-1,0,4.08323,99.1);

   // Adding 83-BI-190-0
   nuc = new Nucleus("BI",190,83,0,189.989,0,6.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.8631,82);
   nuc->AddDecay(0,-1,0,9.63084,18);

   // Adding 83-BI-190-1
   nuc = new Nucleus("BI",190,83,1,189.989,0,6.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.8631,68);
   nuc->AddDecay(0,-1,-1,9.63084,32);

   // Adding 75-RE-191-0
   nuc = new Nucleus("RE",191,75,0,190.963,0,588,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.04522,100);

   // Adding 76-OS-191-0
   nuc = new Nucleus("OS",191,76,0,190.961,0,1.33056e+06,0,5.7e-10,1.8e-09,0,0);
   nuc->AddDecay(0,1,0,0.313801,100);

   // Adding 76-OS-191-1
   nuc = new Nucleus("OS",191,76,1,190.961,0.074,47160,0,9.6e-11,1.5e-10,0,0);
   nuc->AddDecay(0,0,-1,0.074,100);

   // Adding 77-IR-191-0
   nuc = new Nucleus("IR",191,77,0,190.961,0,0,37.3,0,0,0,0);

   // Adding 77-IR-191-1
   nuc = new Nucleus("IR",191,77,1,190.961,0.171,4.94,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.171,100);

   // Adding 77-IR-191-2
   nuc = new Nucleus("IR",191,77,2,190.963,2.047,5.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.047,100);

   // Adding 78-PT-191-0
   nuc = new Nucleus("PT",191,78,0,190.962,0,250560,0,3.4e-10,1.9e-10,0,0);
   nuc->AddDecay(0,-1,0,1.01868,100);

   // Adding 79-AU-191-0
   nuc = new Nucleus("AU",191,79,0,190.964,0,11448,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.82999,100);

   // Adding 79-AU-191-1
   nuc = new Nucleus("AU",191,79,1,190.964,0.266,0.92,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.266,100);

   // Adding 80-HG-191-0
   nuc = new Nucleus("HG",191,80,0,190.967,0,2940,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.17997,100);

   // Adding 80-HG-191-1
   nuc = new Nucleus("HG",191,80,1,190.967,0,3048,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,3.17997,100);

   // Adding 81-TL-191-0
   nuc = new Nucleus("TL",191,81,0,190.972,0,0,0,0,0,0,-2);

   // Adding 81-TL-191-1
   nuc = new Nucleus("TL",191,81,1,190.972,0.299,313.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.79167,100);

   // Adding 82-PB-191-0
   nuc = new Nucleus("PB",191,82,0,190.978,0,79.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.881,99.99);
   nuc->AddDecay(-4,-2,0,5.41312,0.01);

   // Adding 82-PB-191-1
   nuc = new Nucleus("PB",191,82,1,190.978,0,130.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.881,100);

   // Adding 83-BI-191-0
   nuc = new Nucleus("BI",191,83,0,190.986,0,12,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.31581,40);
   nuc->AddDecay(-4,-2,0,6.7811,60);

   // Adding 83-BI-191-1
   nuc = new Nucleus("BI",191,83,1,190.986,0.242,0.15,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,7.0231,50);
   nuc->AddDecay(0,-1,-1,7.55781,50);

   // Adding 75-RE-192-0
   nuc = new Nucleus("RE",192,75,0,191.966,0,16,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.17424,100);

   // Adding 76-OS-192-0
   nuc = new Nucleus("OS",192,76,0,191.961,0,0,41,0,0,0,0);

   // Adding 76-OS-192-1
   nuc = new Nucleus("OS",192,76,1,191.964,2.015,5.9,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.015,87);
   nuc->AddDecay(0,1,-1,0.968792,13);

   // Adding 77-IR-192-0
   nuc = new Nucleus("IR",192,77,0,191.963,0,6.379e+06,0,1.4e-09,6.2e-09,0,0);
   nuc->AddDecay(0,1,0,1.45971,95.24);
   nuc->AddDecay(0,-1,0,1.04621,4.76);

   // Adding 77-IR-192-1
   nuc = new Nucleus("IR",192,77,1,191.963,0.057,87,0,3.1e-10,3.6e-08,0,0);
   nuc->AddDecay(0,0,-1,0.057,99.98);
   nuc->AddDecay(0,1,-1,1.5167,0.02);

   // Adding 77-IR-192-2
   nuc = new Nucleus("IR",192,77,2,191.963,0.155,7.60018e+09,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.155,100);

   // Adding 78-PT-192-0
   nuc = new Nucleus("PT",192,78,0,191.961,0,0,0.79,0,0,0,0);

   // Adding 79-AU-192-0
   nuc = new Nucleus("AU",192,79,0,191.965,0,17784,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.51632,100);

   // Adding 80-HG-192-0
   nuc = new Nucleus("HG",192,80,0,191.966,0,17460,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.710896,100);

   // Adding 81-TL-192-0
   nuc = new Nucleus("TL",192,81,0,191.972,0,576,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.12003,100);

   // Adding 81-TL-192-1
   nuc = new Nucleus("TL",192,81,1,191.972,0,648,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.12003,100);

   // Adding 82-PB-192-0
   nuc = new Nucleus("PB",192,82,0,191.976,0,210,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.36952,99.99);
   nuc->AddDecay(-4,-2,0,5.2211,0.0057);

   // Adding 83-BI-192-0
   nuc = new Nucleus("BI",192,83,0,191.985,0,37,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.94995,82);
   nuc->AddDecay(-4,-2,0,6.3761,18);

   // Adding 83-BI-192-1
   nuc = new Nucleus("BI",192,83,1,191.985,0.105,39.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.4811,9.2);
   nuc->AddDecay(0,-1,-1,9.05495,90.8);

   // Adding 84-PO-192-0
   nuc = new Nucleus("PO",192,84,0,191.992,0,0.034,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.3191,99);
   nuc->AddDecay(0,-1,0,5.73103,1);

   // Adding 76-OS-193-0
   nuc = new Nucleus("OS",193,76,0,192.964,0,109800,0,8.1e-10,6.8e-10,0,0);
   nuc->AddDecay(0,1,0,1.14059,100);

   // Adding 77-IR-193-0
   nuc = new Nucleus("IR",193,77,0,192.963,0,0,62.7,0,0,0,0);

   // Adding 77-IR-193-1
   nuc = new Nucleus("IR",193,77,1,192.963,0.08,909792,0,2.7e-10,1.2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.08,100);

   // Adding 78-PT-193-0
   nuc = new Nucleus("PT",193,78,0,192.963,0,1.5768e+09,0,3.1e-11,2.7e-11,0,0);
   nuc->AddDecay(0,-1,0,0.0567017,100);

   // Adding 78-PT-193-1
   nuc = new Nucleus("PT",193,78,1,192.963,0.15,374112,0,4.5e-10,2.1e-10,0,0);
   nuc->AddDecay(0,0,-1,0.15,100);

   // Adding 79-AU-193-0
   nuc = new Nucleus("AU",193,79,0,192.964,0,63540,0,1.4e-10,1.6e-10,0,0);
   nuc->AddDecay(0,-1,0,1.06844,100);

   // Adding 79-AU-193-1
   nuc = new Nucleus("AU",193,79,1,192.964,0.29,3.9,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.29,99.97);
   nuc->AddDecay(0,-1,-1,1.35844,0.03);

   // Adding 80-HG-193-0
   nuc = new Nucleus("HG",193,80,0,192.967,0,13680,0,8.3e-11,1.1e-09,0,0);
   nuc->AddDecay(0,-1,0,2.34037,100);

   // Adding 80-HG-193-1
   nuc = new Nucleus("HG",193,80,1,192.967,0.141,42480,0,4e-10,3.1e-09,0,0);
   nuc->AddDecay(0,-1,-1,2.48137,92.9);
   nuc->AddDecay(0,0,-1,0.141,7.1);

   // Adding 81-TL-193-0
   nuc = new Nucleus("TL",193,81,0,192.971,0,1296,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.63915,100);

   // Adding 81-TL-193-1
   nuc = new Nucleus("TL",193,81,1,192.971,0.365,126.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.365,75);
   nuc->AddDecay(0,-1,-1,4.00415,25);

   // Adding 82-PB-193-0
   nuc = new Nucleus("PB",193,82,0,192.976,0,0,0,0,0,0,-9);
   nuc->AddDecay(0,-1,0,5.15098,0);

   // Adding 82-PB-193-1
   nuc = new Nucleus("PB",193,82,1,192.976,0.1,348,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.25098,100);

   // Adding 83-BI-193-0
   nuc = new Nucleus("BI",193,83,0,192.983,0,67,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.50184,95);
   nuc->AddDecay(-4,-2,0,6.3051,5);

   // Adding 83-BI-193-1
   nuc = new Nucleus("BI",193,83,1,192.983,0.307,3.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,6.80884,10);
   nuc->AddDecay(-4,-2,-1,6.6121,90);

   // Adding 84-PO-193-0
   nuc = new Nucleus("PO",193,84,0,192.991,0,0.26,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.09853,100);

   // Adding 76-OS-194-0
   nuc = new Nucleus("OS",194,76,0,193.965,0,1.89216e+08,0,2.4e-09,7.9e-08,0,0);
   nuc->AddDecay(0,1,0,0.0965958,100);

   // Adding 77-IR-194-0
   nuc = new Nucleus("IR",194,77,0,193.965,0,68940,0,1.3e-09,7.5e-10,0,0);
   nuc->AddDecay(0,1,0,2.2469,100);

   // Adding 77-IR-194-1
   nuc = new Nucleus("IR",194,77,1,193.965,0.19,1.47744e+07,0,2.1e-09,1.2e-08,0,0);
   nuc->AddDecay(0,1,-1,2.4369,100);

   // Adding 78-PT-194-0
   nuc = new Nucleus("PT",194,78,0,193.963,0,0,32.9,0,0,0,0);

   // Adding 79-AU-194-0
   nuc = new Nucleus("AU",194,79,0,193.965,0,136872,0,4.2e-10,3.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.49212,100);

   // Adding 80-HG-194-0
   nuc = new Nucleus("HG",194,80,0,193.965,0,1.63987e+10,0,5.1e-08,4e-08,0,0);
   nuc->AddDecay(0,-1,0,0.0399895,100);

   // Adding 81-TL-194-0
   nuc = new Nucleus("TL",194,81,0,193.971,0,1980,0,8.1e-12,8.9e-12,1,0);
   nuc->AddDecay(-4,-2,0,3.49314,1e-07);
   nuc->AddDecay(0,-1,0,5.28195,100);

   // Adding 81-TL-194-1
   nuc = new Nucleus("TL",194,81,1,193.971,0,1968,0,4e-11,3.7e-11,0,0);
   nuc->AddDecay(0,-1,-1,5.28195,100);

   // Adding 82-PB-194-0
   nuc = new Nucleus("PB",194,82,0,193.974,0,720,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,4.7371,7.3e-06);
   nuc->AddDecay(0,-1,0,2.71802,100);

   // Adding 83-BI-194-0
   nuc = new Nucleus("BI",194,83,0,193.983,0,106,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.18072,100);

   // Adding 83-BI-194-1
   nuc = new Nucleus("BI",194,83,1,193.983,0,92,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,8.18072,99.93);
   nuc->AddDecay(-4,-2,-1,5.9181,0.07);

   // Adding 83-BI-194-2
   nuc = new Nucleus("BI",194,83,2,193.983,0,125,0,0,0,0,0);
   nuc->AddDecay(0,-1,-2,8.18072,99.79);
   nuc->AddDecay(-4,-2,-2,5.9181,0.21);

   // Adding 84-PO-194-0
   nuc = new Nucleus("PO",194,84,0,193.988,0,0.44,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.98718,100);

   // Adding 85-AT-194-0
   nuc = new Nucleus("AT",194,85,0,193.999,0,0.18,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,7.50005,100);

   // Adding 76-OS-195-0
   nuc = new Nucleus("OS",195,76,0,194.968,0,390,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.99998,100);

   // Adding 77-IR-195-0
   nuc = new Nucleus("IR",195,77,0,194.966,0,9000,0,1e-10,1e-10,0,0);
   nuc->AddDecay(0,1,0,1.1201,100);

   // Adding 77-IR-195-1
   nuc = new Nucleus("IR",195,77,1,194.966,0.1,13680,0,2.1e-10,2.4e-10,0,0);
   nuc->AddDecay(0,1,-1,1.2201,95);
   nuc->AddDecay(0,0,-1,0.1,5);

   // Adding 78-PT-195-0
   nuc = new Nucleus("PT",195,78,0,194.965,0,0,33.8,0,0,0,0);

   // Adding 78-PT-195-1
   nuc = new Nucleus("PT",195,78,1,194.965,0.259,347328,0,6.3e-10,3.1e-10,0,0);
   nuc->AddDecay(0,0,-1,0.259,100);

   // Adding 79-AU-195-0
   nuc = new Nucleus("AU",195,79,0,194.965,0,1.60782e+07,0,2.6e-10,1.6e-09,0,0);
   nuc->AddDecay(0,-1,0,0.226799,100);

   // Adding 79-AU-195-1
   nuc = new Nucleus("AU",195,79,1,194.965,0.319,30.5,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.319,100);

   // Adding 80-HG-195-0
   nuc = new Nucleus("HG",195,80,0,194.967,0,35640,0,9.7e-11,1.4e-09,0,0);
   nuc->AddDecay(0,-1,0,1.50998,100);

   // Adding 80-HG-195-1
   nuc = new Nucleus("HG",195,80,1,194.967,0.176,149760,0,5.6e-10,8.2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.176,54.2);
   nuc->AddDecay(0,-1,-1,1.68598,45.8);

   // Adding 81-TL-195-0
   nuc = new Nucleus("TL",195,81,0,194.97,0,4176,0,2.7e-11,3e-11,0,0);
   nuc->AddDecay(0,-1,0,2.80205,100);

   // Adding 81-TL-195-1
   nuc = new Nucleus("TL",195,81,1,194.97,0.483,3.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.483,100);

   // Adding 82-PB-195-0
   nuc = new Nucleus("PB",195,82,0,194.974,0,900,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.49369,100);

   // Adding 82-PB-195-1
   nuc = new Nucleus("PB",195,82,1,194.975,0.201,900,0,2.9e-11,3e-11,0,0);
   nuc->AddDecay(0,-1,-1,4.69469,100);

   // Adding 83-BI-195-0
   nuc = new Nucleus("BI",195,83,0,194.981,0,183,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.85019,99.97);
   nuc->AddDecay(-4,-2,0,5.8331,0.03);

   // Adding 83-BI-195-1
   nuc = new Nucleus("BI",195,83,1,194.981,0.401,87,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.2341,33);
   nuc->AddDecay(0,-1,-1,6.25119,67);

   // Adding 84-PO-195-0
   nuc = new Nucleus("PO",195,84,0,194.988,0,4.5,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.794,25);
   nuc->AddDecay(-4,-2,0,6.74609,75);

   // Adding 84-PO-195-1
   nuc = new Nucleus("PO",195,84,1,194.988,0.23,2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.97609,90);
   nuc->AddDecay(0,0,-1,0.23,0.01);
   nuc->AddDecay(0,-1,-1,7.024,10);

   // Adding 85-AT-195-0
   nuc = new Nucleus("AT",195,85,0,194.997,0,2.72,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.40005,75);
   nuc->AddDecay(0,-1,0,7.96977,25);

   // Adding 76-OS-196-0
   nuc = new Nucleus("OS",196,76,0,195.97,0,2094,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.1575,100);

   // Adding 77-IR-196-0
   nuc = new Nucleus("IR",196,77,0,195.968,0,52,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.20909,100);

   // Adding 77-IR-196-1
   nuc = new Nucleus("IR",196,77,1,195.969,0.41,5040,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,3.61909,100);

   // Adding 78-PT-196-0
   nuc = new Nucleus("PT",196,78,0,195.965,0,0,25.3,0,0,0,0);

   // Adding 79-AU-196-0
   nuc = new Nucleus("AU",196,79,0,195.967,0,534211,0,4.4e-10,3.7e-10,0,0);
   nuc->AddDecay(0,-1,0,1.50579,92.5);
   nuc->AddDecay(0,1,0,0.6859,7.5);

   // Adding 79-AU-196-1
   nuc = new Nucleus("AU",196,79,1,195.967,0.085,8.1,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.085,100);

   // Adding 79-AU-196-2
   nuc = new Nucleus("AU",196,79,2,195.967,0.596,34920,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.596,100);

   // Adding 80-HG-196-0
   nuc = new Nucleus("HG",196,80,0,195.966,0,0,0.14,0,0,0,0);

   // Adding 81-TL-196-0
   nuc = new Nucleus("TL",196,81,0,195.971,0,6624,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.37651,100);

   // Adding 81-TL-196-1
   nuc = new Nucleus("TL",196,81,1,195.971,0.395,5076,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.77151,95.5);
   nuc->AddDecay(0,0,-1,0.395,4.5);

   // Adding 82-PB-196-0
   nuc = new Nucleus("PB",196,82,0,195.973,0,2220,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.04573,100);
   nuc->AddDecay(-4,-2,0,4.22242,0.0001);

   // Adding 83-BI-196-0
   nuc = new Nucleus("BI",196,83,0,195.981,0,300,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,7.35873,100);

   // Adding 83-BI-196-1
   nuc = new Nucleus("BI",196,83,1,195.981,0,276,0,0,0,0,-8);
   nuc->AddDecay(0,-1,-1,7.35873,50);
   nuc->AddDecay(0,0,-1,0,50);

   // Adding 84-PO-196-0
   nuc = new Nucleus("PO",196,84,0,195.986,0,5.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6561,100);
   nuc->AddDecay(0,-1,0,4.56449,0);

   // Adding 85-AT-196-0
   nuc = new Nucleus("AT",196,85,0,195.996,0,0.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.20209,100);

   // Adding 77-IR-197-0
   nuc = new Nucleus("IR",197,77,0,196.97,0,348,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.15471,100);

   // Adding 77-IR-197-1
   nuc = new Nucleus("IR",197,77,1,196.97,0.115,534,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.26971,99.75);
   nuc->AddDecay(0,0,-1,0.115,0.25);

   // Adding 78-PT-197-0
   nuc = new Nucleus("PT",197,78,0,196.967,0,65880,0,4.1e-10,1.6e-10,0,0);
   nuc->AddDecay(0,1,0,0.718901,100);

   // Adding 78-PT-197-1
   nuc = new Nucleus("PT",197,78,1,196.968,0.4,5724.6,0,8.4e-11,4.4e-11,0,0);
   nuc->AddDecay(0,1,-1,1.1189,3.3);
   nuc->AddDecay(0,0,-1,0.4,96.7);

   // Adding 79-AU-197-0
   nuc = new Nucleus("AU",197,79,0,196.967,0,0,100,0,0,0,0);

   // Adding 79-AU-197-1
   nuc = new Nucleus("AU",197,79,1,196.967,0.409,7.73,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.409,100);

   // Adding 80-HG-197-0
   nuc = new Nucleus("HG",197,80,0,196.967,0,230904,0,2.3e-10,4.4e-09,0,0);
   nuc->AddDecay(0,-1,0,0.599787,100);

   // Adding 80-HG-197-1
   nuc = new Nucleus("HG",197,80,1,196.968,0.299,85680,0,4.7e-10,5.8e-09,0,0);
   nuc->AddDecay(0,0,-1,0.299,93);
   nuc->AddDecay(0,-1,-1,0.898787,7);

   // Adding 81-TL-197-0
   nuc = new Nucleus("TL",197,81,0,196.97,0,10224,0,2.3e-11,2.7e-11,0,0);
   nuc->AddDecay(0,-1,0,2.18282,100);

   // Adding 81-TL-197-1
   nuc = new Nucleus("TL",197,81,1,196.97,0.608,0.54,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.608,100);

   // Adding 82-PB-197-0
   nuc = new Nucleus("PB",197,82,0,196.973,0,480,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.57882,100);

   // Adding 82-PB-197-1
   nuc = new Nucleus("PB",197,82,1,196.974,0.319,2580,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,4.16932,0.0003);
   nuc->AddDecay(0,-1,-1,3.89783,81);
   nuc->AddDecay(0,0,-1,0.319,19);

   // Adding 83-BI-197-0
   nuc = new Nucleus("BI",197,83,0,196.979,0,558,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.17599,100);
   nuc->AddDecay(-4,-2,0,5.38715,0.0001);

   // Adding 83-BI-197-1
   nuc = new Nucleus("BI",197,83,1,196.979,0.5,312,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.67599,44.865);
   nuc->AddDecay(-4,-2,-1,5.88715,54.835);
   nuc->AddDecay(0,0,-1,0.5,0.3);

   // Adding 84-PO-197-0
   nuc = new Nucleus("PO",197,84,0,196.986,0,56,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.17492,56);
   nuc->AddDecay(-4,-2,0,6.41109,44);

   // Adding 84-PO-197-1
   nuc = new Nucleus("PO",197,84,1,196.986,0.204,26,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.204,0.01);
   nuc->AddDecay(-4,-2,-1,6.61509,84);
   nuc->AddDecay(0,-1,-1,6.37892,16);

   // Adding 85-AT-197-0
   nuc = new Nucleus("AT",197,85,0,196.993,0,0.35,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.1031,96);
   nuc->AddDecay(0,-1,0,7.19384,4);

   // Adding 85-AT-197-1
   nuc = new Nucleus("AT",197,85,1,196.993,0.052,3.7,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,7.24584,0);
   nuc->AddDecay(-4,-2,-1,7.1551,100);

   // Adding 77-IR-198-0
   nuc = new Nucleus("IR",198,77,0,197.972,0,8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.10265,100);

   // Adding 78-PT-198-0
   nuc = new Nucleus("PT",198,78,0,197.968,0,0,7.2,0,0,0,0);

   // Adding 79-AU-198-0
   nuc = new Nucleus("AU",198,79,0,197.968,0,232718,0,1e-09,1.1e-09,0,0);
   nuc->AddDecay(0,1,0,1.3724,100);

   // Adding 79-AU-198-1
   nuc = new Nucleus("AU",198,79,1,197.969,0.812,198720,0,1.3e-09,2e-09,0,0);
   nuc->AddDecay(0,0,-1,0.812,100);

   // Adding 80-HG-198-0
   nuc = new Nucleus("HG",198,80,0,197.967,0,0,10.02,0,0,0,0);

   // Adding 81-TL-198-0
   nuc = new Nucleus("TL",198,81,0,197.97,0,19080,0,7.3e-11,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,3.45995,100);

   // Adding 81-TL-198-1
   nuc = new Nucleus("TL",198,81,1,197.971,0.544,6732,0,5.5e-11,7.3e-11,0,0);
   nuc->AddDecay(0,-1,-1,4.00395,56);
   nuc->AddDecay(0,0,-1,0.544,44);

   // Adding 81-TL-198-2
   nuc = new Nucleus("TL",198,81,2,197.971,0.742,0.0321,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.742,100);

   // Adding 82-PB-198-0
   nuc = new Nucleus("PB",198,82,0,197.972,0,8640,0,1e-10,8.7e-11,0,0);
   nuc->AddDecay(0,-1,0,1.40995,100);

   // Adding 83-BI-198-0
   nuc = new Nucleus("BI",198,83,0,197.979,0,711,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.56051,100);

   // Adding 83-BI-198-1
   nuc = new Nucleus("BI",198,83,1,197.979,0.249,7.7,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.249,100);

   // Adding 84-PO-198-0
   nuc = new Nucleus("PO",198,84,0,197.983,0,105.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.02727,30);
   nuc->AddDecay(-4,-2,0,6.3091,70);

   // Adding 85-AT-198-0
   nuc = new Nucleus("AT",198,85,0,197.993,0,4.9,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,8.76472,10);
   nuc->AddDecay(-4,-2,0,6.8931,90);

   // Adding 85-AT-198-1
   nuc = new Nucleus("AT",198,85,1,197.993,0.1,1.5,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.1,33.3333);
   nuc->AddDecay(-4,-2,-1,6.9931,33.3333);
   nuc->AddDecay(0,-1,-1,8.86472,33.3333);

   // Adding 86-RN-198-0
   nuc = new Nucleus("RN",198,86,0,197.999,0,0.05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.3446,99);
   nuc->AddDecay(0,-1,0,5.60381,1);

   // Adding 78-PT-199-0
   nuc = new Nucleus("PT",199,78,0,198.971,0,1848,0,3.9e-11,2.2e-11,0,0);
   nuc->AddDecay(0,1,0,1.70229,100);

   // Adding 78-PT-199-1
   nuc = new Nucleus("PT",199,78,1,198.971,0.424,13.6,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.424,100);

   // Adding 79-AU-199-0
   nuc = new Nucleus("AU",199,79,0,198.969,0,271210,0,4.4e-10,7.6e-10,0,0);
   nuc->AddDecay(0,1,0,0.4526,100);

   // Adding 80-HG-199-0
   nuc = new Nucleus("HG",199,80,0,198.968,0,0,16.84,0,0,0,0);

   // Adding 80-HG-199-1
   nuc = new Nucleus("HG",199,80,1,198.969,0.533,2556,0,3.1e-11,1.8e-10,0,0);
   nuc->AddDecay(0,0,-1,0.533,100);

   // Adding 81-TL-199-0
   nuc = new Nucleus("TL",199,81,0,198.97,0,26712,0,2.6e-11,3.7e-11,0,0);
   nuc->AddDecay(0,-1,0,1.44463,100);

   // Adding 81-TL-199-1
   nuc = new Nucleus("TL",199,81,1,198.971,0.75,0.0284,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.75,100);

   // Adding 82-PB-199-0
   nuc = new Nucleus("PB",199,82,0,198.973,0,5400,0,5.4e-11,4.8e-11,0,0);
   nuc->AddDecay(0,-1,0,2.88333,100);

   // Adding 82-PB-199-1
   nuc = new Nucleus("PB",199,82,1,198.973,0.43,732,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.43,93);
   nuc->AddDecay(0,-1,-1,3.31333,7);

   // Adding 83-BI-199-0
   nuc = new Nucleus("BI",199,83,0,198.978,0,1620,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.34488,100);

   // Adding 83-BI-199-1
   nuc = new Nucleus("BI",199,83,1,198.978,0.68,1482,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.02488,99);
   nuc->AddDecay(0,0,-1,0.68,0.989998);
   nuc->AddDecay(-4,-2,-1,5.63851,0.01);

   // Adding 84-PO-199-0
   nuc = new Nucleus("PO",199,84,0,198.984,0,312,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.60928,88);
   nuc->AddDecay(-4,-2,0,6.0741,12);

   // Adding 84-PO-199-1
   nuc = new Nucleus("PO",199,84,1,198.984,0.31,252,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.91928,59);
   nuc->AddDecay(-4,-2,-1,6.3841,39);
   nuc->AddDecay(0,0,-1,0.31,2.1);

   // Adding 85-AT-199-0
   nuc = new Nucleus("AT",199,85,0,198.991,0,7.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.77909,90);
   nuc->AddDecay(0,-1,0,6.55519,10);

   // Adding 86-RN-199-0
   nuc = new Nucleus("RN",199,86,0,198.998,0,0.62,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.13609,95);
   nuc->AddDecay(0,-1,0,7.151,5);

   // Adding 86-RN-199-1
   nuc = new Nucleus("RN",199,86,1,198.999,0.2,0.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,7.33609,100);
   nuc->AddDecay(0,-1,-1,7.351,1);

   // Adding 78-PT-200-0
   nuc = new Nucleus("PT",200,78,0,199.971,0,45000,0,1.2e-09,4e-10,0,0);
   nuc->AddDecay(0,1,0,0.657331,100);

   // Adding 79-AU-200-0
   nuc = new Nucleus("AU",200,79,0,199.971,0,2904,0,6.8e-11,5.6e-11,0,0);
   nuc->AddDecay(0,1,0,2.24408,100);

   // Adding 79-AU-200-1
   nuc = new Nucleus("AU",200,79,1,199.972,0.99,67320,0,1.1e-09,1e-09,0,0);
   nuc->AddDecay(0,0,-1,0.99,18);
   nuc->AddDecay(0,1,-1,3.23408,82);

   // Adding 80-HG-200-0
   nuc = new Nucleus("HG",200,80,0,199.968,0,0,23.13,0,0,0,0);

   // Adding 81-TL-200-0
   nuc = new Nucleus("TL",200,81,0,199.971,0,93960,0,2e-10,2.5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.45597,100);

   // Adding 81-TL-200-1
   nuc = new Nucleus("TL",200,81,1,199.972,0.754,0.0343,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.754,100);

   // Adding 82-PB-200-0
   nuc = new Nucleus("PB",200,82,0,199.972,0,77400,0,4e-10,2.6e-10,0,0);
   nuc->AddDecay(0,-1,0,0.810452,100);

   // Adding 83-BI-200-0
   nuc = new Nucleus("BI",200,83,0,199.978,0,2184,0,5.1e-11,5.6e-11,0,0);
   nuc->AddDecay(0,-1,0,5.89213,100);

   // Adding 83-BI-200-1
   nuc = new Nucleus("BI",200,83,1,199.978,0.2,1860,0,0,0,0,-4);
   nuc->AddDecay(0,-1,-1,6.09213,90);
   nuc->AddDecay(0,0,-1,0.2,10);

   // Adding 83-BI-200-2
   nuc = new Nucleus("BI",200,83,2,199.979,0.428,0.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,0.428,100);

   // Adding 84-PO-200-0
   nuc = new Nucleus("PO",200,84,0,199.982,0,690,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.34775,85);
   nuc->AddDecay(-4,-2,0,5.98236,15);

   // Adding 85-AT-200-0
   nuc = new Nucleus("AT",200,85,0,199.99,0,43,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.97287,65);
   nuc->AddDecay(-4,-2,0,6.5965,35);

   // Adding 85-AT-200-1
   nuc = new Nucleus("AT",200,85,1,199.991,0.29,4.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.29,80);
   nuc->AddDecay(-4,-2,-1,6.8865,10);
   nuc->AddDecay(0,-1,-1,8.26287,10);

   // Adding 86-RN-200-0
   nuc = new Nucleus("RN",200,86,0,199.996,0,1.06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.0441,98);
   nuc->AddDecay(0,-1,0,5.01209,2);

   // Adding 78-PT-201-0
   nuc = new Nucleus("PT",201,78,0,200.975,0,150,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.65996,100);

   // Adding 79-AU-201-0
   nuc = new Nucleus("AU",201,79,0,200.972,0,1560,0,2.5e-11,2.9e-11,0,0);
   nuc->AddDecay(0,1,0,1.27492,100);

   // Adding 80-HG-201-0
   nuc = new Nucleus("HG",201,80,0,200.97,0,0,13.22,0,0,0,0);

   // Adding 81-TL-201-0
   nuc = new Nucleus("TL",201,81,0,200.971,0,262483,0,9.5e-11,7.6e-11,0,0);
   nuc->AddDecay(0,-1,0,0.482815,100);

   // Adding 81-TL-201-1
   nuc = new Nucleus("TL",201,81,1,200.972,0.92,0.002035,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.92,100);

   // Adding 82-PB-201-0
   nuc = new Nucleus("PB",201,82,0,200.973,0,33588,0,1.6e-10,1.2e-10,0,0);
   nuc->AddDecay(0,-1,0,1.90289,100);

   // Adding 82-PB-201-1
   nuc = new Nucleus("PB",201,82,1,200.974,0.629,61,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.629,99);
   nuc->AddDecay(0,-1,-1,2.53189,1);

   // Adding 83-BI-201-0
   nuc = new Nucleus("BI",201,83,0,200.977,0,6480,0,1.2e-10,1.1e-10,1,0);
   nuc->AddDecay(-4,-2,0,4.5002,1e-06);
   nuc->AddDecay(0,-1,0,3.8438,100);

   // Adding 83-BI-201-1
   nuc = new Nucleus("BI",201,83,1,200.978,0.846,3546,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,4.6898,93);
   nuc->AddDecay(0,0,-1,0.846,6.60562);
   nuc->AddDecay(-4,-2,-1,5.3462,0.295774);

   // Adding 84-PO-201-0
   nuc = new Nucleus("PO",201,84,0,200.982,0,918,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.87773,98.4);
   nuc->AddDecay(-4,-2,0,5.7991,1.6);

   // Adding 84-PO-201-1
   nuc = new Nucleus("PO",201,84,1,200.983,0.424,534,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.424,40);
   nuc->AddDecay(-4,-2,-1,6.2231,2.9);
   nuc->AddDecay(0,-1,-1,5.30173,57);

   // Adding 85-AT-201-0
   nuc = new Nucleus("AT",201,85,0,200.988,0,89,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4734,71);
   nuc->AddDecay(0,-1,0,5.85029,29);

   // Adding 86-RN-201-0
   nuc = new Nucleus("RN",201,86,0,200.996,0,7,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.86109,80);
   nuc->AddDecay(0,-1,0,6.56262,20);

   // Adding 86-RN-201-1
   nuc = new Nucleus("RN",201,86,1,200.996,0.28,3.8,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,7.14109,90);
   nuc->AddDecay(0,-1,-1,6.84262,10);
   nuc->AddDecay(0,0,-1,0.28,0);

   // Adding 87-FR-201-0
   nuc = new Nucleus("FR",201,87,0,201.004,0,0.048,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.87155,1);
   nuc->AddDecay(-4,-2,0,7.5388,99);

   // Adding 78-PT-202-0
   nuc = new Nucleus("PT",202,78,0,201.976,0,156960,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.81612,100);

   // Adding 79-AU-202-0
   nuc = new Nucleus("AU",202,79,0,201.974,0,28.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.94621,100);

   // Adding 80-HG-202-0
   nuc = new Nucleus("HG",202,80,0,201.971,0,0,29.8,0,0,0,0);

   // Adding 81-TL-202-0
   nuc = new Nucleus("TL",202,81,0,201.972,0,1.05667e+06,0,4.5e-10,3.1e-10,0,0);
   nuc->AddDecay(0,-1,0,1.36441,100);

   // Adding 82-PB-202-0
   nuc = new Nucleus("PB",202,82,0,201.972,0,1.65564e+12,0,8.8e-09,1.4e-08,0,0);
   nuc->AddDecay(0,-1,0,0.0496159,99);
   nuc->AddDecay(-4,-2,0,2.59763,1);

   // Adding 82-PB-202-1
   nuc = new Nucleus("PB",202,82,1,201.974,2.17,12708,0,1.4e-10,1.3e-10,0,0);
   nuc->AddDecay(0,0,-1,2.17,90.5);
   nuc->AddDecay(0,-1,-1,2.21962,9.5);

   // Adding 83-BI-202-0
   nuc = new Nucleus("BI",202,83,0,201.978,0,6192,0,8.9e-11,1e-10,0,0);
   nuc->AddDecay(0,-1,0,5.15545,100);
   nuc->AddDecay(-4,-2,0,4.29313,1e-07);

   // Adding 84-PO-202-0
   nuc = new Nucleus("PO",202,84,0,201.981,0,2682,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.81692,98);
   nuc->AddDecay(-4,-2,0,5.7001,2);

   // Adding 85-AT-202-0
   nuc = new Nucleus("AT",202,85,0,201.988,0,181,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.21382,88);
   nuc->AddDecay(-4,-2,0,6.3534,12);

   // Adding 85-AT-202-1
   nuc = new Nucleus("AT",202,85,1,201.989,0.391,1.5,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.391,100);

   // Adding 86-RN-202-0
   nuc = new Nucleus("RN",202,86,0,201.993,0,9.85,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.44796,15);
   nuc->AddDecay(-4,-2,0,6.7741,85);

   // Adding 87-FR-202-0
   nuc = new Nucleus("FR",202,87,0,202.003,0,0.34,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.38895,97);
   nuc->AddDecay(0,-1,0,9.37958,3);

   // Adding 79-AU-203-0
   nuc = new Nucleus("AU",203,79,0,202.975,0,53,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.13932,100);

   // Adding 80-HG-203-0
   nuc = new Nucleus("HG",203,80,0,202.973,0,4.02728e+06,0,1.9e-09,7e-09,0,0);
   nuc->AddDecay(0,1,0,0.491798,100);

   // Adding 81-TL-203-0
   nuc = new Nucleus("TL",203,81,0,202.972,0,0,29.524,0,0,0,0);

   // Adding 82-PB-203-0
   nuc = new Nucleus("PB",203,82,0,202.973,0,186743,0,2.4e-10,1.6e-10,0,0);
   nuc->AddDecay(0,-1,0,0.97496,100);

   // Adding 82-PB-203-1
   nuc = new Nucleus("PB",203,82,1,202.974,0.825,6.3,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.825,100);

   // Adding 82-PB-203-2
   nuc = new Nucleus("PB",203,82,2,202.977,2.949,0.48,0,0,0,0,0);
   nuc->AddDecay(0,0,-2,2.949,100);

   // Adding 83-BI-203-0
   nuc = new Nucleus("BI",203,83,0,202.977,0,42336,0,4.8e-10,4.5e-10,0,0);
   nuc->AddDecay(0,-1,0,3.25305,100);
   nuc->AddDecay(-4,-2,0,4.14628,1e-05);

   // Adding 83-BI-203-1
   nuc = new Nucleus("BI",203,83,1,202.978,1.098,0.303,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.098,100);

   // Adding 84-PO-203-0
   nuc = new Nucleus("PO",203,84,0,202.981,0,2202,0,5.2e-11,6.1e-11,0,0);
   nuc->AddDecay(0,-1,0,4.23306,99.89);
   nuc->AddDecay(-4,-2,0,5.496,0.11);

   // Adding 84-PO-203-1
   nuc = new Nucleus("PO",203,84,1,202.982,0.641,45,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.641,99.96);
   nuc->AddDecay(-4,-2,-1,6.137,0.04);

   // Adding 85-AT-203-0
   nuc = new Nucleus("AT",203,85,0,202.987,0,444,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.05898,69);
   nuc->AddDecay(-4,-2,0,6.2101,31);

   // Adding 86-RN-203-0
   nuc = new Nucleus("RN",203,86,0,202.993,0,45,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6291,66);
   nuc->AddDecay(0,-1,0,6.02828,34);

   // Adding 86-RN-203-1
   nuc = new Nucleus("RN",203,86,1,202.994,0.361,28,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.361,0.1);
   nuc->AddDecay(-4,-2,-1,6.9901,79.92);
   nuc->AddDecay(0,-1,-1,6.38928,19.98);

   // Adding 87-FR-203-0
   nuc = new Nucleus("FR",203,87,0,203.001,0,0.55,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.20364,5);
   nuc->AddDecay(-4,-2,0,7.27755,95);

   // Adding 79-AU-204-0
   nuc = new Nucleus("AU",204,79,0,203.978,0,39.8,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.80033,100);

   // Adding 80-HG-204-0
   nuc = new Nucleus("HG",204,80,0,203.973,0,0,6.85,0,0,0,0);

   // Adding 81-TL-204-0
   nuc = new Nucleus("TL",204,81,0,203.974,0,1.19206e+08,0,1.3e-09,6.2e-10,0,0);
   nuc->AddDecay(0,1,0,0.7637,97.43);
   nuc->AddDecay(0,-1,0,0.347301,2.57);

   // Adding 82-PB-204-0
   nuc = new Nucleus("PB",204,82,0,203.973,0,0,1.4,0,0,0,0);

   // Adding 82-PB-204-1
   nuc = new Nucleus("PB",204,82,1,203.975,2.186,4032,0,5e-11,4.4e-11,0,0);
   nuc->AddDecay(0,0,-1,2.186,100);

   // Adding 83-BI-204-0
   nuc = new Nucleus("BI",204,83,0,203.978,0,40392,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.43771,100);

   // Adding 83-BI-204-1
   nuc = new Nucleus("BI",204,83,1,203.981,2.833,0.00107,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.833,100);

   // Adding 84-PO-204-0
   nuc = new Nucleus("PO",204,84,0,203.98,0,12708,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.4849,0.66);
   nuc->AddDecay(0,-1,0,2.34201,99.34);

   // Adding 85-AT-204-0
   nuc = new Nucleus("AT",204,85,0,203.987,0,552,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.47674,95.7);
   nuc->AddDecay(-4,-2,0,6.06952,4.3);

   // Adding 85-AT-204-1
   nuc = new Nucleus("AT",204,85,1,203.987,0,0.108,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 86-RN-204-0
   nuc = new Nucleus("RN",204,86,0,203.991,0,74.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.5461,68);
   nuc->AddDecay(0,-1,0,3.82433,32);

   // Adding 87-FR-204-0
   nuc = new Nucleus("FR",204,87,0,204.001,0,2.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.17024,80);
   nuc->AddDecay(0,-1,0,8.59701,20);

   // Adding 88-RA-204-0
   nuc = new Nucleus("RA",204,88,0,204.006,0,0,0,0,0,0,-2);

   // Adding 79-AU-205-0
   nuc = new Nucleus("AU",205,79,0,204.98,0,31,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.40395,100);

   // Adding 80-HG-205-0
   nuc = new Nucleus("HG",205,80,0,204.976,0,312,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.53138,100);

   // Adding 81-TL-205-0
   nuc = new Nucleus("TL",205,81,0,204.974,0,0,70.476,0,0,0,0);

   // Adding 82-PB-205-0
   nuc = new Nucleus("PB",205,82,0,204.974,0,4.82501e+14,0,2.8e-10,4.1e-10,0,0);
   nuc->AddDecay(0,-1,0,0.051199,100);

   // Adding 82-PB-205-1
   nuc = new Nucleus("PB",205,82,1,204.976,1.014,0.00554,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.014,100);

   // Adding 83-BI-205-0
   nuc = new Nucleus("BI",205,83,0,204.977,0,1.32278e+06,0,9e-10,1e-09,0,0);
   nuc->AddDecay(0,-1,0,2.70835,100);

   // Adding 84-PO-205-0
   nuc = new Nucleus("PO",205,84,0,204.981,0,5976,0,5.9e-11,8.9e-11,0,0);
   nuc->AddDecay(0,-1,0,3.53105,99.96);
   nuc->AddDecay(-4,-2,0,5.324,0.04);

   // Adding 85-AT-205-0
   nuc = new Nucleus("AT",205,85,0,204.986,0,1572,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.0196,10);
   nuc->AddDecay(0,-1,0,4.5394,90);

   // Adding 86-RN-205-0
   nuc = new Nucleus("RN",205,86,0,204.992,0,168,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.24421,77);
   nuc->AddDecay(-4,-2,0,6.38609,23);

   // Adding 87-FR-205-0
   nuc = new Nucleus("FR",205,87,0,204.999,0,3.85,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.0553,99);
   nuc->AddDecay(0,-1,0,6.5195,1);

   // Adding 88-RA-205-0
   nuc = new Nucleus("RA",205,88,0,205.006,0,0.22,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,7.5065,50);
   nuc->AddDecay(0,-1,0,7.01382,50);

   // Adding 80-HG-206-0
   nuc = new Nucleus("HG",206,80,0,205.977,0,489,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.30811,100);

   // Adding 81-TL-206-0
   nuc = new Nucleus("TL",206,81,0,205.976,0,251.94,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.5332,100);

   // Adding 81-TL-206-1
   nuc = new Nucleus("TL",206,81,1,205.979,2.643,224.4,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.643,100);

   // Adding 82-PB-206-0
   nuc = new Nucleus("PB",206,82,0,205.974,0,0,24.1,0,0,0,0);

   // Adding 83-BI-206-0
   nuc = new Nucleus("BI",206,83,0,205.978,0,539395,0,1.9e-09,2.1e-09,0,0);
   nuc->AddDecay(0,-1,0,3.75745,100);

   // Adding 84-PO-206-0
   nuc = new Nucleus("PO",206,84,0,205.98,0,760320,0,1.3e-07,3.7e-07,1,0);
   nuc->AddDecay(-4,-2,0,5.3264,5.45);
   nuc->AddDecay(0,-1,0,1.84638,94.55);

   // Adding 85-AT-206-0
   nuc = new Nucleus("AT",206,85,0,205.987,0,1800,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.8884,0.89);
   nuc->AddDecay(0,-1,0,5.71745,99.11);

   // Adding 86-RN-206-0
   nuc = new Nucleus("RN",206,86,0,205.99,0,340.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.3841,62);
   nuc->AddDecay(0,-1,0,3.31262,38);

   // Adding 87-FR-206-0
   nuc = new Nucleus("FR",206,87,0,205.998,0,15.9,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.9256,88);
   nuc->AddDecay(0,-1,0,7.75532,12);

   // Adding 87-FR-206-1
   nuc = new Nucleus("FR",206,87,1,205.999,0.531,0.7,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,7.4566,100);

   // Adding 88-RA-206-0
   nuc = new Nucleus("RA",206,88,0,206.004,0,0.24,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.4154,100);

   // Adding 80-HG-207-0
   nuc = new Nucleus("HG",207,80,0,206.983,0,174,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.77504,100);

   // Adding 81-TL-207-0
   nuc = new Nucleus("TL",207,81,0,206.977,0,286.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.42277,100);

   // Adding 81-TL-207-1
   nuc = new Nucleus("TL",207,81,1,206.979,1.348,1.33,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.348,100);

   // Adding 82-PB-207-0
   nuc = new Nucleus("PB",207,82,0,206.976,0,0,22.1,0,0,0,0);

   // Adding 82-PB-207-1
   nuc = new Nucleus("PB",207,82,1,206.978,1.633,0.806,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.633,100);

   // Adding 83-BI-207-0
   nuc = new Nucleus("BI",207,83,0,206.978,0,9.94961e+08,0,1.3e-09,5.3e-09,0,0);
   nuc->AddDecay(0,-1,0,2.39879,7);
   nuc->AddDecay(0,-1,1,0.765793,93);

   // Adding 84-PO-207-0
   nuc = new Nucleus("PO",207,84,0,206.982,0,20880,0,1.4e-10,1.5e-10,0,0);
   nuc->AddDecay(0,-1,0,2.90857,99.98);
   nuc->AddDecay(-4,-2,0,5.2159,0.02);

   // Adding 84-PO-207-1
   nuc = new Nucleus("PO",207,84,1,206.983,1.383,2.79,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.383,100);

   // Adding 85-AT-207-0
   nuc = new Nucleus("AT",207,85,0,206.986,0,6480,0,2.3e-10,2.1e-09,0,0);
   nuc->AddDecay(0,-1,0,3.90975,91.4);
   nuc->AddDecay(-4,-2,0,5.8726,8.6);

   // Adding 86-RN-207-0
   nuc = new Nucleus("RN",207,86,0,206.991,0,555,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.251,21);
   nuc->AddDecay(0,-1,0,4.61145,79);

   // Adding 87-FR-207-0
   nuc = new Nucleus("FR",207,87,0,206.997,0,14.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.70919,5);
   nuc->AddDecay(-4,-2,0,6.90121,95);

   // Adding 88-RA-207-0
   nuc = new Nucleus("RA",207,88,0,207.004,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.40099,10);
   nuc->AddDecay(-4,-2,0,7.27393,90);

   // Adding 88-RA-207-1
   nuc = new Nucleus("RA",207,88,1,207.004,0.47,0.055,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.47,85);
   nuc->AddDecay(-4,-2,-1,7.74393,15);
   nuc->AddDecay(0,-1,-1,6.87099,0.35);

   // Adding 81-TL-208-0
   nuc = new Nucleus("TL",208,81,0,207.982,0,183.18,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.001,100);

   // Adding 82-PB-208-0
   nuc = new Nucleus("PB",208,82,0,207.977,0,0,52.4,0,0,0,0);

   // Adding 83-BI-208-0
   nuc = new Nucleus("BI",208,83,0,207.98,0,1.16052e+13,0,1.4e-09,4e-09,0,0);
   nuc->AddDecay(0,-1,0,2.87969,100);

   // Adding 83-BI-208-1
   nuc = new Nucleus("BI",208,83,1,207.981,1.571,0.00258,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.571,100);

   // Adding 84-PO-208-0
   nuc = new Nucleus("PO",208,84,0,207.981,0,9.13913e+07,0,7.7e-07,2.4e-06,0,0);
   nuc->AddDecay(0,-1,0,1.4006,0.00223);
   nuc->AddDecay(-4,-2,0,5.2155,99.9978);

   // Adding 85-AT-208-0
   nuc = new Nucleus("AT",208,85,0,207.987,0,5868,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.97331,99.45);
   nuc->AddDecay(-4,-2,0,5.7511,0.55);

   // Adding 86-RN-208-0
   nuc = new Nucleus("RN",208,86,0,207.99,0,1461,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.85141,38);
   nuc->AddDecay(-4,-2,0,6.2605,62);

   // Adding 87-FR-208-0
   nuc = new Nucleus("FR",208,87,0,207.997,0,59.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.77112,90);
   nuc->AddDecay(0,-1,0,6.98736,10);

   // Adding 88-RA-208-0
   nuc = new Nucleus("RA",208,88,0,208.002,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.32659,5);
   nuc->AddDecay(-4,-2,0,7.27338,95);

   // Adding 81-TL-209-0
   nuc = new Nucleus("TL",209,81,0,208.985,0,132,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.98033,100);

   // Adding 82-PB-209-0
   nuc = new Nucleus("PB",209,82,0,208.981,0,11710.8,0,5.7e-11,3.2e-11,0,0);
   nuc->AddDecay(0,1,0,0.644098,100);

   // Adding 83-BI-209-0
   nuc = new Nucleus("BI",209,83,0,208.98,0,0,100,0,0,0,0);

   // Adding 84-PO-209-0
   nuc = new Nucleus("PO",209,84,0,208.982,0,3.21667e+09,0,7.7e-07,2.4e-06,1,0);
   nuc->AddDecay(-4,-2,0,4.9793,99.52);
   nuc->AddDecay(0,-1,0,1.8926,0.48);

   // Adding 85-AT-209-0
   nuc = new Nucleus("AT",209,85,0,208.986,0,19476,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.48635,95.9);
   nuc->AddDecay(-4,-2,0,5.7573,4.1);

   // Adding 86-RN-209-0
   nuc = new Nucleus("RN",209,86,0,208.99,0,1710,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.92905,83);
   nuc->AddDecay(-4,-2,0,6.1553,17);

   // Adding 87-FR-209-0
   nuc = new Nucleus("FR",209,87,0,208.996,0,50,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7774,89);
   nuc->AddDecay(0,-1,0,5.1615,11);

   // Adding 88-RA-209-0
   nuc = new Nucleus("RA",209,88,0,209.002,0,4.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.61415,10);
   nuc->AddDecay(-4,-2,0,7.14734,90);

   // Adding 89-AC-209-0
   nuc = new Nucleus("AC",209,89,0,209.01,0,0.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.10549,1);
   nuc->AddDecay(-4,-2,0,7.73333,99);

   // Adding 81-TL-210-0
   nuc = new Nucleus("TL",210,81,0,209.99,0,78,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.48402,99.993);
   nuc->AddDecay(-1,1,0,0.29882,0.007);

   // Adding 82-PB-210-0
   nuc = new Nucleus("PB",210,82,0,209.984,0,7.03253e+08,0,6.8e-07,1.1e-06,0,0);
   nuc->AddDecay(0,1,0,0.0635004,100);
   nuc->AddDecay(-4,-2,0,3.79229,1.9e-06);

   // Adding 83-BI-210-0
   nuc = new Nucleus("BI",210,83,0,209.984,0,433123,0,1.3e-09,8.4e-08,1,0);
   nuc->AddDecay(-4,-2,0,5.0369,0.00013);
   nuc->AddDecay(0,1,0,1.1627,100);

   // Adding 83-BI-210-1
   nuc = new Nucleus("BI",210,83,1,209.984,0.271,9.58694e+13,0,1.5e-08,3.1e-06,1,0);
   nuc->AddDecay(-4,-2,-1,5.3079,100);

   // Adding 84-PO-210-0
   nuc = new Nucleus("PO",210,84,0,209.983,0,1.19557e+07,0,2.4e-07,3e-06,1,0);
   nuc->AddDecay(-4,-2,0,5.4074,100);

   // Adding 85-AT-210-0
   nuc = new Nucleus("AT",210,85,0,209.987,0,29160,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.6314,0.18);
   nuc->AddDecay(0,-1,0,3.98145,99.82);

   // Adding 86-RN-210-0
   nuc = new Nucleus("RN",210,86,0,209.99,0,8640,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.15859,96);
   nuc->AddDecay(0,-1,0,2.37357,4);

   // Adding 87-FR-210-0
   nuc = new Nucleus("FR",210,87,0,209.996,0,190.8,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.70333,60);
   nuc->AddDecay(0,-1,0,6.26219,40);

   // Adding 88-RA-210-0
   nuc = new Nucleus("RA",210,88,0,210,0,3.7,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.15896,96);
   nuc->AddDecay(0,-1,0,3.76825,4);

   // Adding 89-AC-210-0
   nuc = new Nucleus("AC",210,89,0,210.009,0,0.35,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.60733,96);
   nuc->AddDecay(0,-1,0,8.20368,4);

   // Adding 82-PB-211-0
   nuc = new Nucleus("PB",211,82,0,210.989,0,2166,0,1.8e-10,5.6e-09,0,0);
   nuc->AddDecay(0,1,0,1.37263,100);

   // Adding 83-BI-211-0
   nuc = new Nucleus("BI",211,83,0,210.987,0,128.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7506,99.72);
   nuc->AddDecay(0,1,0,0.578873,0.28);

   // Adding 84-PO-211-0
   nuc = new Nucleus("PO",211,84,0,210.987,0,0.516,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.5945,100);

   // Adding 84-PO-211-1
   nuc = new Nucleus("PO",211,84,1,210.988,1.462,25.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,9.0565,99.98);
   nuc->AddDecay(0,0,-1,1.462,0.02);

   // Adding 85-AT-211-0
   nuc = new Nucleus("AT",211,85,0,210.987,0,25970.4,0,1.1e-08,1.1e-07,0,0);
   nuc->AddDecay(0,-1,0,0.786693,58.2);
   nuc->AddDecay(-4,-2,0,5.9824,41.8);

   // Adding 86-RN-211-0
   nuc = new Nucleus("RN",211,86,0,210.991,0,52560,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.89147,72.7);
   nuc->AddDecay(-4,-2,0,5.9653,27.4);

   // Adding 87-FR-211-0
   nuc = new Nucleus("FR",211,87,0,210.996,0,186,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.6604,80);
   nuc->AddDecay(0,-1,0,4.60485,20);

   // Adding 88-RA-211-0
   nuc = new Nucleus("RA",211,88,0,211.001,0,13,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.04593,93);
   nuc->AddDecay(0,-1,0,4.99698,7);

   // Adding 89-AC-211-0
   nuc = new Nucleus("AC",211,89,0,211.008,0,0.25,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.62497,100);

   // Adding 82-PB-212-0
   nuc = new Nucleus("PB",212,82,0,211.992,0,38304,0,5.9e-09,3.2e-07,0,0);
   nuc->AddDecay(0,1,0,0.573702,100);

   // Adding 83-BI-212-0
   nuc = new Nucleus("BI",212,83,0,211.991,0,3633,0,2.6e-10,4e-08,0,0);
   nuc->AddDecay(0,1,0,2.2539,64.046);
   nuc->AddDecay(-4,-1,0,11.2081,0.014);
   nuc->AddDecay(-4,-2,0,6.2071,35.94);

   // Adding 83-BI-212-1
   nuc = new Nucleus("BI",212,83,1,211.992,0.25,1500,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,6.4571,67);
   nuc->AddDecay(0,1,0,-0.4181,33);

   // Adding 83-BI-212-2
   nuc = new Nucleus("BI",212,83,2,211.993,1.91,420,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.158,100);

   // Adding 84-PO-212-0
   nuc = new Nucleus("PO",212,84,0,211.989,0,2.99e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.9542,100);

   // Adding 84-PO-212-1
   nuc = new Nucleus("PO",212,84,1,211.992,2.922,45.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,11.8762,99.93);
   nuc->AddDecay(0,0,-1,2.922,0.07);

   // Adding 84-PO-212-2
   nuc = new Nucleus("PO",212,84,2,212,0,45.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-2,8.95338,100);

   // Adding 85-AT-212-0
   nuc = new Nucleus("AT",212,85,0,211.991,0,0.314,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.0434008,2e-06);
   nuc->AddDecay(-4,-2,0,7.82899,99.97);
   nuc->AddDecay(0,-1,0,1.75449,0.03);

   // Adding 85-AT-212-1
   nuc = new Nucleus("AT",212,85,1,211.991,0.222,0.119,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.222,1);
   nuc->AddDecay(-4,-2,-1,8.05099,99);

   // Adding 86-RN-212-0
   nuc = new Nucleus("RN",212,86,0,211.991,0,1434,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.38499,100);

   // Adding 87-FR-212-0
   nuc = new Nucleus("FR",212,87,0,211.996,0,1200,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.5291,43);
   nuc->AddDecay(0,-1,0,5.11742,57);

   // Adding 88-RA-212-0
   nuc = new Nucleus("RA",212,88,0,212,0,13,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,3.35421,14.2857);
   nuc->AddDecay(-4,-2,0,7.0319,85.7143);

   // Adding 89-AC-212-0
   nuc = new Nucleus("AC",212,89,0,212.008,0,0.93,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,7.47678,3);
   nuc->AddDecay(-4,-2,0,7.52132,97);

   // Adding 90-TH-212-0
   nuc = new Nucleus("TH",212,90,0,212.013,0,0.03,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.9521,100);
   nuc->AddDecay(0,-1,0,4.75737,0.3);

   // Adding 82-PB-213-0
   nuc = new Nucleus("PB",213,82,0,212.997,0,612,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.06958,100);

   // Adding 83-BI-213-0
   nuc = new Nucleus("BI",213,83,0,212.994,0,2735.4,0,2e-10,4.2e-08,0,0);
   nuc->AddDecay(0,1,0,1.42626,97.91);
   nuc->AddDecay(-4,-2,0,5.98252,2.09);

   // Adding 84-PO-213-0
   nuc = new Nucleus("PO",213,84,0,212.993,0,4.2e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.53659,100);

   // Adding 85-AT-213-0
   nuc = new Nucleus("AT",213,85,0,212.993,0,1.25e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.25407,100);

   // Adding 86-RN-213-0
   nuc = new Nucleus("RN",213,86,0,212.994,0,0.025,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.24306,100);

   // Adding 87-FR-213-0
   nuc = new Nucleus("FR",213,87,0,212.996,0,34.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.9051,99.45);
   nuc->AddDecay(0,-1,0,2.14839,0.55);

   // Adding 88-RA-213-0
   nuc = new Nucleus("RA",213,88,0,213,0,164.4,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.88351,20);
   nuc->AddDecay(-4,-2,0,6.85956,80);

   // Adding 88-RA-213-1
   nuc = new Nucleus("RA",213,88,1,213.002,1.77,0.0021,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1.77,99);
   nuc->AddDecay(-4,-2,-1,8.62956,1);

   // Adding 89-AC-213-0
   nuc = new Nucleus("AC",213,89,0,213.007,0,0.8,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.50328,100);

   // Adding 90-TH-213-0
   nuc = new Nucleus("TH",213,90,0,213.013,0,0.14,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.83811,100);

   // Adding 82-PB-214-0
   nuc = new Nucleus("PB",214,82,0,214,0,1608,0,1.4e-10,4.9e-09,0,0);
   nuc->AddDecay(0,1,0,1.02319,100);

   // Adding 83-BI-214-0
   nuc = new Nucleus("BI",214,83,0,213.999,0,1194,0,1.1e-10,2.1e-08,0,0);
   nuc->AddDecay(0,1,0,3.27172,99.98);
   nuc->AddDecay(-4,-2,0,5.6212,0.02);

   // Adding 84-PO-214-0
   nuc = new Nucleus("PO",214,84,0,213.995,0,0.0001643,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.8335,100);

   // Adding 85-AT-214-0
   nuc = new Nucleus("AT",214,85,0,213.996,0,5.58e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.98728,100);

   // Adding 85-AT-214-1
   nuc = new Nucleus("AT",214,85,1,213.996,0.059,2.65e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,9.04628,100);

   // Adding 85-AT-214-2
   nuc = new Nucleus("AT",214,85,2,213.997,0.232,7.6e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-2,9.21928,100);

   // Adding 86-RN-214-0
   nuc = new Nucleus("RN",214,86,0,213.995,0,2.7e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.20843,100);

   // Adding 86-RN-214-1
   nuc = new Nucleus("RN",214,86,1,213.997,1.443,7e-10,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-1,10.6514,100);

   // Adding 86-RN-214-2
   nuc = new Nucleus("RN",214,86,2,213.997,1.625,6.5e-09,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,-2,10.8334,100);

   // Adding 87-FR-214-0
   nuc = new Nucleus("FR",214,87,0,213.999,0,0.005,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.58749,100);

   // Adding 87-FR-214-1
   nuc = new Nucleus("FR",214,87,1,213.999,0.122,0.00335,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,8.70949,100);

   // Adding 88-RA-214-0
   nuc = new Nucleus("RA",214,88,0,214,0,2.46,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.27312,99.94);
   nuc->AddDecay(0,-1,0,1.0592,0.06);

   // Adding 89-AC-214-0
   nuc = new Nucleus("AC",214,89,0,214.007,0,8.2,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,6.34004,11);
   nuc->AddDecay(-4,-2,0,7.35097,89);

   // Adding 90-TH-214-0
   nuc = new Nucleus("TH",214,90,0,214.011,0,0.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.82511,100);

   // Adding 83-BI-215-0
   nuc = new Nucleus("BI",215,83,0,215.002,0,456,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.25222,100);

   // Adding 84-PO-215-0
   nuc = new Nucleus("PO",215,84,0,214.999,0,0.001781,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.5264,100);
   nuc->AddDecay(0,1,0,0.720743,0.00023);

   // Adding 85-AT-215-0
   nuc = new Nucleus("AT",215,85,0,214.999,0,0.0001,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.17829,100);

   // Adding 86-RN-215-0
   nuc = new Nucleus("RN",215,86,0,214.999,0,2.3e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.83895,100);

   // Adding 87-FR-215-0
   nuc = new Nucleus("FR",215,87,0,215,0,8.6e-08,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.54041,100);

   // Adding 88-RA-215-0
   nuc = new Nucleus("RA",215,88,0,215.003,0,0.00159,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.86385,100);

   // Adding 89-AC-215-0
   nuc = new Nucleus("AC",215,89,0,215.006,0,0.17,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.74847,99.91);
   nuc->AddDecay(0,-1,0,3.48947,0.09);

   // Adding 90-TH-215-0
   nuc = new Nucleus("TH",215,90,0,215.012,0,1.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.6656,100);

   // Adding 91-PA-215-0
   nuc = new Nucleus("PA",215,91,0,215.019,0,0.014,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.16722,100);

   // Adding 83-BI-216-0
   nuc = new Nucleus("BI",216,83,0,216.006,0,216,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,3.99977,100);

   // Adding 84-PO-216-0
   nuc = new Nucleus("PO",216,84,0,216.002,0,0.145,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.90656,100);

   // Adding 85-AT-216-0
   nuc = new Nucleus("AT",216,85,0,216.002,0,0.0003,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.94947,100);
   nuc->AddDecay(0,-1,0,0.469214,0.006);
   nuc->AddDecay(0,1,0,2.00326,3e-09);

   // Adding 86-RN-216-0
   nuc = new Nucleus("RN",216,86,0,216,0,4.5e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.20011,100);

   // Adding 87-FR-216-0
   nuc = new Nucleus("FR",216,87,0,216.003,0,7e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.17515,100);
   nuc->AddDecay(0,-1,0,2.72953,2e-09);

   // Adding 88-RA-216-0
   nuc = new Nucleus("RA",216,88,0,216.004,0,1.82e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.52583,100);
   nuc->AddDecay(0,-1,0,0.307279,0);

   // Adding 89-AC-216-0
   nuc = new Nucleus("AC",216,89,0,216.009,0,0.00033,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.24355,100);

   // Adding 89-AC-216-1
   nuc = new Nucleus("AC",216,89,1,216.009,0,0.00033,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,9.24355,100);

   // Adding 90-TH-216-0
   nuc = new Nucleus("TH",216,90,0,216.011,0,0.028,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.07063,100);
   nuc->AddDecay(0,-1,0,2.18129,0.006);

   // Adding 90-TH-216-1
   nuc = new Nucleus("TH",216,90,1,216.013,2.028,0.00018,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,2.028,97);
   nuc->AddDecay(-4,-2,-1,10.0986,3);

   // Adding 91-PA-216-0
   nuc = new Nucleus("PA",216,91,0,216.019,0,0.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.01372,80);
   nuc->AddDecay(0,-1,0,7.41987,20);

   // Adding 84-PO-217-0
   nuc = new Nucleus("PO",217,84,0,217.006,0,10,0,0,0,1,-4);
   nuc->AddDecay(-4,-2,0,6.6603,95);
   nuc->AddDecay(0,1,0,1.52782,5);

   // Adding 85-AT-217-0
   nuc = new Nucleus("AT",217,85,0,217.005,0,0.0323,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.20205,99.99);
   nuc->AddDecay(0,1,0,0.739626,0.01);

   // Adding 86-RN-217-0
   nuc = new Nucleus("RN",217,86,0,217.004,0,0.00054,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.88869,100);

   // Adding 87-FR-217-0
   nuc = new Nucleus("FR",217,87,0,217.005,0,2.2e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.46933,100);

   // Adding 88-RA-217-0
   nuc = new Nucleus("RA",217,88,0,217.006,0,1.6e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.16087,100);

   // Adding 89-AC-217-0
   nuc = new Nucleus("AC",217,89,0,217.009,0,6.9e-08,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.83181,98);
   nuc->AddDecay(0,-1,0,2.81933,2);

   // Adding 90-TH-217-0
   nuc = new Nucleus("TH",217,90,0,217.013,0,0.000252,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.424,100);

   // Adding 91-PA-217-0
   nuc = new Nucleus("PA",217,91,0,217.018,0,0.0049,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.48772,100);

   // Adding 91-PA-217-1
   nuc = new Nucleus("PA",217,91,1,217.02,1.854,0.0016,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,10.3417,100);

   // Adding 84-PO-218-0
   nuc = new Nucleus("PO",218,84,0,218.009,0,186,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.11475,99.98);
   nuc->AddDecay(0,1,0,0.263914,0.02);

   // Adding 85-AT-218-0
   nuc = new Nucleus("AT",218,85,0,218.009,0,1.6,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.88308,0.1);
   nuc->AddDecay(-4,-2,0,6.87402,99.9);

   // Adding 86-RN-218-0
   nuc = new Nucleus("RN",218,86,0,218.006,0,0.035,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.26267,100);

   // Adding 87-FR-218-0
   nuc = new Nucleus("FR",218,87,0,218.008,0,0.001,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.01441,100);

   // Adding 87-FR-218-1
   nuc = new Nucleus("FR",218,87,1,218.008,0.086,0.022,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,8.10041,100);

   // Adding 88-RA-218-0
   nuc = new Nucleus("RA",218,88,0,218.007,0,2.56e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.54601,100);

   // Adding 89-AC-218-0
   nuc = new Nucleus("AC",218,89,0,218.012,0,1.12e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.37754,100);

   // Adding 90-TH-218-0
   nuc = new Nucleus("TH",218,90,0,218.013,0,1.09e-07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.8491,100);

   // Adding 91-PA-218-0
   nuc = new Nucleus("PA",218,91,0,218.02,0,0.00012,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.79412,100);

   // Adding 92-U-218-0
   nuc = new Nucleus("U",218,92,0,218.023,0,0.0048,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.78816,100);

   // Adding 85-AT-219-0
   nuc = new Nucleus("AT",219,85,0,219.011,0,56,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.39089,97);
   nuc->AddDecay(0,1,0,1.69695,3);

   // Adding 86-RN-219-0
   nuc = new Nucleus("RN",219,86,0,219.009,0,3.96,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.94616,100);

   // Adding 87-FR-219-0
   nuc = new Nucleus("FR",219,87,0,219.009,0,0.02,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.44874,100);

   // Adding 88-RA-219-0
   nuc = new Nucleus("RA",219,88,0,219.01,0,0.01,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.13049,100);

   // Adding 89-AC-219-0
   nuc = new Nucleus("AC",219,89,0,219.012,0,1.18e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.82657,100);

   // Adding 90-TH-219-0
   nuc = new Nucleus("TH",219,90,0,219.016,0,1.05e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.51407,100);

   // Adding 91-PA-219-0
   nuc = new Nucleus("PA",219,91,0,219.02,0,5.3e-08,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,10.0846,100);

   // Adding 92-U-219-0
   nuc = new Nucleus("U",219,92,0,219.025,0,5.3e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.90253,100);

   // Adding 85-AT-220-0
   nuc = new Nucleus("AT",220,85,0,220.015,0,222.6,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.0531,100);

   // Adding 86-RN-220-0
   nuc = new Nucleus("RN",220,86,0,220.011,0,55.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4047,100);

   // Adding 87-FR-220-0
   nuc = new Nucleus("FR",220,87,0,220.012,0,27.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.8007,99.65);
   nuc->AddDecay(0,1,0,1.20924,0.35);

   // Adding 88-RA-220-0
   nuc = new Nucleus("RA",220,88,0,220.011,0,0.025,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.59473,100);

   // Adding 89-AC-220-0
   nuc = new Nucleus("AC",220,89,0,220.015,0,0.0261,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.48204,5e-06);
   nuc->AddDecay(-4,-2,0,8.34724,100);

   // Adding 90-TH-220-0
   nuc = new Nucleus("TH",220,90,0,220.016,0,9.7e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.95293,100);
   nuc->AddDecay(0,-1,0,0.912971,2e-09);

   // Adding 91-PA-220-0
   nuc = new Nucleus("PA",220,91,0,220.022,0,7.8e-07,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,9.82942,100);

   // Adding 85-AT-221-0
   nuc = new Nucleus("AT",221,85,0,221.018,0,138,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.1864,100);

   // Adding 86-RN-221-0
   nuc = new Nucleus("RN",221,86,0,221.016,0,1500,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.1471,22);
   nuc->AddDecay(0,1,0,1.21683,78);

   // Adding 87-FR-221-0
   nuc = new Nucleus("FR",221,87,0,221.014,0,294,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.4581,99.9);
   nuc->AddDecay(0,1,0,0.311904,0.1);

   // Adding 88-RA-221-0
   nuc = new Nucleus("RA",221,88,0,221.014,0,28,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.88582,100);

   // Adding 89-AC-221-0
   nuc = new Nucleus("AC",221,89,0,221.016,0,0.052,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.78288,100);

   // Adding 90-TH-221-0
   nuc = new Nucleus("TH",221,90,0,221.018,0,0.00168,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.62761,100);

   // Adding 91-PA-221-0
   nuc = new Nucleus("PA",221,91,0,221.022,0,5.9e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.24762,100);

   // Adding 85-AT-222-0
   nuc = new Nucleus("AT",222,85,0,222.022,0,54,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.43407,100);

   // Adding 86-RN-222-0
   nuc = new Nucleus("RN",222,86,0,222.018,0,330350,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.5904,100);

   // Adding 87-FR-222-0
   nuc = new Nucleus("FR",222,87,0,222.018,0,852,0,7.1e-10,2.1e-08,0,0);
   nuc->AddDecay(0,1,0,2.03132,100);

   // Adding 88-RA-222-0
   nuc = new Nucleus("RA",222,88,0,222.015,0,38,0,0,0,0,0);
   nuc->AddDecay(-14,-6,0,33.0534,3e-08);
   nuc->AddDecay(-4,-2,0,6.68096,100);

   // Adding 89-AC-222-0
   nuc = new Nucleus("AC",222,89,0,222.018,0,5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.12857,99);
   nuc->AddDecay(0,-1,0,2.28962,1);

   // Adding 89-AC-222-1
   nuc = new Nucleus("AC",222,89,1,222.018,0,63,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,7.12857,88);
   nuc->AddDecay(0,0,-1,0,10);
   nuc->AddDecay(0,-1,-1,2.28962,2);

   // Adding 90-TH-222-0
   nuc = new Nucleus("TH",222,90,0,222.018,0,0.0028,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.1291,100);

   // Adding 91-PA-222-0
   nuc = new Nucleus("PA",222,91,0,222.024,0,0.0043,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.79835,100);

   // Adding 92-U-222-0
   nuc = new Nucleus("U",222,92,0,222.026,0,1e-06,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.49969,100);

   // Adding 85-AT-223-0
   nuc = new Nucleus("AT",223,85,0,223.025,0,50,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.2123,100);

   // Adding 86-RN-223-0
   nuc = new Nucleus("RN",223,86,0,223.022,0,1392,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.80407,100);

   // Adding 87-FR-223-0
   nuc = new Nucleus("FR",223,87,0,223.02,0,1308,0,2.3e-09,1.3e-09,0,0);
   nuc->AddDecay(0,1,0,1.1491,99.99);
   nuc->AddDecay(-4,-2,0,5.43145,0.006);

   // Adding 88-RA-223-0
   nuc = new Nucleus("RA",223,88,0,223.018,0,987984,0,1e-07,6.9e-06,1,0);
   nuc->AddDecay(-4,-2,0,5.9793,100);
   nuc->AddDecay(-14,-6,0,31.8387,6.4e-08);

   // Adding 89-AC-223-0
   nuc = new Nucleus("AC",223,89,0,223.019,0,126,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.7833,99);
   nuc->AddDecay(0,-1,0,0.585846,1);

   // Adding 90-TH-223-0
   nuc = new Nucleus("TH",223,90,0,223.021,0,0.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.5667,100);

   // Adding 91-PA-223-0
   nuc = new Nucleus("PA",223,91,0,223.024,0,0.0065,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.34252,100);

   // Adding 92-U-223-0
   nuc = new Nucleus("U",223,92,0,223.028,0,1.8e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.93715,100);

   // Adding 86-RN-224-0
   nuc = new Nucleus("RN",224,86,0,224.024,0,6420,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.824253,100);

   // Adding 87-FR-224-0
   nuc = new Nucleus("FR",224,87,0,224.023,0,198,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.81902,100);

   // Adding 88-RA-224-0
   nuc = new Nucleus("RA",224,88,0,224.02,0,316224,0,6.5e-08,2.9e-06,1,0);
   nuc->AddDecay(-4,-2,0,5.7889,100);
   nuc->AddDecay(-12,-6,0,26.3749,4.3e-09);

   // Adding 89-AC-224-0
   nuc = new Nucleus("AC",224,89,0,224.022,0,10440,0,7e-10,1.2e-07,1,0);
   nuc->AddDecay(-4,-2,0,6.3269,8.9544);
   nuc->AddDecay(0,1,0,0.232033,1.6);
   nuc->AddDecay(0,-1,0,1.40322,89.4456);

   // Adding 90-TH-224-0
   nuc = new Nucleus("TH",224,90,0,224.021,0,1.05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.3041,100);

   // Adding 91-PA-224-0
   nuc = new Nucleus("PA",224,91,0,224.026,0,0.95,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.6948,99.9);
   nuc->AddDecay(0,-1,0,3.87274,0.1);

   // Adding 92-U-224-0
   nuc = new Nucleus("U",224,92,0,224.028,0,0.0007,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.62018,100);

   // Adding 86-RN-225-0
   nuc = new Nucleus("RN",225,86,0,225.028,0,270,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.68529,100);

   // Adding 87-FR-225-0
   nuc = new Nucleus("FR",225,87,0,225.026,0,240,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.86558,100);

   // Adding 88-RA-225-0
   nuc = new Nucleus("RA",225,88,0,225.024,0,1.28736e+06,0,9.6e-08,5.8e-06,0,0);
   nuc->AddDecay(0,1,0,0.356556,100);

   // Adding 89-AC-225-0
   nuc = new Nucleus("AC",225,89,0,225.023,0,864000,0,2.4e-08,7.9e-06,1,0);
   nuc->AddDecay(-4,-2,0,5.9352,100);

   // Adding 90-TH-225-0
   nuc = new Nucleus("TH",225,90,0,225.024,0,523.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.9216,90);
   nuc->AddDecay(0,-1,0,0.674498,10);

   // Adding 91-PA-225-0
   nuc = new Nucleus("PA",225,91,0,225.026,0,1.7,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.39252,100);

   // Adding 92-U-225-0
   nuc = new Nucleus("U",225,92,0,225.029,0,0.05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.01984,100);

   // Adding 86-RN-226-0
   nuc = new Nucleus("RN",226,86,0,226.031,0,360,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.72552,100);

   // Adding 87-FR-226-0
   nuc = new Nucleus("FR",226,87,0,226.029,0,48,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.63366,100);

   // Adding 88-RA-226-0
   nuc = new Nucleus("RA",226,88,0,226.025,0,5.04576e+10,0,2.8e-07,1.6e-05,1,0);
   nuc->AddDecay(-4,-2,0,4.8706,100);
   nuc->AddDecay(-14,-6,0,28.1987,3e-11);

   // Adding 89-AC-226-0
   nuc = new Nucleus("AC",226,89,0,226.026,0,105840,0,1e-08,1.2e-06,0,0);
   nuc->AddDecay(0,1,0,1.11638,83);
   nuc->AddDecay(-4,-2,0,5.53611,0.006);
   nuc->AddDecay(0,-1,0,0.640409,17);

   // Adding 90-TH-226-0
   nuc = new Nucleus("TH",226,90,0,226.025,0,1836,0,4.5e-10,7.7e-08,1,0);
   nuc->AddDecay(-4,-2,0,6.45105,100);

   // Adding 91-PA-226-0
   nuc = new Nucleus("PA",226,91,0,226.028,0,108,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.82537,26);
   nuc->AddDecay(-4,-2,0,6.9868,74);

   // Adding 92-U-226-0
   nuc = new Nucleus("U",226,92,0,226.029,0,0.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.70661,100);

   // Adding 86-RN-227-0
   nuc = new Nucleus("RN",227,86,0,227.036,0,22.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.50261,100);

   // Adding 87-FR-227-0
   nuc = new Nucleus("FR",227,87,0,227.032,0,148.2,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.48626,100);

   // Adding 88-RA-227-0
   nuc = new Nucleus("RA",227,88,0,227.029,0,2532,0,8.4e-11,2.8e-10,0,0);
   nuc->AddDecay(0,1,0,1.3251,100);

   // Adding 89-AC-227-0
   nuc = new Nucleus("AC",227,89,0,227.028,0,6.86633e+08,0,1.1e-06,0.00063,0,0);
   nuc->AddDecay(0,1,0,0.0447979,98.62);
   nuc->AddDecay(-4,-2,0,5.0422,1.38);

   // Adding 90-TH-227-0
   nuc = new Nucleus("TH",227,90,0,227.028,0,1.61741e+06,0,8.9e-09,9.6e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.1465,100);

   // Adding 91-PA-227-0
   nuc = new Nucleus("PA",227,91,0,227.029,0,2298,0,4.6e-10,9.7e-08,1,0);
   nuc->AddDecay(-4,-2,0,6.58,85);
   nuc->AddDecay(0,-1,0,1.01935,15);

   // Adding 92-U-227-0
   nuc = new Nucleus("U",227,92,0,227.031,0,66,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.21131,100);

   // Adding 93-NP-227-0
   nuc = new Nucleus("NP",227,93,0,227.035,0,0.51,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.8164,100);

   // Adding 86-RN-228-0
   nuc = new Nucleus("RN",228,86,0,228,0,65,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 87-FR-228-0
   nuc = new Nucleus("FR",228,87,0,228.036,0,39,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.33792,100);

   // Adding 88-RA-228-0
   nuc = new Nucleus("RA",228,88,0,228.031,0,1.81332e+08,0,6.7e-07,2.6e-06,0,0);
   nuc->AddDecay(0,1,0,0.0458984,100);

   // Adding 89-AC-228-0
   nuc = new Nucleus("AC",228,89,0,228.031,0,22140,0,4.3e-10,2.9e-08,0,0);
   nuc->AddDecay(0,1,0,2.1266,100);
   nuc->AddDecay(-4,-2,0,4.82768,5.5e-06);

   // Adding 90-TH-228-0
   nuc = new Nucleus("TH",228,90,0,228.029,0,6.03315e+07,0,7e-08,3.9e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.5201,100);

   // Adding 91-PA-228-0
   nuc = new Nucleus("PA",228,91,0,228.031,0,79200,0,7.8e-10,6.9e-08,0,0);
   nuc->AddDecay(0,-1,0,2.11143,98.15);
   nuc->AddDecay(-4,-2,0,6.22831,1.85);

   // Adding 92-U-228-0
   nuc = new Nucleus("U",228,92,0,228.031,0,546,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.8035,95);
   nuc->AddDecay(0,-1,0,0.343163,5);

   // Adding 93-NP-228-0
   nuc = new Nucleus("NP",228,93,0,228.036,0,60,0,0,0,0,-8);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 87-FR-229-0
   nuc = new Nucleus("FR",229,87,0,229.039,0,50,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.72564,100);

   // Adding 88-RA-229-0
   nuc = new Nucleus("RA",229,88,0,229.035,0,240,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.76001,100);

   // Adding 89-AC-229-0
   nuc = new Nucleus("AC",229,89,0,229.033,0,3762,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.09502,100);

   // Adding 90-TH-229-0
   nuc = new Nucleus("TH",229,90,0,229.032,0,2.31474e+11,0,4.8e-07,9.9e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.1679,100);

   // Adding 91-PA-229-0
   nuc = new Nucleus("PA",229,91,0,229.032,0,129600,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.316158,99.52);
   nuc->AddDecay(-4,-2,0,5.84061,0.48);

   // Adding 92-U-229-0
   nuc = new Nucleus("U",229,92,0,229.033,0,3480,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.30909,80);
   nuc->AddDecay(-4,-2,0,6.47521,20);

   // Adding 93-NP-229-0
   nuc = new Nucleus("NP",229,93,0,229.036,0,240,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.01272,50);
   nuc->AddDecay(0,-1,0,2.55951,50);

   // Adding 87-FR-230-0
   nuc = new Nucleus("FR",230,87,0,230.043,0,19.1,0,0,0,0,0);
   nuc->AddDecay(0,1,0,5.46787,100);

   // Adding 88-RA-230-0
   nuc = new Nucleus("RA",230,88,0,230.037,0,5580,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.987022,100);

   // Adding 89-AC-230-0
   nuc = new Nucleus("AC",230,89,0,230.036,0,122,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.69999,100);

   // Adding 90-TH-230-0
   nuc = new Nucleus("TH",230,90,0,230.033,0,2.37718e+12,0,2.1e-07,4e-05,1,0);
   nuc->AddDecay(-4,-2,0,4.77,100);
   nuc->AddDecay(0,1000,0,207,5e-11);

   // Adding 91-PA-230-0
   nuc = new Nucleus("PA",230,91,0,230.035,0,1.50336e+06,0,9.2e-10,7.1e-07,0,0);
   nuc->AddDecay(0,-1,0,1.30981,91.6);
   nuc->AddDecay(0,1,0,0.562984,8.4);
   nuc->AddDecay(-4,-2,0,5.4394,0.0032);

   // Adding 92-U-230-0
   nuc = new Nucleus("U",230,92,0,230.034,0,1.79712e+06,0,5.5e-08,1.5e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.9928,100);

   // Adding 93-NP-230-0
   nuc = new Nucleus("NP",230,93,0,230.038,0,276,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.61071,97);
   nuc->AddDecay(-4,-2,0,6.77813,3);

   // Adding 94-PU-230-0
   nuc = new Nucleus("PU",230,94,0,230.04,0,0.2,0,0,0,1,-5);
   nuc->AddDecay(-4,-2,0,7.1749,100);

   // Adding 87-FR-231-0
   nuc = new Nucleus("FR",231,87,0,231.046,0,17.5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.463,100);

   // Adding 88-RA-231-0
   nuc = new Nucleus("RA",231,88,0,231.041,0,103,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.65189,100);

   // Adding 89-AC-231-0
   nuc = new Nucleus("AC",231,89,0,231.039,0,450,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.09999,100);

   // Adding 90-TH-231-0
   nuc = new Nucleus("TH",231,90,0,231.036,0,91872,0,3.4e-10,4e-10,0,0);
   nuc->AddDecay(0,1,0,0.389492,100);
   nuc->AddDecay(-4,-2,0,4.21329,1e-08);

   // Adding 91-PA-231-0
   nuc = new Nucleus("PA",231,91,0,231.036,0,1.03312e+12,0,7.1e-07,0.00013,1,0);
   nuc->AddDecay(-4,-2,0,5.1489,97);
   nuc->AddDecay(0,1000,0,207,3);

   // Adding 92-U-231-0
   nuc = new Nucleus("U",231,92,0,231.036,0,362880,0,2.8e-10,4e-10,0,0);
   nuc->AddDecay(0,-1,0,0.357624,100);
   nuc->AddDecay(-4,-2,0,5.55132,0.0055);

   // Adding 93-NP-231-0
   nuc = new Nucleus("NP",231,93,0,231.038,0,2928,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.36837,2);
   nuc->AddDecay(0,-1,0,1.8364,98);

   // Adding 94-PU-231-0
   nuc = new Nucleus("PU",231,94,0,231.041,0,0,0,0,0,0,-2);

   // Adding 87-FR-232-0
   nuc = new Nucleus("FR",232,87,0,232.049,0,5,0,0,0,0,0);
   nuc->AddDecay(0,1,0,4.2085,100);

   // Adding 88-RA-232-0
   nuc = new Nucleus("RA",232,88,0,232.044,0,250,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.94859,100);

   // Adding 89-AC-232-0
   nuc = new Nucleus("AC",232,89,0,232.042,0,119,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.69999,100);

   // Adding 90-TH-232-0
   nuc = new Nucleus("TH",232,90,0,232.038,0,4.43081e+17,100,2.3e-07,4.2e-05,1,0);
   nuc->AddDecay(-4,-2,0,4.0828,100);
   nuc->AddDecay(0,1000,0,207,1e-09);

   // Adding 91-PA-232-0
   nuc = new Nucleus("PA",232,91,0,232.039,0,113184,0,7.2e-10,9.6e-09,0,0);
   nuc->AddDecay(0,1,0,1.33715,100);
   nuc->AddDecay(0,-1,0,0.495457,0.003);

   // Adding 92-U-232-0
   nuc = new Nucleus("U",232,92,0,232.037,0,2.17283e+09,0,2.9e-07,3.5e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.4136,100);

   // Adding 93-NP-232-0
   nuc = new Nucleus("NP",232,93,0,232.04,0,882,0,9.7e-12,4.7e-11,0,0);
   nuc->AddDecay(0,-1,0,2.69787,100);

   // Adding 94-PU-232-0
   nuc = new Nucleus("PU",232,94,0,232.041,0,2046,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.05912,80);
   nuc->AddDecay(-4,-2,0,6.716,20);

   // Adding 95-AM-232-0
   nuc = new Nucleus("AM",232,95,0,232.047,0,79,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.07848,98);
   nuc->AddDecay(-4,-2,0,7.3106,2);

   // Adding 88-RA-233-0
   nuc = new Nucleus("RA",233,88,0,233.048,0,30,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.0968,100);

   // Adding 89-AC-233-0
   nuc = new Nucleus("AC",233,89,0,233.045,0,145,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.87528,100);

   // Adding 90-TH-233-0
   nuc = new Nucleus("TH",233,90,0,233.042,0,1338,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.2453,100);

   // Adding 91-PA-233-0
   nuc = new Nucleus("PA",233,91,0,233.04,0,2.32995e+06,0,8.8e-10,3.7e-09,0,0);
   nuc->AddDecay(0,1,0,0.570496,100);

   // Adding 92-U-233-0
   nuc = new Nucleus("U",233,92,0,233.04,0,5.02053e+12,0,5e-08,8.7e-06,1,0);
   nuc->AddDecay(-4,-2,0,4.9085,100);
   nuc->AddDecay(0,1000,0,207,6e-09);

   // Adding 93-NP-233-0
   nuc = new Nucleus("NP",233,93,0,233.041,0,2172,0,2.2e-12,3e-12,0,0);
   nuc->AddDecay(0,-1,0,1.23403,100);
   nuc->AddDecay(-4,-2,0,5.82637,0.001);

   // Adding 94-PU-233-0
   nuc = new Nucleus("PU",233,94,0,233.043,0,1254,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.41627,0.12);
   nuc->AddDecay(0,-1,0,1.89899,99.88);

   // Adding 95-AM-233-0
   nuc = new Nucleus("AM",233,95,0,233.046,0,0,0,0,0,0,-2);

   // Adding 88-RA-234-0
   nuc = new Nucleus("RA",234,88,0,234.051,0,30,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 89-AC-234-0
   nuc = new Nucleus("AC",234,89,0,234,0,44,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 90-TH-234-0
   nuc = new Nucleus("TH",234,90,0,234.044,0,2.08224e+06,0,3.4e-09,7.3e-09,0,0);
   nuc->AddDecay(0,1,1,0.199089,100);

   // Adding 91-PA-234-0
   nuc = new Nucleus("PA",234,91,0,234.043,0,24120,0,5.1e-10,5.8e-10,0,0);
   nuc->AddDecay(0,1,0,2.19703,100);

   // Adding 91-PA-234-1
   nuc = new Nucleus("PA",234,91,1,234.043,0.074,70.2,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.27103,99.87);
   nuc->AddDecay(0,0,-1,0.074,0.13);

   // Adding 92-U-234-0
   nuc = new Nucleus("U",234,92,0,234.041,0,7.72632e+12,0.0055,4.9e-08,8.5e-06,0,0);
   nuc->AddDecay(0,1000,0,207,1.73e-09);
   nuc->AddDecay(-4,-2,0,4.8585,100);

   // Adding 93-NP-234-0
   nuc = new Nucleus("NP",234,93,0,234.043,0,380160,0,8.1e-10,7.3e-10,0,0);
   nuc->AddDecay(0,-1,0,1.80996,100);

   // Adding 94-PU-234-0
   nuc = new Nucleus("PU",234,94,0,234.043,0,31680,0,1.6e-10,2.2e-08,0,0);
   nuc->AddDecay(0,-1,0,0.388287,94);
   nuc->AddDecay(-4,-2,0,6.30992,6);

   // Adding 95-AM-234-0
   nuc = new Nucleus("AM",234,95,0,234.048,0,156,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.87245,0.039);
   nuc->AddDecay(0,-1,0,4.17324,99.961);

   // Adding 90-TH-235-0
   nuc = new Nucleus("TH",235,90,0,235.048,0,426,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.92825,100);

   // Adding 91-PA-235-0
   nuc = new Nucleus("PA",235,91,0,235.045,0,1470,0,0,0,0,0);
   nuc->AddDecay(0,1,1,1.41003,100);

   // Adding 92-U-235-0
   nuc = new Nucleus("U",235,92,0,235.044,0,2.2195e+16,0.72,4.6e-08,7.7e-06,1,0);
   nuc->AddDecay(-4,-2,0,4.6787,100);
   nuc->AddDecay(0,1000,0,207,7e-09);

   // Adding 92-U-235-1
   nuc = new Nucleus("U",235,92,1,235.044,0,1500,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0,100);

   // Adding 93-NP-235-0
   nuc = new Nucleus("NP",235,93,0,235.044,0,3.4223e+07,0,5.3e-11,4e-10,0,0);
   nuc->AddDecay(0,-1,0,0.123703,100);
   nuc->AddDecay(-4,-2,0,5.19189,0.0026);

   // Adding 94-PU-235-0
   nuc = new Nucleus("PU",235,94,0,235.045,0,1518,0,2.1e-12,2.6e-12,0,0);
   nuc->AddDecay(0,-1,0,1.16782,100);
   nuc->AddDecay(-4,-2,0,6.00209,0.0027);

   // Adding 95-AM-235-0
   nuc = new Nucleus("AM",235,95,0,235.048,0,0,0,0,0,0,-2);

   // Adding 96-CM-235-0
   nuc = new Nucleus("CM",235,96,0,235.052,0,0,0,0,0,0,-2);

   // Adding 90-TH-236-0
   nuc = new Nucleus("TH",236,90,0,236.05,0,2250,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.36598,100);

   // Adding 91-PA-236-0
   nuc = new Nucleus("PA",236,91,0,236.049,0,546,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.9,100);

   // Adding 92-U-236-0
   nuc = new Nucleus("U",236,92,0,236.046,0,7.38573e+14,0,4.6e-08,7.9e-06,0,0);
   nuc->AddDecay(0,1000,0,207,9.6e-08);
   nuc->AddDecay(-4,-2,0,4.572,100);

   // Adding 93-NP-236-0
   nuc = new Nucleus("NP",236,93,0,236.047,0,4.85654e+12,0,1.7e-08,3e-06,0,0);
   nuc->AddDecay(0,-1,0,0.940033,87.3);
   nuc->AddDecay(0,1,0,0.486626,12.5);
   nuc->AddDecay(-4,-2,0,5.01657,0.16);

   // Adding 93-NP-236-1
   nuc = new Nucleus("NP",236,93,1,236.047,0.06,81000,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,1.00003,52);
   nuc->AddDecay(0,1,-1,0.546627,48);

   // Adding 94-PU-236-0
   nuc = new Nucleus("PU",236,94,0,236.046,0,9.01299e+07,0,8.6e-08,1.8e-05,0,0);
   nuc->AddDecay(0,1000,0,207,1.36e-07);
   nuc->AddDecay(-4,-2,0,5.8671,100);

   // Adding 95-AM-236-0
   nuc = new Nucleus("AM",236,95,0,236.05,0,532,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,6.4491,50);
   nuc->AddDecay(0,-1,0,3.27987,50);

   // Adding 96-CM-236-0
   nuc = new Nucleus("CM",236,96,0,236.051,0,196,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,1.7101,50);
   nuc->AddDecay(-4,-2,0,7.10008,50);

   // Adding 90-TH-237-0
   nuc = new Nucleus("TH",237,90,0,237.054,0,300,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.56479,100);

   // Adding 91-PA-237-0
   nuc = new Nucleus("PA",237,91,0,237.051,0,522,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.24999,100);

   // Adding 92-U-237-0
   nuc = new Nucleus("U",237,92,0,237.049,0,583200,0,7.7e-10,1.8e-09,0,0);
   nuc->AddDecay(0,1,0,0.518501,100);

   // Adding 93-NP-237-0
   nuc = new Nucleus("NP",237,93,0,237.048,0,6.7487e+13,0,1.1e-07,2.1e-05,1,0);
   nuc->AddDecay(-4,-2,0,4.95919,100);
   nuc->AddDecay(0,1000,0,207,2e-12);

   // Adding 94-PU-237-0
   nuc = new Nucleus("PU",237,94,0,237.048,0,3.90528e+06,0,1e-10,3.6e-10,0,0);
   nuc->AddDecay(0,-1,0,0.220306,100);
   nuc->AddDecay(-4,-2,0,5.75,0.004);

   // Adding 94-PU-237-1
   nuc = new Nucleus("PU",237,94,1,237.049,0.146,0.18,0,0,0,0,-8);
   nuc->AddDecay(0,0,-1,0.146,100);

   // Adding 95-AM-237-0
   nuc = new Nucleus("AM",237,95,0,237.05,0,4380,0,1.8e-11,3.7e-11,1,0);
   nuc->AddDecay(-4,-2,0,6.24613,0.02);
   nuc->AddDecay(0,-1,0,1.73016,99.98);

   // Adding 96-CM-237-0
   nuc = new Nucleus("CM",237,96,0,237.053,0,0,0,0,0,0,-2);

   // Adding 97-BK-237-0
   nuc = new Nucleus("BK",237,97,0,237.057,0,0,0,0,0,0,-2);

   // Adding 91-PA-238-0
   nuc = new Nucleus("PA",238,91,0,238.055,0,138,0,0,0,0,0);
   nuc->AddDecay(0,1,0,3.46004,100);

   // Adding 92-U-238-0
   nuc = new Nucleus("U",238,92,0,238.051,0,1.40903e+17,99.2745,4.4e-08,7.3e-06,1,0);
   nuc->AddDecay(-4,-2,0,4.26978,100);
   nuc->AddDecay(0,1000,0,207,5.45e-05);

   // Adding 93-NP-238-0
   nuc = new Nucleus("NP",238,93,0,238.051,0,182909,0,9.1e-10,2e-09,0,0);
   nuc->AddDecay(0,1,0,1.2921,100);

   // Adding 94-PU-238-0
   nuc = new Nucleus("PU",238,94,0,238.05,0,2.76697e+09,0,2.3e-07,4.3e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.5932,100);
   nuc->AddDecay(0,1000,0,207,1.9e-07);

   // Adding 95-AM-238-0
   nuc = new Nucleus("AM",238,95,0,238.052,0,5880,0,3.2e-11,8.5e-11,0,0);
   nuc->AddDecay(0,-1,0,2.25843,99.99);
   nuc->AddDecay(-4,-2,0,6.04167,0.0001);

   // Adding 96-CM-238-0
   nuc = new Nucleus("CM",238,96,0,238.053,0,8640,0,8e-11,4.9e-09,0,0);
   nuc->AddDecay(0,-1,0,0.968086,90);
   nuc->AddDecay(-4,-2,0,6.62147,10);

   // Adding 97-BK-238-0
   nuc = new Nucleus("BK",238,97,0,238.058,0,144,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.95195,50);
   nuc->AddDecay(-4,-2,0,7.40018,50);

   // Adding 92-U-239-0
   nuc = new Nucleus("U",239,92,0,239.054,0,1407,0,2.8e-11,3.5e-11,0,0);
   nuc->AddDecay(0,1,0,1.2652,100);

   // Adding 93-NP-239-0
   nuc = new Nucleus("NP",239,93,0,239.053,0,203602,0,8e-10,1.1e-09,0,0);
   nuc->AddDecay(0,1,0,0.721802,100);

   // Adding 94-PU-239-0
   nuc = new Nucleus("PU",239,94,0,239.052,0,7.60333e+11,0,2.5e-07,4.7e-05,0,0);
   nuc->AddDecay(0,1000,0,207,3.1e-10);
   nuc->AddDecay(-4,-2,0,5.2445,0.0115);
   nuc->AddDecay(-4,-2,1,5.2445,99.9885);

   // Adding 95-AM-239-0
   nuc = new Nucleus("AM",239,95,0,239.053,0,42840,0,2.4e-10,2.9e-10,1,0);
   nuc->AddDecay(-4,-2,0,5.92371,0.01);
   nuc->AddDecay(0,-1,0,0.80291,99.99);

   // Adding 96-CM-239-0
   nuc = new Nucleus("CM",239,96,0,239.055,0,10440,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.70057,99.9);
   nuc->AddDecay(-4,-2,0,6.45646,0.1);

   // Adding 97-BK-239-0
   nuc = new Nucleus("BK",239,97,0,239.058,0,91.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.27819,50);
   nuc->AddDecay(-4,-2,0,7.20018,50);

   // Adding 98-CF-239-0
   nuc = new Nucleus("CF",239,98,0,239.063,0,39,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.8101,100);
   nuc->AddDecay(0,-1,0,3.91994,0);

   // Adding 92-U-240-0
   nuc = new Nucleus("U",240,92,0,240.057,0,50760,0,1.1e-09,8.4e-10,1,0);
   nuc->AddDecay(-4,-2,0,3.57775,1e-10);
   nuc->AddDecay(0,1,1,0.388336,100);

   // Adding 93-NP-240-0
   nuc = new Nucleus("NP",240,93,0,240.056,0,3714,0,8.2e-11,1.3e-10,0,0);
   nuc->AddDecay(0,1,0,2.19959,100);

   // Adding 93-NP-240-1
   nuc = new Nucleus("NP",240,93,1,240.056,0,433.2,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.19959,99.89);
   nuc->AddDecay(0,0,-1,0,0.11);

   // Adding 94-PU-240-0
   nuc = new Nucleus("PU",240,94,0,240.054,0,2.07002e+11,0,2.5e-07,4.7e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.2558,100);
   nuc->AddDecay(0,1000,0,207,5.7e-06);

   // Adding 95-AM-240-0
   nuc = new Nucleus("AM",240,95,0,240.055,0,182880,0,5.8e-10,5.9e-10,0,0);
   nuc->AddDecay(0,-1,0,1.37889,100);
   nuc->AddDecay(-4,-2,0,5.69466,0.00019);

   // Adding 96-CM-240-0
   nuc = new Nucleus("CM",240,96,0,240.056,0,2.3328e+06,0,8.3e-09,3e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.3972,99.5);
   nuc->AddDecay(0,-1,0,0.215916,0.5);
   nuc->AddDecay(0,1000,0,207,3.9e-06);

   // Adding 97-BK-240-0
   nuc = new Nucleus("BK",240,97,0,240.06,0,288,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.93982,100);
   nuc->AddDecay(0,2000,0,3.9,0);

   // Adding 98-CF-240-0
   nuc = new Nucleus("CF",240,98,0,240.062,0,63.6,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.7191,100);

   // Adding 93-NP-241-0
   nuc = new Nucleus("NP",241,93,0,241.058,0,834,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.30505,100);

   // Adding 94-PU-241-0
   nuc = new Nucleus("PU",241,94,0,241.057,0,4.52542e+08,0,4.7e-09,8.5e-07,0,0);
   nuc->AddDecay(0,1,0,0.0207977,99.998);
   nuc->AddDecay(-4,-2,0,5.1401,0.00245);

   // Adding 95-AM-241-0
   nuc = new Nucleus("AM",241,95,0,241.057,0,1.36456e+10,0,2e-07,3.9e-05,0,0);
   nuc->AddDecay(0,1000,0,207,3.77e-10);
   nuc->AddDecay(-4,-2,0,5.6378,100);

   // Adding 96-CM-241-0
   nuc = new Nucleus("CM",241,96,0,241.058,0,2.83392e+06,0,9.3e-10,3.8e-08,0,0);
   nuc->AddDecay(0,-1,0,0.767502,99);
   nuc->AddDecay(-4,-2,0,6.185,1);

   // Adding 97-BK-241-0
   nuc = new Nucleus("BK",241,97,0,241.06,0,0,0,0,0,0,-2);

   // Adding 98-CF-241-0
   nuc = new Nucleus("CF",241,98,0,241.064,0,226.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.25706,90);
   nuc->AddDecay(-4,-2,0,7.65915,10);

   // Adding 99-ES-241-0
   nuc = new Nucleus("ES",241,99,0,241.069,0,8.5,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,4.54808,50);
   nuc->AddDecay(-4,-2,0,8.26413,50);

   // Adding 92-U-242-0
   nuc = new Nucleus("U",242,92,0,242.063,0,1008,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.41788,100);

   // Adding 93-NP-242-0
   nuc = new Nucleus("NP",242,93,0,242.062,0,330,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.7,100);

   // Adding 93-NP-242-1
   nuc = new Nucleus("NP",242,93,1,242.062,0,132,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,2.7,100);

   // Adding 94-PU-242-0
   nuc = new Nucleus("PU",242,94,0,242.059,0,1.17724e+13,0,2.4e-07,4.5e-05,1,0);
   nuc->AddDecay(-4,-2,0,4.9827,100);
   nuc->AddDecay(0,1000,0,207,0.00055);

   // Adding 95-AM-242-0
   nuc = new Nucleus("AM",242,95,0,242.06,0,57672,0,3e-10,1.6e-08,0,0);
   nuc->AddDecay(0,1,0,0.664799,82.7);
   nuc->AddDecay(0,-1,0,0.750999,17.3);

   // Adding 95-AM-242-1
   nuc = new Nucleus("AM",242,95,1,242.06,0.049,4.44658e+09,0,1.9e-07,3.5e-05,0,0);
   nuc->AddDecay(0,1000,-1,207,0);
   nuc->AddDecay(0,0,-1,0.049,99.54);
   nuc->AddDecay(-4,-2,-1,5.6373,0.46);

   // Adding 96-CM-242-0
   nuc = new Nucleus("CM",242,96,0,242.059,0,1.40651e+07,0,1.3e-08,4.9e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.2156,100);
   nuc->AddDecay(0,1000,0,207,6.2e-06);

   // Adding 97-BK-242-0
   nuc = new Nucleus("BK",242,97,0,242.062,0,420,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.99988,100);

   // Adding 98-CF-242-0
   nuc = new Nucleus("CF",242,98,0,242.064,0,209.4,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,7.5163,50);
   nuc->AddDecay(0,-1,0,1.52734,50);

   // Adding 99-ES-242-0
   nuc = new Nucleus("ES",242,99,0,242.07,0,40,0,0,0,1,-8);
   nuc->AddDecay(-4,-2,0,8.18214,100);

   // Adding 100-FM-242-0
   nuc = new Nucleus("FM",242,100,0,242.073,0,0.0008,0,0,0,0,-8);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 93-NP-243-0
   nuc = new Nucleus("NP",243,93,0,243.064,0,111,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.16978,100);

   // Adding 94-PU-243-0
   nuc = new Nucleus("PU",243,94,0,243.062,0,17841.6,0,8.5e-11,1.1e-10,0,0);
   nuc->AddDecay(0,1,0,0.581509,100);

   // Adding 95-AM-243-0
   nuc = new Nucleus("AM",243,95,0,243.061,0,2.3242e+11,0,2e-07,3.9e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.4381,100);
   nuc->AddDecay(0,1000,0,207,3.7e-09);

   // Adding 96-CM-243-0
   nuc = new Nucleus("CM",243,96,0,243.061,0,9.17698e+08,0,2e-07,3.9e-05,1,0);
   nuc->AddDecay(-4,-2,0,6.1688,99.71);
   nuc->AddDecay(0,-1,0,0.00889969,0.29);
   nuc->AddDecay(0,1000,0,207,5.3e-09);

   // Adding 97-BK-243-0
   nuc = new Nucleus("BK",243,97,0,243.063,0,16200,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.50843,99.85);
   nuc->AddDecay(-4,-2,0,6.87432,0.15);

   // Adding 98-CF-243-0
   nuc = new Nucleus("CF",243,98,0,243.065,0,642,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.21637,86);
   nuc->AddDecay(-4,-2,0,7.39012,14);

   // Adding 99-ES-243-0
   nuc = new Nucleus("ES",243,99,0,243.07,0,21,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.96017,70);
   nuc->AddDecay(-4,-2,0,8.0721,30);

   // Adding 100-FM-243-0
   nuc = new Nucleus("FM",243,100,0,243.075,0,0.18,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.68911,100);

   // Adding 94-PU-244-0
   nuc = new Nucleus("PU",244,94,0,244.064,0,2.54811e+15,0,2.4e-07,4.4e-05,0,0);
   nuc->AddDecay(0,1000,0,207,0.12);
   nuc->AddDecay(-4,-2,0,4.6655,99.88);

   // Adding 95-AM-244-0
   nuc = new Nucleus("AM",244,95,0,244.064,0,36360,0,4.6e-10,1.9e-09,0,0);
   nuc->AddDecay(0,1,0,1.42782,100);

   // Adding 95-AM-244-1
   nuc = new Nucleus("AM",244,95,1,244.064,0.088,1560,0,2.9e-11,7.9e-11,0,0);
   nuc->AddDecay(0,1,-1,1.51582,99.96);
   nuc->AddDecay(0,-1,-1,0.164173,0.04);

   // Adding 96-CM-244-0
   nuc = new Nucleus("CM",244,96,0,244.063,0,5.70802e+08,0,1.6e-07,3.2e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.90178,99.9999);
   nuc->AddDecay(0,1000,0,207,0.0001347);

   // Adding 97-BK-244-0
   nuc = new Nucleus("BK",244,97,0,244.065,0,15660,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.25545,99.99);
   nuc->AddDecay(-4,-2,0,6.77834,0.006);

   // Adding 98-CF-244-0
   nuc = new Nucleus("CF",244,98,0,244.066,0,1164,0,7e-11,1.8e-08,1,0);
   nuc->AddDecay(-4,-2,0,7.32911,100);

   // Adding 99-ES-244-0
   nuc = new Nucleus("ES",244,99,0,244.071,0,37,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.94713,4);
   nuc->AddDecay(0,-1,0,4.55785,96);

   // Adding 100-FM-244-0
   nuc = new Nucleus("FM",244,100,0,244.074,0,0.0037,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 94-PU-245-0
   nuc = new Nucleus("PU",245,94,0,245.068,0,37800,0,7.2e-10,6.5e-10,0,0);
   nuc->AddDecay(0,1,0,1.20468,100);

   // Adding 95-AM-245-0
   nuc = new Nucleus("AM",245,95,0,245.066,0,7380,0,6.2e-11,7.6e-11,0,0);
   nuc->AddDecay(0,1,0,0.893906,100);

   // Adding 96-CM-245-0
   nuc = new Nucleus("CM",245,96,0,245.065,0,2.68056e+11,0,3e-07,5.5e-05,0,0);
   nuc->AddDecay(0,1000,0,207,6.1e-07);
   nuc->AddDecay(-4,-2,0,5.62351,100);

   // Adding 97-BK-245-0
   nuc = new Nucleus("BK",245,97,0,245.066,0,426816,0,5.7e-10,2e-09,1,0);
   nuc->AddDecay(-4,-2,0,6.45451,0.12);
   nuc->AddDecay(0,-1,0,0.8102,99.88);

   // Adding 98-CF-245-0
   nuc = new Nucleus("CF",245,98,0,245.068,0,2700,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.25561,36);
   nuc->AddDecay(0,-1,0,1.5686,64);

   // Adding 99-ES-245-0
   nuc = new Nucleus("ES",245,99,0,245.071,0,66,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.05377,60);
   nuc->AddDecay(-4,-2,0,7.9091,40);

   // Adding 100-FM-245-0
   nuc = new Nucleus("FM",245,100,0,245.075,0,4.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.43512,100);

   // Adding 94-PU-246-0
   nuc = new Nucleus("PU",246,94,0,246.07,0,936576,0,3.3e-09,7.6e-09,0,0);
   nuc->AddDecay(0,1,1,0.40049,100);

   // Adding 95-AM-246-0
   nuc = new Nucleus("AM",246,95,0,246.07,0,2340,0,5.8e-11,1.1e-10,0,0);
   nuc->AddDecay(0,1,0,2.3762,100);

   // Adding 95-AM-246-1
   nuc = new Nucleus("AM",246,95,1,246.07,0,1500,0,3.5e-11,3.8e-11,0,0);
   nuc->AddDecay(0,1,-1,2.3762,100);
   nuc->AddDecay(0,0,-1,0,0.01);

   // Adding 96-CM-246-0
   nuc = new Nucleus("CM",246,96,0,246.067,0,1.49165e+11,0,2.9e-07,5.5e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.4748,99.97);
   nuc->AddDecay(0,1000,0,207,0.03);

   // Adding 97-BK-246-0
   nuc = new Nucleus("BK",246,97,0,246.069,0,155520,0,4.8e-10,4.6e-10,1,0);
   nuc->AddDecay(-4,-2,0,6.07384,0.2);
   nuc->AddDecay(0,-1,0,1.35004,99.8);

   // Adding 98-CF-246-0
   nuc = new Nucleus("CF",246,98,0,246.069,0,128520,0,3.3e-09,4.2e-07,1,0);
   nuc->AddDecay(-4,-2,0,6.8616,100);
   nuc->AddDecay(0,-1,0,0.122963,0.0005);
   nuc->AddDecay(0,1000,0,207,0.0002);

   // Adding 99-ES-246-0
   nuc = new Nucleus("ES",246,99,0,246.073,0,462,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.8804,90.1);
   nuc->AddDecay(-4,-2,0,7.74213,9.9);

   // Adding 100-FM-246-0
   nuc = new Nucleus("FM",246,100,0,246.075,0,1.1,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.15911,1);
   nuc->AddDecay(-4,-2,0,8.3739,91.08);
   nuc->AddDecay(0,1000,0,207,7.92);

   // Adding 94-PU-247-0
   nuc = new Nucleus("PU",247,94,0,247.075,0,196128,0,0,0,0,0);
   nuc->AddDecay(0,1,0,2.1778,100);

   // Adding 95-AM-247-0
   nuc = new Nucleus("AM",247,95,0,247.072,0,1380,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.70036,100);

   // Adding 96-CM-247-0
   nuc = new Nucleus("CM",247,96,0,247.07,0,4.91962e+14,0,2.7e-07,5.1e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.35291,100);

   // Adding 97-BK-247-0
   nuc = new Nucleus("BK",247,97,0,247.07,0,4.35197e+10,0,3.5e-07,6.5e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.88953,100);

   // Adding 98-CF-247-0
   nuc = new Nucleus("CF",247,98,0,247.071,0,11196,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.646027,99.97);
   nuc->AddDecay(-4,-2,0,6.52666,0.04);

   // Adding 99-ES-247-0
   nuc = new Nucleus("ES",247,99,0,247.074,0,273,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.47542,93);
   nuc->AddDecay(-4,-2,0,7.49365,7);

   // Adding 100-FM-247-0
   nuc = new Nucleus("FM",247,100,0,247.077,0,35,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.19111,50);
   nuc->AddDecay(0,-1,0,2.91383,50);

   // Adding 100-FM-247-1
   nuc = new Nucleus("FM",247,100,1,247.077,0,9.2,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,8.19111,100);

   // Adding 101-MD-247-0
   nuc = new Nucleus("MD",247,101,0,247.082,0,2.9,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.81716,100);

   // Adding 95-AM-248-0
   nuc = new Nucleus("AM",248,95,0,248.076,0,0,0,0,0,0,-9);
   nuc->AddDecay(0,1,0,3.09965,100);

   // Adding 96-CM-248-0
   nuc = new Nucleus("CM",248,96,0,248.072,0,1.07222e+13,0,1.1e-06,0.0002,0,0);
   nuc->AddDecay(0,1000,0,207,8.26);
   nuc->AddDecay(-4,-2,0,5.1618,91.74);

   // Adding 97-BK-248-0
   nuc = new Nucleus("BK",248,97,0,248.073,0,2.83824e+08,0,0,0,1,-5);
   nuc->AddDecay(-4,-2,0,5.8027,100);

   // Adding 97-BK-248-1
   nuc = new Nucleus("BK",248,97,1,248.073,0,85320,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,5.8027,0.001);
   nuc->AddDecay(0,-1,-1,0.717072,30);
   nuc->AddDecay(0,1,-1,0.869965,70);

   // Adding 98-CF-248-0
   nuc = new Nucleus("CF",248,98,0,248.072,0,2.88144e+07,0,2.8e-08,8.2e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.36056,100);
   nuc->AddDecay(0,1000,0,207,0.0029);

   // Adding 99-ES-248-0
   nuc = new Nucleus("ES",248,99,0,248.075,0,1620,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.0606,99.75);
   nuc->AddDecay(-4,-2,0,7.16571,0.25);

   // Adding 100-FM-248-0
   nuc = new Nucleus("FM",248,100,0,248.077,0,36,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.00228,99);
   nuc->AddDecay(0,-1,0,1.60326,1);
   nuc->AddDecay(0,1000,0,207,0.05);

   // Adding 101-MD-248-0
   nuc = new Nucleus("MD",248,101,0,248.083,0,7,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,5.25272,80);
   nuc->AddDecay(-4,-2,0,8.69715,20);

   // Adding 96-CM-249-0
   nuc = new Nucleus("CM",249,96,0,249.076,0,3849,0,3.1e-11,5.5e-11,0,0);
   nuc->AddDecay(0,1,0,0.899925,100);

   // Adding 97-BK-249-0
   nuc = new Nucleus("BK",249,97,0,249.075,0,2.7648e+07,0,9.7e-10,1.5e-07,0,0);
   nuc->AddDecay(0,1000,0,207,4.7e-08);
   nuc->AddDecay(0,1,0,0.124901,99.9986);
   nuc->AddDecay(-4,-2,0,5.526,0.00145);

   // Adding 98-CF-249-0
   nuc = new Nucleus("CF",249,98,0,249.075,0,1.10691e+10,0,3.5e-07,6.6e-05,0,0);
   nuc->AddDecay(0,1000,0,207,5.2e-07);
   nuc->AddDecay(-4,-2,0,6.295,100);

   // Adding 99-ES-249-0
   nuc = new Nucleus("ES",249,99,0,249.076,0,6132,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.45447,99.43);
   nuc->AddDecay(-4,-2,0,6.93927,0.57);

   // Adding 100-FM-249-0
   nuc = new Nucleus("FM",249,100,0,249.079,0,156,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.4371,85);
   nuc->AddDecay(-4,-2,0,7.80777,15);

   // Adding 101-MD-249-0
   nuc = new Nucleus("MD",249,101,0,249.083,0,24,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.45919,70);
   nuc->AddDecay(0,-1,0,3.70518,30);

   // Adding 102-NO-249-0
   nuc = new Nucleus("NO",249,102,0,249,0,0,0,0,0,0,-6);

   // Adding 96-CM-250-0
   nuc = new Nucleus("CM",250,96,0,250.078,0,3.05899e+11,0,6.3e-06,0.0011,0,0);
   nuc->AddDecay(0,1000,0,207,80);
   nuc->AddDecay(-4,-2,0,5.1689,11);
   nuc->AddDecay(0,1,0,0.037468,9);

   // Adding 97-BK-250-0
   nuc = new Nucleus("BK",250,97,0,250.078,0,11581.2,0,1.4e-10,9.6e-10,0,0);
   nuc->AddDecay(0,1,0,1.77972,100);

   // Adding 98-CF-250-0
   nuc = new Nucleus("CF",250,98,0,250.076,0,4.12491e+08,0,1.6e-07,3.2e-05,1,0);
   nuc->AddDecay(-4,-2,0,6.1284,99.92);
   nuc->AddDecay(0,1000,0,207,0.08);

   // Adding 99-ES-250-0
   nuc = new Nucleus("ES",250,99,0,250.079,0,30960,0,2.1e-11,5.9e-10,0,0);
   nuc->AddDecay(0,-1,0,2.09998,97);
   nuc->AddDecay(-4,-2,0,6.87834,3);

   // Adding 99-ES-250-1
   nuc = new Nucleus("ES",250,99,1,250.079,0,7992,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.09998,99);
   nuc->AddDecay(-4,-2,-1,6.87834,1);

   // Adding 100-FM-250-0
   nuc = new Nucleus("FM",250,100,0,250.08,0,1800,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,0.0006);
   nuc->AddDecay(-4,-2,0,7.55699,90);
   nuc->AddDecay(0,-1,0,0.801613,10);

   // Adding 100-FM-250-1
   nuc = new Nucleus("FM",250,100,1,250.081,1,1.8,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,1,100);

   // Adding 101-MD-250-0
   nuc = new Nucleus("MD",250,101,0,250.084,0,52,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.63259,93);
   nuc->AddDecay(-4,-2,0,8.30918,7);

   // Adding 102-NO-250-0
   nuc = new Nucleus("NO",250,102,0,250,0,0.00025,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9,0.05);
   nuc->AddDecay(0,1000,0,207,99.95);

   // Adding 96-CM-251-0
   nuc = new Nucleus("CM",251,96,0,251.082,0,1008,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.42001,100);

   // Adding 97-BK-251-0
   nuc = new Nucleus("BK",251,97,0,251.081,0,3336,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,5.56842,1e-05);
   nuc->AddDecay(0,1,0,1.09296,100);

   // Adding 98-CF-251-0
   nuc = new Nucleus("CF",251,98,0,251.08,0,2.83193e+10,0,3.6e-07,6.7e-05,1,0);
   nuc->AddDecay(-4,-2,0,6.17581,100);

   // Adding 99-ES-251-0
   nuc = new Nucleus("ES",251,99,0,251.08,0,118800,0,1.7e-10,2e-09,0,0);
   nuc->AddDecay(0,-1,0,0.376213,99.51);
   nuc->AddDecay(-4,-2,0,6.5969,0.49);

   // Adding 100-FM-251-0
   nuc = new Nucleus("FM",251,100,0,251.082,0,19080,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.4251,1.8);
   nuc->AddDecay(0,-1,0,1.47422,98.2);

   // Adding 101-MD-251-0
   nuc = new Nucleus("MD",251,101,0,251.085,0,240,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.07332,90);
   nuc->AddDecay(-4,-2,0,8.023,10);

   // Adding 102-NO-251-0
   nuc = new Nucleus("NO",251,102,0,251.089,0,0.8,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.77594,1);
   nuc->AddDecay(-4,-2,0,8.88511,100);

   // Adding 103-LR-251-0
   nuc = new Nucleus("LR",251,103,0,251.094,0,0,0,0,0,0,-6);

   // Adding 96-CM-252-0
   nuc = new Nucleus("CM",252,96,0,252,0,172800,0,0,0,0,-4);
   nuc->AddDecay(0,1,0,0,100);

   // Adding 97-BK-252-0
   nuc = new Nucleus("BK",252,97,0,252.084,0,108,0,0,0,0,-8);
   nuc->AddDecay(0,1,0,2.49995,100);

   // Adding 98-CF-252-0
   nuc = new Nucleus("CF",252,98,0,252.082,0,8.34127e+07,0,9e-08,1.8e-05,1,0);
   nuc->AddDecay(-4,-2,0,6.2168,96.91);
   nuc->AddDecay(0,1000,0,207,3.09);

   // Adding 99-ES-252-0
   nuc = new Nucleus("ES",252,99,0,252.083,0,4.07549e+07,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,6.75972,76);
   nuc->AddDecay(0,-1,0,1.25999,24);
   nuc->AddDecay(0,1,0,0.47699,0.01);

   // Adding 100-FM-252-0
   nuc = new Nucleus("FM",252,100,0,252.082,0,91404,0,2.7e-09,3e-07,1,0);
   nuc->AddDecay(-4,-2,0,7.1527,100);
   nuc->AddDecay(0,1000,0,207,0.0023);

   // Adding 101-MD-252-0
   nuc = new Nucleus("MD",252,101,0,252.087,0,138,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.97704,50);
   nuc->AddDecay(0,-1,0,3.88494,50);

   // Adding 102-NO-252-0
   nuc = new Nucleus("NO",252,102,0,252.089,0,2.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.5495,73.1);
   nuc->AddDecay(0,1000,0,207,26.9);

   // Adding 103-LR-252-0
   nuc = new Nucleus("LR",252,103,0,252,0,1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.15,89.1);
   nuc->AddDecay(0,-1,0,0,9.9);
   nuc->AddDecay(0,1000,0,207,1);

   // Adding 98-CF-253-0
   nuc = new Nucleus("CF",253,98,0,253.085,0,1.53878e+06,0,1.4e-09,0.0001,0,0);
   nuc->AddDecay(0,1,0,0.285141,99.69);
   nuc->AddDecay(-4,-2,0,6.12442,0.31);

   // Adding 99-ES-253-0
   nuc = new Nucleus("ES",253,99,0,253.085,0,1.76861e+06,0,6.1e-09,2.5e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.7392,100);
   nuc->AddDecay(0,1000,0,207,8.7e-06);

   // Adding 100-FM-253-0
   nuc = new Nucleus("FM",253,100,0,253.085,0,259200,0,9.1e-10,3.7e-07,1,0);
   nuc->AddDecay(-4,-2,0,7.19692,12);
   nuc->AddDecay(0,-1,0,0.332817,88);

   // Adding 101-MD-253-0
   nuc = new Nucleus("MD",253,101,0,253.087,0,600,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,1.96086,50);
   nuc->AddDecay(-4,-2,0,7.70331,50);

   // Adding 102-NO-253-0
   nuc = new Nucleus("NO",253,102,0,253.091,0,102,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.44323,80);
   nuc->AddDecay(0,-1,0,3.17702,20);

   // Adding 103-LR-253-0
   nuc = new Nucleus("LR",253,103,0,253.095,0,1.3,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,4.25406,1);
   nuc->AddDecay(-4,-2,0,8.9921,90);
   nuc->AddDecay(0,1000,0,207,9);

   // Adding 104-04-253-0
   nuc = new Nucleus("04",253,104,0,253,0,1.8,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.5,50);
   nuc->AddDecay(0,1000,0,207,50);

   // Adding 98-CF-254-0
   nuc = new Nucleus("CF",254,98,0,254.087,0,5.2272e+06,0,4e-07,3.7e-05,1,0);
   nuc->AddDecay(-4,-2,0,5.9264,0.31);
   nuc->AddDecay(0,1000,0,207,99.69);

   // Adding 99-ES-254-0
   nuc = new Nucleus("ES",254,99,0,254.088,0,2.38205e+07,0,2.8e-08,8e-06,0,0);
   nuc->AddDecay(0,1,0,1.09052,1.74e-06);
   nuc->AddDecay(-4,-2,0,6.61801,100);
   nuc->AddDecay(0,-1,0,0.654137,0.0001);
   nuc->AddDecay(0,1000,0,207,3e-06);

   // Adding 99-ES-254-1
   nuc = new Nucleus("ES",254,99,1,254.088,0.078,141480,0,4.2e-09,4.4e-07,0,0);
   nuc->AddDecay(0,1000,-1,207,0.0260656);
   nuc->AddDecay(0,-1,-1,0.73214,0.08);
   nuc->AddDecay(-4,-2,-1,6.69601,0.33);
   nuc->AddDecay(0,1,-1,1.16852,98);
   nuc->AddDecay(0,0,-1,0.078,1.56394);

   // Adding 100-FM-254-0
   nuc = new Nucleus("FM",254,100,0,254.087,0,11664,0,4.4e-10,7.7e-08,1,0);
   nuc->AddDecay(-4,-2,0,7.30721,99.94);
   nuc->AddDecay(0,1000,0,207,0.06);

   // Adding 101-MD-254-0
   nuc = new Nucleus("MD",254,101,0,254.09,0,600,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.68291,100);

   // Adding 101-MD-254-1
   nuc = new Nucleus("MD",254,101,1,254.09,0,1680,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,2.68291,100);

   // Adding 102-NO-254-0
   nuc = new Nucleus("NO",254,102,0,254.091,0,55,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.2258,90);
   nuc->AddDecay(0,-1,0,1.13728,10);
   nuc->AddDecay(0,1000,0,207,0.25);

   // Adding 102-NO-254-1
   nuc = new Nucleus("NO",254,102,1,254.091,0.5,0.28,0,0,0,0,0);
   nuc->AddDecay(0,0,-1,0.5,100);

   // Adding 103-LR-254-0
   nuc = new Nucleus("LR",254,103,0,254.096,0,13,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.74614,77.922);
   nuc->AddDecay(0,-1,0,5.15292,21.978);
   nuc->AddDecay(0,1000,0,207,0.1);

   // Adding 104-04-254-0
   nuc = new Nucleus("04",254,104,0,254,0,0.0005,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.3,0.3);
   nuc->AddDecay(0,1000,0,207,99.7);

   // Adding 98-CF-255-0
   nuc = new Nucleus("CF",255,98,0,255.091,0,5100,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.700287,100);

   // Adding 99-ES-255-0
   nuc = new Nucleus("ES",255,99,0,255.09,0,3.43872e+06,0,0,0,0,0);
   nuc->AddDecay(0,1,0,0.287964,92);
   nuc->AddDecay(-4,-2,0,6.4356,8);
   nuc->AddDecay(0,1000,0,207,0.0041);

   // Adding 100-FM-255-0
   nuc = new Nucleus("FM",255,100,0,255.09,0,72252,0,2.6e-09,2.6e-07,1,0);
   nuc->AddDecay(-4,-2,0,7.2406,100);
   nuc->AddDecay(0,1000,0,207,2.4e-05);

   // Adding 101-MD-255-0
   nuc = new Nucleus("MD",255,101,0,255.091,0,1620,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,1.04253,92);
   nuc->AddDecay(-4,-2,0,7.90691,8);

   // Adding 102-NO-255-0
   nuc = new Nucleus("NO",255,102,0,255.093,0,186,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.44453,61.4);
   nuc->AddDecay(0,-1,0,2.01184,38.6);

   // Adding 103-LR-255-0
   nuc = new Nucleus("LR",255,103,0,255.097,0,22,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,3.24291,15);
   nuc->AddDecay(-4,-2,0,8.61412,85);

   // Adding 104-04-255-0
   nuc = new Nucleus("04",255,104,0,255.102,0,1.5,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.29813,48);
   nuc->AddDecay(0,1000,0,207,52);

   // Adding 105-05-255-0
   nuc = new Nucleus("05",255,105,0,255,0,1.6,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,20);
   nuc->AddDecay(-4,-2,0,9.6,80);

   // Adding 98-CF-256-0
   nuc = new Nucleus("CF",256,98,0,256,0,738,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,99);
   nuc->AddDecay(-4,-2,0,5.6,9.9e-07);
   nuc->AddDecay(0,1,0,0,1);

   // Adding 99-ES-256-0
   nuc = new Nucleus("ES",256,99,0,256.094,0,1524,0,0,0,0,0);
   nuc->AddDecay(0,1,0,1.66993,100);

   // Adding 99-ES-256-1
   nuc = new Nucleus("ES",256,99,1,256.094,0,27360,0,0,0,0,0);
   nuc->AddDecay(0,1,-1,1.66993,100);

   // Adding 100-FM-256-0
   nuc = new Nucleus("FM",256,100,0,256.092,0,9456,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,91.9);
   nuc->AddDecay(-4,-2,0,7.02702,8.1);

   // Adding 101-MD-256-0
   nuc = new Nucleus("MD",256,101,0,256.094,0,4560,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,2.12958,87.979);
   nuc->AddDecay(-4,-2,0,7.8966,9.021);
   nuc->AddDecay(0,1000,0,207,3);

   // Adding 102-NO-256-0
   nuc = new Nucleus("NO",256,102,0,256.094,0,3.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.58142,99.8);
   nuc->AddDecay(0,1000,0,207,0.199997);

   // Adding 103-LR-256-0
   nuc = new Nucleus("LR",256,103,0,256.099,0,28,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,0.0299102);
   nuc->AddDecay(-4,-2,0,8.88512,80);
   nuc->AddDecay(0,-1,0,4.18864,19.9401);

   // Adding 104-04-256-0
   nuc = new Nucleus("04",256,104,0,256.101,0,0.0067,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.95201,2.2);
   nuc->AddDecay(0,1000,0,207,98);

   // Adding 105-05-256-0
   nuc = new Nucleus("05",256,105,0,256,0,2.6,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0,10);
   nuc->AddDecay(0,1000,0,207,27.6923);
   nuc->AddDecay(-4,-2,0,9.5,62.3077);

   // Adding 99-ES-257-0
   nuc = new Nucleus("ES",257,99,0,257.096,0,0,0,0,0,0,-2);

   // Adding 100-FM-257-0
   nuc = new Nucleus("FM",257,100,0,257.095,0,8.6832e+06,0,1.5e-08,6.6e-06,1,0);
   nuc->AddDecay(-4,-2,0,6.8639,99.79);
   nuc->AddDecay(0,1000,0,207,0.21);

   // Adding 101-MD-257-0
   nuc = new Nucleus("MD",257,101,0,257.096,0,19080,0,1.2e-10,2.3e-08,0,0);
   nuc->AddDecay(0,1000,0,207,4);
   nuc->AddDecay(0,-1,0,0.408562,86.4);
   nuc->AddDecay(-4,-2,0,7.55761,9.6);

   // Adding 102-NO-257-0
   nuc = new Nucleus("NO",257,102,0,257.097,0,25,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.45169,100);

   // Adding 103-LR-257-0
   nuc = new Nucleus("LR",257,103,0,257.1,0,0.646,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.0061,100);

   // Adding 104-04-257-0
   nuc = new Nucleus("04",257,104,0,257.103,0,4.7,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.24958,79.6);
   nuc->AddDecay(0,1000,0,207,2.4);
   nuc->AddDecay(0,-1,0,3.42049,18);

   // Adding 105-05-257-0
   nuc = new Nucleus("05",257,105,0,257.108,0,1.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.30511,82);
   nuc->AddDecay(0,1000,0,207,17);
   nuc->AddDecay(0,-1,0,4.30959,1);

   // Adding 100-FM-258-0
   nuc = new Nucleus("FM",258,100,0,258.097,0,0.00037,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 101-MD-258-0
   nuc = new Nucleus("MD",258,101,0,258.098,0,3600,0,1.3e-08,5.5e-06,0,-8);
   nuc->AddDecay(0,-1,0,1.22466,50);
   nuc->AddDecay(-4,-2,0,7.27121,50);

   // Adding 102-NO-258-0
   nuc = new Nucleus("NO",258,102,0,258.098,0,0.0012,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.20007,0.001);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 103-LR-258-0
   nuc = new Nucleus("LR",258,103,0,258.102,0,4.3,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.9001,95);
   nuc->AddDecay(0,-1,0,3.38294,5);

   // Adding 104-04-258-0
   nuc = new Nucleus("04",258,104,0,258.103,0,0.012,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.24984,13);
   nuc->AddDecay(0,1000,0,207,87);

   // Adding 105-05-258-0
   nuc = new Nucleus("05",258,105,0,258.109,0,4.4,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.54511,66.33);
   nuc->AddDecay(0,1000,0,207,1);
   nuc->AddDecay(0,-1,0,5.4482,32.67);

   // Adding 105-05-258-1
   nuc = new Nucleus("05",258,105,1,258.109,0,20,0,0,0,0,0);
   nuc->AddDecay(0,-1,-1,5.4482,100);

   // Adding 100-FM-259-0
   nuc = new Nucleus("FM",259,100,0,259.101,0,1.5,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 101-MD-259-0
   nuc = new Nucleus("MD",259,101,0,259.1,0,6180,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.11059,3);
   nuc->AddDecay(0,1000,0,207,97);

   // Adding 102-NO-259-0
   nuc = new Nucleus("NO",259,102,0,259.101,0,3480,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,7.90525,67.5);
   nuc->AddDecay(0,-1,0,0.506699,22.5);
   nuc->AddDecay(0,1000,0,207,10);

   // Adding 103-LR-259-0
   nuc = new Nucleus("LR",259,103,0,259.103,0,5.4,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,49.0148);
   nuc->AddDecay(-4,-2,0,8.67434,50);
   nuc->AddDecay(0,-1,0,1.81161,0.490148);

   // Adding 104-04-259-0
   nuc = new Nucleus("04",259,104,0,259.106,0,3.1,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.10851,93);
   nuc->AddDecay(0,1000,0,207,7);
   nuc->AddDecay(0,-1,0,2.44601,0.3);

   // Adding 105-05-259-0
   nuc = new Nucleus("05",259,105,0,259.11,0,0,0,0,0,0,-2);

   // Adding 106-06-259-0
   nuc = new Nucleus("06",259,106,0,259.115,0,0.48,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,10);
   nuc->AddDecay(-4,-2,0,9.87114,90);

   // Adding 100-FM-260-0
   nuc = new Nucleus("FM",260,100,0,260.103,0,0,0,0,0,0,-6);

   // Adding 101-MD-260-0
   nuc = new Nucleus("MD",260,101,0,260.104,0,2.74752e+06,0,0,0,0,0);
   nuc->AddDecay(0,-1,0,0.896599,9);
   nuc->AddDecay(0,1000,0,207,70);
   nuc->AddDecay(0,1,0,0.992393,6);
   nuc->AddDecay(-4,-2,0,7.02259,15);

   // Adding 102-NO-260-0
   nuc = new Nucleus("NO",260,102,0,260.103,0,0.106,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 103-LR-260-0
   nuc = new Nucleus("LR",260,103,0,260.106,0,180,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.30546,75);
   nuc->AddDecay(0,1000,0,207,10);
   nuc->AddDecay(0,-1,0,2.73491,15);

   // Adding 104-04-260-0
   nuc = new Nucleus("04",260,104,0,260.107,0,0.0201,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,98);
   nuc->AddDecay(-4,-2,0,8.99972,2);

   // Adding 105-05-260-0
   nuc = new Nucleus("05",260,105,0,260.111,0,1.52,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,6.55693);
   nuc->AddDecay(-4,-2,0,9.37111,90);
   nuc->AddDecay(0,-1,0,4.56003,1.70753);

   // Adding 106-06-260-0
   nuc = new Nucleus("06",260,106,0,260.114,0,0.0036,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.92292,50);
   nuc->AddDecay(0,1000,0,207,50);

   // Adding 107-07-260-0
   nuc = new Nucleus("07",260,107,0,260,0,0,0,0,0,1,-9);
   nuc->AddDecay(-4,-2,0,10,100);

   // Adding 103-LR-261-0
   nuc = new Nucleus("LR",261,103,0,261.107,0,2340,0,0,0,0,-8);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 104-04-261-0
   nuc = new Nucleus("04",261,104,0,261.109,0,65,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,8.80927,80);
   nuc->AddDecay(0,-1,0,1.8357,10);
   nuc->AddDecay(0,1000,0,207,10);

   // Adding 105-05-261-0
   nuc = new Nucleus("05",261,105,0,261.112,0,1.8,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,50);
   nuc->AddDecay(-4,-2,0,9.26915,50);

   // Adding 106-06-261-0
   nuc = new Nucleus("06",261,106,0,261.116,0,0.23,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.80667,95);
   nuc->AddDecay(0,1000,0,207,5);

   // Adding 107-07-261-0
   nuc = new Nucleus("07",261,107,0,261.122,0,0.0118,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,10.5621,95);
   nuc->AddDecay(0,1000,0,207,5);

   // Adding 102-NO-262-0
   nuc = new Nucleus("NO",262,102,0,262.108,0,0,0,0,0,0,-6);

   // Adding 103-LR-262-0
   nuc = new Nucleus("LR",262,103,0,262.11,0,12960,0,0,0,0,-8);
   nuc->AddDecay(0,-1,0,2.0976,100);

   // Adding 104-04-262-0
   nuc = new Nucleus("04",262,104,0,262.11,0,0.047,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,100);

   // Adding 105-05-262-0
   nuc = new Nucleus("05",262,105,0,262.114,0,34,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,71);
   nuc->AddDecay(0,-1,0,3.987,3);
   nuc->AddDecay(-4,-2,0,9.2052,26);

   // Adding 106-06-262-0
   nuc = new Nucleus("06",262,106,0,262.117,0,0,0,0,0,0,-2);

   // Adding 107-07-262-0
   nuc = new Nucleus("07",262,107,0,262.123,0,0.102,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,20);
   nuc->AddDecay(-4,-2,0,10.4161,80);

   // Adding 107-07-262-1
   nuc = new Nucleus("07",262,107,1,262.123,0.315,0.008,0,0,0,1,0);
   nuc->AddDecay(-4,-2,-1,10.7311,70);
   nuc->AddDecay(0,1000,-1,207,30);

   // Adding 105-05-263-0
   nuc = new Nucleus("05",263,105,0,263.115,0,1560,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,9.03356,100);

   // Adding 106-06-263-0
   nuc = new Nucleus("06",263,106,0,263.119,0,0.8,0,0,0,0,0);
   nuc->AddDecay(0,1000,0,207,70);
   nuc->AddDecay(-4,-2,0,9.69251,30);

   // Adding 107-07-263-0
   nuc = new Nucleus("07",263,107,0,263.123,0,0,0,0,0,0,-2);

   // Adding 108-08-263-0
   nuc = new Nucleus("08",263,108,0,263,0,0.0018,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,0,100);

   // Adding 107-07-264-0
   nuc = new Nucleus("07",264,107,0,264.125,0,0,0,0,0,0,-2);

   // Adding 108-08-264-0
   nuc = new Nucleus("08",264,108,0,264.129,0,8e-05,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,10.8,100);

   // Adding 108-08-265-0
   nuc = new Nucleus("08",265,108,0,265.131,0,0.0018,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,10.8241,100);

   // Adding 109-09-266-0
   nuc = new Nucleus("09",266,109,0,266.138,0,0.0034,0,0,0,1,0);
   nuc->AddDecay(-4,-2,0,11.2831,100);
   for(map<int,Nucleus*>::const_iterator inuc=Nucleus::Nuclei().begin(); inuc != Nucleus::Nuclei().end(); ++inuc)
      inuc->second->NormDecay();
#else
    std::cerr << "CREATION OF NUCLEUS DB NOT COMPILED IN; CHANGE CMAKE OPTION \"GENERATE_MATERIALS_DB\" to ON \n";
#endif

}
 } // End of inline namespace
 } // End of vecgeom namespace
#ifdef __clang__
#pragma clang optimize on
#endif
