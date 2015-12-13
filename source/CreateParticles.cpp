#if defined(__clang__) && !defined(__APPLE__)
#pragma clang optimize off
#endif
#include "materials/Particle.h"
namespace vecgeom {
   inline namespace VECGEOM_IMPL_NAMESPACE {


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0000() {

   // Creating Alpha_bar
   new Particle("Alpha_bar", -1000020040, 0, "ion", 100, -2, 3.7284, 1.6916e-33, 100, 100, 0, 100, 1);

   // Creating HE3_bar
   new Particle("HE3_bar", -1000020030, 0, "ion", 100, -2, 2.80941, 0, 100, 100, 0, 100, 1);

   // Creating Triton_bar
   new Particle("Triton_bar", -1000010030, 0, "ion", 100, -1, 2.80941, 1.6916e-33, 100, 100, 0, 100, 1);

   // Creating Deuteron_bar
   new Particle("Deuteron_bar", -1000010020, 0, "ion", 100, -1, 1.87106, 0, 100, 100, 0, 100, 1);

   // Creating N(2250)+_bar
   new Particle("N(2250)+_bar", -100012210, 0, "Unknown", 100, -1, 2.275, 0.5, 100, 100, 0, 100, 1);

   // Creating N(2250)0_bar
   new Particle("N(2250)0_bar", -100012110, 0, "Unknown", 100, 0, 2.275, 0.5, 100, 100, 0, 100, 1);

   // Creating N(2220)+_bar
   new Particle("N(2220)+_bar", -100002210, 0, "Unknown", 100, -1, 2.25, 0.4, 100, 100, 0, 100, 1);

   // Creating N(2220)0_bar
   new Particle("N(2220)0_bar", -100002110, 0, "Unknown", 100, 0, 2.25, 0.4, 100, 100, 0, 100, 1);

   // Creating GenericIon_bar
   new Particle("GenericIon_bar", -50000060, 0, "Unknown", 100, -0.333333, 0.938272, 0, 100, 100, 0, 100, 1);

   // Creating Cherenkov_bar
   new Particle("Cherenkov_bar", -50000050, 0, "Unknown", 100, 0, 0, 0, 100, 100, 0, 100, 1);

   // Creating f2(2010)_bar
   new Particle("f2(2010)_bar", -9060225, 0, "Unknown", 100, 0, 2.01, 0.2, 100, 100, 0, 100, 1);

   // Creating f2(1810)_bar
   new Particle("f2(1810)_bar", -9030225, 0, "Unknown", 100, 0, 1.815, 0.197, 100, 100, 0, 100, 1);

   // Creating f0(1500)_bar
   new Particle("f0(1500)_bar", -9030221, 0, "Unknown", 100, 0, 1.505, 0.109, 100, 100, 0, 100, 1);

   // Creating eta(1405)_bar
   new Particle("eta(1405)_bar", -9020221, 0, "Unknown", 100, 0, 1.4098, 0.0511, 100, 100, 0, 100, 1);

   // Creating f0(980)_bar
   new Particle("f0(980)_bar", -9010221, 0, "Unknown", 100, 0, 0.98, 0.07, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0001() {

   // Creating f0(600)_bar
   new Particle("f0(600)_bar", -9000221, 0, "Unknown", 100, 0, 0.8, 0.8, 100, 100, 0, 100, 1);

   // Creating a0(980)-_bar
   new Particle("a0(980)-_bar", -9000211, 0, "Unknown", 100, -1, 0.98, 0.06, 100, 100, 0, 100, 1);

   // Creating a0(980)0_bar
   new Particle("a0(980)0_bar", -9000111, 0, "Unknown", 100, 0, 0.98, 0.075, 100, 100, 0, 100, 1);

   // Creating nu*_e0_bar
   new Particle("nu*_e0_bar", -4000012, 0, "Excited", 100, 0, 400, 0.41917, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-4000012));
   part->AddDecay(Particle::Decay(0, 0.610139,  vector<int>{-24,-11}));
   part->AddDecay(Particle::Decay(0, 0.389861,  vector<int>{-23,-12}));

   // Creating e*+
   new Particle("e*+", -4000011, 0, "Excited", 100, 1, 400, 0.42901, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4000011));
   part->AddDecay(Particle::Decay(0, 0.596149,  vector<int>{24,-12}));
   part->AddDecay(Particle::Decay(0, 0.294414,  vector<int>{-22,-11}));
   part->AddDecay(Particle::Decay(0, 0.109437,  vector<int>{-23,-11}));

   // Creating u*_bar
   new Particle("u*_bar", -4000002, 0, "Excited", 100, -0.666667, 400, 2.65499, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4000002));
   part->AddDecay(Particle::Decay(0, 0.853166,  vector<int>{-21,-2}));
   part->AddDecay(Particle::Decay(0, 0.0963291,  vector<int>{-24,-1}));
   part->AddDecay(Particle::Decay(0, 0.029361,  vector<int>{-23,-2}));
   part->AddDecay(Particle::Decay(0, 0.021144,  vector<int>{-22,-2}));

   // Creating d*_bar
   new Particle("d*_bar", -4000001, 0, "Excited", 100, 0.333333, 400, 2.65171, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4000001));
   part->AddDecay(Particle::Decay(53, 0.85422,  vector<int>{-21,-1}));
   part->AddDecay(Particle::Decay(0, 0.096449,  vector<int>{24,-2}));
   part->AddDecay(Particle::Decay(0, 0.044039,  vector<int>{-23,-1}));
   part->AddDecay(Particle::Decay(0, 0.005292,  vector<int>{-22,-1}));

   // Creating ~nu_tauR_bar
   new Particle("~nu_tauR_bar", -2000016, 0, "Sparticle", 100, 0, 500, 0, 100, 100, 1, 100, 1);

   // Creating ~tau_2+
   new Particle("~tau_2+", -2000015, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000015));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,37}));

   // Creating ~nu_muR_bar
   new Particle("~nu_muR_bar", -2000014, 0, "Sparticle", 100, 0, 500, 0, 100, 100, 1, 100, 1);

   // Creating ~mu_R+
   new Particle("~mu_R+", -2000013, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000013));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,37}));

   // Creating ~nu_eR_bar
   new Particle("~nu_eR_bar", -2000012, 0, "Sparticle", 100, 0, 500, 0, 100, 100, 1, 100, 1);

   // Creating ~e_R+
   new Particle("~e_R+", -2000011, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000011));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,37}));

   // Creating ~t_2_bar
   new Particle("~t_2_bar", -2000006, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000006));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-6}));

   // Creating ~b_2_bar
   new Particle("~b_2_bar", -2000005, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000005));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-5}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0002() {

   // Creating ~c_R_bar
   new Particle("~c_R_bar", -2000004, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-2000004));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-4}));

   // Creating ~s_R_bar
   new Particle("~s_R_bar", -2000003, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000003));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-3}));

   // Creating ~u_R_bar
   new Particle("~u_R_bar", -2000002, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000002));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-2}));

   // Creating ~d_R_bar
   new Particle("~d_R_bar", -2000001, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2000001));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-1}));

   // Creating ~chi_2-
   new Particle("~chi_2-", -1000037, 0, "Sparticle", 100, -1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000037));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-11,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-13,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-15,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-12,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-14,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-16,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-5,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-2,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-4,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,3,-4}));

   // Creating ~chi_1-
   new Particle("~chi_1-", -1000024, 0, "Sparticle", 100, -1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000024));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,3,-4}));

   // Creating ~nu_tauL_bar
   new Particle("~nu_tauL_bar", -1000016, 0, "Sparticle", 100, 0, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000016));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,-37}));

   // Creating ~tau_1+
   new Particle("~tau_1+", -1000015, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000015));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,37}));

   // Creating ~nu_muL_bar
   new Particle("~nu_muL_bar", -1000014, 0, "Sparticle", 100, 0, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000014));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,-37}));

   // Creating ~mu_L+
   new Particle("~mu_L+", -1000013, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000013));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,37}));

   // Creating ~nu_eL_bar
   new Particle("~nu_eL_bar", -1000012, 0, "Sparticle", 100, 0, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000012));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,-37}));

   // Creating ~e_L+
   new Particle("~e_L+", -1000011, 0, "Sparticle", 100, 1, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000011));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,37}));

   // Creating ~t_1_bar
   new Particle("~t_1_bar", -1000006, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000006));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,15,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-16,-5}));

   // Creating ~b_1_bar
   new Particle("~b_1_bar", -1000005, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000005));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-5}));

   // Creating ~c_L_bar
   new Particle("~c_L_bar", -1000004, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000004));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-4}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0003() {

   // Creating ~s_L_bar
   new Particle("~s_L_bar", -1000003, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-1000003));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-3}));

   // Creating ~u_L_bar
   new Particle("~u_L_bar", -1000002, 0, "Sparticle", 100, -0.666667, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000002));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-2}));

   // Creating ~d_L_bar
   new Particle("~d_L_bar", -1000001, 0, "Sparticle", 100, 0.333333, 500, 1, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1000001));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-1}));

   // Creating phi(1680)_bar
   new Particle("phi(1680)_bar", -100333, 0, "Unknown", 100, 0, 1.68, 0.15, 100, 100, 0, 100, 1);

   // Creating eta(1475)_bar
   new Particle("eta(1475)_bar", -100331, 0, "Unknown", 100, 0, 1.476, 0.085, 100, 100, 0, 100, 1);

   // Creating k2_star(1980)-_bar
   new Particle("k2_star(1980)-_bar", -100325, 0, "Unknown", 100, -1, 1.973, 0.373, 100, 100, 0, 100, 1);

   // Creating k_star(1410)-_bar
   new Particle("k_star(1410)-_bar", -100323, 0, "Unknown", 100, -1, 1.414, 0.232, 100, 100, 0, 100, 1);

   // Creating k(1460)-_bar
   new Particle("k(1460)-_bar", -100321, 0, "Unknown", 100, -1, 1.46, 0.26, 100, 100, 0, 100, 1);

   // Creating k2_star(1980)0_bar
   new Particle("k2_star(1980)0_bar", -100315, 0, "Unknown", 100, 0, 1.973, 0.373, 100, 100, 0, 100, 1);

   // Creating k_star(1410)0_bar
   new Particle("k_star(1410)0_bar", -100313, 0, "Unknown", 100, 0, 1.414, 0.232, 100, 100, 0, 100, 1);

   // Creating k(1460)0_bar
   new Particle("k(1460)0_bar", -100311, 0, "Unknown", 100, 0, 1.46, 0.26, 100, 100, 0, 100, 1);

   // Creating omega(1420)_bar
   new Particle("omega(1420)_bar", -100223, 0, "Unknown", 100, 0, 1.425, 0.215, 100, 100, 0, 100, 1);

   // Creating eta(1295)_bar
   new Particle("eta(1295)_bar", -100221, 0, "Unknown", 100, 0, 1.294, 0.055, 100, 100, 0, 100, 1);

   // Creating rho(1450)-_bar
   new Particle("rho(1450)-_bar", -100213, 0, "Unknown", 100, -1, 1.465, 0.4, 100, 100, 0, 100, 1);

   // Creating pi(1300)-_bar
   new Particle("pi(1300)-_bar", -100211, 0, "Unknown", 100, -1, 1.3, 0.4, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0004() {

   // Creating rho(1450)0_bar
   new Particle("rho(1450)0_bar", -100113, 0, "Unknown", 100, 0, 1.465, 0.4, 100, 100, 0, 100, 1);

   // Creating pi(1300)0_bar
   new Particle("pi(1300)0_bar", -100111, 0, "Unknown", 100, 0, 1.3, 0.4, 100, 100, 0, 100, 1);

   // Creating lambda(1810)_bar
   new Particle("lambda(1810)_bar", -53122, 0, "Unknown", 100, 0, 1.81, 0.15, 100, 100, 0, 100, 1);

   // Creating N(2090)+_bar
   new Particle("N(2090)+_bar", -52214, 0, "Unknown", 100, -1, 2.08, 0.35, 100, 100, 0, 100, 1);

   // Creating N(2090)0_bar
   new Particle("N(2090)0_bar", -52114, 0, "Unknown", 100, 0, 2.08, 0.35, 100, 100, 0, 100, 1);

   // Creating lambda(1800)_bar
   new Particle("lambda(1800)_bar", -43122, 0, "Unknown", 100, 0, 1.8, 0.3, 100, 100, 0, 100, 1);

   // Creating N(1710)+_bar
   new Particle("N(1710)+_bar", -42212, 0, "Unknown", 100, -1, 1.71, 0.1, 100, 100, 0, 100, 1);

   // Creating N(1900)+_bar
   new Particle("N(1900)+_bar", -42124, 0, "Unknown", 100, -1, 1.9, 0.5, 100, 100, 0, 100, 1);

   // Creating N(1710)0_bar
   new Particle("N(1710)0_bar", -42112, 0, "Unknown", 100, 0, 1.71, 0.1, 100, 100, 0, 100, 1);

   // Creating N(1900)0_bar
   new Particle("N(1900)0_bar", -41214, 0, "Unknown", 100, 0, 1.9, 0.5, 100, 100, 0, 100, 1);

   // Creating xi(1950)0_bar
   new Particle("xi(1950)0_bar", -33324, 0, "Unknown", 100, 0, 1.95, 0.06, 100, 100, 0, 100, 1);

   // Creating xi(1950)-_bar
   new Particle("xi(1950)-_bar", -33314, 0, "Unknown", 100, 1, 1.95, 0.06, 100, 100, 0, 100, 1);

   // Creating lambda(1670)_bar
   new Particle("lambda(1670)_bar", -33122, 0, "Unknown", 100, 0, 1.67, 0.035, 100, 100, 0, 100, 1);

   // Creating delta(1600)++_bar
   new Particle("delta(1600)++_bar", -32224, 0, "Unknown", 100, -2, 1.6, 0.35, 100, 100, 0, 100, 1);

   // Creating delta(1600)+_bar
   new Particle("delta(1600)+_bar", -32214, 0, "Unknown", 100, -1, 1.6, 0.35, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0005() {

   // Creating N(1650)+_bar
   new Particle("N(1650)+_bar", -32212, 0, "Unknown", 100, -1, 1.655, 0.165, 100, 100, 0, 100, 1);

   // Creating N(1720)+_bar
   new Particle("N(1720)+_bar", -32124, 0, "Unknown", 100, -1, 1.72, 0.2, 100, 100, 0, 100, 1);

   // Creating delta(1600)0_bar
   new Particle("delta(1600)0_bar", -32114, 0, "Unknown", 100, 0, 1.6, 0.35, 100, 100, 0, 100, 1);

   // Creating N(1650)0_bar
   new Particle("N(1650)0_bar", -32112, 0, "Unknown", 100, 0, 1.655, 0.165, 100, 100, 0, 100, 1);

   // Creating N(1720)0_bar
   new Particle("N(1720)0_bar", -31214, 0, "Unknown", 100, 0, 1.72, 0.2, 100, 100, 0, 100, 1);

   // Creating delta(1600)-_bar
   new Particle("delta(1600)-_bar", -31114, 0, "Unknown", 100, 1, 1.6, 0.35, 100, 100, 0, 100, 1);

   // Creating k_star(1680)-_bar
   new Particle("k_star(1680)-_bar", -30323, 0, "Unknown", 100, -1, 1.717, 0.32, 100, 100, 0, 100, 1);

   // Creating k_star(1680)0_bar
   new Particle("k_star(1680)0_bar", -30313, 0, "Unknown", 100, 0, 1.717, 0.32, 100, 100, 0, 100, 1);

   // Creating omega(1650)_bar
   new Particle("omega(1650)_bar", -30223, 0, "Unknown", 100, 0, 1.67, 0.315, 100, 100, 0, 100, 1);

   // Creating rho(1700)-_bar
   new Particle("rho(1700)-_bar", -30213, 0, "Unknown", 100, -1, 1.72, 0.25, 100, 100, 0, 100, 1);

   // Creating rho(1700)0_bar
   new Particle("rho(1700)0_bar", -30113, 0, "Unknown", 100, 0, 1.72, 0.25, 100, 100, 0, 100, 1);

   // Creating xi(1690)0_bar
   new Particle("xi(1690)0_bar", -23324, 0, "Unknown", 100, 0, 1.69, 0.05, 100, 100, 0, 100, 1);

   // Creating xi(1690)-_bar
   new Particle("xi(1690)-_bar", -23314, 0, "Unknown", 100, 1, 1.69, 0.05, 100, 100, 0, 100, 1);

   // Creating sigma(1940)+_bar
   new Particle("sigma(1940)+_bar", -23224, 0, "Unknown", 100, -1, 1.94, 0.22, 100, 100, 0, 100, 1);

   // Creating sigma(1750)+_bar
   new Particle("sigma(1750)+_bar", -23222, 0, "Unknown", 100, -1, 1.75, 0.09, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0006() {

   // Creating sigma(1940)0_bar
   new Particle("sigma(1940)0_bar", -23214, 0, "Unknown", 100, 0, 1.94, 0.22, 100, 100, 0, 100, 1);

   // Creating sigma(1750)0_bar
   new Particle("sigma(1750)0_bar", -23212, 0, "Unknown", 100, 0, 1.75, 0.09, 100, 100, 0, 100, 1);

   // Creating lambda(2110)_bar
   new Particle("lambda(2110)_bar", -23126, 0, "Unknown", 100, 0, 2.11, 0.2, 100, 100, 0, 100, 1);

   // Creating lambda(1890)_bar
   new Particle("lambda(1890)_bar", -23124, 0, "Unknown", 100, 0, 1.89, 0.1, 100, 100, 0, 100, 1);

   // Creating lambda(1600)_bar
   new Particle("lambda(1600)_bar", -23122, 0, "Unknown", 100, 0, 1.6, 0.15, 100, 100, 0, 100, 1);

   // Creating sigma(1940)-_bar
   new Particle("sigma(1940)-_bar", -23114, 0, "Unknown", 100, 1, 1.94, 0.22, 100, 100, 0, 100, 1);

   // Creating sigma(1750)-_bar
   new Particle("sigma(1750)-_bar", -23112, 0, "Unknown", 100, 1, 1.75, 0.09, 100, 100, 0, 100, 1);

   // Creating delta(1920)++_bar
   new Particle("delta(1920)++_bar", -22224, 0, "Unknown", 100, -2, 1.92, 0.2, 100, 100, 0, 100, 1);

   // Creating delta(1910)++_bar
   new Particle("delta(1910)++_bar", -22222, 0, "Unknown", 100, -2, 1.91, 0.25, 100, 100, 0, 100, 1);

   // Creating delta(1920)+_bar
   new Particle("delta(1920)+_bar", -22214, 0, "Unknown", 100, -1, 1.92, 0.2, 100, 100, 0, 100, 1);

   // Creating N(1535)+_bar
   new Particle("N(1535)+_bar", -22212, 0, "Unknown", 100, -1, 1.535, 0.15, 100, 100, 0, 100, 1);

   // Creating N(1700)+_bar
   new Particle("N(1700)+_bar", -22124, 0, "Unknown", 100, -1, 1.7, 0.1, 100, 100, 0, 100, 1);

   // Creating delta(1910)+_bar
   new Particle("delta(1910)+_bar", -22122, 0, "Unknown", 100, -1, 1.91, 0.25, 100, 100, 0, 100, 1);

   // Creating delta(1920)0_bar
   new Particle("delta(1920)0_bar", -22114, 0, "Unknown", 100, 0, 1.92, 0.2, 100, 100, 0, 100, 1);

   // Creating N(1535)0_bar
   new Particle("N(1535)0_bar", -22112, 0, "Unknown", 100, 0, 1.535, 0.15, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0007() {

   // Creating N(1700)0_bar
   new Particle("N(1700)0_bar", -21214, 0, "Unknown", 100, 0, 1.7, 0.1, 100, 100, 0, 100, 1);

   // Creating delta(1910)0_bar
   new Particle("delta(1910)0_bar", -21212, 0, "Unknown", 100, 0, 1.91, 0.25, 100, 100, 0, 100, 1);

   // Creating delta(1920)-_bar
   new Particle("delta(1920)-_bar", -21114, 0, "Unknown", 100, 1, 1.92, 0.2, 100, 100, 0, 100, 1);

   // Creating delta(1910)-_bar
   new Particle("delta(1910)-_bar", -21112, 0, "Unknown", 100, 1, 1.91, 0.25, 100, 100, 0, 100, 1);

   // Creating B*_1c-
   new Particle("B*_1c-", -20543, 0, "Unknown", 100, -1, 7.3, 0.05, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-20543));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-513,-411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-523,-421}));

   // Creating B*_1s0_bar
   new Particle("B*_1s0_bar", -20533, 0, "Unknown", 100, 0, 6.02, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20533));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-523,321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-513,311}));

   // Creating B*_1-
   new Particle("B*_1-", -20523, 0, "Unknown", 100, -1, 5.78, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20523));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-513,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-523,-111}));

   // Creating B*_10_bar
   new Particle("B*_10_bar", -20513, 0, "Unknown", 100, 0, 5.78, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20513));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-523,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-513,-111}));

   // Creating D*_1s-
   new Particle("D*_1s-", -20433, 0, "Unknown", 100, -1, 2.4596, 0.0055, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20433));
   part->AddDecay(Particle::Decay(0, 0.8,  vector<int>{-433,-111}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-433,-22}));

   // Creating D*_10_bar
   new Particle("D*_10_bar", -20423, 0, "Unknown", 100, 0, 2.372, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20423));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-413,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-423,-111}));

   // Creating D*_1-
   new Particle("D*_1-", -20413, 0, "Unknown", 100, -1, 2.372, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20413));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-423,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-413,-111}));

   // Creating K*_1-
   new Particle("K*_1-", -20323, 0, "Unknown", 100, -1, 1.403, 0.174, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20323));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-313,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-323,-111}));

   // Creating K*_10_bar
   new Particle("K*_10_bar", -20313, 0, "Unknown", 100, 0, 1.403, 0.174, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20313));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-323,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-313,-111}));

   // Creating a_1-
   new Particle("a_1-", -20213, 0, "Unknown", 100, -1, 1.23, 0.4, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-20213));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-113,-211}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-213,-111}));

   // Creating xi(2030)0_bar
   new Particle("xi(2030)0_bar", -13326, 0, "Unknown", 100, 0, 2.025, 0.02, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0008() {

   // Creating xi(1820)0_bar
   new Particle("xi(1820)0_bar", -13324, 0, "Unknown", 100, 0, 1.823, 0.024, 100, 100, 0, 100, 1);

   // Creating xi(2030)-_bar
   new Particle("xi(2030)-_bar", -13316, 0, "Unknown", 100, 1, 2.025, 0.02, 100, 100, 0, 100, 1);

   // Creating xi(1820)-_bar
   new Particle("xi(1820)-_bar", -13314, 0, "Unknown", 100, 1, 1.823, 0.024, 100, 100, 0, 100, 1);

   // Creating sigma(1915)+_bar
   new Particle("sigma(1915)+_bar", -13226, 0, "Unknown", 100, -1, 1.915, 0.12, 100, 100, 0, 100, 1);

   // Creating sigma(1670)+_bar
   new Particle("sigma(1670)+_bar", -13224, 0, "Unknown", 100, -1, 1.67, 0.06, 100, 100, 0, 100, 1);

   // Creating sigma(1660)+_bar
   new Particle("sigma(1660)+_bar", -13222, 0, "Unknown", 100, -1, 1.66, 0.1, 100, 100, 0, 100, 1);

   // Creating sigma(1915)0_bar
   new Particle("sigma(1915)0_bar", -13216, 0, "Unknown", 100, 0, 1.915, 0.12, 100, 100, 0, 100, 1);

   // Creating sigma(1670)0_bar
   new Particle("sigma(1670)0_bar", -13214, 0, "Unknown", 100, 0, 1.67, 0.06, 100, 100, 0, 100, 1);

   // Creating sigma(1660)0_bar
   new Particle("sigma(1660)0_bar", -13212, 0, "Unknown", 100, 0, 1.66, 0.1, 100, 100, 0, 100, 1);

   // Creating lambda(1830)_bar
   new Particle("lambda(1830)_bar", -13126, 0, "Unknown", 100, 0, 1.83, 0.095, 100, 100, 0, 100, 1);

   // Creating lambda(1690)_bar
   new Particle("lambda(1690)_bar", -13124, 0, "Unknown", 100, 0, 1.69, 0.06, 100, 100, 0, 100, 1);

   // Creating lambda(1405)_bar
   new Particle("lambda(1405)_bar", -13122, 0, "Unknown", 100, 0, 1.4051, 0.05, 100, 100, 0, 100, 1);

   // Creating sigma(1915)-_bar
   new Particle("sigma(1915)-_bar", -13116, 0, "Unknown", 100, 1, 1.915, 0.12, 100, 100, 0, 100, 1);

   // Creating sigma(1670)-_bar
   new Particle("sigma(1670)-_bar", -13114, 0, "Unknown", 100, 1, 1.67, 0.06, 100, 100, 0, 100, 1);

   // Creating sigma(1660)-_bar
   new Particle("sigma(1660)-_bar", -13112, 0, "Unknown", 100, 1, 1.66, 0.1, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0009() {

   // Creating delta(1930)++_bar
   new Particle("delta(1930)++_bar", -12226, 0, "Unknown", 100, -2, 1.96, 0.36, 100, 100, 0, 100, 1);

   // Creating delta(1700)++_bar
   new Particle("delta(1700)++_bar", -12224, 0, "Unknown", 100, -2, 1.7, 0.3, 100, 100, 0, 100, 1);

   // Creating delta(1900)++_bar
   new Particle("delta(1900)++_bar", -12222, 0, "Unknown", 100, -2, 1.9, 0.2, 100, 100, 0, 100, 1);

   // Creating N(1990)+_bar
   new Particle("N(1990)+_bar", -12218, 0, "Unknown", 100, -1, 1.95, 0.555, 100, 100, 0, 100, 1);

   // Creating N(1680)+_bar
   new Particle("N(1680)+_bar", -12216, 0, "Unknown", 100, -1, 1.685, 0.13, 100, 100, 0, 100, 1);

   // Creating delta(1700)+_bar
   new Particle("delta(1700)+_bar", -12214, 0, "Unknown", 100, -1, 1.7, 0.3, 100, 100, 0, 100, 1);

   // Creating N(1440)+_bar
   new Particle("N(1440)+_bar", -12212, 0, "Unknown", 100, -1, 1.44, 0.3, 100, 100, 0, 100, 1);

   // Creating delta(1930)+_bar
   new Particle("delta(1930)+_bar", -12126, 0, "Unknown", 100, -1, 1.96, 0.36, 100, 100, 0, 100, 1);

   // Creating delta(1900)+_bar
   new Particle("delta(1900)+_bar", -12122, 0, "Unknown", 100, -1, 1.9, 0.2, 100, 100, 0, 100, 1);

   // Creating N(1990)0_bar
   new Particle("N(1990)0_bar", -12118, 0, "Unknown", 100, 0, 1.95, 0.555, 100, 100, 0, 100, 1);

   // Creating N(1680)0_bar
   new Particle("N(1680)0_bar", -12116, 0, "Unknown", 100, 0, 1.685, 0.13, 100, 100, 0, 100, 1);

   // Creating delta(1700)0_bar
   new Particle("delta(1700)0_bar", -12114, 0, "Unknown", 100, 0, 1.7, 0.3, 100, 100, 0, 100, 1);

   // Creating N(1440)0_bar
   new Particle("N(1440)0_bar", -12112, 0, "Unknown", 100, 0, 1.44, 0.3, 100, 100, 0, 100, 1);

   // Creating delta(1930)0_bar
   new Particle("delta(1930)0_bar", -11216, 0, "Unknown", 100, 0, 1.96, 0.36, 100, 100, 0, 100, 1);

   // Creating delta(1900)0_bar
   new Particle("delta(1900)0_bar", -11212, 0, "Unknown", 100, 0, 1.9, 0.2, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0010() {

   // Creating delta(1930)-_bar
   new Particle("delta(1930)-_bar", -11116, 0, "Unknown", 100, 1, 1.96, 0.36, 100, 100, 0, 100, 1);

   // Creating delta(1700)-_bar
   new Particle("delta(1700)-_bar", -11114, 0, "Unknown", 100, 1, 1.7, 0.3, 100, 100, 0, 100, 1);

   // Creating delta(1900)-_bar
   new Particle("delta(1900)-_bar", -11112, 0, "Unknown", 100, 1, 1.9, 0.2, 100, 100, 0, 100, 1);

   // Creating B_1c-
   new Particle("B_1c-", -10543, 0, "Unknown", 100, -1, 7.3, 0.05, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-10543));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-513,-411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-523,-421}));

   // Creating B*_0c-
   new Particle("B*_0c-", -10541, 0, "Unknown", 100, -1, 7.25, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10541));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-511,-411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-521,-421}));

   // Creating B_1s0_bar
   new Particle("B_1s0_bar", -10533, 0, "Unknown", 100, 0, 5.97, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10533));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-523,321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-513,311}));

   // Creating B*_0s0_bar
   new Particle("B*_0s0_bar", -10531, 0, "Unknown", 100, 0, 5.92, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10531));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-521,321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-511,311}));

   // Creating B_1-
   new Particle("B_1-", -10523, 0, "Unknown", 100, -1, 5.73, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10523));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-513,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-523,-111}));

   // Creating B*_0-
   new Particle("B*_0-", -10521, 0, "Unknown", 100, -1, 5.68, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10521));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-511,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-521,-111}));

   // Creating B_10_bar
   new Particle("B_10_bar", -10513, 0, "Unknown", 100, 0, 5.73, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10513));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-523,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-513,-111}));

   // Creating B*_00_bar
   new Particle("B*_00_bar", -10511, 0, "Unknown", 100, 0, 5.68, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10511));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-521,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-511,-111}));

   // Creating D_1s-
   new Particle("D_1s-", -10433, 0, "Unknown", 100, -1, 2.5353, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10433));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-423,-321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-413,-311}));

   // Creating D*_0s-
   new Particle("D*_0s-", -10431, 0, "Unknown", 100, -1, 2.3178, 0.0046, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10431));
   part->AddDecay(Particle::Decay(0, 0.8,  vector<int>{-431,-111}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-431,-22}));

   // Creating D_10_bar
   new Particle("D_10_bar", -10423, 0, "Unknown", 100, 0, 2.4223, 0.02, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10423));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-413,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-423,-111}));

   // Creating D*_00_bar
   new Particle("D*_00_bar", -10421, 0, "Unknown", 100, 0, 2.272, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10421));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-411,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-421,-111}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0011() {

   // Creating D_1-
   new Particle("D_1-", -10413, 0, "Unknown", 100, -1, 2.424, 0.02, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-10413));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-423,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-413,-111}));

   // Creating D*_0-
   new Particle("D*_0-", -10411, 0, "Unknown", 100, -1, 2.272, 0.05, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10411));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-421,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-411,-111}));

   // Creating eta2(1870)_bar
   new Particle("eta2(1870)_bar", -10335, 0, "Unknown", 100, 0, 1.842, 0.225, 100, 100, 0, 100, 1);

   // Creating k2(1770)-_bar
   new Particle("k2(1770)-_bar", -10325, 0, "Unknown", 100, -1, 1.773, 0.186, 100, 100, 0, 100, 1);

   // Creating K_1-
   new Particle("K_1-", -10323, 0, "Unknown", 100, -1, 1.272, 0.09, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10323));
   part->AddDecay(Particle::Decay(0, 0.313,  vector<int>{-313,-211}));
   part->AddDecay(Particle::Decay(0, 0.28,  vector<int>{-311,-213}));
   part->AddDecay(Particle::Decay(0, 0.157,  vector<int>{-323,-111}));
   part->AddDecay(Particle::Decay(0, 0.14,  vector<int>{-321,-113}));
   part->AddDecay(Particle::Decay(0, 0.11,  vector<int>{-321,-223}));

   // Creating K*_0-
   new Particle("K*_0-", -10321, 0, "Unknown", 100, -1, 1.42, 0.287, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10321));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-311,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-321,-111}));

   // Creating k2(1770)0_bar
   new Particle("k2(1770)0_bar", -10315, 0, "Unknown", 100, 0, 1.773, 0.186, 100, 100, 0, 100, 1);

   // Creating K_10_bar
   new Particle("K_10_bar", -10313, 0, "Unknown", 100, 0, 1.272, 0.09, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10313));
   part->AddDecay(Particle::Decay(0, 0.313,  vector<int>{-323,211}));
   part->AddDecay(Particle::Decay(0, 0.28,  vector<int>{-321,213}));
   part->AddDecay(Particle::Decay(0, 0.157,  vector<int>{-313,-111}));
   part->AddDecay(Particle::Decay(0, 0.14,  vector<int>{-311,-113}));
   part->AddDecay(Particle::Decay(0, 0.11,  vector<int>{-311,-223}));

   // Creating K*_00_bar
   new Particle("K*_00_bar", -10311, 0, "Unknown", 100, 0, 1.42, 0.287, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10311));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-321,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-311,-111}));

   // Creating eta2(1645)_bar
   new Particle("eta2(1645)_bar", -10225, 0, "Unknown", 100, 0, 1.617, 0.181, 100, 100, 0, 100, 1);

   // Creating pi2(1670)-_bar
   new Particle("pi2(1670)-_bar", -10215, 0, "Unknown", 100, -1, 1.6722, 0.26, 100, 100, 0, 100, 1);

   // Creating b_1-
   new Particle("b_1-", -10213, 0, "Unknown", 100, -1, 1.2295, 0.142, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10213));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-223,-211}));

   // Creating a_0-
   new Particle("a_0-", -10211, 0, "Unknown", 100, -1, 0.9835, 0.06, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-10211));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-221,-211}));

   // Creating pi2(1670)0_bar
   new Particle("pi2(1670)0_bar", -10115, 0, "Unknown", 100, 0, 1.6722, 0.26, 100, 100, 0, 100, 1);

   // Creating Omega*_bbb+
   new Particle("Omega*_bbb+", -5554, 0, "B-Baryon", 100, 1, 15.1106, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5554));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0012() {

   // Creating Omega*_bbc0_bar
   new Particle("Omega*_bbc0_bar", -5544, 0, "B-Baryon", 100, 0, 11.7115, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-5544));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega_bbc0_bar
   new Particle("Omega_bbc0_bar", -5542, 0, "B-Baryon", 100, 0, 11.7077, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5542));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega*_bb+
   new Particle("Omega*_bb+", -5534, 0, "B-Baryon", 100, 1, 10.6143, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5534));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega_bb+
   new Particle("Omega_bb+", -5532, 0, "B-Baryon", 100, 1, 10.6021, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5532));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi*_bb0_bar
   new Particle("Xi*_bb0_bar", -5524, 0, "B-Baryon", 100, 0, 10.4414, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5524));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi_bb0_bar
   new Particle("Xi_bb0_bar", -5522, 0, "B-Baryon", 100, 0, 10.4227, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5522));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi*_bb+
   new Particle("Xi*_bb+", -5514, 0, "B-Baryon", 100, 1, 10.4414, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5514));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi_bb+
   new Particle("Xi_bb+", -5512, 0, "Unknown", 100, 1, 10.4227, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5512));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating bb_1_bar
   new Particle("bb_1_bar", -5503, 0, "Unknown", 100, 0.666667, 10.0735, 0, 100, 100, 1, 100, 1);

   // Creating Omega*_bcc-
   new Particle("Omega*_bcc-", -5444, 0, "B-Baryon", 100, -1, 8.31325, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5444));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega_bcc+_bar
   new Particle("Omega_bcc+_bar", -5442, 0, "B-Baryon", 100, -1, 8.30945, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5442));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega*_bc0_bar
   new Particle("Omega*_bc0_bar", -5434, 0, "B-Baryon", 100, 0, 7.219, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5434));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega'_bc0_bar
   new Particle("Omega'_bc0_bar", -5432, 0, "B-Baryon", 100, 0, 7.21101, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5432));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi*_bc-
   new Particle("Xi*_bc-", -5424, 0, "B-Baryon", 100, -1, 7.0485, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5424));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi'_bc-
   new Particle("Xi'_bc-", -5422, 0, "B-Baryon", 100, -1, 7.03724, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5422));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0013() {

   // Creating Xi*_bc0_bar
   new Particle("Xi*_bc0_bar", -5414, 0, "B-Baryon", 100, 0, 7.0485, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-5414));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi'_bc0_bar
   new Particle("Xi'_bc0_bar", -5412, 0, "B-Baryon", 100, 0, 7.03724, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5412));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating bc_1_bar
   new Particle("bc_1_bar", -5403, 0, "Unknown", 100, -0.333333, 6.67397, 0, 100, 100, 1, 100, 1);

   // Creating bc_0_bar
   new Particle("bc_0_bar", -5401, 0, "Unknown", 100, -0.333333, 6.67143, 0, 100, 100, 1, 100, 1);

   // Creating Omega_bc0_bar
   new Particle("Omega_bc0_bar", -5342, 0, "B-Baryon", 100, 0, 7.19099, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5342));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Omega*_b+
   new Particle("Omega*_b+", -5334, 0, "B-Baryon", 100, 1, 6.13, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5334));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5332,-22}));

   // Creating Omega_b+
   new Particle("Omega_b+", -5332, 0, "B-Baryon", 100, 1, 6.12, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5332));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi*_b0_bar
   new Particle("Xi*_b0_bar", -5324, 0, "B-Baryon", 100, 0, 5.97, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5324));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5232,-22}));

   // Creating Xi'_b0_bar
   new Particle("Xi'_b0_bar", -5322, 0, "B-Baryon", 100, 0, 5.96, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5322));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5232,-22}));

   // Creating Xi*_b+
   new Particle("Xi*_b+", -5314, 0, "B-Baryon", 100, 1, 5.97, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5314));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5132,-22}));

   // Creating Xi'_b+
   new Particle("Xi'_b+", -5312, 0, "B-Baryon", 100, 1, 5.96, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5312));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5132,-22}));

   // Creating bs_1_bar
   new Particle("bs_1_bar", -5303, 0, "Unknown", 100, 0.666667, 5.57536, 0, 100, 100, 1, 100, 1);

   // Creating bs_0_bar
   new Particle("bs_0_bar", -5301, 0, "Unknown", 100, 0.666667, 5.56725, 0, 100, 100, 1, 100, 1);

   // Creating Xi_bc-
   new Particle("Xi_bc-", -5242, 0, "B-Baryon", 100, -1, 7.00575, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5242));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi_b0_bar
   new Particle("Xi_b0_bar", -5232, 0, "B-Baryon", 100, 0, 5.7924, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5232));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0014() {

   // Creating Sigma*_b+_bar
   new Particle("Sigma*_b+_bar", -5224, 0, "B-Baryon", 100, -1, 5.829, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-5224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,-211}));

   // Creating Sigma_b+_bar
   new Particle("Sigma_b+_bar", -5222, 0, "B-Baryon", 100, -1, 5.8078, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5222));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,-211}));

   // Creating Sigma*_b0_bar
   new Particle("Sigma*_b0_bar", -5214, 0, "B-Baryon", 100, 0, 5.829, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5214));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,-111}));

   // Creating Sigma_b0_bar
   new Particle("Sigma_b0_bar", -5212, 0, "B-Baryon", 100, 0, 5.8078, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,-111}));

   // Creating bu_1_bar
   new Particle("bu_1_bar", -5203, 0, "Unknown", 100, -0.333333, 5.40145, 0, 100, 100, 1, 100, 1);

   // Creating bu_0_bar
   new Particle("bu_0_bar", -5201, 0, "Unknown", 100, -0.333333, 5.38897, 0, 100, 100, 1, 100, 1);

   // Creating Xi_bc0_bar
   new Particle("Xi_bc0_bar", -5142, 0, "B-Baryon", 100, 0, 7.00575, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5142));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Xi_b+
   new Particle("Xi_b+", -5132, 0, "B-Baryon", 100, 1, 5.7924, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5132));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating Lambda_b0_bar
   new Particle("Lambda_b0_bar", -5122, 0, "B-Baryon", 100, 0, 5.6202, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5122));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{2,-1,-4,-2101}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4122}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4122}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{2,-4,-1,-2101}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{4,-3,-4,-2101}));
   part->AddDecay(Particle::Decay(0, 0.0435,  vector<int>{-4122,433}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4122}));
   part->AddDecay(Particle::Decay(0, 0.0285,  vector<int>{-4122,431}));
   part->AddDecay(Particle::Decay(0, 0.0235,  vector<int>{-4122,20213}));
   part->AddDecay(Particle::Decay(0, 0.02,  vector<int>{-4122,213}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{4,-4,-3,-2101}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-2101}));
   part->AddDecay(Particle::Decay(0, 0.0077,  vector<int>{-4122,211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-2101}));
   part->AddDecay(Particle::Decay(0, 0.0044,  vector<int>{-20443,-3122}));
   part->AddDecay(Particle::Decay(0, 0.0022,  vector<int>{-443,-3122}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{-441,-3122}));

   // Creating Sigma*_b-_bar
   new Particle("Sigma*_b-_bar", -5114, 0, "B-Baryon", 100, 1, 5.8364, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,211}));

   // Creating Sigma_b-_bar
   new Particle("Sigma_b-_bar", -5112, 0, "B-Baryon", 100, 1, 5.8152, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5112));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-5122,211}));

   // Creating bd_1_bar
   new Particle("bd_1_bar", -5103, 0, "Unknown", 100, 0.666667, 5.40145, 0, 100, 100, 1, 100, 1);

   // Creating bd_0_bar
   new Particle("bd_0_bar", -5101, 0, "Unknown", 100, 0.666667, 5.38897, 0, 100, 100, 1, 100, 1);

   // Creating Omega*_ccc--
   new Particle("Omega*_ccc--", -4444, 0, "CharmedBaryon", 100, -2, 4.91594, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4444));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Omega*_cc-
   new Particle("Omega*_cc-", -4434, 0, "CharmedBaryon", 100, -1, 3.82466, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4434));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0015() {

   // Creating Omega_cc-
   new Particle("Omega_cc-", -4432, 0, "CharmedBaryon", 100, -1, 3.78663, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-4432));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Xi*_cc--
   new Particle("Xi*_cc--", -4424, 0, "CharmedBaryon", 100, -2, 3.65648, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4424));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Xi_cc--
   new Particle("Xi_cc--", -4422, 0, "CharmedBaryon", 100, -2, 3.59798, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4422));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Xi*_cc-
   new Particle("Xi*_cc-", -4414, 0, "CharmedBaryon", 100, -1, 3.65648, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4414));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Xi_cc-
   new Particle("Xi_cc-", -4412, 0, "CharmedBaryon", 100, -1, 3.59798, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4412));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating cc_1_bar
   new Particle("cc_1_bar", -4403, 0, "Unknown", 100, -1.33333, 3.27531, 0, 100, 100, 1, 100, 1);

   // Creating Omega*_c0_bar
   new Particle("Omega*_c0_bar", -4334, 0, "CharmedBaryon", 100, 0, 2.7683, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4334));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4332,-22}));

   // Creating Omega_c0_bar
   new Particle("Omega_c0_bar", -4332, 0, "CharmedBaryon", 100, 0, 2.6975, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4332));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Xi*_c-
   new Particle("Xi*_c-", -4324, 0, "CharmedBaryon", 100, -1, 2.6466, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4324));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-4232,-111}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-4232,-22}));

   // Creating Xi'_c-
   new Particle("Xi'_c-", -4322, 0, "CharmedBaryon", 100, -1, 2.5757, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4322));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4232,-22}));

   // Creating Xi*_c0_bar
   new Particle("Xi*_c0_bar", -4314, 0, "CharmedBaryon", 100, 0, 2.6461, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4314));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-4132,-111}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-4132,-22}));

   // Creating Xi'_c0_bar
   new Particle("Xi'_c0_bar", -4312, 0, "CharmedBaryon", 100, 0, 2.578, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4312));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4132,-22}));

   // Creating cs_1_bar
   new Particle("cs_1_bar", -4303, 0, "Unknown", 100, -0.333333, 2.17967, 0, 100, 100, 1, 100, 1);

   // Creating cs_0_bar
   new Particle("cs_0_bar", -4301, 0, "CharmedBaryon", 100, -0.333333, 2.15432, 0, 100, 100, 1, 100, 1);

   // Creating Xi_c-
   new Particle("Xi_c-", -4232, 0, "CharmedBaryon", 100, -1, 2.4679, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4232));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0016() {

   // Creating Sigma*_c--
   new Particle("Sigma*_c--", -4224, 0, "CharmedBaryon", 100, -2, 2.5184, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-4224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,-211}));

   // Creating Sigma_c--
   new Particle("Sigma_c--", -4222, 0, "CharmedBaryon", 100, -2, 2.45402, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4222));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,-211}));

   // Creating Sigma*_c-
   new Particle("Sigma*_c-", -4214, 0, "CharmedBaryon", 100, -1, 2.5175, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4214));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,-111}));

   // Creating Sigma_c-
   new Particle("Sigma_c-", -4212, 0, "CharmedBaryon", 100, -1, 2.4529, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,-111}));

   // Creating cu_1_bar
   new Particle("cu_1_bar", -4203, 0, "Unknown", 100, -1.33333, 2.00808, 0, 100, 100, 1, 100, 1);

   // Creating cu_0_bar
   new Particle("cu_0_bar", -4201, 0, "Unknown", 100, -1.33333, 1.96908, 0, 100, 100, 1, 100, 1);

   // Creating Xi_c0_bar
   new Particle("Xi_c0_bar", -4132, 0, "CharmedBaryon", 100, 0, 2.471, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4132));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating Lambda_c-
   new Particle("Lambda_c-", -4122, 0, "CharmedBaryon", 100, -1, 2.28646, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4122));
   part->AddDecay(Particle::Decay(13, 0.2432,  vector<int>{-2,1,-3,-2101}));
   part->AddDecay(Particle::Decay(13, 0.15,  vector<int>{-3,-2203}));
   part->AddDecay(Particle::Decay(13, 0.075,  vector<int>{-2,-3201}));
   part->AddDecay(Particle::Decay(13, 0.075,  vector<int>{-2,-3203}));
   part->AddDecay(Particle::Decay(13, 0.057,  vector<int>{-2,1,-3,-2103}));
   part->AddDecay(Particle::Decay(13, 0.035,  vector<int>{-2,1,-1,-2101}));
   part->AddDecay(Particle::Decay(13, 0.035,  vector<int>{-2,3,-3,-2101}));
   part->AddDecay(Particle::Decay(13, 0.03,  vector<int>{-1,-2203}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{-2224,323}));
   part->AddDecay(Particle::Decay(42, 0.018,  vector<int>{13,-14,-3122}));
   part->AddDecay(Particle::Decay(42, 0.018,  vector<int>{11,-12,-3122}));
   part->AddDecay(Particle::Decay(0, 0.016,  vector<int>{-2212,311}));
   part->AddDecay(Particle::Decay(13, 0.015,  vector<int>{-2,-2101}));
   part->AddDecay(Particle::Decay(13, 0.015,  vector<int>{-2,-2103}));
   part->AddDecay(Particle::Decay(0, 0.0088,  vector<int>{-2212,313}));
   part->AddDecay(Particle::Decay(0, 0.0066,  vector<int>{-2224,321}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{13,-14,-2212,211}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{13,-14,-2112,-111}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{11,-12,-2112,-111}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{11,-12,-2212,211}));
   part->AddDecay(Particle::Decay(0, 0.0058,  vector<int>{-3122,-211}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{-3212,-211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,-3212}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-2214,311}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-2214,313}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,-3212}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-3122,-213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,-3214}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-3122,-321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-3122,-323}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,-3214}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3212,-213}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3214,-211}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3214,-213}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3222,-111}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3222,-113}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-3222,-223}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-3224,-113}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-3224,-223}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-2112,-211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-2112,-213}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-2114,-211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-2114,-213}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{13,-14,-2112}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{11,-12,-2112}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-3224,-111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3322,-321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3212,-321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3212,-323}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3222,-311}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3222,-313}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3322,-323}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3324,-321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-2212,-111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-2212,-113}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-2212,-223}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,-2114}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3222,-221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3224,-221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-3222,-331}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,-2114}));
   part->AddDecay(Particle::Decay(0, 0.0018,  vector<int>{-2212,-10221}));
   part->AddDecay(Particle::Decay(0, 0.0013,  vector<int>{-2212,-333}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2224,213}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-3224,-311}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-3224,-313}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2224,211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2212,-221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2212,-331}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2214,-111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2214,-221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2214,-331}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2214,-113}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-3214,-321}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-3214,-323}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2214,-223}));

   // Creating Sigma*_c0_bar
   new Particle("Sigma*_c0_bar", -4114, 0, "CharmedBaryon", 100, 0, 2.518, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,211}));

   // Creating Sigma_c0_bar
   new Particle("Sigma_c0_bar", -4112, 0, "CharmedBaryon", 100, 0, 2.45376, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4112));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-4122,211}));

   // Creating cd_1_bar
   new Particle("cd_1_bar", -4103, 0, "Unknown", 100, -0.333333, 2.00808, 0, 100, 100, 1, 100, 1);

   // Creating cd_0_bar
   new Particle("cd_0_bar", -4101, 0, "Unknown", 100, -0.333333, 1.96908, 0, 100, 100, 1, 100, 1);

   // Creating Omega+
   new Particle("Omega+", -3334, 0, "Baryon", 100, 1, 1.67245, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3334));
   part->AddDecay(Particle::Decay(0, 0.676,  vector<int>{-3122,321}));
   part->AddDecay(Particle::Decay(0, 0.234,  vector<int>{-3322,211}));
   part->AddDecay(Particle::Decay(0, 0.085,  vector<int>{-3312,-111}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{12,-11,-3322}));

   // Creating Xi*0_bar
   new Particle("Xi*0_bar", -3324, 0, "Baryon", 100, 0, 1.5318, 0.0091, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3324));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-3312,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-3322,-111}));

   // Creating Xi0_bar
   new Particle("Xi0_bar", -3322, 0, "Baryon", 100, 0, 1.31486, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3322));
   part->AddDecay(Particle::Decay(0, 0.9954,  vector<int>{-3122,-111}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{-3212,-22}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{-3122,-22}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0017() {

   // Creating Xi*+
   new Particle("Xi*+", -3314, 0, "Baryon", 100, 1, 1.535, 0.0099, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-3314));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{-3322,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-3312,-111}));

   // Creating Xi-_bar
   new Particle("Xi-_bar", -3312, 0, "Baryon", 100, 1, 1.32171, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3312));
   part->AddDecay(Particle::Decay(0, 0.9988,  vector<int>{-3122,211}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{12,-11,-3122}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{14,-13,-3122}));
   part->AddDecay(Particle::Decay(0, 0.0001,  vector<int>{-3112,-22}));
   part->AddDecay(Particle::Decay(0, 0.0001,  vector<int>{12,-11,-3212}));

   // Creating ss_1_bar
   new Particle("ss_1_bar", -3303, 0, "Unknown", 100, 0.666667, 2.08, 0, 100, 100, 1, 100, 1);

   // Creating sigma(2030)+_bar
   new Particle("sigma(2030)+_bar", -3228, 0, "Unknown", 100, -1, 2.03, 0.18, 100, 100, 0, 100, 1);

   // Creating sigma(1775)+_bar
   new Particle("sigma(1775)+_bar", -3226, 0, "Unknown", 100, -1, 1.775, 0.12, 100, 100, 0, 100, 1);

   // Creating Sigma*+_bar
   new Particle("Sigma*+_bar", -3224, 0, "Baryon", 100, -1, 1.3828, 0.0358, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3224));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{-3122,-211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3222,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3212,-211}));

   // Creating Sigma+_bar
   new Particle("Sigma+_bar", -3222, 0, "Baryon", 100, -1, 1.18937, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3222));
   part->AddDecay(Particle::Decay(0, 0.516,  vector<int>{-2212,-111}));
   part->AddDecay(Particle::Decay(0, 0.483,  vector<int>{-2112,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-2212,-22}));

   // Creating sigma(2030)0_bar
   new Particle("sigma(2030)0_bar", -3218, 0, "Unknown", 100, 0, 2.03, 0.18, 100, 100, 0, 100, 1);

   // Creating sigma(1775)0_bar
   new Particle("sigma(1775)0_bar", -3216, 0, "Unknown", 100, 0, 1.775, 0.12, 100, 100, 0, 100, 1);

   // Creating Sigma*0_bar
   new Particle("Sigma*0_bar", -3214, 0, "Baryon", 100, 0, 1.3837, 0.036, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3214));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{-3122,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3222,211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3112,-211}));

   // Creating Sigma0_bar
   new Particle("Sigma0_bar", -3212, 0, "Baryon", 100, 0, 1.19264, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-3122,-22}));

   // Creating su_1_bar
   new Particle("su_1_bar", -3203, 0, "Unknown", 100, -0.333333, 0.1064, 0, 100, 100, 1, 100, 1);

   // Creating su_0_bar
   new Particle("su_0_bar", -3201, 0, "Unknown", 100, -0.333333, 0.1064, 0, 100, 100, 1, 100, 1);

   // Creating lambda(2100)_bar
   new Particle("lambda(2100)_bar", -3128, 0, "Unknown", 100, 0, 2.1, 0.2, 100, 100, 0, 100, 1);

   // Creating lambda(1820)_bar
   new Particle("lambda(1820)_bar", -3126, 0, "Unknown", 100, 0, 1.82, 0.08, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0018() {

   // Creating lambda(1520)_bar
   new Particle("lambda(1520)_bar", -3124, 0, "Unknown", 100, 0, 1.5195, 0.0156, 100, 100, 0, 100, 1);

   // Creating Lambda0_bar
   new Particle("Lambda0_bar", -3122, 0, "Baryon", 100, 0, 1.11568, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-3122));
   part->AddDecay(Particle::Decay(0, 0.639,  vector<int>{-2212,211}));
   part->AddDecay(Particle::Decay(0, 0.358,  vector<int>{-2112,-111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-2112,-22}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{12,-11,-2212}));

   // Creating sigma(2030)-_bar
   new Particle("sigma(2030)-_bar", -3118, 0, "Unknown", 100, 1, 2.03, 0.18, 100, 100, 0, 100, 1);

   // Creating sigma(1775)-_bar
   new Particle("sigma(1775)-_bar", -3116, 0, "Unknown", 100, 1, 1.775, 0.12, 100, 100, 0, 100, 1);

   // Creating Sigma*-_bar
   new Particle("Sigma*-_bar", -3114, 0, "Baryon", 100, 1, 1.3872, 0.0394, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3114));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{-3122,211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3212,211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-3112,-111}));

   // Creating Sigma-_bar
   new Particle("Sigma-_bar", -3112, 0, "Baryon", 100, 1, 1.19744, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3112));
   part->AddDecay(Particle::Decay(0, 0.999,  vector<int>{-2112,211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{12,-11,-2112}));

   // Creating sd_1_bar
   new Particle("sd_1_bar", -3103, 0, "Unknown", 100, 0.666667, 0.1088, 0, 100, 100, 1, 100, 1);

   // Creating sd_0_bar
   new Particle("sd_0_bar", -3101, 0, "Unknown", 100, 0.666667, 0.108, 0, 100, 100, 1, 100, 1);

   // Creating delta(1950)++_bar
   new Particle("delta(1950)++_bar", -2228, 0, "Unknown", 100, -2, 1.93, 0.28, 100, 100, 0, 100, 1);

   // Creating delta(1905)++_bar
   new Particle("delta(1905)++_bar", -2226, 0, "Unknown", 100, -2, 1.89, 0.33, 100, 100, 0, 100, 1);

   // Creating Delta--
   new Particle("Delta--", -2224, 0, "Baryon", 100, -2, 1.232, 0.12, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-2212,-211}));

   // Creating delta(1620)++_bar
   new Particle("delta(1620)++_bar", -2222, 0, "Unknown", 100, -2, 1.63, 0.145, 100, 100, 0, 100, 1);

   // Creating delta(1950)+_bar
   new Particle("delta(1950)+_bar", -2218, 0, "Unknown", 100, -1, 1.93, 0.28, 100, 100, 0, 100, 1);

   // Creating N(1675)+_bar
   new Particle("N(1675)+_bar", -2216, 0, "Unknown", 100, -1, 1.675, 0.15, 100, 100, 0, 100, 1);

   // Creating Delta+_bar
   new Particle("Delta+_bar", -2214, 0, "Baryon", 100, -1, 1.232, 0.12, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-2214));
   part->AddDecay(Particle::Decay(0, 0.663,  vector<int>{-2212,-111}));
   part->AddDecay(Particle::Decay(0, 0.331,  vector<int>{-2112,-211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{-2212,-22}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0019() {

   // Creating antiproton
   new Particle("antiproton", -2212, 0, "Baryon", 100, -1, 0.938272, 0, 100, 100, 1, 100, 1);

   // Creating p_diffr+_bar
   new Particle("p_diffr+_bar", -2210, 0, "Unknown", 100, -1, 0, 0, 100, 100, 1, 100, 1);

   // Creating uu_1_bar
   new Particle("uu_1_bar", -2203, 0, "Unknown", 100, -1.33333, 0.0048, 0, 100, 100, 1, 100, 1);

   // Creating N(2190)+_bar
   new Particle("N(2190)+_bar", -2128, 0, "Unknown", 100, -1, 2.19, 0.5, 100, 100, 0, 100, 1);

   // Creating delta(1905)+_bar
   new Particle("delta(1905)+_bar", -2126, 0, "Unknown", 100, -1, 1.89, 0.33, 100, 100, 0, 100, 1);

   // Creating N(1520)+_bar
   new Particle("N(1520)+_bar", -2124, 0, "Unknown", 100, -1, 1.52, 0.115, 100, 100, 0, 100, 1);

   // Creating delta(1620)+_bar
   new Particle("delta(1620)+_bar", -2122, 0, "Unknown", 100, -1, 1.63, 0.145, 100, 100, 0, 100, 1);

   // Creating delta(1950)0_bar
   new Particle("delta(1950)0_bar", -2118, 0, "Unknown", 100, 0, 1.93, 0.28, 100, 100, 0, 100, 1);

   // Creating N(1675)0_bar
   new Particle("N(1675)0_bar", -2116, 0, "Unknown", 100, 0, 1.675, 0.15, 100, 100, 0, 100, 1);

   // Creating Delta0_bar
   new Particle("Delta0_bar", -2114, 0, "Baryon", 100, 0, 1.232, 0.12, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-2114));
   part->AddDecay(Particle::Decay(0, 0.663,  vector<int>{-2112,-111}));
   part->AddDecay(Particle::Decay(0, 0.331,  vector<int>{-2212,211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{-2112,-22}));

   // Creating antineutron
   new Particle("antineutron", -2112, 0, "Baryon", 100, 0, 0.939565, 0, 100, 100, 1, 100, 1);

   // Creating n_diffr0_bar
   new Particle("n_diffr0_bar", -2110, 0, "Unknown", 100, 0, 0, 0, 100, 100, 1, 100, 1);

   // Creating ud_1_bar
   new Particle("ud_1_bar", -2103, 0, "Unknown", 100, -0.333333, 0.0072, 0, 100, 100, 1, 100, 1);

   // Creating ud_0_bar
   new Particle("ud_0_bar", -2101, 0, "Unknown", 100, -0.333333, 0.0073, 0, 100, 100, 1, 100, 1);

   // Creating N(2190)0_bar
   new Particle("N(2190)0_bar", -1218, 0, "Unknown", 100, 0, 2.19, 0.5, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0020() {

   // Creating delta(1905)0_bar
   new Particle("delta(1905)0_bar", -1216, 0, "Unknown", 100, 0, 1.89, 0.33, 100, 100, 0, 100, 1);

   // Creating N(1520)0_bar
   new Particle("N(1520)0_bar", -1214, 0, "Unknown", 100, 0, 1.52, 0.115, 100, 100, 0, 100, 1);

   // Creating delta(1620)0_bar
   new Particle("delta(1620)0_bar", -1212, 0, "Unknown", 100, 0, 1.63, 0.145, 100, 100, 0, 100, 1);

   // Creating delta(1950)-_bar
   new Particle("delta(1950)-_bar", -1118, 0, "Unknown", 100, 1, 1.93, 0.28, 100, 100, 0, 100, 1);

   // Creating delta(1905)-_bar
   new Particle("delta(1905)-_bar", -1116, 0, "Unknown", 100, 1, 1.89, 0.33, 100, 100, 0, 100, 1);

   // Creating Delta-_bar
   new Particle("Delta-_bar", -1114, 0, "Unknown", 100, 1, 1.232, 0.12, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-1114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-2112,211}));

   // Creating delta(1620)-_bar
   new Particle("delta(1620)-_bar", -1112, 0, "Unknown", 100, 1, 1.63, 0.145, 100, 100, 0, 100, 1);

   // Creating dd_1_bar
   new Particle("dd_1_bar", -1103, 0, "Unknown", 100, 0.666667, 0.96, 0, 100, 100, 1, 100, 1);

   // Creating B*_2c-
   new Particle("B*_2c-", -545, 0, "B-Meson", 100, -1, 7.35, 0.02, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-545));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-511,-411}));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-521,-421}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-513,-411}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-523,-421}));

   // Creating B*_c-
   new Particle("B*_c-", -543, 0, "B-Meson", 100, -1, 6.602, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-543));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-541,-22}));

   // Creating B_c-
   new Particle("B_c-", -541, 0, "B-Meson", 100, -1, 6.276, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-541));
   part->AddDecay(Particle::Decay(42, 0.24,  vector<int>{1,-2,-3,5}));
   part->AddDecay(Particle::Decay(42, 0.15,  vector<int>{-2,1,4,-4}));
   part->AddDecay(Particle::Decay(11, 0.122,  vector<int>{-4,3}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{1,-3,-2,5}));
   part->AddDecay(Particle::Decay(42, 0.05,  vector<int>{-4,3,4,-4}));
   part->AddDecay(Particle::Decay(0, 0.047,  vector<int>{-16,15}));
   part->AddDecay(Particle::Decay(42, 0.042,  vector<int>{11,-12,-533}));
   part->AddDecay(Particle::Decay(42, 0.042,  vector<int>{13,-14,-533}));
   part->AddDecay(Particle::Decay(42, 0.037,  vector<int>{-2,4,1,-4}));
   part->AddDecay(Particle::Decay(42, 0.035,  vector<int>{-14,13,-443}));
   part->AddDecay(Particle::Decay(42, 0.035,  vector<int>{-12,11,-443}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-4,4,3,-4}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{13,-14,-531}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{11,-12,-531}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{1,-2,-1,5}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-12,11,-441}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{3,-2,-3,5}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-14,13,-441}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-2,3,4,-4}));
   part->AddDecay(Particle::Decay(42, 0.007,  vector<int>{-16,15,-443}));
   part->AddDecay(Particle::Decay(11, 0.006,  vector<int>{-4,1}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-16,15,-441}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{3,-3,-2,5}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-4,1,4,-4}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{1,-1,-2,5}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,-513}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-2,4,3,-4}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,-513}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-511}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-4,4,1,-4}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-511}));

   // Creating B*_2s0_bar
   new Particle("B*_2s0_bar", -535, 0, "B-Meson", 100, 0, 5.8397, 0.02, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-535));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-521,321}));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-511,311}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-523,321}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{-513,311}));

   // Creating B*_s0_bar
   new Particle("B*_s0_bar", -533, 0, "B-Meson", 100, 0, 5.4128, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-533));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-531,-22}));

   // Creating B_s0_bar
   new Particle("B_s0_bar", -531, 0, "B-Meson", 100, 0, 5.3663, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-531));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{-2,1,4,-3}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{-2,4,1,-3}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{-4,3,4,-3}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-14,13,433}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-12,11,433}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{-16,15,433}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{433,-433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-12,11,431}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-14,13,431}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{-4,4,3,-3}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{431,-433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{433,-20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{431,-431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,-3}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{433,-431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-14,13,435}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-12,11,435}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{431,-213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-16,15,431}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{433,-213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-14,13,20433}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-12,11,20433}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{431,-20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10431}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10433}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10431}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10433}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,-3}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{433,-211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{431,-211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{-20443,-333}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{-443,-333}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-20443,-221}));
   part->AddDecay(Particle::Decay(0, 0.0009,  vector<int>{-20443,-331}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{-441,-333}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-443,-221}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-443,-331}));
   part->AddDecay(Particle::Decay(0, 0.0002,  vector<int>{-441,-331}));
   part->AddDecay(Particle::Decay(0, 0.0002,  vector<int>{-441,-221}));

   // Creating B*_2-
   new Particle("B*_2-", -525, 0, "B-Meson", 100, -1, 5.7469, 0.02, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-525));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-511,-211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{-513,-211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{-521,-111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{-513,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-523,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-511,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-523,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{-521,-211,211}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0021() {

   // Creating B*-
   new Particle("B*-", -523, 0, "Meson", 100, -1, 5.3251, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-523));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-521,-22}));

   // Creating B-
   new Particle("B-", -521, 0, "B-Meson", 100, -1, 5.27915, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-521));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{-2,1,4,-2}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{-2,4,1,-2}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{-4,3,4,-2}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-14,13,423}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-12,11,423}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{-16,15,423}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{423,-433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-12,11,421}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-14,13,421}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{-4,4,3,-2}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{421,-433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{423,-20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{421,-431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,-2}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{423,-431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-14,13,425}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-12,11,425}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{421,-213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-16,15,421}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{423,-213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-14,13,20423}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-12,11,20423}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{421,-20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10421}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10423}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10421}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10423}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,-2}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{423,-211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{421,-211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{-20443,-323}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{-20443,-321}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{-443,-323}));
   part->AddDecay(Particle::Decay(0, 0.0008,  vector<int>{-443,-321}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{-441,-323}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-441,-321}));

   // Creating B*_20_bar
   new Particle("B*_20_bar", -515, 0, "B-Meson", 100, 0, 5.7469, 0.02, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-515));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-521,211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{-523,211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{-511,-111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{-523,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-513,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-521,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-513,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{-511,-211,211}));

   // Creating B*0_bar
   new Particle("B*0_bar", -513, 0, "B-Meson", 100, 0, 5.3251, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-513));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-511,-22}));

   // Creating B0_bar
   new Particle("B0_bar", -511, 0, "B-Meson", 100, 0, 5.27953, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-511));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{-2,1,4,-1}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{-2,4,1,-1}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{-4,3,4,-1}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-14,13,413}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{-12,11,413}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{-16,15,413}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{413,-433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-12,11,411}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-14,13,411}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{-4,4,3,-1}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{411,-433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{413,-20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{411,-431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,-1}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{413,-431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-14,13,415}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-12,11,415}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{411,-213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-16,15,411}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{413,-213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-14,13,20413}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{-12,11,20413}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{411,-20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10411}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10413}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-14,13,10411}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-12,11,10413}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,-1}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{413,-211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{411,-211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{-20443,-313}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{-20443,-311}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{-443,-313}));
   part->AddDecay(Particle::Decay(0, 0.0008,  vector<int>{-443,-311}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{-441,-313}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-441,-311}));

   // Creating D*_2s-
   new Particle("D*_2s-", -435, 0, "CharmedMeson", 100, -1, 2.5726, 0.015, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-435));
   part->AddDecay(Particle::Decay(0, 0.4,  vector<int>{-421,-321}));
   part->AddDecay(Particle::Decay(0, 0.4,  vector<int>{-411,-311}));
   part->AddDecay(Particle::Decay(0, 0.1,  vector<int>{-423,-321}));
   part->AddDecay(Particle::Decay(0, 0.1,  vector<int>{-413,-311}));

   // Creating D*_s-
   new Particle("D*_s-", -433, 0, "CharmedMeson", 100, -1, 2.1123, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-433));
   part->AddDecay(Particle::Decay(0, 0.94,  vector<int>{-431,-22}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-431,-111}));

   // Creating D_s-
   new Particle("D_s-", -431, 0, "CharmedMeson", 100, -1, 1.9685, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-431));
   part->AddDecay(Particle::Decay(13, 0.25,  vector<int>{-2,1,-3,3}));
   part->AddDecay(Particle::Decay(13, 0.0952,  vector<int>{-2,1}));
   part->AddDecay(Particle::Decay(0, 0.095,  vector<int>{-331,-213}));
   part->AddDecay(Particle::Decay(0, 0.079,  vector<int>{-221,-213}));
   part->AddDecay(Particle::Decay(0, 0.052,  vector<int>{-333,-213}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{-323,313}));
   part->AddDecay(Particle::Decay(0, 0.037,  vector<int>{-331,-211}));
   part->AddDecay(Particle::Decay(0, 0.033,  vector<int>{-323,311}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{11,-12,-333}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{13,-14,-333}));
   part->AddDecay(Particle::Decay(0, 0.028,  vector<int>{-333,-211}));
   part->AddDecay(Particle::Decay(0, 0.028,  vector<int>{-321,311}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{-321,313}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{11,-12,-331}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{11,-12,-221}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{13,-14,-221}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{13,-14,-331}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-221,-211}));
   part->AddDecay(Particle::Decay(0, 0.01,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(0, 0.01,  vector<int>{-2212,2112}));
   part->AddDecay(Particle::Decay(0, 0.0078,  vector<int>{-10221,-211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,-321,321}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,-311,311}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-221,-321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-331,-321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-333,-321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-221,-323}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,-311,311}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,-321,321}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-213,-113}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-213,-111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-211,-113}));

   // Creating D*_20_bar
   new Particle("D*_20_bar", -425, 0, "CharmedMeson", 100, 0, 2.4611, 0.023, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-425));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-411,211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{-413,211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{-421,-111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{-413,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-423,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-411,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-423,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{-421,-211,211}));

   // Creating D*0_bar
   new Particle("D*0_bar", -423, 0, "CharmedMeson", 100, 0, 2.00697, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-423));
   part->AddDecay(Particle::Decay(3, 0.619,  vector<int>{-421,-111}));
   part->AddDecay(Particle::Decay(0, 0.381,  vector<int>{-421,-22}));

   // Creating D0_bar
   new Particle("D0_bar", -421, 0, "CharmedMeson", 100, 0, 1.86484, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-421));
   part->AddDecay(Particle::Decay(0, 0.0923,  vector<int>{321,-211,-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.074,  vector<int>{321,-20213}));
   part->AddDecay(Particle::Decay(0, 0.073,  vector<int>{321,-213}));
   part->AddDecay(Particle::Decay(0, 0.067,  vector<int>{311,-211,211,-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.062,  vector<int>{323,-213}));
   part->AddDecay(Particle::Decay(0, 0.0511,  vector<int>{311,-113,-111,-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.045,  vector<int>{323,-211}));
   part->AddDecay(Particle::Decay(0, 0.0365,  vector<int>{321,-211}));
   part->AddDecay(Particle::Decay(42, 0.034,  vector<int>{11,-12,321}));
   part->AddDecay(Particle::Decay(42, 0.034,  vector<int>{13,-14,321}));
   part->AddDecay(Particle::Decay(42, 0.027,  vector<int>{11,-12,323}));
   part->AddDecay(Particle::Decay(42, 0.027,  vector<int>{13,-14,323}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{311,-223}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{321,-211,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.022,  vector<int>{311,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{313,-111}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{313,-221}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{311,-111}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{311,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{321,-211,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.017,  vector<int>{-211,-211,211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.016,  vector<int>{313,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{313,-113}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{321,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0109,  vector<int>{10323,-211}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{311,-321,321,-111}));
   part->AddDecay(Particle::Decay(0, 0.0088,  vector<int>{311,-333}));
   part->AddDecay(Particle::Decay(0, 0.0085,  vector<int>{311,-211,-211,211,211}));
   part->AddDecay(Particle::Decay(0, 0.0077,  vector<int>{313,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0075,  vector<int>{-211,-211,211,211}));
   part->AddDecay(Particle::Decay(0, 0.0063,  vector<int>{321,-211,-113}));
   part->AddDecay(Particle::Decay(0, 0.0061,  vector<int>{311,-113}));
   part->AddDecay(Particle::Decay(0, 0.0052,  vector<int>{321,-321,311}));
   part->AddDecay(Particle::Decay(0, 0.0041,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{13,-14,323,-111}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{11,-12,313,211}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{11,-12,323,-111}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{13,-14,313,211}));
   part->AddDecay(Particle::Decay(0, 0.0036,  vector<int>{313,-321,211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{321,-323}));
   part->AddDecay(Particle::Decay(0, 0.0034,  vector<int>{321,-311,-211}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{-321,321,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0027,  vector<int>{313,-313}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,311,211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,321,-111}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,213}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{323,-321}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{13,-14,213}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,311,211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{11,-12,321,-111}));
   part->AddDecay(Particle::Decay(0, 0.0018,  vector<int>{-333,-113}));
   part->AddDecay(Particle::Decay(0, 0.0016,  vector<int>{-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.0016,  vector<int>{-211,211}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{311,-311}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{313,-311}));
   part->AddDecay(Particle::Decay(0, 0.0009,  vector<int>{-310,-310,-310}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{-333,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-113,-211,-211,211,211}));

   // Creating D*_2-
   new Particle("D*_2-", -415, 0, "CharmedMeson", 100, -1, 2.4601, 0.023, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-415));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{-421,-211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{-423,-211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{-411,-111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{-423,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-413,-111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{-421,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{-413,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{-411,-211,211}));

   // Creating D*-
   new Particle("D*-", -413, 0, "CharmedMeson", 100, -1, 2.01027, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-413));
   part->AddDecay(Particle::Decay(3, 0.683,  vector<int>{-421,-211}));
   part->AddDecay(Particle::Decay(3, 0.306,  vector<int>{-411,-111}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{-411,-22}));

   // Creating D-
   new Particle("D-", -411, 0, "CharmedMeson", 100, -1, 1.86962, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-411));
   part->AddDecay(Particle::Decay(0, 0.087,  vector<int>{311,-211,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.076,  vector<int>{311,-20213}));
   part->AddDecay(Particle::Decay(42, 0.07,  vector<int>{11,-12,311}));
   part->AddDecay(Particle::Decay(42, 0.07,  vector<int>{13,-14,311}));
   part->AddDecay(Particle::Decay(0, 0.067,  vector<int>{321,-211,-211}));
   part->AddDecay(Particle::Decay(0, 0.066,  vector<int>{311,-213}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{11,-12,313}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{13,-14,313}));
   part->AddDecay(Particle::Decay(0, 0.045,  vector<int>{20313,-211}));
   part->AddDecay(Particle::Decay(0, 0.041,  vector<int>{313,-213}));
   part->AddDecay(Particle::Decay(0, 0.027,  vector<int>{311,-321,311}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{313,-323}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{311,-211}));
   part->AddDecay(Particle::Decay(0, 0.022,  vector<int>{321,-211,-211,-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.0218,  vector<int>{-211,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.019,  vector<int>{-333,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.019,  vector<int>{313,-211}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{311,-211,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{311,-211,-111}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{13,-14,323,-211}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{11,-12,313,-111}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{11,-12,323,-211}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{13,-14,313,-111}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{321,-211,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.008,  vector<int>{321,-213,-211}));
   part->AddDecay(Particle::Decay(0, 0.0073,  vector<int>{311,-321}));
   part->AddDecay(Particle::Decay(0, 0.0066,  vector<int>{-221,-211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{-333,-211}));
   part->AddDecay(Particle::Decay(0, 0.0057,  vector<int>{313,-211,-113}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-333,-213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,311,-111}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{11,-12,321,-211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,311,-111}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-221,-213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{13,-14,321,-211}));
   part->AddDecay(Particle::Decay(0, 0.0047,  vector<int>{313,-321}));
   part->AddDecay(Particle::Decay(0, 0.0047,  vector<int>{311,-323}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{321,-321,-211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-331,-211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{-331,-213}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{-113,-211,-211,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0022,  vector<int>{-211,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{313,-211,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{321,-113,-211,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0015,  vector<int>{-211,-211,-211,211,211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-111,-211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-221}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-331}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-113}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-223}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-223,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-223,-213}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{11,-12,-111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{321,-211,-211,-211,211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-111}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{311,-113,-211,-211,211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-331}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-113}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{13,-14,-223}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{-113,-213}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{-111,-213}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{-113,-211}));

   // Creating phi3(1850)_bar
   new Particle("phi3(1850)_bar", -337, 0, "Unknown", 100, 0, 1.854, 0.087, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0022() {

   // Creating k3_star(1780)-_bar
   new Particle("k3_star(1780)-_bar", -327, 0, "Unknown", 100, -1, 1.776, 0.159, 100, 100, 0, 100, 1);

   // Creating K*_2-
   new Particle("K*_2-", -325, 0, "Meson", 100, -1, 1.4256, 0.098, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-325));
   part->AddDecay(Particle::Decay(0, 0.332,  vector<int>{-311,-211}));
   part->AddDecay(Particle::Decay(0, 0.168,  vector<int>{-313,-211}));
   part->AddDecay(Particle::Decay(0, 0.166,  vector<int>{-321,-111}));
   part->AddDecay(Particle::Decay(0, 0.086,  vector<int>{-313,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.084,  vector<int>{-323,-111}));
   part->AddDecay(Particle::Decay(0, 0.059,  vector<int>{-311,-213}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{-323,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{-321,-113}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{-321,-223}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-321,-221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-321,-22}));

   // Creating K*-
   new Particle("K*-", -323, 0, "Meson", 100, -1, 0.89166, 0.0498, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-323));
   part->AddDecay(Particle::Decay(3, 0.666,  vector<int>{-311,-211}));
   part->AddDecay(Particle::Decay(3, 0.333,  vector<int>{-321,-111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-321,-22}));

   // Creating K-
   new Particle("K-", -321, 0, "Meson", 100, -1, 0.493677, 5.31674e-17, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-321));
   part->AddDecay(Particle::Decay(0, 0.6352,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(0, 0.2116,  vector<int>{-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.0559,  vector<int>{-211,-211,211}));
   part->AddDecay(Particle::Decay(42, 0.0482,  vector<int>{-12,11,-111}));
   part->AddDecay(Particle::Decay(42, 0.0318,  vector<int>{-14,13,-111}));
   part->AddDecay(Particle::Decay(0, 0.0173,  vector<int>{-211,-111,-111}));

   // Creating k3_star(1780)0_bar
   new Particle("k3_star(1780)0_bar", -317, 0, "Unknown", 100, 0, 1.776, 0.159, 100, 100, 0, 100, 1);

   // Creating K*_20_bar
   new Particle("K*_20_bar", -315, 0, "Meson", 100, 0, 1.4324, 0.109, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-315));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-321,211}));
   part->AddDecay(Particle::Decay(0, 0.168,  vector<int>{-323,211}));
   part->AddDecay(Particle::Decay(0, 0.166,  vector<int>{-311,-111}));
   part->AddDecay(Particle::Decay(0, 0.087,  vector<int>{-323,211,-111}));
   part->AddDecay(Particle::Decay(0, 0.084,  vector<int>{-313,-111}));
   part->AddDecay(Particle::Decay(0, 0.059,  vector<int>{-321,213}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{-313,-211,211}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{-311,-113}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{-311,-223}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-311,-221}));

   // Creating K*0_bar
   new Particle("K*0_bar", -313, 0, "Meson", 100, 0, 0.896, 0.0505, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-313));
   part->AddDecay(Particle::Decay(3, 0.665,  vector<int>{-321,211}));
   part->AddDecay(Particle::Decay(3, 0.333,  vector<int>{-311,-111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-311,-22}));

   // Creating K0_bar
   new Particle("K0_bar", -311, 0, "Meson", 100, 0, 0.497614, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-311));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-130}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-310}));

   // Creating omega3(1670)_bar
   new Particle("omega3(1670)_bar", -227, 0, "Unknown", 100, 0, 1.667, 0.168, 100, 100, 0, 100, 1);

   // Creating rho3(1690)-_bar
   new Particle("rho3(1690)-_bar", -217, 0, "Unknown", 100, -1, 1.6888, 0.161, 100, 100, 0, 100, 1);

   // Creating a_2-
   new Particle("a_2-", -215, 0, "Meson", 100, -1, 1.3183, 0.107, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-215));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{-213,-111}));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{-113,-211}));
   part->AddDecay(Particle::Decay(0, 0.144,  vector<int>{-221,-211}));
   part->AddDecay(Particle::Decay(0, 0.104,  vector<int>{-223,-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.049,  vector<int>{-321,311}));
   part->AddDecay(Particle::Decay(0, 0.0057,  vector<int>{-331,-211}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{-211,-22}));

   // Creating rho-
   new Particle("rho-", -213, 0, "Meson", 100, -1, 0.77549, 0.149, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-213));
   part->AddDecay(Particle::Decay(3, 0.99955,  vector<int>{-211,-111}));
   part->AddDecay(Particle::Decay(0, 0.00045,  vector<int>{-211,-22}));

   // Creating pi-
   new Particle("pi-", -211, 0, "Meson", 100, -1, 0.13957, 2.52837e-17, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-211));
   part->AddDecay(Particle::Decay(0, 0.999877,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(0, 0.000123,  vector<int>{11,-12}));

   // Creating pi_diffr-
   new Particle("pi_diffr-", -210, 0, "Meson", 100, -1, 0, 0, 100, 100, 1, 100, 1);

   // Creating rho3(1690)0_bar
   new Particle("rho3(1690)0_bar", -117, 0, "Unknown", 100, 0, 1.6888, 0.161, 100, 100, 0, 100, 1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0023() {

   // Creating b-hadron_bar
   new Particle("b-hadron_bar", -85, 0, "Generator", 100, 0.333333, 5, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-85));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{2,-1,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{4,-3,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{12,-11,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{14,-13,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{2,-4,-1,-81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{16,-15,-4,-81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,-81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{4,-4,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,-81}));

   // Creating c-hadron_bar
   new Particle("c-hadron_bar", -84, 0, "Generator", 100, -0.666667, 2, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-84));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{-2,1,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{13,-14,-3,-81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{11,-12,-3,-81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{-2,3,-3,-81}));

   // Creating rndmflav_bar
   new Particle("rndmflav_bar", -82, 0, "Generator", 100, 0, 0, 0, 100, 100, 1, 100, 1);

   // Creating nu_Rtau_bar
   new Particle("nu_Rtau_bar", -66, 0, "Unknown", 100, 0, 750, 0, 100, 100, 1, 100, 1);

   // Creating nu_Rmu_bar
   new Particle("nu_Rmu_bar", -65, 0, "Unknown", 100, 0, 750, 0, 100, 100, 1, 100, 1);

   // Creating nu_Re_bar
   new Particle("nu_Re_bar", -64, 0, "Unknown", 100, 0, 750, 0, 100, 100, 1, 100, 1);

   // Creating W_R-
   new Particle("W_R-", -63, 0, "Unknown", 100, -1, 750, 19.3391, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-63));
   part->AddDecay(Particle::Decay(32, 0.325914,  vector<int>{1,-2}));
   part->AddDecay(Particle::Decay(32, 0.32532,  vector<int>{3,-4}));
   part->AddDecay(Particle::Decay(32, 0.314118,  vector<int>{5,-6}));
   part->AddDecay(Particle::Decay(32, 0.016736,  vector<int>{3,-2}));
   part->AddDecay(Particle::Decay(32, 0.016735,  vector<int>{1,-4}));
   part->AddDecay(Particle::Decay(32, 0.000603001,  vector<int>{5,-4}));
   part->AddDecay(Particle::Decay(32, 0.000554001,  vector<int>{3,-6}));
   part->AddDecay(Particle::Decay(32, 1e-05,  vector<int>{5,-2}));
   part->AddDecay(Particle::Decay(32, 9.00001e-06,  vector<int>{1,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-64}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{13,-65}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{15,-66}));

   // Creating H_R--
   new Particle("H_R--", -62, 0, "Unknown", 100, -2, 200, 0.88001, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-62));
   part->AddDecay(Particle::Decay(0, 0.813719,  vector<int>{15,15}));
   part->AddDecay(Particle::Decay(0, 0.0904279,  vector<int>{13,13}));
   part->AddDecay(Particle::Decay(0, 0.0904279,  vector<int>{11,11}));
   part->AddDecay(Particle::Decay(0, 0.001809,  vector<int>{11,13}));
   part->AddDecay(Particle::Decay(0, 0.001808,  vector<int>{13,15}));
   part->AddDecay(Particle::Decay(0, 0.001808,  vector<int>{11,15}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-63,-63}));

   // Creating H_L--
   new Particle("H_L--", -61, 0, "Unknown", 100, -2, 200, 0.88161, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-61));
   part->AddDecay(Particle::Decay(0, 0.812251,  vector<int>{15,15}));
   part->AddDecay(Particle::Decay(0, 0.0902641,  vector<int>{13,13}));
   part->AddDecay(Particle::Decay(0, 0.0902641,  vector<int>{11,11}));
   part->AddDecay(Particle::Decay(0, 0.001806,  vector<int>{-24,-24}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{13,15}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{11,15}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{11,13}));

   // Creating rho_tech-
   new Particle("rho_tech-", -55, 0, "Unknown", 100, -1, 210, 0.64973, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-55));
   part->AddDecay(Particle::Decay(0, 0.474101,  vector<int>{-24,-51}));
   part->AddDecay(Particle::Decay(0, 0.176299,  vector<int>{-52,-23}));
   part->AddDecay(Particle::Decay(0, 0.138845,  vector<int>{-24,-23}));
   part->AddDecay(Particle::Decay(0, 0.109767,  vector<int>{-52,-22}));
   part->AddDecay(Particle::Decay(32, 0.0285839,  vector<int>{1,-2}));
   part->AddDecay(Particle::Decay(32, 0.0285299,  vector<int>{3,-4}));
   part->AddDecay(Particle::Decay(0, 0.00966098,  vector<int>{11,-12}));
   part->AddDecay(Particle::Decay(0, 0.00966098,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(0, 0.00965998,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(0, 0.00816098,  vector<int>{-24,-53}));
   part->AddDecay(Particle::Decay(32, 0.00373499,  vector<int>{5,-6}));
   part->AddDecay(Particle::Decay(32, 0.001468,  vector<int>{3,-2}));
   part->AddDecay(Particle::Decay(32, 0.001468,  vector<int>{1,-4}));
   part->AddDecay(Particle::Decay(32, 5.29999e-05,  vector<int>{5,-4}));
   part->AddDecay(Particle::Decay(32, 6.99999e-06,  vector<int>{3,-6}));
   part->AddDecay(Particle::Decay(32, 9.99998e-07,  vector<int>{5,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{5,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-52,-51}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-18}));

   // Creating pi_tech-
   new Particle("pi_tech-", -52, 0, "Unknown", 100, -1, 110, 0.0105, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-52));
   part->AddDecay(Particle::Decay(32, 0.90916,  vector<int>{-4,5}));
   part->AddDecay(Particle::Decay(0, 0.048905,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(32, 0.041762,  vector<int>{-4,3}));
   part->AddDecay(Particle::Decay(0, 0.000173,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-24,-5,5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-12}));

   // Creating R0_bar
   new Particle("R0_bar", -40, 0, "Unknown", 100, 0, 5000, 417.465, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-40));
   part->AddDecay(Particle::Decay(32, 0.215134,  vector<int>{-1,3}));
   part->AddDecay(Particle::Decay(32, 0.215134,  vector<int>{-2,4}));
   part->AddDecay(Particle::Decay(32, 0.215133,  vector<int>{-3,5}));
   part->AddDecay(Particle::Decay(32, 0.214738,  vector<int>{-4,6}));
   part->AddDecay(Particle::Decay(0, 0.0699301,  vector<int>{-11,13}));
   part->AddDecay(Particle::Decay(0, 0.0699301,  vector<int>{-13,15}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-5,7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-6,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-15,17}));

   // Creating LQ_ue_bar
   new Particle("LQ_ue_bar", -39, 0, "Unknown", 100, 0.333333, 200, 0.39162, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-39));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{-2,-11}));

   // Creating H-
   new Particle("H-", -37, 0, "GaugeBoson", 100, -1, 300, 5.75967, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-37));
   part->AddDecay(Particle::Decay(0, 0.929792,  vector<int>{-24,-25}));
   part->AddDecay(Particle::Decay(32, 0.067484,  vector<int>{5,-6}));
   part->AddDecay(Particle::Decay(0, 0.002701,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(32, 1.3e-05,  vector<int>{3,-4}));
   part->AddDecay(Particle::Decay(0, 1e-05,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-1000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-1000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-1000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-1000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-1000016}));

   // Creating W'-
   new Particle("W'-", -34, 0, "GaugeBoson", 100, -1, 500, 16.6708, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-34));
   part->AddDecay(Particle::Decay(32, 0.251276,  vector<int>{1,-2}));
   part->AddDecay(Particle::Decay(32, 0.250816,  vector<int>{3,-4}));
   part->AddDecay(Particle::Decay(32, 0.215459,  vector<int>{5,-6}));
   part->AddDecay(Particle::Decay(0, 0.085262,  vector<int>{11,-12}));
   part->AddDecay(Particle::Decay(0, 0.085262,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(0, 0.08526,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(32, 0.012903,  vector<int>{3,-2}));
   part->AddDecay(Particle::Decay(32, 0.012903,  vector<int>{1,-4}));
   part->AddDecay(Particle::Decay(32, 0.000465,  vector<int>{5,-4}));
   part->AddDecay(Particle::Decay(32, 0.00038,  vector<int>{3,-6}));
   part->AddDecay(Particle::Decay(32, 8e-06,  vector<int>{5,-2}));
   part->AddDecay(Particle::Decay(32, 6e-06,  vector<int>{1,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{5,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-23}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-22}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-25}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0024() {

   // Creating W-
   new Particle("W-", -24, 0, "GaugeBoson", 100, -1, 80.398, 2.07002, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-24));
   part->AddDecay(Particle::Decay(32, 0.321502,  vector<int>{1,-2}));
   part->AddDecay(Particle::Decay(32, 0.320778,  vector<int>{3,-4}));
   part->AddDecay(Particle::Decay(0, 0.108062,  vector<int>{11,-12}));
   part->AddDecay(Particle::Decay(0, 0.108062,  vector<int>{13,-14}));
   part->AddDecay(Particle::Decay(0, 0.107983,  vector<int>{15,-16}));
   part->AddDecay(Particle::Decay(32, 0.016509,  vector<int>{3,-2}));
   part->AddDecay(Particle::Decay(32, 0.016502,  vector<int>{1,-4}));
   part->AddDecay(Particle::Decay(32, 0.000591001,  vector<int>{5,-4}));
   part->AddDecay(Particle::Decay(32, 1e-05,  vector<int>{5,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{5,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{5,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-18}));

   // Creating nu'_tau_bar
   new Particle("nu'_tau_bar", -18, 0, "Lepton", 100, 0, 0, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-18));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,-17}));

   // Creating tau'+
   new Particle("tau'+", -17, 0, "Lepton", 100, 1, 400, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-17));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-17}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-18}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,-18}));

   // Creating nu_tau_bar
   new Particle("nu_tau_bar", -16, 0, "Lepton", 100, 0, 0, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-16));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-16}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-15}));

   // Creating tau+
   new Particle("tau+", -15, 0, "Lepton", 100, 1, 1.77684, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-15));
   part->AddDecay(Particle::Decay(0, 0.2494,  vector<int>{-16,213}));
   part->AddDecay(Particle::Decay(42, 0.1783,  vector<int>{12,-11,-16}));
   part->AddDecay(Particle::Decay(42, 0.1735,  vector<int>{14,-13,-16}));
   part->AddDecay(Particle::Decay(0, 0.1131,  vector<int>{-16,211}));
   part->AddDecay(Particle::Decay(41, 0.09,  vector<int>{-16,213,-111}));
   part->AddDecay(Particle::Decay(41, 0.08,  vector<int>{-16,211,-113}));
   part->AddDecay(Particle::Decay(41, 0.0191,  vector<int>{-16,211,-223}));
   part->AddDecay(Particle::Decay(41, 0.0133,  vector<int>{-16,211,-113,-111}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{-16,323}));
   part->AddDecay(Particle::Decay(41, 0.011,  vector<int>{-16,211,-211,211}));
   part->AddDecay(Particle::Decay(41, 0.01,  vector<int>{-16,213,-111,-111}));
   part->AddDecay(Particle::Decay(0, 0.0071,  vector<int>{-16,321}));
   part->AddDecay(Particle::Decay(41, 0.0067,  vector<int>{-16,213,-211,211}));
   part->AddDecay(Particle::Decay(41, 0.005,  vector<int>{-16,213,-113}));
   part->AddDecay(Particle::Decay(41, 0.0035,  vector<int>{-16,213,-223}));
   part->AddDecay(Particle::Decay(41, 0.0034,  vector<int>{-16,321,-321,211}));
   part->AddDecay(Particle::Decay(41, 0.003,  vector<int>{-16,211,-111}));
   part->AddDecay(Particle::Decay(41, 0.0027,  vector<int>{-16,211,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.00205,  vector<int>{-16,211,-310,-111}));
   part->AddDecay(Particle::Decay(41, 0.00205,  vector<int>{-16,211,-130,-111}));
   part->AddDecay(Particle::Decay(41, 0.0015,  vector<int>{-16,213,-221}));
   part->AddDecay(Particle::Decay(41, 0.0014,  vector<int>{-16,211,-111,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.0012,  vector<int>{-16,213,-111,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.0011,  vector<int>{-16,213,-113,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.00078,  vector<int>{-16,321,-310}));
   part->AddDecay(Particle::Decay(41, 0.00078,  vector<int>{-16,321,-130}));
   part->AddDecay(Particle::Decay(41, 0.00075,  vector<int>{-16,211,-113,-113}));
   part->AddDecay(Particle::Decay(41, 0.00075,  vector<int>{-16,323,-111}));
   part->AddDecay(Particle::Decay(41, 0.00069,  vector<int>{-16,321,-310,-111}));
   part->AddDecay(Particle::Decay(41, 0.00069,  vector<int>{-16,321,-130,-111}));
   part->AddDecay(Particle::Decay(41, 0.0006,  vector<int>{-16,211,-223,-111}));
   part->AddDecay(Particle::Decay(41, 0.00051,  vector<int>{-16,211,-310,-130}));
   part->AddDecay(Particle::Decay(41, 0.0005,  vector<int>{-16,211,-211,211,-111}));
   part->AddDecay(Particle::Decay(41, 0.0004,  vector<int>{-16,323,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.0004,  vector<int>{-16,321,-111}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{-16,211,-130}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{-16,211,-310,-310}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{-16,211,-310}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{-16,211,-130,-130}));
   part->AddDecay(Particle::Decay(41, 0.00022,  vector<int>{-16,211,-113,-113,-111}));
   part->AddDecay(Particle::Decay(41, 0.00021,  vector<int>{-16,211,-221,-111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{-16,211,213,-211,-111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{-16,211,-113,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{-16,213,-113,-111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{-16,211,-213,213}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{-16,211,-213,211,-111}));
   part->AddDecay(Particle::Decay(41, 0.0001,  vector<int>{-16,321,-111,-111,-111}));
   part->AddDecay(Particle::Decay(41, 0.0001,  vector<int>{-16,211,-221,-221}));
   part->AddDecay(Particle::Decay(41, 6e-05,  vector<int>{-16,323,-111,-111}));
   part->AddDecay(Particle::Decay(41, 6e-05,  vector<int>{-16,211,-221}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-15}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-15}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-16}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-15}));

   // Creating nu_mu_bar
   new Particle("nu_mu_bar", -14, 0, "Lepton", 100, 0, 0, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-14));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-13}));

   // Creating mu+
   new Particle("mu+", -13, 0, "Lepton", 100, 1, 0.105658, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-13));
   part->AddDecay(Particle::Decay(42, 1,  vector<int>{12,-11,-14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-13}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-13}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-13}));

   // Creating nu_e_bar
   new Particle("nu_e_bar", -12, 0, "Lepton", 100, 0, 0, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-12));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-12}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-11}));

   // Creating e+
   new Particle("e+", -11, 0, "Lepton", 100, 1, 0.000510999, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-11));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-11}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-11}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-12}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-11}));

   // Creating t'_bar
   new Particle("t'_bar", -8, 0, "Quark", 100, -0.666667, 171.2, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-8));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-1}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-3}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,-5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,-7}));

   // Creating b'_bar
   new Particle("b'_bar", -7, 0, "Quark", 100, 0.333333, 468, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-7));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-7}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-2}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-4}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-7}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,-4}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,-6}));

   // Creating t_bar
   new Particle("t_bar", -6, 0, "Quark", 100, -0.666667, 171.2, 1.39883, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-6));
   part->AddDecay(Particle::Decay(0, 0.998205,  vector<int>{-24,-5}));
   part->AddDecay(Particle::Decay(0, 0.001765,  vector<int>{-24,-3}));
   part->AddDecay(Particle::Decay(0, 3e-05,  vector<int>{-24,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000022,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000023,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000025,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000035,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000021,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000039,-1000006}));

   // Creating b_bar
   new Particle("b_bar", -5, 0, "Quark", 100, 0.333333, 4.68, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-5));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-5}));

   // Creating c_bar
   new Particle("c_bar", -4, 0, "Quark", 100, -0.666667, 1.27, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-4));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-4}));

   // Creating s_bar
   new Particle("s_bar", -3, 0, "Quark", 100, 0.333333, 0.104, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-3));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-3}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0025() {

   // Creating u_bar
   new Particle("u_bar", -2, 0, "Quark", 100, -0.666667, 0.0024, 0, 100, 100, 1, 100, 1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(-2));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-2}));

   // Creating d_bar
   new Particle("d_bar", -1, 0, "Quark", 100, 0.333333, 0.0048, 0, 100, 100, 1, 100, 1);
   part = const_cast<Particle*>(&Particle::Particles().at(-1));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-21,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-22,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-23,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-25,-1}));

   // Creating Rootino
   new Particle("Rootino", 0, 0, "Unknown", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating d
   new Particle("d", 1, 1, "Quark", 100, -0.333333, 0.0048, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,1}));

   // Creating u
   new Particle("u", 2, 1, "Quark", 100, 0.666667, 0.0024, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,2}));

   // Creating s
   new Particle("s", 3, 1, "Quark", 100, -0.333333, 0.104, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,3}));

   // Creating c
   new Particle("c", 4, 1, "Quark", 100, 0.666667, 1.27, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,4}));

   // Creating b
   new Particle("b", 5, 1, "Quark", 100, -0.333333, 4.68, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,5}));

   // Creating t
   new Particle("t", 6, 1, "Quark", 100, 0.666667, 171.2, 1.39883, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(6));
   part->AddDecay(Particle::Decay(0, 0.998205,  vector<int>{24,5}));
   part->AddDecay(Particle::Decay(0, 0.001765,  vector<int>{24,3}));
   part->AddDecay(Particle::Decay(0, 3e-05,  vector<int>{24,1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,1000006}));

   // Creating b'
   new Particle("b'", 7, 1, "Quark", 100, -0.333333, 468, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(7));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,7}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,2}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,4}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,7}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,4}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,6}));

   // Creating t'
   new Particle("t'", 8, 1, "Quark", 100, 0.666667, 171.2, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(8));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,1}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,3}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,7}));

   // Creating e-
   new Particle("e-", 11, 1, "Lepton", 100, -1, 0.000510999, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(11));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,11}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,11}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,12}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,11}));

   // Creating nu_e
   new Particle("nu_e", 12, 1, "Lepton", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(12));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,12}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,11}));

   // Creating mu-
   new Particle("mu-", 13, 1, "Lepton", 100, -1, 0.105658, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(13));
   part->AddDecay(Particle::Decay(42, 1,  vector<int>{-12,11,14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,13}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,13}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,13}));

   // Creating nu_mu
   new Particle("nu_mu", 14, 1, "Lepton", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(14));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,14}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,13}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0026() {

   // Creating tau-
   new Particle("tau-", 15, 1, "Lepton", 100, -1, 1.77684, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(15));
   part->AddDecay(Particle::Decay(0, 0.2494,  vector<int>{16,-213}));
   part->AddDecay(Particle::Decay(42, 0.1783,  vector<int>{-12,11,16}));
   part->AddDecay(Particle::Decay(42, 0.1735,  vector<int>{-14,13,16}));
   part->AddDecay(Particle::Decay(0, 0.1131,  vector<int>{16,-211}));
   part->AddDecay(Particle::Decay(41, 0.09,  vector<int>{16,-213,111}));
   part->AddDecay(Particle::Decay(41, 0.08,  vector<int>{16,-211,113}));
   part->AddDecay(Particle::Decay(41, 0.0191,  vector<int>{16,-211,223}));
   part->AddDecay(Particle::Decay(41, 0.0133,  vector<int>{16,-211,113,111}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{16,-323}));
   part->AddDecay(Particle::Decay(41, 0.011,  vector<int>{16,-211,211,-211}));
   part->AddDecay(Particle::Decay(41, 0.01,  vector<int>{16,-213,111,111}));
   part->AddDecay(Particle::Decay(0, 0.0071,  vector<int>{16,-321}));
   part->AddDecay(Particle::Decay(41, 0.0067,  vector<int>{16,-213,211,-211}));
   part->AddDecay(Particle::Decay(41, 0.005,  vector<int>{16,-213,113}));
   part->AddDecay(Particle::Decay(41, 0.0035,  vector<int>{16,-213,223}));
   part->AddDecay(Particle::Decay(41, 0.0034,  vector<int>{16,-321,321,-211}));
   part->AddDecay(Particle::Decay(41, 0.003,  vector<int>{16,-211,111}));
   part->AddDecay(Particle::Decay(41, 0.0027,  vector<int>{16,-211,111,111}));
   part->AddDecay(Particle::Decay(41, 0.00205,  vector<int>{16,-211,310,111}));
   part->AddDecay(Particle::Decay(41, 0.00205,  vector<int>{16,-211,130,111}));
   part->AddDecay(Particle::Decay(41, 0.0015,  vector<int>{16,-213,221}));
   part->AddDecay(Particle::Decay(41, 0.0014,  vector<int>{16,-211,111,111,111}));
   part->AddDecay(Particle::Decay(41, 0.0012,  vector<int>{16,-213,111,111,111}));
   part->AddDecay(Particle::Decay(41, 0.0011,  vector<int>{16,-213,113,111,111}));
   part->AddDecay(Particle::Decay(41, 0.00078,  vector<int>{16,-321,310}));
   part->AddDecay(Particle::Decay(41, 0.00078,  vector<int>{16,-321,130}));
   part->AddDecay(Particle::Decay(41, 0.00075,  vector<int>{16,-211,113,113}));
   part->AddDecay(Particle::Decay(41, 0.00075,  vector<int>{16,-323,111}));
   part->AddDecay(Particle::Decay(41, 0.00069,  vector<int>{16,-321,310,111}));
   part->AddDecay(Particle::Decay(41, 0.00069,  vector<int>{16,-321,130,111}));
   part->AddDecay(Particle::Decay(41, 0.0006,  vector<int>{16,-211,223,111}));
   part->AddDecay(Particle::Decay(41, 0.00051,  vector<int>{16,-211,310,130}));
   part->AddDecay(Particle::Decay(41, 0.0005,  vector<int>{16,-211,211,-211,111}));
   part->AddDecay(Particle::Decay(41, 0.0004,  vector<int>{16,-323,111,111}));
   part->AddDecay(Particle::Decay(41, 0.0004,  vector<int>{16,-321,111}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{16,-211,130}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{16,-211,310,310}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{16,-211,310}));
   part->AddDecay(Particle::Decay(41, 0.00025,  vector<int>{16,-211,130,130}));
   part->AddDecay(Particle::Decay(41, 0.00022,  vector<int>{16,-211,113,113,111}));
   part->AddDecay(Particle::Decay(41, 0.00021,  vector<int>{16,-211,221,111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{16,-211,-213,211,111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{16,-211,113,111,111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{16,-213,113,111}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{16,-211,213,-213}));
   part->AddDecay(Particle::Decay(41, 0.0002,  vector<int>{16,-211,213,-211,111}));
   part->AddDecay(Particle::Decay(41, 0.0001,  vector<int>{16,-321,111,111,111}));
   part->AddDecay(Particle::Decay(41, 0.0001,  vector<int>{16,-211,221,221}));
   part->AddDecay(Particle::Decay(41, 6e-05,  vector<int>{16,-323,111,111}));
   part->AddDecay(Particle::Decay(41, 6e-05,  vector<int>{16,-211,221}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,15}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,15}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{-24,16}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,15}));

   // Creating nu_tau
   new Particle("nu_tau", 16, 1, "Lepton", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(16));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,16}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{24,15}));

   // Creating tau'-
   new Particle("tau'-", 17, 1, "Lepton", 100, -1, 400, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(17));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{22,17}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-24,18}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{25,17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-37,18}));

   // Creating nu'_tau
   new Particle("nu'_tau", 18, 1, "Lepton", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(18));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{23,18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,17}));

   // Creating g
   new Particle("g", 21, 0, "GaugeBoson", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(21));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{21,21}));

   // Creating gamma
   new Particle("gamma", 22, 0, "GaugeBoson", 100, 0, 0, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(22));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(102, 0,  vector<int>{17,-17}));

   // Creating Z0
   new Particle("Z0", 23, 0, "GaugeBoson", 100, 0, 91.187, 2.48009, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(23));
   part->AddDecay(Particle::Decay(32, 0.154075,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0.154072,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(32, 0.152196,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(32, 0.119483,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0.119346,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 0.0667521,  vector<int>{12,-12}));
   part->AddDecay(Particle::Decay(0, 0.0667521,  vector<int>{14,-14}));
   part->AddDecay(Particle::Decay(0, 0.0667521,  vector<int>{16,-16}));
   part->AddDecay(Particle::Decay(0, 0.033549,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.033549,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.033473,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{18,-18}));

   // Creating W+
   new Particle("W+", 24, 1, "GaugeBoson", 100, 1, 80.398, 2.07002, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(24));
   part->AddDecay(Particle::Decay(32, 0.321502,  vector<int>{-1,2}));
   part->AddDecay(Particle::Decay(32, 0.320778,  vector<int>{-3,4}));
   part->AddDecay(Particle::Decay(0, 0.108062,  vector<int>{-11,12}));
   part->AddDecay(Particle::Decay(0, 0.108062,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(0, 0.107983,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(32, 0.016509,  vector<int>{-3,2}));
   part->AddDecay(Particle::Decay(32, 0.016502,  vector<int>{-1,4}));
   part->AddDecay(Particle::Decay(32, 0.000591001,  vector<int>{-5,4}));
   part->AddDecay(Particle::Decay(32, 1e-05,  vector<int>{-5,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-5,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-5,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-3,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-3,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-17,18}));

   // Creating h0
   new Particle("h0", 25, 0, "GaugeBoson", 100, 0, 80, 0.00237, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(25));
   part->AddDecay(Particle::Decay(32, 0.852249,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.06883,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0.053489,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 0.023981,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 0.000879,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(32, 0.000327,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(0, 0.000244,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(32, 1e-06,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{22,23}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{23,23}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000035}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-1000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-1000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-1000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-1000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-1000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-1000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-1000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-1000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-1000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,2000016}));

   // Creating reggeon
   new Particle("reggeon", 28, 0, "GaugeBoson", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating pomeron
   new Particle("pomeron", 29, 0, "GaugeBoson", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating Z'0
   new Particle("Z'0", 32, 0, "GaugeBoson", 100, 0, 500, 14.5485, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(32));
   part->AddDecay(Particle::Decay(32, 0.145869,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0.145869,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(32, 0.14581,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(32, 0.113303,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0.113298,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 0.0636061,  vector<int>{12,-12}));
   part->AddDecay(Particle::Decay(0, 0.0636061,  vector<int>{14,-14}));
   part->AddDecay(Particle::Decay(0, 0.0636061,  vector<int>{16,-16}));
   part->AddDecay(Particle::Decay(32, 0.0490131,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0.0320071,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0320071,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.0320041,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{18,-18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{37,-37}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{23,22}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{23,25}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{25,36}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{35,36}));

   // Creating Z"0
   new Particle("Z\"0", 33, 0, "GaugeBoson", 100, 0, 900, 0, -100, -1, -100, -1, -1);

   // Creating W'+
   new Particle("W'+", 34, 1, "GaugeBoson", 100, 1, 500, 16.6708, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(34));
   part->AddDecay(Particle::Decay(32, 0.251276,  vector<int>{-1,2}));
   part->AddDecay(Particle::Decay(32, 0.250816,  vector<int>{-3,4}));
   part->AddDecay(Particle::Decay(32, 0.215459,  vector<int>{-5,6}));
   part->AddDecay(Particle::Decay(0, 0.085262,  vector<int>{-11,12}));
   part->AddDecay(Particle::Decay(0, 0.085262,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(0, 0.08526,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(32, 0.012903,  vector<int>{-3,2}));
   part->AddDecay(Particle::Decay(32, 0.012903,  vector<int>{-1,4}));
   part->AddDecay(Particle::Decay(32, 0.000465,  vector<int>{-5,4}));
   part->AddDecay(Particle::Decay(32, 0.00038,  vector<int>{-3,6}));
   part->AddDecay(Particle::Decay(32, 8e-06,  vector<int>{-5,2}));
   part->AddDecay(Particle::Decay(32, 6e-06,  vector<int>{-1,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-3,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-5,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-17,18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,23}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,22}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,25}));

   // Creating H0
   new Particle("H0", 35, 0, "GaugeBoson", 100, 0, 300, 8.42842, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(35));
   part->AddDecay(Particle::Decay(0, 0.688641,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(0, 0.306171,  vector<int>{23,23}));
   part->AddDecay(Particle::Decay(0, 0.003799,  vector<int>{25,25}));
   part->AddDecay(Particle::Decay(32, 0.000754001,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.000439,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 7.40001e-05,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(0, 6.10001e-05,  vector<int>{22,23}));
   part->AddDecay(Particle::Decay(32, 4.6e-05,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 1.5e-05,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{23,25}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{36,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000035}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-1000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-1000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-1000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-1000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-1000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-1000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-1000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-1000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-1000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,2000016}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0027() {

   // Creating A0
   new Particle("A0", 36, 0, "GaugeBoson", 100, 0, 300, 4.92026, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(36));
   part->AddDecay(Particle::Decay(0, 0.996235,  vector<int>{23,25}));
   part->AddDecay(Particle::Decay(0, 0.002256,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(32, 0.001292,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.000126,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 7.90002e-05,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 1e-05,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(0, 2e-06,  vector<int>{22,23}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{23,23}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000022}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000023}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000025}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000035}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,2000001}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-1000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-1000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,2000003}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-1000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,2000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-1000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,2000006}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-1000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,2000011}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-1000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,2000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-1000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,2000013}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-1000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,2000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-1000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,2000015}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-1000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-2000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,2000016}));

   // Creating H+
   new Particle("H+", 37, 1, "GaugeBoson", 100, 1, 300, 5.75967, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(37));
   part->AddDecay(Particle::Decay(0, 0.929792,  vector<int>{24,25}));
   part->AddDecay(Particle::Decay(32, 0.067484,  vector<int>{-5,6}));
   part->AddDecay(Particle::Decay(0, 0.002701,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(32, 1.3e-05,  vector<int>{-3,4}));
   part->AddDecay(Particle::Decay(0, 1e-05,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-17,18}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000024}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1000037}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-1000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-2000005}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,1000002}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,1000004}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,1000012}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,1000014}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,1000016}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,1000016}));

   // Creating eta_tech0
   new Particle("eta_tech0", 38, 0, "Unknown", 100, 0, 350, 0.10158, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(38));
   part->AddDecay(Particle::Decay(32, 0.547101,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(32, 0.452899,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));

   // Creating LQ_ue
   new Particle("LQ_ue", 39, 1, "Unknown", 100, -0.333333, 200, 0.39162, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(39));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{2,11}));

   // Creating R0
   new Particle("R0", 40, 1, "Unknown", 100, 0, 5000, 417.465, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(40));
   part->AddDecay(Particle::Decay(32, 0.215134,  vector<int>{1,-3}));
   part->AddDecay(Particle::Decay(32, 0.215134,  vector<int>{2,-4}));
   part->AddDecay(Particle::Decay(32, 0.215133,  vector<int>{3,-5}));
   part->AddDecay(Particle::Decay(32, 0.214738,  vector<int>{4,-6}));
   part->AddDecay(Particle::Decay(0, 0.0699301,  vector<int>{11,-13}));
   part->AddDecay(Particle::Decay(0, 0.0699301,  vector<int>{13,-15}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{5,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{15,-17}));

   // Creating pi_tech0
   new Particle("pi_tech0", 51, 0, "Unknown", 100, 0, 110, 0.04104, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(51));
   part->AddDecay(Particle::Decay(32, 0.596654,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(32, 0.316112,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 0.050055,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0.036777,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(32, 0.000225,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(0, 0.000177,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));

   // Creating pi_tech+
   new Particle("pi_tech+", 52, 1, "Unknown", 100, 1, 110, 0.0105, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(52));
   part->AddDecay(Particle::Decay(32, 0.90916,  vector<int>{4,-5}));
   part->AddDecay(Particle::Decay(0, 0.048905,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(32, 0.041762,  vector<int>{4,-3}));
   part->AddDecay(Particle::Decay(0, 0.000173,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{24,5,-5}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-11,12}));

   // Creating pi'_tech0
   new Particle("pi'_tech0", 53, 0, "Unknown", 100, 0, 110, 0.02807, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(53));
   part->AddDecay(Particle::Decay(32, 0.872445,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.0731921,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0.0537761,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(32, 0.000328,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(0, 0.000259,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{21,21}));

   // Creating rho_tech0
   new Particle("rho_tech0", 54, 0, "Unknown", 100, 0, 210, 0.82101, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(54));
   part->AddDecay(Particle::Decay(0, 0.342802,  vector<int>{24,-52}));
   part->AddDecay(Particle::Decay(0, 0.342802,  vector<int>{52,-24}));
   part->AddDecay(Particle::Decay(0, 0.153373,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(0, 0.0868672,  vector<int>{22,51}));
   part->AddDecay(Particle::Decay(0, 0.0312801,  vector<int>{22,53}));
   part->AddDecay(Particle::Decay(32, 0.00691101,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0.00691101,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(32, 0.00478901,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(32, 0.00478901,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0.00478901,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.00307701,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.00307701,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.00307701,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(0, 0.001598,  vector<int>{23,51}));
   part->AddDecay(Particle::Decay(0, 0.00103,  vector<int>{14,-14}));
   part->AddDecay(Particle::Decay(0, 0.00103,  vector<int>{12,-12}));
   part->AddDecay(Particle::Decay(0, 0.00103,  vector<int>{16,-16}));
   part->AddDecay(Particle::Decay(0, 0.000768002,  vector<int>{23,53}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{52,-52}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{18,-18}));

   // Creating rho_tech+
   new Particle("rho_tech+", 55, 1, "Unknown", 100, 1, 210, 0.64973, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(55));
   part->AddDecay(Particle::Decay(0, 0.474101,  vector<int>{24,51}));
   part->AddDecay(Particle::Decay(0, 0.176299,  vector<int>{52,23}));
   part->AddDecay(Particle::Decay(0, 0.138845,  vector<int>{24,23}));
   part->AddDecay(Particle::Decay(0, 0.109767,  vector<int>{52,22}));
   part->AddDecay(Particle::Decay(32, 0.0285839,  vector<int>{-1,2}));
   part->AddDecay(Particle::Decay(32, 0.0285299,  vector<int>{-3,4}));
   part->AddDecay(Particle::Decay(0, 0.00966098,  vector<int>{-11,12}));
   part->AddDecay(Particle::Decay(0, 0.00966098,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(0, 0.00965998,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(0, 0.00816098,  vector<int>{24,53}));
   part->AddDecay(Particle::Decay(32, 0.00373499,  vector<int>{-5,6}));
   part->AddDecay(Particle::Decay(32, 0.001468,  vector<int>{-3,2}));
   part->AddDecay(Particle::Decay(32, 0.001468,  vector<int>{-1,4}));
   part->AddDecay(Particle::Decay(32, 5.29999e-05,  vector<int>{-5,4}));
   part->AddDecay(Particle::Decay(32, 6.99999e-06,  vector<int>{-3,6}));
   part->AddDecay(Particle::Decay(32, 9.99998e-07,  vector<int>{-5,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-5,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,2}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,4}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,6}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-7,8}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-3,8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{52,51}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{-1,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-17,18}));

   // Creating omega_tech
   new Particle("omega_tech", 56, 0, "Unknown", 100, 0, 210, 0.1575, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(56));
   part->AddDecay(Particle::Decay(0, 0.45294,  vector<int>{22,53}));
   part->AddDecay(Particle::Decay(0, 0.163019,  vector<int>{22,51}));
   part->AddDecay(Particle::Decay(32, 0.045908,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0.045908,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(0, 0.038354,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.038354,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.038354,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(0, 0.038042,  vector<int>{52,-24}));
   part->AddDecay(Particle::Decay(0, 0.038042,  vector<int>{24,-52}));
   part->AddDecay(Particle::Decay(32, 0.017733,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(32, 0.017733,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0.017733,  vector<int>{5,-5}));
   part->AddDecay(Particle::Decay(0, 0.011181,  vector<int>{14,-14}));
   part->AddDecay(Particle::Decay(0, 0.011181,  vector<int>{12,-12}));
   part->AddDecay(Particle::Decay(0, 0.011181,  vector<int>{16,-16}));
   part->AddDecay(Particle::Decay(0, 0.00833401,  vector<int>{23,53}));
   part->AddDecay(Particle::Decay(0, 0.004003,  vector<int>{23,51}));
   part->AddDecay(Particle::Decay(0, 0.001999,  vector<int>{24,-24}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{7,-7}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{8,-8}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{52,-52}));
   part->AddDecay(Particle::Decay(32, 0,  vector<int>{6,-6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{17,-17}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{18,-18}));

   // Creating H_L++
   new Particle("H_L++", 61, 1, "Unknown", 100, 2, 200, 0.88161, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(61));
   part->AddDecay(Particle::Decay(0, 0.812251,  vector<int>{-15,-15}));
   part->AddDecay(Particle::Decay(0, 0.0902641,  vector<int>{-13,-13}));
   part->AddDecay(Particle::Decay(0, 0.0902641,  vector<int>{-11,-11}));
   part->AddDecay(Particle::Decay(0, 0.001806,  vector<int>{24,24}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{-13,-15}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{-11,-15}));
   part->AddDecay(Particle::Decay(0, 0.001805,  vector<int>{-11,-13}));

   // Creating H_R++
   new Particle("H_R++", 62, 1, "Unknown", 100, 2, 200, 0.88001, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(62));
   part->AddDecay(Particle::Decay(0, 0.813719,  vector<int>{-15,-15}));
   part->AddDecay(Particle::Decay(0, 0.0904279,  vector<int>{-13,-13}));
   part->AddDecay(Particle::Decay(0, 0.0904279,  vector<int>{-11,-11}));
   part->AddDecay(Particle::Decay(0, 0.001809,  vector<int>{-11,-13}));
   part->AddDecay(Particle::Decay(0, 0.001808,  vector<int>{-13,-15}));
   part->AddDecay(Particle::Decay(0, 0.001808,  vector<int>{-11,-15}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{63,63}));

   // Creating W_R+
   new Particle("W_R+", 63, 1, "Unknown", 100, 1, 750, 19.3391, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(63));
   part->AddDecay(Particle::Decay(32, 0.325914,  vector<int>{-1,2}));
   part->AddDecay(Particle::Decay(32, 0.32532,  vector<int>{-3,4}));
   part->AddDecay(Particle::Decay(32, 0.314118,  vector<int>{-5,6}));
   part->AddDecay(Particle::Decay(32, 0.016736,  vector<int>{-3,2}));
   part->AddDecay(Particle::Decay(32, 0.016735,  vector<int>{-1,4}));
   part->AddDecay(Particle::Decay(32, 0.000603001,  vector<int>{-5,4}));
   part->AddDecay(Particle::Decay(32, 0.000554001,  vector<int>{-3,6}));
   part->AddDecay(Particle::Decay(32, 1e-05,  vector<int>{-5,2}));
   part->AddDecay(Particle::Decay(32, 9.00001e-06,  vector<int>{-1,6}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-11,64}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-13,65}));
   part->AddDecay(Particle::Decay(0, 0,  vector<int>{-15,66}));

   // Creating nu_Re
   new Particle("nu_Re", 64, 1, "Unknown", 100, 0, 750, 0, -100, -1, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0028() {

   // Creating nu_Rmu
   new Particle("nu_Rmu", 65, 1, "Unknown", 100, 0, 750, 0, -100, -1, -100, -1, -1);

   // Creating nu_Rtau
   new Particle("nu_Rtau", 66, 1, "Unknown", 100, 0, 750, 0, -100, -1, -100, -1, -1);

   // Creating specflav
   new Particle("specflav", 81, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating rndmflav
   new Particle("rndmflav", 82, 1, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating phasespa
   new Particle("phasespa", 83, 0, "Generator", 100, 0, 1, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(83));
   part->AddDecay(Particle::Decay(12, 1,  vector<int>{82,-82}));

   // Creating c-hadron
   new Particle("c-hadron", 84, 1, "Generator", 100, 0.666667, 2, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(84));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating b-hadron
   new Particle("b-hadron", 85, 1, "Generator", 100, -0.333333, 5, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(85));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating cluster
   new Particle("cluster", 91, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating string
   new Particle("string", 92, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating indep.
   new Particle("indep.", 93, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating CMshower
   new Particle("CMshower", 94, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating SPHEaxis
   new Particle("SPHEaxis", 95, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating THRUaxis
   new Particle("THRUaxis", 96, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating CLUSjet
   new Particle("CLUSjet", 97, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating CELLjet
   new Particle("CELLjet", 98, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0029() {

   // Creating table
   new Particle("table", 99, 0, "Generator", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating rho_diff0
   new Particle("rho_diff0", 110, 0, "Unknown", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating pi0
   new Particle("pi0", 111, 0, "Meson", 100, 0, 0.134977, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(111));
   part->AddDecay(Particle::Decay(0, 0.988,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(2, 0.012,  vector<int>{22,11,-11}));

   // Creating rho0
   new Particle("rho0", 113, 0, "Meson", 100, 0, 0.77549, 0.151, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(113));
   part->AddDecay(Particle::Decay(3, 0.998739,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.00079,  vector<int>{111,22}));
   part->AddDecay(Particle::Decay(0, 0.00038,  vector<int>{221,22}));
   part->AddDecay(Particle::Decay(0, 4.6e-05,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 4.5e-05,  vector<int>{11,-11}));

   // Creating a_20
   new Particle("a_20", 115, 0, "Meson", 100, 0, 1.3183, 0.107, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(115));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{213,-211}));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{-213,211}));
   part->AddDecay(Particle::Decay(0, 0.144,  vector<int>{221,111}));
   part->AddDecay(Particle::Decay(0, 0.104,  vector<int>{223,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0245,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(0, 0.01225,  vector<int>{130,130}));
   part->AddDecay(Particle::Decay(0, 0.01225,  vector<int>{310,310}));
   part->AddDecay(Particle::Decay(0, 0.0057,  vector<int>{331,111}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{111,22}));

   // Creating rho3(1690)0
   new Particle("rho3(1690)0", 117, 1, "Unknown", 100, 0, 1.6888, 0.161, -100, 0, -100, -1, -1);

   // Creating K_L0
   new Particle("K_L0", 130, 0, "Meson", 100, 0, 0.497614, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(130));
   part->AddDecay(Particle::Decay(0, 0.2112,  vector<int>{111,111,111}));
   part->AddDecay(Particle::Decay(42, 0.1939,  vector<int>{-12,11,211}));
   part->AddDecay(Particle::Decay(42, 0.1939,  vector<int>{12,-11,-211}));
   part->AddDecay(Particle::Decay(42, 0.1359,  vector<int>{-14,13,211}));
   part->AddDecay(Particle::Decay(42, 0.1359,  vector<int>{14,-13,-211}));
   part->AddDecay(Particle::Decay(0, 0.1256,  vector<int>{211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{111,111}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{22,22}));

   // Creating pi_diffr+
   new Particle("pi_diffr+", 210, 1, "Meson", 100, 1, 0, 0, -100, -1, -100, -1, -1);

   // Creating pi+
   new Particle("pi+", 211, 1, "Meson", 100, 1, 0.13957, 2.52837e-17, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(211));
   part->AddDecay(Particle::Decay(0, 0.999877,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(0, 0.000123,  vector<int>{-11,12}));

   // Creating rho+
   new Particle("rho+", 213, 1, "Meson", 100, 1, 0.77549, 0.149, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(213));
   part->AddDecay(Particle::Decay(3, 0.99955,  vector<int>{211,111}));
   part->AddDecay(Particle::Decay(0, 0.00045,  vector<int>{211,22}));

   // Creating a_2+
   new Particle("a_2+", 215, 1, "Meson", 100, 1, 1.3183, 0.107, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(215));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{213,111}));
   part->AddDecay(Particle::Decay(0, 0.34725,  vector<int>{113,211}));
   part->AddDecay(Particle::Decay(0, 0.144,  vector<int>{221,211}));
   part->AddDecay(Particle::Decay(0, 0.104,  vector<int>{223,211,111}));
   part->AddDecay(Particle::Decay(0, 0.049,  vector<int>{321,-311}));
   part->AddDecay(Particle::Decay(0, 0.0057,  vector<int>{331,211}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{211,22}));

   // Creating rho3(1690)+
   new Particle("rho3(1690)+", 217, 1, "Unknown", 100, 1, 1.6888, 0.161, -100, 0, -100, -1, -1);

   // Creating omega_di
   new Particle("omega_di", 220, 0, "Meson", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating eta
   new Particle("eta", 221, 0, "Meson", 100, 0, 0.547853, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(221));
   part->AddDecay(Particle::Decay(0, 0.3923,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(0, 0.321,  vector<int>{111,111,111}));
   part->AddDecay(Particle::Decay(0, 0.2317,  vector<int>{211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.0478,  vector<int>{22,211,-211}));
   part->AddDecay(Particle::Decay(2, 0.0049,  vector<int>{22,11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0013,  vector<int>{211,-211,11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{111,22,22}));
   part->AddDecay(Particle::Decay(0, 0.0003,  vector<int>{22,13,-13}));

   // Creating omega
   new Particle("omega", 223, 0, "Meson", 100, 0, 0.78265, 0.00843, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(223));
   part->AddDecay(Particle::Decay(1, 0.89,  vector<int>{211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.08693,  vector<int>{22,111}));
   part->AddDecay(Particle::Decay(3, 0.0221,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.00083,  vector<int>{221,22}));
   part->AddDecay(Particle::Decay(0, 7e-05,  vector<int>{111,111,22}));
   part->AddDecay(Particle::Decay(0, 7e-05,  vector<int>{11,-11}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0030() {

   // Creating f_2
   new Particle("f_2", 225, 0, "Meson", 100, 0, 1.2751, 0.185, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(225));
   part->AddDecay(Particle::Decay(0, 0.564,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.282,  vector<int>{111,111}));
   part->AddDecay(Particle::Decay(0, 0.072,  vector<int>{211,-211,111,111}));
   part->AddDecay(Particle::Decay(0, 0.028,  vector<int>{211,-211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.023,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(0, 0.0115,  vector<int>{130,130}));
   part->AddDecay(Particle::Decay(0, 0.0115,  vector<int>{310,310}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{221,221}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{111,111,111,111}));

   // Creating omega3(1670)
   new Particle("omega3(1670)", 227, 1, "Unknown", 100, 0, 1.667, 0.168, -100, 0, -100, -1, -1);

   // Creating K_S0
   new Particle("K_S0", 310, 0, "Meson", 100, 0, 0.497614, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(310));
   part->AddDecay(Particle::Decay(0, 0.6861,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.3139,  vector<int>{111,111}));

   // Creating K0
   new Particle("K0", 311, 1, "Meson", 100, 0, 0.497614, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(311));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{130}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{310}));

   // Creating K*0
   new Particle("K*0", 313, 1, "Meson", 100, 0, 0.896, 0.0505, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(313));
   part->AddDecay(Particle::Decay(3, 0.665,  vector<int>{321,-211}));
   part->AddDecay(Particle::Decay(3, 0.333,  vector<int>{311,111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{311,22}));

   // Creating K*_20
   new Particle("K*_20", 315, 1, "Meson", 100, 0, 1.4324, 0.109, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(315));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{321,-211}));
   part->AddDecay(Particle::Decay(0, 0.168,  vector<int>{323,-211}));
   part->AddDecay(Particle::Decay(0, 0.166,  vector<int>{311,111}));
   part->AddDecay(Particle::Decay(0, 0.087,  vector<int>{323,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.084,  vector<int>{313,111}));
   part->AddDecay(Particle::Decay(0, 0.059,  vector<int>{321,-213}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{313,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{311,113}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{311,223}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{311,221}));

   // Creating k3_star(1780)0
   new Particle("k3_star(1780)0", 317, 1, "Unknown", 100, 0, 1.776, 0.159, -100, 0, -100, -1, -1);

   // Creating K+
   new Particle("K+", 321, 1, "Meson", 100, 1, 0.493677, 5.31674e-17, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(321));
   part->AddDecay(Particle::Decay(0, 0.6352,  vector<int>{-13,14}));
   part->AddDecay(Particle::Decay(0, 0.2116,  vector<int>{211,111}));
   part->AddDecay(Particle::Decay(0, 0.0559,  vector<int>{211,211,-211}));
   part->AddDecay(Particle::Decay(42, 0.0482,  vector<int>{12,-11,111}));
   part->AddDecay(Particle::Decay(42, 0.0318,  vector<int>{14,-13,111}));
   part->AddDecay(Particle::Decay(0, 0.0173,  vector<int>{211,111,111}));

   // Creating K*+
   new Particle("K*+", 323, 1, "Meson", 100, 1, 0.89166, 0.0498, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(323));
   part->AddDecay(Particle::Decay(3, 0.666,  vector<int>{311,211}));
   part->AddDecay(Particle::Decay(3, 0.333,  vector<int>{321,111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{321,22}));

   // Creating K*_2+
   new Particle("K*_2+", 325, 1, "Meson", 100, 1, 1.4256, 0.098, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(325));
   part->AddDecay(Particle::Decay(0, 0.332,  vector<int>{311,211}));
   part->AddDecay(Particle::Decay(0, 0.168,  vector<int>{313,211}));
   part->AddDecay(Particle::Decay(0, 0.166,  vector<int>{321,111}));
   part->AddDecay(Particle::Decay(0, 0.086,  vector<int>{313,211,111}));
   part->AddDecay(Particle::Decay(0, 0.084,  vector<int>{323,111}));
   part->AddDecay(Particle::Decay(0, 0.059,  vector<int>{311,213}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{323,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{321,113}));
   part->AddDecay(Particle::Decay(0, 0.029,  vector<int>{321,223}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{321,221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{321,22}));

   // Creating k3_star(1780)+
   new Particle("k3_star(1780)+", 327, 1, "Unknown", 100, 1, 1.776, 0.159, -100, 0, -100, -1, -1);

   // Creating phi_diff
   new Particle("phi_diff", 330, 0, "Meson", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating eta'
   new Particle("eta'", 331, 0, "Meson", 100, 0, 0.95766, 0.0002, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(331));
   part->AddDecay(Particle::Decay(0, 0.437,  vector<int>{211,-211,221}));
   part->AddDecay(Particle::Decay(0, 0.302,  vector<int>{22,113}));
   part->AddDecay(Particle::Decay(0, 0.208,  vector<int>{111,111,221}));
   part->AddDecay(Particle::Decay(0, 0.0302,  vector<int>{22,223}));
   part->AddDecay(Particle::Decay(0, 0.0212,  vector<int>{22,22}));
   part->AddDecay(Particle::Decay(0, 0.0016,  vector<int>{111,111,111}));

   // Creating phi
   new Particle("phi", 333, 0, "Meson", 100, 0, 1.01945, 0.00443, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(333));
   part->AddDecay(Particle::Decay(3, 0.48947,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(3, 0.34,  vector<int>{130,310}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{-213,211}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{113,111}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{213,-211}));
   part->AddDecay(Particle::Decay(1, 0.027,  vector<int>{211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.0126,  vector<int>{22,221}));
   part->AddDecay(Particle::Decay(0, 0.0013,  vector<int>{111,22}));
   part->AddDecay(Particle::Decay(0, 0.0003,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.00025,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 8e-05,  vector<int>{211,-211}));

   // Creating f'_2
   new Particle("f'_2", 335, 0, "Meson", 100, 0, 1.525, 0.076, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(335));
   part->AddDecay(Particle::Decay(0, 0.444,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(0, 0.222,  vector<int>{130,130}));
   part->AddDecay(Particle::Decay(0, 0.222,  vector<int>{310,310}));
   part->AddDecay(Particle::Decay(0, 0.104,  vector<int>{221,221}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{111,111}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0031() {

   // Creating phi3(1850)
   new Particle("phi3(1850)", 337, 1, "Unknown", 100, 0, 1.854, 0.087, -100, 0, -100, -1, -1);

   // Creating D+
   new Particle("D+", 411, 1, "CharmedMeson", 100, 1, 1.86962, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(411));
   part->AddDecay(Particle::Decay(0, 0.087,  vector<int>{-311,211,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.076,  vector<int>{-311,20213}));
   part->AddDecay(Particle::Decay(42, 0.07,  vector<int>{-11,12,-311}));
   part->AddDecay(Particle::Decay(42, 0.07,  vector<int>{-13,14,-311}));
   part->AddDecay(Particle::Decay(0, 0.067,  vector<int>{-321,211,211}));
   part->AddDecay(Particle::Decay(0, 0.066,  vector<int>{-311,213}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{-11,12,-313}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{-13,14,-313}));
   part->AddDecay(Particle::Decay(0, 0.045,  vector<int>{-20313,211}));
   part->AddDecay(Particle::Decay(0, 0.041,  vector<int>{-313,213}));
   part->AddDecay(Particle::Decay(0, 0.027,  vector<int>{-311,321,-311}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{-313,323}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{-311,211}));
   part->AddDecay(Particle::Decay(0, 0.022,  vector<int>{-321,211,211,111,111}));
   part->AddDecay(Particle::Decay(0, 0.0218,  vector<int>{211,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.019,  vector<int>{333,211,111}));
   part->AddDecay(Particle::Decay(0, 0.019,  vector<int>{-313,211}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{-311,211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.012,  vector<int>{-311,211,111}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{-13,14,-323,211}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{-11,12,-313,111}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{-11,12,-323,211}));
   part->AddDecay(Particle::Decay(42, 0.011,  vector<int>{-13,14,-313,111}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{-321,211,211,111}));
   part->AddDecay(Particle::Decay(0, 0.008,  vector<int>{-321,213,211}));
   part->AddDecay(Particle::Decay(0, 0.0073,  vector<int>{-311,321}));
   part->AddDecay(Particle::Decay(0, 0.0066,  vector<int>{221,211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{333,211}));
   part->AddDecay(Particle::Decay(0, 0.0057,  vector<int>{-313,211,113}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{333,213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,-311,111}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,-321,211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,-311,111}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{221,213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,-321,211}));
   part->AddDecay(Particle::Decay(0, 0.0047,  vector<int>{-313,321}));
   part->AddDecay(Particle::Decay(0, 0.0047,  vector<int>{-311,323}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{-321,321,211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{331,211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{331,213}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{113,211,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.0022,  vector<int>{211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-313,211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{-321,113,211,211,111}));
   part->AddDecay(Particle::Decay(0, 0.0015,  vector<int>{211,211,211,-211,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{111,211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,221}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,331}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,113}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,223}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{223,211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{223,213}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-321,211,211,211,-211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,111}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-311,113,211,211,-211}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,331}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,113}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,223}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{113,213}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{111,213}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{113,211}));

   // Creating D*+
   new Particle("D*+", 413, 1, "CharmedMeson", 100, 1, 2.01027, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(413));
   part->AddDecay(Particle::Decay(3, 0.683,  vector<int>{421,211}));
   part->AddDecay(Particle::Decay(3, 0.306,  vector<int>{411,111}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{411,22}));

   // Creating D*_2+
   new Particle("D*_2+", 415, 1, "CharmedMeson", 100, 1, 2.4601, 0.023, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(415));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{421,211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{423,211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{411,111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{423,211,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{413,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{421,211,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{413,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{411,211,-211}));

   // Creating D0
   new Particle("D0", 421, 1, "CharmedMeson", 100, 0, 1.86484, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(421));
   part->AddDecay(Particle::Decay(0, 0.0923,  vector<int>{-321,211,111,111}));
   part->AddDecay(Particle::Decay(0, 0.074,  vector<int>{-321,20213}));
   part->AddDecay(Particle::Decay(0, 0.073,  vector<int>{-321,213}));
   part->AddDecay(Particle::Decay(0, 0.067,  vector<int>{-311,211,-211,111,111}));
   part->AddDecay(Particle::Decay(0, 0.062,  vector<int>{-323,213}));
   part->AddDecay(Particle::Decay(0, 0.0511,  vector<int>{-311,113,111,111,111}));
   part->AddDecay(Particle::Decay(0, 0.045,  vector<int>{-323,211}));
   part->AddDecay(Particle::Decay(0, 0.0365,  vector<int>{-321,211}));
   part->AddDecay(Particle::Decay(42, 0.034,  vector<int>{-11,12,-321}));
   part->AddDecay(Particle::Decay(42, 0.034,  vector<int>{-13,14,-321}));
   part->AddDecay(Particle::Decay(42, 0.027,  vector<int>{-11,12,-323}));
   part->AddDecay(Particle::Decay(42, 0.027,  vector<int>{-13,14,-323}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{-311,223}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{-321,211,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.022,  vector<int>{-311,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{-313,111}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{-313,221}));
   part->AddDecay(Particle::Decay(0, 0.021,  vector<int>{-311,111}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{-311,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{-321,211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.017,  vector<int>{211,211,-211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.016,  vector<int>{-313,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-313,113}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{-321,211,111}));
   part->AddDecay(Particle::Decay(0, 0.0109,  vector<int>{-10323,211}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{-311,321,-321,111}));
   part->AddDecay(Particle::Decay(0, 0.0088,  vector<int>{-311,333}));
   part->AddDecay(Particle::Decay(0, 0.0085,  vector<int>{-311,211,211,-211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0077,  vector<int>{-313,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.0075,  vector<int>{211,211,-211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0063,  vector<int>{-321,211,113}));
   part->AddDecay(Particle::Decay(0, 0.0061,  vector<int>{-311,113}));
   part->AddDecay(Particle::Decay(0, 0.0052,  vector<int>{-321,321,-311}));
   part->AddDecay(Particle::Decay(0, 0.0041,  vector<int>{-321,321}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{-13,14,-323,111}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{-11,12,-313,-211}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{-11,12,-323,111}));
   part->AddDecay(Particle::Decay(42, 0.004,  vector<int>{-13,14,-313,-211}));
   part->AddDecay(Particle::Decay(0, 0.0036,  vector<int>{-313,321,-211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{-321,323}));
   part->AddDecay(Particle::Decay(0, 0.0034,  vector<int>{-321,311,211}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{321,-321,211,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.0027,  vector<int>{-313,313}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,-311,-211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,-321,111}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,-211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,-213}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{-323,321}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,-211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,-213}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,-311,-211}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,-321,111}));
   part->AddDecay(Particle::Decay(0, 0.0018,  vector<int>{333,113}));
   part->AddDecay(Particle::Decay(0, 0.0016,  vector<int>{111,111}));
   part->AddDecay(Particle::Decay(0, 0.0016,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{-311,311}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-313,311}));
   part->AddDecay(Particle::Decay(0, 0.0009,  vector<int>{310,310,310}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{333,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{113,211,211,-211,-211}));

   // Creating D*0
   new Particle("D*0", 423, 1, "CharmedMeson", 100, 0, 2.00697, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(423));
   part->AddDecay(Particle::Decay(3, 0.619,  vector<int>{421,111}));
   part->AddDecay(Particle::Decay(0, 0.381,  vector<int>{421,22}));

   // Creating D*_20
   new Particle("D*_20", 425, 1, "CharmedMeson", 100, 0, 2.4611, 0.023, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(425));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{411,-211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{413,-211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{421,111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{413,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{423,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{411,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{423,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{421,211,-211}));

   // Creating D_s+
   new Particle("D_s+", 431, 1, "CharmedMeson", 100, 1, 1.9685, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(431));
   part->AddDecay(Particle::Decay(13, 0.25,  vector<int>{2,-1,3,-3}));
   part->AddDecay(Particle::Decay(13, 0.0952,  vector<int>{2,-1}));
   part->AddDecay(Particle::Decay(0, 0.095,  vector<int>{331,213}));
   part->AddDecay(Particle::Decay(0, 0.079,  vector<int>{221,213}));
   part->AddDecay(Particle::Decay(0, 0.052,  vector<int>{333,213}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{323,-313}));
   part->AddDecay(Particle::Decay(0, 0.037,  vector<int>{331,211}));
   part->AddDecay(Particle::Decay(0, 0.033,  vector<int>{323,-311}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{-11,12,333}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{-13,14,333}));
   part->AddDecay(Particle::Decay(0, 0.028,  vector<int>{333,211}));
   part->AddDecay(Particle::Decay(0, 0.028,  vector<int>{321,-311}));
   part->AddDecay(Particle::Decay(0, 0.026,  vector<int>{321,-313}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-11,12,331}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-11,12,221}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-13,14,221}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{-13,14,331}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{221,211}));
   part->AddDecay(Particle::Decay(0, 0.01,  vector<int>{-15,16}));
   part->AddDecay(Particle::Decay(0, 0.01,  vector<int>{2212,-2112}));
   part->AddDecay(Particle::Decay(0, 0.0078,  vector<int>{10221,211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,321,-321}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,311,-311}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{221,321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{331,321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{333,321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{221,323}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,311,-311}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,321,-321}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{213,113}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{211,111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{213,111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{211,113}));

   // Creating D*_s+
   new Particle("D*_s+", 433, 1, "CharmedMeson", 100, 1, 2.1123, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(433));
   part->AddDecay(Particle::Decay(0, 0.94,  vector<int>{431,22}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{431,111}));

   // Creating D*_2s+
   new Particle("D*_2s+", 435, 1, "CharmedMeson", 100, 1, 2.5726, 0.015, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(435));
   part->AddDecay(Particle::Decay(0, 0.4,  vector<int>{421,321}));
   part->AddDecay(Particle::Decay(0, 0.4,  vector<int>{411,311}));
   part->AddDecay(Particle::Decay(0, 0.1,  vector<int>{423,321}));
   part->AddDecay(Particle::Decay(0, 0.1,  vector<int>{413,311}));

   // Creating J/psi_di
   new Particle("J/psi_di", 440, 0, "CharmedMeson", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating eta_c
   new Particle("eta_c", 441, 0, "CharmedMeson", 100, 0, 2.9803, 0.0013, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(441));
   part->AddDecay(Particle::Decay(12, 1,  vector<int>{82,-82}));

   // Creating J/psi
   new Particle("J/psi", 443, 0, "Meson", 100, 0, 3.09692, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(443));
   part->AddDecay(Particle::Decay(12, 0.8797,  vector<int>{82,-82}));
   part->AddDecay(Particle::Decay(0, 0.0602,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0601,  vector<int>{13,-13}));

   // Creating chi_2c
   new Particle("chi_2c", 445, 0, "CharmedMeson", 100, 0, 3.5562, 0.002, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(445));
   part->AddDecay(Particle::Decay(12, 0.865,  vector<int>{82,-82}));
   part->AddDecay(Particle::Decay(0, 0.135,  vector<int>{443,22}));

   // Creating B0
   new Particle("B0", 511, 1, "B-Meson", 100, 0, 5.27953, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(511));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{2,-1,-4,1}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{2,-4,-1,1}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{4,-3,-4,1}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{14,-13,-413}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{12,-11,-413}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{16,-15,-413}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{-413,433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{12,-11,-411}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{14,-13,-411}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{4,-4,-3,1}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{-411,433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{-413,20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-411,431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,1}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{-413,431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{14,-13,-415}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{12,-11,-415}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{-411,213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{16,-15,-411}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{-413,213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{14,-13,-20413}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{12,-11,-20413}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{-411,20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10411}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10413}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10411}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10413}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,1}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{-413,211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{-411,211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{20443,313}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{20443,311}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{443,313}));
   part->AddDecay(Particle::Decay(0, 0.0008,  vector<int>{443,311}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{441,313}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{441,311}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0032() {

   // Creating B*0
   new Particle("B*0", 513, 1, "B-Meson", 100, 0, 5.3251, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(513));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{511,22}));

   // Creating B*_20
   new Particle("B*_20", 515, 1, "B-Meson", 100, 0, 5.7469, 0.02, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(515));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{521,-211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{523,-211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{511,111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{523,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{513,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{521,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{513,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{511,211,-211}));

   // Creating B+
   new Particle("B+", 521, 1, "B-Meson", 100, 1, 5.27915, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(521));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{2,-1,-4,2}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{2,-4,-1,2}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{4,-3,-4,2}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{14,-13,-423}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{12,-11,-423}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{16,-15,-423}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{-423,433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{12,-11,-421}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{14,-13,-421}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{4,-4,-3,2}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{-421,433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{-423,20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-421,431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,2}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{-423,431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{14,-13,-425}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{12,-11,-425}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{-421,213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{16,-15,-421}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{-423,213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{14,-13,-20423}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{12,-11,-20423}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{-421,20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10421}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10423}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10421}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10423}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,2}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{-423,211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{-421,211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{20443,323}));
   part->AddDecay(Particle::Decay(0, 0.0019,  vector<int>{20443,321}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{443,323}));
   part->AddDecay(Particle::Decay(0, 0.0008,  vector<int>{443,321}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{441,323}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{441,321}));

   // Creating B*+
   new Particle("B*+", 523, 1, "Meson", 100, 1, 5.3251, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(523));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{521,22}));

   // Creating B*_2+
   new Particle("B*_2+", 525, 1, "B-Meson", 100, 1, 5.7469, 0.02, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(525));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{511,211}));
   part->AddDecay(Particle::Decay(0, 0.16,  vector<int>{513,211}));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{521,111}));
   part->AddDecay(Particle::Decay(0, 0.13,  vector<int>{513,211,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{523,111}));
   part->AddDecay(Particle::Decay(0, 0.08,  vector<int>{511,211,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{523,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.04,  vector<int>{521,211,-211}));

   // Creating B_s0
   new Particle("B_s0", 531, 1, "B-Meson", 100, 0, 5.3663, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(531));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{2,-1,-4,3}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{2,-4,-1,3}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{4,-3,-4,3}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{14,-13,-433}));
   part->AddDecay(Particle::Decay(42, 0.055,  vector<int>{12,-11,-433}));
   part->AddDecay(Particle::Decay(42, 0.03,  vector<int>{16,-15,-433}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{-433,433}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{12,-11,-431}));
   part->AddDecay(Particle::Decay(42, 0.02,  vector<int>{14,-13,-431}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{4,-4,-3,3}));
   part->AddDecay(Particle::Decay(0, 0.0185,  vector<int>{-431,433}));
   part->AddDecay(Particle::Decay(0, 0.018,  vector<int>{-433,20213}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{-431,431}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{2,-1,-2,3}));
   part->AddDecay(Particle::Decay(0, 0.0135,  vector<int>{-433,431}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{14,-13,-435}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{12,-11,-435}));
   part->AddDecay(Particle::Decay(0, 0.011,  vector<int>{-431,213}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{16,-15,-431}));
   part->AddDecay(Particle::Decay(0, 0.009,  vector<int>{-433,213}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{14,-13,-20433}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{12,-11,-20433}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{-431,20213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10431}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10433}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{14,-13,-10431}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{12,-11,-10433}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{4,-3,-2,3}));
   part->AddDecay(Particle::Decay(0, 0.0042,  vector<int>{-433,211}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{-431,211}));
   part->AddDecay(Particle::Decay(0, 0.0025,  vector<int>{20443,333}));
   part->AddDecay(Particle::Decay(0, 0.0014,  vector<int>{443,333}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{20443,221}));
   part->AddDecay(Particle::Decay(0, 0.0009,  vector<int>{20443,331}));
   part->AddDecay(Particle::Decay(0, 0.0007,  vector<int>{441,333}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{443,221}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{443,331}));
   part->AddDecay(Particle::Decay(0, 0.0002,  vector<int>{441,331}));
   part->AddDecay(Particle::Decay(0, 0.0002,  vector<int>{441,221}));

   // Creating B*_s0
   new Particle("B*_s0", 533, 1, "B-Meson", 100, 0, 5.4128, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(533));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{531,22}));

   // Creating B*_2s0
   new Particle("B*_2s0", 535, 1, "B-Meson", 100, 0, 5.8397, 0.02, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(535));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{521,-321}));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{511,-311}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{523,-321}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{513,-311}));

   // Creating ChargedRootino_bar-50000052
   new Particle("ChargedRootino_bar-50000052", 540, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0);

   // Creating B_c+
   new Particle("B_c+", 541, 1, "B-Meson", 100, 1, 6.276, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(541));
   part->AddDecay(Particle::Decay(42, 0.24,  vector<int>{-1,2,3,-5}));
   part->AddDecay(Particle::Decay(42, 0.15,  vector<int>{2,-1,-4,4}));
   part->AddDecay(Particle::Decay(11, 0.122,  vector<int>{4,-3}));
   part->AddDecay(Particle::Decay(42, 0.065,  vector<int>{-1,3,2,-5}));
   part->AddDecay(Particle::Decay(42, 0.05,  vector<int>{4,-3,-4,4}));
   part->AddDecay(Particle::Decay(0, 0.047,  vector<int>{16,-15}));
   part->AddDecay(Particle::Decay(42, 0.042,  vector<int>{-11,12,533}));
   part->AddDecay(Particle::Decay(42, 0.042,  vector<int>{-13,14,533}));
   part->AddDecay(Particle::Decay(42, 0.037,  vector<int>{2,-4,-1,4}));
   part->AddDecay(Particle::Decay(42, 0.035,  vector<int>{14,-13,443}));
   part->AddDecay(Particle::Decay(42, 0.035,  vector<int>{12,-11,443}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{4,-4,-3,4}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{-13,14,531}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{-11,12,531}));
   part->AddDecay(Particle::Decay(42, 0.014,  vector<int>{-1,2,1,-5}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{12,-11,441}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{-3,2,3,-5}));
   part->AddDecay(Particle::Decay(42, 0.012,  vector<int>{14,-13,441}));
   part->AddDecay(Particle::Decay(42, 0.008,  vector<int>{2,-3,-4,4}));
   part->AddDecay(Particle::Decay(42, 0.007,  vector<int>{16,-15,443}));
   part->AddDecay(Particle::Decay(11, 0.006,  vector<int>{4,-1}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{16,-15,441}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-3,3,2,-5}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{4,-1,-4,4}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-1,1,2,-5}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,513}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{2,-4,-3,4}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,513}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-11,12,511}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{4,-4,-1,4}));
   part->AddDecay(Particle::Decay(42, 0.001,  vector<int>{-13,14,511}));

   // Creating B*_c+
   new Particle("B*_c+", 543, 1, "B-Meson", 100, 1, 6.602, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(543));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{541,22}));

   // Creating B*_2c+
   new Particle("B*_2c+", 545, 1, "B-Meson", 100, 1, 7.35, 0.02, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(545));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{511,411}));
   part->AddDecay(Particle::Decay(0, 0.3,  vector<int>{521,421}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{513,411}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{523,421}));

   // Creating eta_b
   new Particle("eta_b", 551, 0, "B-Meson", 100, 0, 9.4, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(551));
   part->AddDecay(Particle::Decay(32, 1,  vector<int>{21,21}));

   // Creating Upsilon
   new Particle("Upsilon", 553, 0, "B-Meson", 100, 0, 9.4603, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(553));
   part->AddDecay(Particle::Decay(4, 0.7743,  vector<int>{21,21,21}));
   part->AddDecay(Particle::Decay(32, 0.045,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(32, 0.045,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(4, 0.029,  vector<int>{22,21,21}));
   part->AddDecay(Particle::Decay(0, 0.0267,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(0, 0.0252,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0248,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(32, 0.015,  vector<int>{3,-3}));
   part->AddDecay(Particle::Decay(32, 0.015,  vector<int>{1,-1}));

   // Creating chi_2b
   new Particle("chi_2b", 555, 0, "B-Meson", 100, 0, 9.9122, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(555));
   part->AddDecay(Particle::Decay(32, 0.78,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 0.22,  vector<int>{553,22}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0033() {

   // Creating dd_1
   new Particle("dd_1", 1103, 1, "Unknown", 100, -0.666667, 0.96, 0, -100, -1, -100, -1, -1);

   // Creating delta(1620)-
   new Particle("delta(1620)-", 1112, 1, "Unknown", 100, -1, 1.63, 0.145, -100, 0, -100, -1, -1);

   // Creating Delta-
   new Particle("Delta-", 1114, 1, "Unknown", 100, -1, 1.232, 0.12, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(1114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{2112,-211}));

   // Creating delta(1905)-
   new Particle("delta(1905)-", 1116, 1, "Unknown", 100, -1, 1.89, 0.33, -100, 0, -100, -1, -1);

   // Creating delta(1950)-
   new Particle("delta(1950)-", 1118, 1, "Unknown", 100, -1, 1.93, 0.28, -100, 0, -100, -1, -1);

   // Creating delta(1620)0
   new Particle("delta(1620)0", 1212, 1, "Unknown", 100, 0, 1.63, 0.145, -100, 0, -100, -1, -1);

   // Creating N(1520)0
   new Particle("N(1520)0", 1214, 1, "Unknown", 100, 0, 1.52, 0.115, -100, 0, -100, -1, -1);

   // Creating delta(1905)0
   new Particle("delta(1905)0", 1216, 1, "Unknown", 100, 0, 1.89, 0.33, -100, 0, -100, -1, -1);

   // Creating N(2190)0
   new Particle("N(2190)0", 1218, 1, "Unknown", 100, 0, 2.19, 0.5, -100, 0, -100, -1, -1);

   // Creating ud_0
   new Particle("ud_0", 2101, 1, "Unknown", 100, 0.333333, 0.0073, 0, -100, -1, -100, -1, -1);

   // Creating ud_1
   new Particle("ud_1", 2103, 1, "Unknown", 100, 0.333333, 0.0072, 0, -100, -1, -100, -1, -1);

   // Creating n_diffr0
   new Particle("n_diffr0", 2110, 1, "Unknown", 100, 0, 0, 0, -100, -1, -100, -1, -1);

   // Creating neutron
   new Particle("neutron", 2112, 1, "Baryon", 100, 0, 0.939565, 0, -100, -1, -100, -1, -1);

   // Creating Delta0
   new Particle("Delta0", 2114, 1, "Baryon", 100, 0, 1.232, 0.12, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2114));
   part->AddDecay(Particle::Decay(0, 0.663,  vector<int>{2112,111}));
   part->AddDecay(Particle::Decay(0, 0.331,  vector<int>{2212,-211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{2112,22}));

   // Creating N(1675)0
   new Particle("N(1675)0", 2116, 1, "Unknown", 100, 0, 1.675, 0.15, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0034() {

   // Creating delta(1950)0
   new Particle("delta(1950)0", 2118, 1, "Unknown", 100, 0, 1.93, 0.28, -100, 0, -100, -1, -1);

   // Creating delta(1620)+
   new Particle("delta(1620)+", 2122, 1, "Unknown", 100, 1, 1.63, 0.145, -100, 0, -100, -1, -1);

   // Creating N(1520)+
   new Particle("N(1520)+", 2124, 1, "Unknown", 100, 1, 1.52, 0.115, -100, 0, -100, -1, -1);

   // Creating delta(1905)+
   new Particle("delta(1905)+", 2126, 1, "Unknown", 100, 1, 1.89, 0.33, -100, 0, -100, -1, -1);

   // Creating N(2190)+
   new Particle("N(2190)+", 2128, 1, "Unknown", 100, 1, 2.19, 0.5, -100, 0, -100, -1, -1);

   // Creating uu_1
   new Particle("uu_1", 2203, 1, "Unknown", 100, 1.33333, 0.0048, 0, -100, -1, -100, -1, -1);

   // Creating p_diffr+
   new Particle("p_diffr+", 2210, 1, "Unknown", 100, 1, 0, 0, -100, -1, -100, -1, -1);

   // Creating proton
   new Particle("proton", 2212, 1, "Baryon", 100, 1, 0.938272, 0, -100, -1, -100, -1, -1);

   // Creating Delta+
   new Particle("Delta+", 2214, 1, "Baryon", 100, 1, 1.232, 0.12, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(2214));
   part->AddDecay(Particle::Decay(0, 0.663,  vector<int>{2212,111}));
   part->AddDecay(Particle::Decay(0, 0.331,  vector<int>{2112,211}));
   part->AddDecay(Particle::Decay(0, 0.006,  vector<int>{2212,22}));

   // Creating N(1675)+
   new Particle("N(1675)+", 2216, 1, "Unknown", 100, 1, 1.675, 0.15, -100, 0, -100, -1, -1);

   // Creating delta(1950)+
   new Particle("delta(1950)+", 2218, 1, "Unknown", 100, 1, 1.93, 0.28, -100, 0, -100, -1, -1);

   // Creating delta(1620)++
   new Particle("delta(1620)++", 2222, 1, "Unknown", 100, 2, 1.63, 0.145, -100, 0, -100, -1, -1);

   // Creating Delta++
   new Particle("Delta++", 2224, 1, "Baryon", 100, 2, 1.232, 0.12, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{2212,211}));

   // Creating delta(1905)++
   new Particle("delta(1905)++", 2226, 1, "Unknown", 100, 2, 1.89, 0.33, -100, 0, -100, -1, -1);

   // Creating delta(1950)++
   new Particle("delta(1950)++", 2228, 1, "Unknown", 100, 2, 1.93, 0.28, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0035() {

   // Creating sd_0
   new Particle("sd_0", 3101, 1, "Unknown", 100, -0.666667, 0.108, 0, -100, -1, -100, -1, -1);

   // Creating sd_1
   new Particle("sd_1", 3103, 1, "Unknown", 100, -0.666667, 0.1088, 0, -100, -1, -100, -1, -1);

   // Creating Sigma-
   new Particle("Sigma-", 3112, 1, "Baryon", 100, -1, 1.19744, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(3112));
   part->AddDecay(Particle::Decay(0, 0.999,  vector<int>{2112,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-12,11,2112}));

   // Creating Sigma*-
   new Particle("Sigma*-", 3114, 1, "Baryon", 100, -1, 1.3872, 0.0394, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3114));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{3122,-211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3212,-211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3112,111}));

   // Creating sigma(1775)-
   new Particle("sigma(1775)-", 3116, 1, "Unknown", 100, -1, 1.775, 0.12, -100, 0, -100, -1, -1);

   // Creating sigma(2030)-
   new Particle("sigma(2030)-", 3118, 1, "Unknown", 100, -1, 2.03, 0.18, -100, 0, -100, -1, -1);

   // Creating Lambda0
   new Particle("Lambda0", 3122, 1, "Baryon", 100, 0, 1.11568, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3122));
   part->AddDecay(Particle::Decay(0, 0.639,  vector<int>{2212,-211}));
   part->AddDecay(Particle::Decay(0, 0.358,  vector<int>{2112,111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{2112,22}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{-12,11,2212}));

   // Creating lambda(1520)
   new Particle("lambda(1520)", 3124, 1, "Unknown", 100, 0, 1.5195, 0.0156, -100, 0, -100, -1, -1);

   // Creating lambda(1820)
   new Particle("lambda(1820)", 3126, 1, "Unknown", 100, 0, 1.82, 0.08, -100, 0, -100, -1, -1);

   // Creating lambda(2100)
   new Particle("lambda(2100)", 3128, 1, "Unknown", 100, 0, 2.1, 0.2, -100, 0, -100, -1, -1);

   // Creating su_0
   new Particle("su_0", 3201, 1, "Unknown", 100, 0.333333, 0.1064, 0, -100, -1, -100, -1, -1);

   // Creating su_1
   new Particle("su_1", 3203, 1, "Unknown", 100, 0.333333, 0.1064, 0, -100, -1, -100, -1, -1);

   // Creating Sigma0
   new Particle("Sigma0", 3212, 1, "Baryon", 100, 0, 1.19264, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{3122,22}));

   // Creating Sigma*0
   new Particle("Sigma*0", 3214, 1, "Baryon", 100, 0, 1.3837, 0.036, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3214));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{3122,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3222,-211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3112,211}));

   // Creating sigma(1775)0
   new Particle("sigma(1775)0", 3216, 1, "Unknown", 100, 0, 1.775, 0.12, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0036() {

   // Creating sigma(2030)0
   new Particle("sigma(2030)0", 3218, 1, "Unknown", 100, 0, 2.03, 0.18, -100, 0, -100, -1, -1);

   // Creating Sigma+
   new Particle("Sigma+", 3222, 1, "Baryon", 100, 1, 1.18937, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(3222));
   part->AddDecay(Particle::Decay(0, 0.516,  vector<int>{2212,111}));
   part->AddDecay(Particle::Decay(0, 0.483,  vector<int>{2112,211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2212,22}));

   // Creating Sigma*+
   new Particle("Sigma*+", 3224, 1, "Baryon", 100, 1, 1.3828, 0.0358, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3224));
   part->AddDecay(Particle::Decay(0, 0.88,  vector<int>{3122,211}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3222,111}));
   part->AddDecay(Particle::Decay(0, 0.06,  vector<int>{3212,211}));

   // Creating sigma(1775)+
   new Particle("sigma(1775)+", 3226, 1, "Unknown", 100, 1, 1.775, 0.12, -100, 0, -100, -1, -1);

   // Creating sigma(2030)+
   new Particle("sigma(2030)+", 3228, 1, "Unknown", 100, 1, 2.03, 0.18, -100, 0, -100, -1, -1);

   // Creating ss_1
   new Particle("ss_1", 3303, 1, "Unknown", 100, -0.666667, 2.08, 0, -100, -1, -100, -1, -1);

   // Creating Xi-
   new Particle("Xi-", 3312, 1, "Baryon", 100, -1, 1.32171, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3312));
   part->AddDecay(Particle::Decay(0, 0.9988,  vector<int>{3122,-211}));
   part->AddDecay(Particle::Decay(0, 0.0006,  vector<int>{-12,11,3122}));
   part->AddDecay(Particle::Decay(0, 0.0004,  vector<int>{-14,13,3122}));
   part->AddDecay(Particle::Decay(0, 0.0001,  vector<int>{3112,22}));
   part->AddDecay(Particle::Decay(0, 0.0001,  vector<int>{-12,11,3212}));

   // Creating Xi*-
   new Particle("Xi*-", 3314, 1, "Baryon", 100, -1, 1.535, 0.0099, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3314));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{3322,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{3312,111}));

   // Creating Xi0
   new Particle("Xi0", 3322, 1, "Baryon", 100, 0, 1.31486, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3322));
   part->AddDecay(Particle::Decay(0, 0.9954,  vector<int>{3122,111}));
   part->AddDecay(Particle::Decay(0, 0.0035,  vector<int>{3212,22}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{3122,22}));

   // Creating Xi*0
   new Particle("Xi*0", 3324, 1, "Baryon", 100, 0, 1.5318, 0.0091, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3324));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{3312,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{3322,111}));

   // Creating Omega-
   new Particle("Omega-", 3334, 1, "Baryon", 100, -1, 1.67245, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(3334));
   part->AddDecay(Particle::Decay(0, 0.676,  vector<int>{3122,-321}));
   part->AddDecay(Particle::Decay(0, 0.234,  vector<int>{3322,-211}));
   part->AddDecay(Particle::Decay(0, 0.085,  vector<int>{3312,111}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{-12,11,3322}));

   // Creating cd_0
   new Particle("cd_0", 4101, 1, "Unknown", 100, 0.333333, 1.96908, 0, -100, -1, -100, -1, -1);

   // Creating cd_1
   new Particle("cd_1", 4103, 1, "Unknown", 100, 0.333333, 2.00808, 0, -100, -1, -100, -1, -1);

   // Creating Sigma_c0
   new Particle("Sigma_c0", 4112, 1, "CharmedBaryon", 100, 0, 2.45376, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4112));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,-211}));

   // Creating Sigma*_c0
   new Particle("Sigma*_c0", 4114, 1, "CharmedBaryon", 100, 0, 2.518, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,-211}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0037() {

   // Creating Lambda_c+
   new Particle("Lambda_c+", 4122, 1, "CharmedBaryon", 100, 1, 2.28646, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(4122));
   part->AddDecay(Particle::Decay(13, 0.2432,  vector<int>{2,-1,3,2101}));
   part->AddDecay(Particle::Decay(13, 0.15,  vector<int>{3,2203}));
   part->AddDecay(Particle::Decay(13, 0.075,  vector<int>{2,3201}));
   part->AddDecay(Particle::Decay(13, 0.075,  vector<int>{2,3203}));
   part->AddDecay(Particle::Decay(13, 0.057,  vector<int>{2,-1,3,2103}));
   part->AddDecay(Particle::Decay(13, 0.035,  vector<int>{2,-1,1,2101}));
   part->AddDecay(Particle::Decay(13, 0.035,  vector<int>{2,-3,3,2101}));
   part->AddDecay(Particle::Decay(13, 0.03,  vector<int>{1,2203}));
   part->AddDecay(Particle::Decay(0, 0.025,  vector<int>{2224,-323}));
   part->AddDecay(Particle::Decay(42, 0.018,  vector<int>{-13,14,3122}));
   part->AddDecay(Particle::Decay(42, 0.018,  vector<int>{-11,12,3122}));
   part->AddDecay(Particle::Decay(0, 0.016,  vector<int>{2212,-311}));
   part->AddDecay(Particle::Decay(13, 0.015,  vector<int>{2,2101}));
   part->AddDecay(Particle::Decay(13, 0.015,  vector<int>{2,2103}));
   part->AddDecay(Particle::Decay(0, 0.0088,  vector<int>{2212,-313}));
   part->AddDecay(Particle::Decay(0, 0.0066,  vector<int>{2224,-321}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{-13,14,2212,-211}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{-13,14,2112,111}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{-11,12,2112,111}));
   part->AddDecay(Particle::Decay(42, 0.006,  vector<int>{-11,12,2212,-211}));
   part->AddDecay(Particle::Decay(0, 0.0058,  vector<int>{3122,211}));
   part->AddDecay(Particle::Decay(0, 0.0055,  vector<int>{3212,211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,3212}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{2214,-311}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{2214,-313}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,3212}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{3122,213}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-13,14,3214}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{3122,321}));
   part->AddDecay(Particle::Decay(0, 0.005,  vector<int>{3122,323}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-11,12,3214}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3212,213}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3214,211}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3214,213}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3222,111}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3222,113}));
   part->AddDecay(Particle::Decay(0, 0.004,  vector<int>{3222,223}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{3224,113}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{3224,223}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{2112,211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{2112,213}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{2114,211}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{2114,213}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-13,14,2112}));
   part->AddDecay(Particle::Decay(42, 0.003,  vector<int>{-11,12,2112}));
   part->AddDecay(Particle::Decay(0, 0.003,  vector<int>{3224,111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3322,321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3212,321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3212,323}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3222,311}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3222,313}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3322,323}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3324,321}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{2212,111}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{2212,113}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{2212,223}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-11,12,2114}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3222,221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3224,221}));
   part->AddDecay(Particle::Decay(0, 0.002,  vector<int>{3222,331}));
   part->AddDecay(Particle::Decay(42, 0.002,  vector<int>{-13,14,2114}));
   part->AddDecay(Particle::Decay(0, 0.0018,  vector<int>{2212,10221}));
   part->AddDecay(Particle::Decay(0, 0.0013,  vector<int>{2212,333}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2224,-213}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{3224,311}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{3224,313}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2224,-211}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2212,221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2212,331}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2214,111}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2214,221}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2214,331}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2214,113}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{3214,321}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{3214,323}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{2214,223}));

   // Creating Xi_c0
   new Particle("Xi_c0", 4132, 1, "CharmedBaryon", 100, 0, 2.471, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4132));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating cu_0
   new Particle("cu_0", 4201, 1, "Unknown", 100, 1.33333, 1.96908, 0, -100, -1, -100, -1, -1);

   // Creating cu_1
   new Particle("cu_1", 4203, 1, "Unknown", 100, 1.33333, 2.00808, 0, -100, -1, -100, -1, -1);

   // Creating Sigma_c+
   new Particle("Sigma_c+", 4212, 1, "CharmedBaryon", 100, 1, 2.4529, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,111}));

   // Creating Sigma*_c+
   new Particle("Sigma*_c+", 4214, 1, "CharmedBaryon", 100, 1, 2.5175, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4214));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,111}));

   // Creating Sigma_c++
   new Particle("Sigma_c++", 4222, 1, "CharmedBaryon", 100, 2, 2.45402, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4222));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,211}));

   // Creating Sigma*_c++
   new Particle("Sigma*_c++", 4224, 1, "CharmedBaryon", 100, 2, 2.5184, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4122,211}));

   // Creating Xi_c+
   new Particle("Xi_c+", 4232, 1, "CharmedBaryon", 100, 1, 2.4679, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4232));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating cs_0
   new Particle("cs_0", 4301, 1, "CharmedBaryon", 100, 0.333333, 2.15432, 0, -100, -1, -100, -1, -1);

   // Creating cs_1
   new Particle("cs_1", 4303, 1, "Unknown", 100, 0.333333, 2.17967, 0, -100, -1, -100, -1, -1);

   // Creating Xi'_c0
   new Particle("Xi'_c0", 4312, 1, "CharmedBaryon", 100, 0, 2.578, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4312));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4132,22}));

   // Creating Xi*_c0
   new Particle("Xi*_c0", 4314, 1, "CharmedBaryon", 100, 0, 2.6461, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4314));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{4132,111}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{4132,22}));

   // Creating Xi'_c+
   new Particle("Xi'_c+", 4322, 1, "CharmedBaryon", 100, 1, 2.5757, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4322));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4232,22}));

   // Creating Xi*_c+
   new Particle("Xi*_c+", 4324, 1, "CharmedBaryon", 100, 1, 2.6466, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4324));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{4232,111}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{4232,22}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0038() {

   // Creating Omega_c0
   new Particle("Omega_c0", 4332, 1, "CharmedBaryon", 100, 0, 2.6975, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(4332));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Omega*_c0
   new Particle("Omega*_c0", 4334, 1, "CharmedBaryon", 100, 0, 2.7683, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4334));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{4332,22}));

   // Creating cc_1
   new Particle("cc_1", 4403, 1, "Unknown", 100, 1.33333, 3.27531, 0, -100, -1, -100, -1, -1);

   // Creating Xi_cc+
   new Particle("Xi_cc+", 4412, 1, "CharmedBaryon", 100, 1, 3.59798, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4412));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Xi*_cc+
   new Particle("Xi*_cc+", 4414, 1, "CharmedBaryon", 100, 1, 3.65648, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4414));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Xi_cc++
   new Particle("Xi_cc++", 4422, 1, "CharmedBaryon", 100, 2, 3.59798, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4422));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Xi*_cc++
   new Particle("Xi*_cc++", 4424, 1, "CharmedBaryon", 100, 2, 3.65648, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4424));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Omega_cc+
   new Particle("Omega_cc+", 4432, 1, "CharmedBaryon", 100, 1, 3.78663, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4432));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Omega*_cc+
   new Particle("Omega*_cc+", 4434, 1, "CharmedBaryon", 100, 1, 3.82466, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4434));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating Omega*_ccc++
   new Particle("Omega*_ccc++", 4444, 1, "CharmedBaryon", 100, 2, 4.91594, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4444));
   part->AddDecay(Particle::Decay(11, 0.76,  vector<int>{2,-1,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-13,14,3,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-11,12,3,81}));
   part->AddDecay(Particle::Decay(11, 0.08,  vector<int>{2,-3,3,81}));

   // Creating bd_0
   new Particle("bd_0", 5101, 1, "Unknown", 100, -0.666667, 5.38897, 0, -100, -1, -100, -1, -1);

   // Creating bd_1
   new Particle("bd_1", 5103, 1, "Unknown", 100, -0.666667, 5.40145, 0, -100, -1, -100, -1, -1);

   // Creating Sigma_b-
   new Particle("Sigma_b-", 5112, 1, "B-Baryon", 100, -1, 5.8152, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5112));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,-211}));

   // Creating Sigma*_b-
   new Particle("Sigma*_b-", 5114, 1, "B-Baryon", 100, -1, 5.8364, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5114));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,-211}));

   // Creating Lambda_b0
   new Particle("Lambda_b0", 5122, 1, "B-Baryon", 100, 0, 5.6202, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5122));
   part->AddDecay(Particle::Decay(48, 0.4291,  vector<int>{-2,1,4,2101}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4122}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4122}));
   part->AddDecay(Particle::Decay(13, 0.08,  vector<int>{-2,4,1,2101}));
   part->AddDecay(Particle::Decay(13, 0.07,  vector<int>{-4,3,4,2101}));
   part->AddDecay(Particle::Decay(0, 0.0435,  vector<int>{4122,-433}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4122}));
   part->AddDecay(Particle::Decay(0, 0.0285,  vector<int>{4122,-431}));
   part->AddDecay(Particle::Decay(0, 0.0235,  vector<int>{4122,-20213}));
   part->AddDecay(Particle::Decay(0, 0.02,  vector<int>{4122,-213}));
   part->AddDecay(Particle::Decay(13, 0.02,  vector<int>{-4,4,3,2101}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,2101}));
   part->AddDecay(Particle::Decay(0, 0.0077,  vector<int>{4122,-211}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,2101}));
   part->AddDecay(Particle::Decay(0, 0.0044,  vector<int>{20443,3122}));
   part->AddDecay(Particle::Decay(0, 0.0022,  vector<int>{443,3122}));
   part->AddDecay(Particle::Decay(0, 0.0011,  vector<int>{441,3122}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0039() {

   // Creating Xi_b-
   new Particle("Xi_b-", 5132, 1, "B-Baryon", 100, -1, 5.7924, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(5132));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi_bc0
   new Particle("Xi_bc0", 5142, 1, "B-Baryon", 100, 0, 7.00575, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5142));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating bu_0
   new Particle("bu_0", 5201, 1, "Unknown", 100, 0.333333, 5.38897, 0, -100, -1, -100, -1, -1);

   // Creating bu_1
   new Particle("bu_1", 5203, 1, "Unknown", 100, 0.333333, 5.40145, 0, -100, -1, -100, -1, -1);

   // Creating Sigma_b0
   new Particle("Sigma_b0", 5212, 1, "B-Baryon", 100, 0, 5.8078, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5212));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,111}));

   // Creating Sigma*_b0
   new Particle("Sigma*_b0", 5214, 1, "B-Baryon", 100, 0, 5.829, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5214));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,111}));

   // Creating Sigma_b+
   new Particle("Sigma_b+", 5222, 1, "B-Baryon", 100, 1, 5.8078, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5222));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,211}));

   // Creating Sigma*_b+
   new Particle("Sigma*_b+", 5224, 1, "B-Baryon", 100, 1, 5.829, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5224));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5122,211}));

   // Creating Xi_b0
   new Particle("Xi_b0", 5232, 1, "B-Baryon", 100, 0, 5.7924, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5232));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi_bc+
   new Particle("Xi_bc+", 5242, 1, "B-Baryon", 100, 1, 7.00575, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5242));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating bs_0
   new Particle("bs_0", 5301, 1, "Unknown", 100, -0.666667, 5.56725, 0, -100, -1, -100, -1, -1);

   // Creating bs_1
   new Particle("bs_1", 5303, 1, "Unknown", 100, -0.666667, 5.57536, 0, -100, -1, -100, -1, -1);

   // Creating Xi'_b-
   new Particle("Xi'_b-", 5312, 1, "B-Baryon", 100, -1, 5.96, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5312));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5132,22}));

   // Creating Xi*_b-
   new Particle("Xi*_b-", 5314, 1, "B-Baryon", 100, -1, 5.97, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5314));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5132,22}));

   // Creating Xi'_b0
   new Particle("Xi'_b0", 5322, 1, "B-Baryon", 100, 0, 5.96, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5322));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5232,22}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0040() {

   // Creating Xi*_b0
   new Particle("Xi*_b0", 5324, 1, "B-Baryon", 100, 0, 5.97, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(5324));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5232,22}));

   // Creating Omega_b-
   new Particle("Omega_b-", 5332, 1, "B-Baryon", 100, -1, 6.12, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5332));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_b-
   new Particle("Omega*_b-", 5334, 1, "B-Baryon", 100, -1, 6.13, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5334));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{5332,22}));

   // Creating Omega_bc0
   new Particle("Omega_bc0", 5342, 1, "B-Baryon", 100, 0, 7.19099, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5342));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating bc_0
   new Particle("bc_0", 5401, 1, "Unknown", 100, 0.333333, 6.67143, 0, -100, -1, -100, -1, -1);

   // Creating bc_1
   new Particle("bc_1", 5403, 1, "Unknown", 100, 0.333333, 6.67397, 0, -100, -1, -100, -1, -1);

   // Creating Xi'_bc0
   new Particle("Xi'_bc0", 5412, 1, "B-Baryon", 100, 0, 7.03724, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5412));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi*_bc0
   new Particle("Xi*_bc0", 5414, 1, "B-Baryon", 100, 0, 7.0485, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5414));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi'_bc+
   new Particle("Xi'_bc+", 5422, 1, "B-Baryon", 100, 1, 7.03724, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5422));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi*_bc+
   new Particle("Xi*_bc+", 5424, 1, "B-Baryon", 100, 1, 7.0485, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5424));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega'_bc0
   new Particle("Omega'_bc0", 5432, 1, "B-Baryon", 100, 0, 7.21101, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5432));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_bc0
   new Particle("Omega*_bc0", 5434, 1, "B-Baryon", 100, 0, 7.219, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5434));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega_bcc+
   new Particle("Omega_bcc+", 5442, 1, "B-Baryon", 100, 1, 8.30945, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5442));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_bcc+
   new Particle("Omega*_bcc+", 5444, 1, "B-Baryon", 100, 1, 8.31325, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5444));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating bb_1
   new Particle("bb_1", 5503, 1, "Unknown", 100, -0.666667, 10.0735, 0, -100, -1, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0041() {

   // Creating Xi_bb-
   new Particle("Xi_bb-", 5512, 1, "Unknown", 100, -1, 10.4227, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(5512));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi*_bb-
   new Particle("Xi*_bb-", 5514, 1, "B-Baryon", 100, -1, 10.4414, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5514));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi_bb0
   new Particle("Xi_bb0", 5522, 1, "B-Baryon", 100, 0, 10.4227, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5522));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Xi*_bb0
   new Particle("Xi*_bb0", 5524, 1, "B-Baryon", 100, 0, 10.4414, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5524));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega_bb-
   new Particle("Omega_bb-", 5532, 1, "B-Baryon", 100, -1, 10.6021, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5532));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_bb-
   new Particle("Omega*_bb-", 5534, 1, "B-Baryon", 100, -1, 10.6143, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5534));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega_bbc0
   new Particle("Omega_bbc0", 5542, 1, "B-Baryon", 100, 0, 11.7077, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5542));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_bbc0
   new Particle("Omega*_bbc0", 5544, 1, "B-Baryon", 100, 0, 11.7115, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5544));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating Omega*_bbb-
   new Particle("Omega*_bbb-", 5554, 1, "B-Baryon", 100, -1, 15.1106, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(5554));
   part->AddDecay(Particle::Decay(42, 0.5,  vector<int>{-2,1,4,81}));
   part->AddDecay(Particle::Decay(42, 0.14,  vector<int>{-4,3,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-12,11,4,81}));
   part->AddDecay(Particle::Decay(42, 0.105,  vector<int>{-14,13,4,81}));
   part->AddDecay(Particle::Decay(42, 0.08,  vector<int>{-2,4,1,81}));
   part->AddDecay(Particle::Decay(42, 0.04,  vector<int>{-16,15,4,81}));
   part->AddDecay(Particle::Decay(42, 0.015,  vector<int>{-2,1,2,81}));
   part->AddDecay(Particle::Decay(42, 0.01,  vector<int>{-4,4,3,81}));
   part->AddDecay(Particle::Decay(42, 0.005,  vector<int>{-4,3,2,81}));

   // Creating a_00
   new Particle("a_00", 10111, 0, "Unknown", 100, 0, 0.9835, 0.06, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10111));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{221,111}));

   // Creating b_10
   new Particle("b_10", 10113, 0, "Unknown", 100, 0, 1.2295, 0.142, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10113));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{223,111}));

   // Creating pi2(1670)0
   new Particle("pi2(1670)0", 10115, 1, "Unknown", 100, 0, 1.6722, 0.26, -100, 0, -100, -1, -1);

   // Creating a_0+
   new Particle("a_0+", 10211, 1, "Unknown", 100, 1, 0.9835, 0.06, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10211));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{221,211}));

   // Creating b_1+
   new Particle("b_1+", 10213, 1, "Unknown", 100, 1, 1.2295, 0.142, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10213));
   part->AddDecay(Particle::Decay(0, 1,  vector<int>{223,211}));

   // Creating pi2(1670)+
   new Particle("pi2(1670)+", 10215, 1, "Unknown", 100, 1, 1.6722, 0.26, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0042() {

   // Creating f_0
   new Particle("f_0", 10221, 0, "Unknown", 100, 0, 1.3, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(10221));
   part->AddDecay(Particle::Decay(0, 0.52,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.26,  vector<int>{111,111}));
   part->AddDecay(Particle::Decay(0, 0.11,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(0, 0.055,  vector<int>{130,130}));
   part->AddDecay(Particle::Decay(0, 0.055,  vector<int>{310,310}));

   // Creating h_1
   new Particle("h_1", 10223, 0, "Unknown", 100, 0, 1.17, 0.36, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10223));
   part->AddDecay(Particle::Decay(0, 0.334,  vector<int>{113,111}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{213,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{-213,211}));

   // Creating eta2(1645)
   new Particle("eta2(1645)", 10225, 1, "Unknown", 100, 0, 1.617, 0.181, -100, 0, -100, -1, -1);

   // Creating K*_00
   new Particle("K*_00", 10311, 1, "Unknown", 100, 0, 1.42, 0.287, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10311));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{321,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{311,111}));

   // Creating K_10
   new Particle("K_10", 10313, 1, "Unknown", 100, 0, 1.272, 0.09, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10313));
   part->AddDecay(Particle::Decay(0, 0.313,  vector<int>{323,-211}));
   part->AddDecay(Particle::Decay(0, 0.28,  vector<int>{321,-213}));
   part->AddDecay(Particle::Decay(0, 0.157,  vector<int>{313,111}));
   part->AddDecay(Particle::Decay(0, 0.14,  vector<int>{311,113}));
   part->AddDecay(Particle::Decay(0, 0.11,  vector<int>{311,223}));

   // Creating k2(1770)0
   new Particle("k2(1770)0", 10315, 1, "Unknown", 100, 0, 1.773, 0.186, -100, 0, -100, -1, -1);

   // Creating K*_0+
   new Particle("K*_0+", 10321, 1, "Unknown", 100, 1, 1.42, 0.287, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10321));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{311,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{321,111}));

   // Creating K_1+
   new Particle("K_1+", 10323, 1, "Unknown", 100, 1, 1.272, 0.09, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10323));
   part->AddDecay(Particle::Decay(0, 0.313,  vector<int>{313,211}));
   part->AddDecay(Particle::Decay(0, 0.28,  vector<int>{311,213}));
   part->AddDecay(Particle::Decay(0, 0.157,  vector<int>{323,111}));
   part->AddDecay(Particle::Decay(0, 0.14,  vector<int>{321,113}));
   part->AddDecay(Particle::Decay(0, 0.11,  vector<int>{321,223}));

   // Creating k2(1770)+
   new Particle("k2(1770)+", 10325, 1, "Unknown", 100, 1, 1.773, 0.186, -100, 0, -100, -1, -1);

   // Creating f'_0
   new Particle("f'_0", 10331, 0, "Unknown", 100, 0, 1.724, 0.25, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10331));
   part->AddDecay(Particle::Decay(0, 0.36,  vector<int>{211,-211}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{211,-211,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{211,-211,111,111}));
   part->AddDecay(Particle::Decay(0, 0.18,  vector<int>{111,111}));
   part->AddDecay(Particle::Decay(0, 0.03,  vector<int>{321,-321}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{130,130}));
   part->AddDecay(Particle::Decay(0, 0.015,  vector<int>{310,310}));

   // Creating h'_1
   new Particle("h'_1", 10333, 0, "Unknown", 100, 0, 1.4, 0.08, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10333));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{313,-311}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{-313,311}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{323,-321}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{-323,321}));

   // Creating eta2(1870)
   new Particle("eta2(1870)", 10335, 1, "Unknown", 100, 0, 1.842, 0.225, -100, 0, -100, -1, -1);

   // Creating D*_0+
   new Particle("D*_0+", 10411, 1, "Unknown", 100, 1, 2.272, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10411));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{421,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{411,111}));

   // Creating D_1+
   new Particle("D_1+", 10413, 1, "Unknown", 100, 1, 2.424, 0.02, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10413));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{423,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{413,111}));

   // Creating D*_00
   new Particle("D*_00", 10421, 1, "Unknown", 100, 0, 2.272, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10421));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{411,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{421,111}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0043() {

   // Creating D_10
   new Particle("D_10", 10423, 1, "Unknown", 100, 0, 2.4223, 0.02, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(10423));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{413,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{423,111}));

   // Creating D*_0s+
   new Particle("D*_0s+", 10431, 1, "Unknown", 100, 1, 2.3178, 0.0046, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10431));
   part->AddDecay(Particle::Decay(0, 0.8,  vector<int>{431,111}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{431,22}));

   // Creating D_1s+
   new Particle("D_1s+", 10433, 1, "Unknown", 100, 1, 2.5353, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10433));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{423,321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{413,311}));

   // Creating chi_0c
   new Particle("chi_0c", 10441, 0, "Unknown", 100, 0, 3.41475, 0.014, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10441));
   part->AddDecay(Particle::Decay(12, 0.993,  vector<int>{82,-82}));
   part->AddDecay(Particle::Decay(0, 0.007,  vector<int>{443,22}));

   // Creating h_1c
   new Particle("h_1c", 10443, 0, "Unknown", 100, 0, 3.52593, 0.01, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10443));
   part->AddDecay(Particle::Decay(12, 1,  vector<int>{82,-82}));

   // Creating B*_00
   new Particle("B*_00", 10511, 1, "Unknown", 100, 0, 5.68, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10511));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{521,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{511,111}));

   // Creating B_10
   new Particle("B_10", 10513, 1, "Unknown", 100, 0, 5.73, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10513));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{523,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{513,111}));

   // Creating B*_0+
   new Particle("B*_0+", 10521, 1, "Unknown", 100, 1, 5.68, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10521));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{511,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{521,111}));

   // Creating B_1+
   new Particle("B_1+", 10523, 1, "Unknown", 100, 1, 5.73, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10523));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{513,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{523,111}));

   // Creating B*_0s0
   new Particle("B*_0s0", 10531, 1, "Unknown", 100, 0, 5.92, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10531));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{521,-321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{511,-311}));

   // Creating B_1s0
   new Particle("B_1s0", 10533, 1, "Unknown", 100, 0, 5.97, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10533));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{523,-321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{513,-311}));

   // Creating B*_0c+
   new Particle("B*_0c+", 10541, 1, "Unknown", 100, 1, 7.25, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10541));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{511,411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{521,421}));

   // Creating B_1c+
   new Particle("B_1c+", 10543, 1, "Unknown", 100, 1, 7.3, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10543));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{513,411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{523,421}));

   // Creating chi_0b
   new Particle("chi_0b", 10551, 0, "Unknown", 100, 0, 9.8594, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10551));
   part->AddDecay(Particle::Decay(32, 0.98,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 0.02,  vector<int>{553,22}));

   // Creating h_1b
   new Particle("h_1b", 10553, 0, "Unknown", 100, 0, 9.875, 0.01, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(10553));
   part->AddDecay(Particle::Decay(32, 1,  vector<int>{21,21}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0044() {

   // Creating delta(1900)-
   new Particle("delta(1900)-", 11112, 1, "Unknown", 100, -1, 1.9, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1700)-
   new Particle("delta(1700)-", 11114, 1, "Unknown", 100, -1, 1.7, 0.3, -100, 0, -100, -1, -1);

   // Creating delta(1930)-
   new Particle("delta(1930)-", 11116, 1, "Unknown", 100, -1, 1.96, 0.36, -100, 0, -100, -1, -1);

   // Creating delta(1900)0
   new Particle("delta(1900)0", 11212, 1, "Unknown", 100, 0, 1.9, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1930)0
   new Particle("delta(1930)0", 11216, 1, "Unknown", 100, 0, 1.96, 0.36, -100, 0, -100, -1, -1);

   // Creating N(1440)0
   new Particle("N(1440)0", 12112, 1, "Unknown", 100, 0, 1.44, 0.3, -100, 0, -100, -1, -1);

   // Creating delta(1700)0
   new Particle("delta(1700)0", 12114, 1, "Unknown", 100, 0, 1.7, 0.3, -100, 0, -100, -1, -1);

   // Creating N(1680)0
   new Particle("N(1680)0", 12116, 1, "Unknown", 100, 0, 1.685, 0.13, -100, 0, -100, -1, -1);

   // Creating N(1990)0
   new Particle("N(1990)0", 12118, 1, "Unknown", 100, 0, 1.95, 0.555, -100, 0, -100, -1, -1);

   // Creating delta(1900)+
   new Particle("delta(1900)+", 12122, 1, "Unknown", 100, 1, 1.9, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1930)+
   new Particle("delta(1930)+", 12126, 1, "Unknown", 100, 1, 1.96, 0.36, -100, 0, -100, -1, -1);

   // Creating N(1440)+
   new Particle("N(1440)+", 12212, 1, "Unknown", 100, 1, 1.44, 0.3, -100, 0, -100, -1, -1);

   // Creating delta(1700)+
   new Particle("delta(1700)+", 12214, 1, "Unknown", 100, 1, 1.7, 0.3, -100, 0, -100, -1, -1);

   // Creating N(1680)+
   new Particle("N(1680)+", 12216, 1, "Unknown", 100, 1, 1.685, 0.13, -100, 0, -100, -1, -1);

   // Creating N(1990)+
   new Particle("N(1990)+", 12218, 1, "Unknown", 100, 1, 1.95, 0.555, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0045() {

   // Creating delta(1900)++
   new Particle("delta(1900)++", 12222, 1, "Unknown", 100, 2, 1.9, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1700)++
   new Particle("delta(1700)++", 12224, 1, "Unknown", 100, 2, 1.7, 0.3, -100, 0, -100, -1, -1);

   // Creating delta(1930)++
   new Particle("delta(1930)++", 12226, 1, "Unknown", 100, 2, 1.96, 0.36, -100, 0, -100, -1, -1);

   // Creating sigma(1660)-
   new Particle("sigma(1660)-", 13112, 1, "Unknown", 100, -1, 1.66, 0.1, -100, 0, -100, -1, -1);

   // Creating sigma(1670)-
   new Particle("sigma(1670)-", 13114, 1, "Unknown", 100, -1, 1.67, 0.06, -100, 0, -100, -1, -1);

   // Creating sigma(1915)-
   new Particle("sigma(1915)-", 13116, 1, "Unknown", 100, -1, 1.915, 0.12, -100, 0, -100, -1, -1);

   // Creating lambda(1405)
   new Particle("lambda(1405)", 13122, 1, "Unknown", 100, 0, 1.4051, 0.05, -100, 0, -100, -1, -1);

   // Creating lambda(1690)
   new Particle("lambda(1690)", 13124, 1, "Unknown", 100, 0, 1.69, 0.06, -100, 0, -100, -1, -1);

   // Creating lambda(1830)
   new Particle("lambda(1830)", 13126, 1, "Unknown", 100, 0, 1.83, 0.095, -100, 0, -100, -1, -1);

   // Creating sigma(1660)0
   new Particle("sigma(1660)0", 13212, 1, "Unknown", 100, 0, 1.66, 0.1, -100, 0, -100, -1, -1);

   // Creating sigma(1670)0
   new Particle("sigma(1670)0", 13214, 1, "Unknown", 100, 0, 1.67, 0.06, -100, 0, -100, -1, -1);

   // Creating sigma(1915)0
   new Particle("sigma(1915)0", 13216, 1, "Unknown", 100, 0, 1.915, 0.12, -100, 0, -100, -1, -1);

   // Creating sigma(1660)+
   new Particle("sigma(1660)+", 13222, 1, "Unknown", 100, 1, 1.66, 0.1, -100, 0, -100, -1, -1);

   // Creating sigma(1670)+
   new Particle("sigma(1670)+", 13224, 1, "Unknown", 100, 1, 1.67, 0.06, -100, 0, -100, -1, -1);

   // Creating sigma(1915)+
   new Particle("sigma(1915)+", 13226, 1, "Unknown", 100, 1, 1.915, 0.12, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0046() {

   // Creating xi(1820)-
   new Particle("xi(1820)-", 13314, 1, "Unknown", 100, -1, 1.823, 0.024, -100, 0, -100, -1, -1);

   // Creating xi(2030)-
   new Particle("xi(2030)-", 13316, 1, "Unknown", 100, -1, 2.025, 0.02, -100, 0, -100, -1, -1);

   // Creating xi(1820)0
   new Particle("xi(1820)0", 13324, 1, "Unknown", 100, 0, 1.823, 0.024, -100, 0, -100, -1, -1);

   // Creating xi(2030)0
   new Particle("xi(2030)0", 13326, 1, "Unknown", 100, 0, 2.025, 0.02, -100, 0, -100, -1, -1);

   // Creating a_10
   new Particle("a_10", 20113, 0, "Unknown", 100, 0, 1.23, 0.4, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(20113));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{213,-211}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{-213,211}));

   // Creating a_1+
   new Particle("a_1+", 20213, 1, "Unknown", 100, 1, 1.23, 0.4, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20213));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{113,211}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{213,111}));

   // Creating f_1
   new Particle("f_1", 20223, 0, "Unknown", 100, 0, 1.2818, 0.025, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20223));
   part->AddDecay(Particle::Decay(0, 0.15,  vector<int>{113,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.146,  vector<int>{10111,111}));
   part->AddDecay(Particle::Decay(0, 0.146,  vector<int>{-10211,211}));
   part->AddDecay(Particle::Decay(0, 0.146,  vector<int>{10211,-211}));
   part->AddDecay(Particle::Decay(0, 0.066,  vector<int>{113,22}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{213,-211,111}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{221,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{113,111,111}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{-213,211,111}));
   part->AddDecay(Particle::Decay(0, 0.05,  vector<int>{221,111,111}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{321,-311,-211}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{311,-311,111}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{311,-321,211}));
   part->AddDecay(Particle::Decay(0, 0.024,  vector<int>{321,-321,111}));

   // Creating K*_10
   new Particle("K*_10", 20313, 1, "Unknown", 100, 0, 1.403, 0.174, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20313));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{323,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{313,111}));

   // Creating K*_1+
   new Particle("K*_1+", 20323, 1, "Unknown", 100, 1, 1.403, 0.174, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20323));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{313,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{323,111}));

   // Creating f'_1
   new Particle("f'_1", 20333, 0, "Unknown", 100, 0, 1.4264, 0.053, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20333));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{313,-311}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{-313,311}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{323,-321}));
   part->AddDecay(Particle::Decay(0, 0.25,  vector<int>{-323,321}));

   // Creating D*_1+
   new Particle("D*_1+", 20413, 1, "Unknown", 100, 1, 2.372, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20413));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{423,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{413,111}));

   // Creating D*_10
   new Particle("D*_10", 20423, 1, "Unknown", 100, 0, 2.372, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20423));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{413,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{423,111}));

   // Creating D*_1s+
   new Particle("D*_1s+", 20433, 1, "Unknown", 100, 1, 2.4596, 0.0055, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20433));
   part->AddDecay(Particle::Decay(0, 0.8,  vector<int>{433,111}));
   part->AddDecay(Particle::Decay(0, 0.2,  vector<int>{433,22}));

   // Creating chi_1c
   new Particle("chi_1c", 20443, 0, "Unknown", 100, 0, 3.51066, 0.0009, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20443));
   part->AddDecay(Particle::Decay(12, 0.727,  vector<int>{82,-82}));
   part->AddDecay(Particle::Decay(0, 0.273,  vector<int>{443,22}));

   // Creating B*_10
   new Particle("B*_10", 20513, 1, "Unknown", 100, 0, 5.78, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20513));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{523,-211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{513,111}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0047() {

   // Creating B*_1+
   new Particle("B*_1+", 20523, 1, "Unknown", 100, 1, 5.78, 0.05, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(20523));
   part->AddDecay(Particle::Decay(0, 0.667,  vector<int>{513,211}));
   part->AddDecay(Particle::Decay(0, 0.333,  vector<int>{523,111}));

   // Creating B*_1s0
   new Particle("B*_1s0", 20533, 1, "Unknown", 100, 0, 6.02, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20533));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{523,-321}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{513,-311}));

   // Creating B*_1c+
   new Particle("B*_1c+", 20543, 1, "Unknown", 100, 1, 7.3, 0.05, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20543));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{513,411}));
   part->AddDecay(Particle::Decay(0, 0.5,  vector<int>{523,421}));

   // Creating chi_1b
   new Particle("chi_1b", 20553, 0, "Unknown", 100, 0, 9.8928, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(20553));
   part->AddDecay(Particle::Decay(32, 0.65,  vector<int>{21,21}));
   part->AddDecay(Particle::Decay(0, 0.35,  vector<int>{553,22}));

   // Creating delta(1910)-
   new Particle("delta(1910)-", 21112, 1, "Unknown", 100, -1, 1.91, 0.25, -100, 0, -100, -1, -1);

   // Creating delta(1920)-
   new Particle("delta(1920)-", 21114, 1, "Unknown", 100, -1, 1.92, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1910)0
   new Particle("delta(1910)0", 21212, 1, "Unknown", 100, 0, 1.91, 0.25, -100, 0, -100, -1, -1);

   // Creating N(1700)0
   new Particle("N(1700)0", 21214, 1, "Unknown", 100, 0, 1.7, 0.1, -100, 0, -100, -1, -1);

   // Creating N(1535)0
   new Particle("N(1535)0", 22112, 1, "Unknown", 100, 0, 1.535, 0.15, -100, 0, -100, -1, -1);

   // Creating delta(1920)0
   new Particle("delta(1920)0", 22114, 1, "Unknown", 100, 0, 1.92, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1910)+
   new Particle("delta(1910)+", 22122, 1, "Unknown", 100, 1, 1.91, 0.25, -100, 0, -100, -1, -1);

   // Creating N(1700)+
   new Particle("N(1700)+", 22124, 1, "Unknown", 100, 1, 1.7, 0.1, -100, 0, -100, -1, -1);

   // Creating N(1535)+
   new Particle("N(1535)+", 22212, 1, "Unknown", 100, 1, 1.535, 0.15, -100, 0, -100, -1, -1);

   // Creating delta(1920)+
   new Particle("delta(1920)+", 22214, 1, "Unknown", 100, 1, 1.92, 0.2, -100, 0, -100, -1, -1);

   // Creating delta(1910)++
   new Particle("delta(1910)++", 22222, 1, "Unknown", 100, 2, 1.91, 0.25, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0048() {

   // Creating delta(1920)++
   new Particle("delta(1920)++", 22224, 1, "Unknown", 100, 2, 1.92, 0.2, -100, 0, -100, -1, -1);

   // Creating sigma(1750)-
   new Particle("sigma(1750)-", 23112, 1, "Unknown", 100, -1, 1.75, 0.09, -100, 0, -100, -1, -1);

   // Creating sigma(1940)-
   new Particle("sigma(1940)-", 23114, 1, "Unknown", 100, -1, 1.94, 0.22, -100, 0, -100, -1, -1);

   // Creating lambda(1600)
   new Particle("lambda(1600)", 23122, 1, "Unknown", 100, 0, 1.6, 0.15, -100, 0, -100, -1, -1);

   // Creating lambda(1890)
   new Particle("lambda(1890)", 23124, 1, "Unknown", 100, 0, 1.89, 0.1, -100, 0, -100, -1, -1);

   // Creating lambda(2110)
   new Particle("lambda(2110)", 23126, 1, "Unknown", 100, 0, 2.11, 0.2, -100, 0, -100, -1, -1);

   // Creating sigma(1750)0
   new Particle("sigma(1750)0", 23212, 1, "Unknown", 100, 0, 1.75, 0.09, -100, 0, -100, -1, -1);

   // Creating sigma(1940)0
   new Particle("sigma(1940)0", 23214, 1, "Unknown", 100, 0, 1.94, 0.22, -100, 0, -100, -1, -1);

   // Creating sigma(1750)+
   new Particle("sigma(1750)+", 23222, 1, "Unknown", 100, 1, 1.75, 0.09, -100, 0, -100, -1, -1);

   // Creating sigma(1940)+
   new Particle("sigma(1940)+", 23224, 1, "Unknown", 100, 1, 1.94, 0.22, -100, 0, -100, -1, -1);

   // Creating xi(1690)-
   new Particle("xi(1690)-", 23314, 1, "Unknown", 100, -1, 1.69, 0.05, -100, 0, -100, -1, -1);

   // Creating xi(1690)0
   new Particle("xi(1690)0", 23324, 1, "Unknown", 100, 0, 1.69, 0.05, -100, 0, -100, -1, -1);

   // Creating rho(1700)0
   new Particle("rho(1700)0", 30113, 1, "Unknown", 100, 0, 1.72, 0.25, -100, 0, -100, -1, -1);

   // Creating rho(1700)+
   new Particle("rho(1700)+", 30213, 1, "Unknown", 100, 1, 1.72, 0.25, -100, 0, -100, -1, -1);

   // Creating omega(1650)
   new Particle("omega(1650)", 30223, 1, "Unknown", 100, 0, 1.67, 0.315, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0049() {

   // Creating k_star(1680)0
   new Particle("k_star(1680)0", 30313, 1, "Unknown", 100, 0, 1.717, 0.32, -100, 0, -100, -1, -1);

   // Creating k_star(1680)+
   new Particle("k_star(1680)+", 30323, 1, "Unknown", 100, 1, 1.717, 0.32, -100, 0, -100, -1, -1);

   // Creating delta(1600)-
   new Particle("delta(1600)-", 31114, 1, "Unknown", 100, -1, 1.6, 0.35, -100, 0, -100, -1, -1);

   // Creating N(1720)0
   new Particle("N(1720)0", 31214, 1, "Unknown", 100, 0, 1.72, 0.2, -100, 0, -100, -1, -1);

   // Creating N(1650)0
   new Particle("N(1650)0", 32112, 1, "Unknown", 100, 0, 1.655, 0.165, -100, 0, -100, -1, -1);

   // Creating delta(1600)0
   new Particle("delta(1600)0", 32114, 1, "Unknown", 100, 0, 1.6, 0.35, -100, 0, -100, -1, -1);

   // Creating N(1720)+
   new Particle("N(1720)+", 32124, 1, "Unknown", 100, 1, 1.72, 0.2, -100, 0, -100, -1, -1);

   // Creating N(1650)+
   new Particle("N(1650)+", 32212, 1, "Unknown", 100, 1, 1.655, 0.165, -100, 0, -100, -1, -1);

   // Creating delta(1600)+
   new Particle("delta(1600)+", 32214, 1, "Unknown", 100, 1, 1.6, 0.35, -100, 0, -100, -1, -1);

   // Creating delta(1600)++
   new Particle("delta(1600)++", 32224, 1, "Unknown", 100, 2, 1.6, 0.35, -100, 0, -100, -1, -1);

   // Creating lambda(1670)
   new Particle("lambda(1670)", 33122, 1, "Unknown", 100, 0, 1.67, 0.035, -100, 0, -100, -1, -1);

   // Creating xi(1950)-
   new Particle("xi(1950)-", 33314, 1, "Unknown", 100, -1, 1.95, 0.06, -100, 0, -100, -1, -1);

   // Creating xi(1950)0
   new Particle("xi(1950)0", 33324, 1, "Unknown", 100, 0, 1.95, 0.06, -100, 0, -100, -1, -1);

   // Creating N(1900)0
   new Particle("N(1900)0", 41214, 1, "Unknown", 100, 0, 1.9, 0.5, -100, 0, -100, -1, -1);

   // Creating N(1710)0
   new Particle("N(1710)0", 42112, 1, "Unknown", 100, 0, 1.71, 0.1, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0050() {

   // Creating N(1900)+
   new Particle("N(1900)+", 42124, 1, "Unknown", 100, 1, 1.9, 0.5, -100, 0, -100, -1, -1);

   // Creating N(1710)+
   new Particle("N(1710)+", 42212, 1, "Unknown", 100, 1, 1.71, 0.1, -100, 0, -100, -1, -1);

   // Creating lambda(1800)
   new Particle("lambda(1800)", 43122, 1, "Unknown", 100, 0, 1.8, 0.3, -100, 0, -100, -1, -1);

   // Creating N(2090)0
   new Particle("N(2090)0", 52114, 1, "Unknown", 100, 0, 2.08, 0.35, -100, 0, -100, -1, -1);

   // Creating N(2090)+
   new Particle("N(2090)+", 52214, 1, "Unknown", 100, 1, 2.08, 0.35, -100, 0, -100, -1, -1);

   // Creating lambda(1810)
   new Particle("lambda(1810)", 53122, 1, "Unknown", 100, 0, 1.81, 0.15, -100, 0, -100, -1, -1);

   // Creating pi(1300)0
   new Particle("pi(1300)0", 100111, 1, "Unknown", 100, 0, 1.3, 0.4, -100, 0, -100, -1, -1);

   // Creating rho(1450)0
   new Particle("rho(1450)0", 100113, 1, "Unknown", 100, 0, 1.465, 0.4, -100, 0, -100, -1, -1);

   // Creating pi(1300)+
   new Particle("pi(1300)+", 100211, 1, "Unknown", 100, 1, 1.3, 0.4, -100, 0, -100, -1, -1);

   // Creating rho(1450)+
   new Particle("rho(1450)+", 100213, 1, "Unknown", 100, 1, 1.465, 0.4, -100, 0, -100, -1, -1);

   // Creating eta(1295)
   new Particle("eta(1295)", 100221, 1, "Unknown", 100, 0, 1.294, 0.055, -100, 0, -100, -1, -1);

   // Creating omega(1420)
   new Particle("omega(1420)", 100223, 1, "Unknown", 100, 0, 1.425, 0.215, -100, 0, -100, -1, -1);

   // Creating k(1460)0
   new Particle("k(1460)0", 100311, 1, "Unknown", 100, 0, 1.46, 0.26, -100, 0, -100, -1, -1);

   // Creating k_star(1410)0
   new Particle("k_star(1410)0", 100313, 1, "Unknown", 100, 0, 1.414, 0.232, -100, 0, -100, -1, -1);

   // Creating k2_star(1980)0
   new Particle("k2_star(1980)0", 100315, 1, "Unknown", 100, 0, 1.973, 0.373, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0051() {

   // Creating k(1460)+
   new Particle("k(1460)+", 100321, 1, "Unknown", 100, 1, 1.46, 0.26, -100, 0, -100, -1, -1);

   // Creating k_star(1410)+
   new Particle("k_star(1410)+", 100323, 1, "Unknown", 100, 1, 1.414, 0.232, -100, 0, -100, -1, -1);

   // Creating k2_star(1980)+
   new Particle("k2_star(1980)+", 100325, 1, "Unknown", 100, 1, 1.973, 0.373, -100, 0, -100, -1, -1);

   // Creating eta(1475)
   new Particle("eta(1475)", 100331, 1, "Unknown", 100, 0, 1.476, 0.085, -100, 0, -100, -1, -1);

   // Creating phi(1680)
   new Particle("phi(1680)", 100333, 1, "Unknown", 100, 0, 1.68, 0.15, -100, 0, -100, -1, -1);

   // Creating psi'
   new Particle("psi'", 100443, 0, "Unknown", 100, 0, 3.68609, 0, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(100443));
   part->AddDecay(Particle::Decay(0, 0.324,  vector<int>{443,211,-211}));
   part->AddDecay(Particle::Decay(12, 0.1866,  vector<int>{82,-82}));
   part->AddDecay(Particle::Decay(0, 0.184,  vector<int>{443,111,111}));
   part->AddDecay(Particle::Decay(0, 0.093,  vector<int>{10441,22}));
   part->AddDecay(Particle::Decay(0, 0.087,  vector<int>{20443,22}));
   part->AddDecay(Particle::Decay(0, 0.078,  vector<int>{445,22}));
   part->AddDecay(Particle::Decay(0, 0.027,  vector<int>{443,221}));
   part->AddDecay(Particle::Decay(0, 0.0083,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.0083,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.0028,  vector<int>{441,22}));
   part->AddDecay(Particle::Decay(0, 0.001,  vector<int>{443,111}));

   // Creating Upsilon'
   new Particle("Upsilon'", 100553, 0, "Unknown", 100, 0, 10.0233, 0, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(100553));
   part->AddDecay(Particle::Decay(4, 0.425,  vector<int>{21,21,21}));
   part->AddDecay(Particle::Decay(0, 0.185,  vector<int>{553,211,-211}));
   part->AddDecay(Particle::Decay(0, 0.088,  vector<int>{553,111,111}));
   part->AddDecay(Particle::Decay(0, 0.067,  vector<int>{20553,22}));
   part->AddDecay(Particle::Decay(0, 0.066,  vector<int>{555,22}));
   part->AddDecay(Particle::Decay(0, 0.043,  vector<int>{10551,22}));
   part->AddDecay(Particle::Decay(32, 0.024,  vector<int>{2,-2}));
   part->AddDecay(Particle::Decay(32, 0.024,  vector<int>{4,-4}));
   part->AddDecay(Particle::Decay(4, 0.02,  vector<int>{22,21,21}));
   part->AddDecay(Particle::Decay(0, 0.014,  vector<int>{11,-11}));
   part->AddDecay(Particle::Decay(0, 0.014,  vector<int>{13,-13}));
   part->AddDecay(Particle::Decay(0, 0.014,  vector<int>{15,-15}));
   part->AddDecay(Particle::Decay(32, 0.008,  vector<int>{1,-1}));
   part->AddDecay(Particle::Decay(32, 0.008,  vector<int>{3,-3}));

   // Creating ~d_L
   new Particle("~d_L", 1000001, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000001));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1}));

   // Creating ~u_L
   new Particle("~u_L", 1000002, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000002));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,2}));

   // Creating ~s_L
   new Particle("~s_L", 1000003, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000003));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,3}));

   // Creating ~c_L
   new Particle("~c_L", 1000004, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000004));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,4}));

   // Creating ~b_1
   new Particle("~b_1", 1000005, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000005));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,5}));

   // Creating ~t_1
   new Particle("~t_1", 1000006, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000006));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-15,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,16,5}));

   // Creating ~e_L-
   new Particle("~e_L-", 1000011, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000011));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-37}));

   // Creating ~nu_eL
   new Particle("~nu_eL", 1000012, 1, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000012));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,37}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0052() {

   // Creating ~mu_L-
   new Particle("~mu_L-", 1000013, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(1000013));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-37}));

   // Creating ~nu_muL
   new Particle("~nu_muL", 1000014, 1, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000014));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,37}));

   // Creating ~tau_1-
   new Particle("~tau_1-", 1000015, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000015));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-37}));

   // Creating ~nu_tauL
   new Particle("~nu_tauL", 1000016, 1, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000016));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,37}));

   // Creating ~g
   new Particle("~g", 1000021, 0, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000021));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,21}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,6,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,6,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,6,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,6,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,5,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-5,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,5,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-5,6}));

   // Creating ~chi_10
   new Particle("~chi_10", 1000022, 0, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000022));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{4,-1,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1,-3,12}));

   // Creating ~chi_20
   new Particle("~chi_20", 1000023, 0, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000023));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,4,-4}));

   // Creating ~chi_1+
   new Particle("~chi_1+", 1000024, 1, "Sparticle", 100, 1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000024));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,-3,4}));

   // Creating ~chi_30
   new Particle("~chi_30", 1000025, 0, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000025));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,4,-4}));

   // Creating ~chi_40
   new Particle("~chi_40", 1000035, 0, "Sparticle", 100, 0, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000035));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,22}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,11,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,13,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,15,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000002,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000004,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000006,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000011,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000012,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000013,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000014,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000015,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000016,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,4,-4}));

   // Creating ~chi_2+
   new Particle("~chi_2+", 1000037, 1, "Sparticle", 100, 1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(1000037));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,11,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,13,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,15,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,12,-12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,14,-14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,16,-16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,5,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,2,-2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,4,-4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-11,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-13,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-15,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,-3,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000001,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000001,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000003,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000003,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000005,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000005,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000011,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000011,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000013,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000013,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000015,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-2000015,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,-1,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,-3,4}));

   // Creating ~gravitino
   new Particle("~gravitino", 1000039, 0, "Sparticle", 100, 0, 500, 0, -100, -1, -100, -1, -1);

   // Creating ~d_R
   new Particle("~d_R", 2000001, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000001));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000002,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,1}));

   // Creating ~u_R
   new Particle("~u_R", 2000002, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000002));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,1}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,2}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000002,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000001,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000001,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,2}));

   // Creating ~s_R
   new Particle("~s_R", 2000003, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000003));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000004,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,3}));
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0053() {

   // Creating ~c_R
   new Particle("~c_R", 2000004, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   Particle *part = 0;
   part = const_cast<Particle*>(&Particle::Particles().at(2000004));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,3}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,4}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000004,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000003,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000003,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,4}));

   // Creating ~b_2
   new Particle("~b_2", 2000005, 1, "Sparticle", 100, -0.333333, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000005));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000006,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,5}));

   // Creating ~t_2
   new Particle("~t_2", 2000006, 1, "Sparticle", 100, 0.666667, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000006));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000024,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000037,5}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,6}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000006,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000005,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000005,37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000021,6}));

   // Creating ~e_R-
   new Particle("~e_R-", 2000011, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000011));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,12}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,11}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000011,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000012,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000012,-37}));

   // Creating ~nu_eR
   new Particle("~nu_eR", 2000012, 1, "Sparticle", 100, 0, 500, 0, -100, -1, -100, -1, -1);

   // Creating ~mu_R-
   new Particle("~mu_R-", 2000013, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000013));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,14}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,13}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000013,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000014,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000014,-37}));

   // Creating ~nu_muR
   new Particle("~nu_muR", 2000014, 1, "Sparticle", 100, 0, 500, 0, -100, -1, -100, -1, -1);

   // Creating ~tau_2-
   new Particle("~tau_2-", 2000015, 1, "Sparticle", 100, -1, 500, 1, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(2000015));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000039,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000024,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{-1000037,16}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000022,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000023,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000025,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000035,15}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,23}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,25}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,35}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000015,36}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-24}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{1000016,-37}));
   part->AddDecay(Particle::Decay(53, 0,  vector<int>{2000016,-37}));

   // Creating ~nu_tauR
   new Particle("~nu_tauR", 2000016, 1, "Sparticle", 100, 0, 500, 0, -100, -1, -100, -1, -1);

   // Creating d*
   new Particle("d*", 4000001, 1, "Excited", 100, -0.333333, 400, 2.65171, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4000001));
   part->AddDecay(Particle::Decay(53, 0.85422,  vector<int>{21,1}));
   part->AddDecay(Particle::Decay(0, 0.096449,  vector<int>{-24,2}));
   part->AddDecay(Particle::Decay(0, 0.044039,  vector<int>{23,1}));
   part->AddDecay(Particle::Decay(0, 0.005292,  vector<int>{22,1}));

   // Creating u*
   new Particle("u*", 4000002, 1, "Excited", 100, 0.666667, 400, 2.65499, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4000002));
   part->AddDecay(Particle::Decay(0, 0.853166,  vector<int>{21,2}));
   part->AddDecay(Particle::Decay(0, 0.0963291,  vector<int>{24,1}));
   part->AddDecay(Particle::Decay(0, 0.029361,  vector<int>{23,2}));
   part->AddDecay(Particle::Decay(0, 0.021144,  vector<int>{22,2}));

   // Creating e*-
   new Particle("e*-", 4000011, 1, "Excited", 100, -1, 400, 0.42901, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4000011));
   part->AddDecay(Particle::Decay(0, 0.596149,  vector<int>{-24,12}));
   part->AddDecay(Particle::Decay(0, 0.294414,  vector<int>{22,11}));
   part->AddDecay(Particle::Decay(0, 0.109437,  vector<int>{23,11}));

   // Creating nu*_e0
   new Particle("nu*_e0", 4000012, 1, "Excited", 100, 0, 400, 0.41917, -100, -1, -100, -1, -1);
   part = const_cast<Particle*>(&Particle::Particles().at(4000012));
   part->AddDecay(Particle::Decay(0, 0.610139,  vector<int>{24,11}));
   part->AddDecay(Particle::Decay(0, 0.389861,  vector<int>{23,12}));

   // Creating a0(980)0
   new Particle("a0(980)0", 9000111, 1, "Unknown", 100, 0, 0.98, 0.075, -100, 0, -100, -1, -1);

   // Creating a0(980)+
   new Particle("a0(980)+", 9000211, 1, "Unknown", 100, 1, 0.98, 0.06, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0054() {

   // Creating f0(600)
   new Particle("f0(600)", 9000221, 1, "Unknown", 100, 0, 0.8, 0.8, -100, 0, -100, -1, -1);

   // Creating f0(980)
   new Particle("f0(980)", 9010221, 1, "Unknown", 100, 0, 0.98, 0.07, -100, 0, -100, -1, -1);

   // Creating eta(1405)
   new Particle("eta(1405)", 9020221, 1, "Unknown", 100, 0, 1.4098, 0.0511, -100, 0, -100, -1, -1);

   // Creating f0(1500)
   new Particle("f0(1500)", 9030221, 1, "Unknown", 100, 0, 1.505, 0.109, -100, 0, -100, -1, -1);

   // Creating f2(1810)
   new Particle("f2(1810)", 9030225, 1, "Unknown", 100, 0, 1.815, 0.197, -100, 0, -100, -1, -1);

   // Creating f2(2010)
   new Particle("f2(2010)", 9060225, 1, "Unknown", 100, 0, 2.01, 0.2, -100, 0, -100, -1, -1);

   // Creating Cherenkov
   new Particle("Cherenkov", 50000050, 1, "Unknown", 100, 0, 0, 0, -100, 0, -100, -1, -1);

   // Creating ChargedRootino
   new Particle("ChargedRootino", 50000052, 1, "Unknown", 100, -0.333333, 0, 0, -100, 0, -100, -1, -1);

   // Creating GenericIon
   new Particle("GenericIon", 50000060, 1, "Unknown", 100, 0.333333, 0.938272, 0, -100, 0, -100, -1, -1);

   // Creating N(2220)0
   new Particle("N(2220)0", 100002110, 1, "Unknown", 100, 0, 2.25, 0.4, -100, 0, -100, -1, -1);

   // Creating N(2220)+
   new Particle("N(2220)+", 100002210, 1, "Unknown", 100, 1, 2.25, 0.4, -100, 0, -100, -1, -1);

   // Creating N(2250)0
   new Particle("N(2250)0", 100012110, 1, "Unknown", 100, 0, 2.275, 0.5, -100, 0, -100, -1, -1);

   // Creating N(2250)+
   new Particle("N(2250)+", 100012210, 1, "Unknown", 100, 1, 2.275, 0.5, -100, 0, -100, -1, -1);

   // Creating Deuteron
   new Particle("Deuteron", 1000010020, 1, "ion", 100, 1, 1.87106, 0, -100, 0, -100, -1, -1);

   // Creating Triton
   new Particle("Triton", 1000010030, 1, "ion", 100, 1, 2.80941, 1.6916e-33, -100, 0, -100, -1, -1);
}


//________________________________________________________________________________
 VECGEOM_CUDA_HEADER_BOTH
static void CreateParticle0055() {

   // Creating HE3
   new Particle("HE3", 1000020030, 1, "ion", 100, 2, 2.80941, 0, -100, 0, -100, -1, -1);

   // Creating Alpha
   new Particle("Alpha", 1000020040, 1, "ion", 100, 2, 3.7284, 1.6916e-33, -100, 0, -100, -1, -1);
}
#ifdef VECGEOM_NVCC_DEVICE
VECGEOM_CUDA_HEADER_DEVICE bool initDone=false;
#endif
void Particle::CreateParticles() {
#ifndef VECGEOM_NVCC_DEVICE
   static bool initDone=false;
#endif
   if(initDone) return;
   initDone = true;
  CreateParticle0000();
  CreateParticle0001();
  CreateParticle0002();
  CreateParticle0003();
  CreateParticle0004();
  CreateParticle0005();
  CreateParticle0006();
  CreateParticle0007();
  CreateParticle0008();
  CreateParticle0009();
  CreateParticle0010();
  CreateParticle0011();
  CreateParticle0012();
  CreateParticle0013();
  CreateParticle0014();
  CreateParticle0015();
  CreateParticle0016();
  CreateParticle0017();
  CreateParticle0018();
  CreateParticle0019();
  CreateParticle0020();
  CreateParticle0021();
  CreateParticle0022();
  CreateParticle0023();
  CreateParticle0024();
  CreateParticle0025();
  CreateParticle0026();
  CreateParticle0027();
  CreateParticle0028();
  CreateParticle0029();
  CreateParticle0030();
  CreateParticle0031();
  CreateParticle0032();
  CreateParticle0033();
  CreateParticle0034();
  CreateParticle0035();
  CreateParticle0036();
  CreateParticle0037();
  CreateParticle0038();
  CreateParticle0039();
  CreateParticle0040();
  CreateParticle0041();
  CreateParticle0042();
  CreateParticle0043();
  CreateParticle0044();
  CreateParticle0045();
  CreateParticle0046();
  CreateParticle0047();
  CreateParticle0048();
  CreateParticle0049();
  CreateParticle0050();
  CreateParticle0051();
  CreateParticle0052();
  CreateParticle0053();
  CreateParticle0054();
  CreateParticle0055();
}
 } // End of inline namespace
 } // End of vecgeom namespace
#if defined(__clang__) && !defined(__APPLE__)
#pragma clang optimize on
#endif
