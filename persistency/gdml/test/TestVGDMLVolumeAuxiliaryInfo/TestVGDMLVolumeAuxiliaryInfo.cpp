#include <iostream>

#include "Backend.h"
#include "Middleware.h"
#include "Auxiliary.h"
#include "VecGeom/management/GeoManager.h"

void exit_error(int line, const std::string &msg)
{
  std::cerr << "error(line : " << line << "): " << msg << std::endl;
  exit(1);
}

using vgdml::Auxiliary;
using vgdml::Backend;
using vgdml::Middleware;

// Construct expected list of auxiliary info for "Boxvol"
std::vector<Auxiliary> makeExpectedBoxVolAux() {
  std::vector<Auxiliary> tmp;
  Auxiliary a;
  a.SetType("SensDet");
  a.SetValue("Tracker");
  tmp.push_back(a);

  Auxiliary b;
  b.SetType("Color");
  b.SetValue("Blue");
  tmp.push_back(b);

  return tmp;
}

int main(int argc, char* argv[]) {
  // Here we simply check that after reading in the GDML that the auxiliaries
  // are attached correctly to the volume
  // *Assumes* the gdmls/auxiliary.gdml file is passed as argv[1]!
  if(argc != 2) {
    exit_error(__LINE__, "argc is not 2!");
  }

  // Load the gdml
  // To speed up testing and ensure the middleware and backend work without
  // schema validation, skip it for the middleware test.
  std::string filename(argv[1]);
  auto aBackend       = Backend(false);
  auto const aDOMDoc  = aBackend.Load(filename);
  auto aMiddleware      = Middleware();
  auto loadedMiddleware = aMiddleware.Load(aDOMDoc);

  if(!loadedMiddleware) {
    exit_error(__LINE__, "Middleware::Load failed to load file: " + filename);
  }

  // Get the volume auxiliary info
  const auto& volAuxInfo = aMiddleware.GetVolumeAuxiliaryInfo();

  // Expect: one entry
  if(volAuxInfo.size() != 1) {
    exit_error(__LINE__, std::to_string(volAuxInfo.size()) + " volumes have auxiliaries, expected 1");
  }

  // Expect: volume named "Boxvol"
  // - VecGeom only guarantees a unique int Id, so find the Id for 'BoxVol'
  auto& gm = vecgeom::GeoManager::Instance();
  auto* boxVolLV = gm.FindLogicalVolume("Boxvol");
  if (boxVolLV == nullptr) {
    exit_error(__LINE__, "expected logical volume named BoxVol, but it is not present in vecgeom::GeoManager");
  }

  // ... that has auxiliary tags
  int boxVolId = boxVolLV->id();
  auto iter = volAuxInfo.find(boxVolId);
  if(iter == volAuxInfo.end()) {
    exit_error(__LINE__, "expected 'Boxvol' to have auxiliary info, but it does not");
  }

  // ... and has two auxiliary tags
  auto actualSize = (*iter).second.size();
  if(actualSize != 2) {
    exit_error(__LINE__, "expected 'Boxvol' to have 2 auxiliaries, but it has " + std::to_string(actualSize));
  }

  // ... that have the expected value(s)
  const auto& expectedAux = makeExpectedBoxVolAux();
  if(expectedAux != (*iter).second) {
    exit_error(__LINE__, "'BoxVol' auxiliary info is not as expected");
  }

  return 0;
}