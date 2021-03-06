//!    \file Frontend.cpp
//!    \brief implements loading GDML to VecGeom
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Frontend.h"
#include "Backend.h"
#include "Middleware.h"

#include <string>

namespace vgdml {

namespace Frontend {
bool Load(std::string const &aFilename, bool validate)
{
  auto aBackend      = vgdml::Backend(validate);
  auto const aDOMDoc = aBackend.Load(aFilename);
  if (!aDOMDoc) {
    std::cerr << "== Error: GDML file " << aFilename << " could not be loaded\n";
    return false;
  }
  auto aMiddleware            = vgdml::Middleware();
  auto const loadedMiddleware = aMiddleware.Load(aDOMDoc);
  return loadedMiddleware;
} // namespace Frontend

} // namespace Frontend

} // namespace vgdml
