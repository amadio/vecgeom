//!    \file Frontend.h
//!    \brief Declares the interfaces for loading GDML to VecGeom
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once
#ifndef VGDMLFrontend_h
#define VGDMLFrontend_h

#include <string>
#include <memory>
#include "Middleware.h"

namespace vgdml {

namespace Frontend {
/// Construct a VecGeom geometry tree from a GDML file
/// \return true if the geometry was read and constructed without error
bool Load(std::string const &aFilename, bool validate = true, double mm_unit = 0.1, bool verbose = 1);
} // namespace Frontend

struct Parser {
public:
  /// Construct a VecGeom geometry tree from a GDML file
  /// \return unique_ptr holding VGDML representation of material/auxiliary data, or nullptr if construction fails
  std::unique_ptr<Middleware> Load(std::string const &aFilename, bool validate = true, double mm_unit = 0.1,
                                   bool verbose = 1);
};

} // namespace vgdml

#endif // VGDMLFrontend_h
