//!    \file Frontend.h
//!    \brief Declares the interface for loading GDML to VecGeom
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

class Middleware;

namespace Frontend {
std::unique_ptr<Middleware> Load(std::string const &aFilename, bool validate = true, double mm_unit = 0.1,
                                 bool verbose = 1);
} // namespace Frontend

} // namespace vgdml

#endif // VGDMLFrontend_h
