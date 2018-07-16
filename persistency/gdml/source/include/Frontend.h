//!    \file Frontend.h
//!    \brief Declares the interface for loading GDML to VecGeom
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once
#ifndef VGDMLFrontend_h
#define VGDMLFrontend_h

#include <string>

namespace vgdml {

namespace Frontend {
bool Load(std::string const &aFilename);
} // namespace Frontend

} // namespace vgdml

#endif // VGDMLFrontend_h
