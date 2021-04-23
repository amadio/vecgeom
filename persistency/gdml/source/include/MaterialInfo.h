//!    \file MaterialInfo.h
//!    \brief Declares helper functions to manipulate data from XML
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once

#ifndef MaterialInfo_h
#define MaterialInfo_h

#include <string>
#include <vector>
#include <utility>
#include <array>
#include <map>

// TODO use std::optional and std::any

namespace vgdml {

struct Isotope {
  std::map<std::string, std::string> attributes;
};

struct Element {
  std::map<std::string, std::string> attributes;
  std::map<std::string, std::string> isotopeFractions;
};

struct Material {
  std::string name;
  std::map<std::string, std::string> attributes;
  std::map<std::string, std::string> fractions;
  std::map<std::string, std::string> components;
};

} // namespace vgdml

#endif
