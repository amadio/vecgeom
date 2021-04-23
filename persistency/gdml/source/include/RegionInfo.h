//!    \file RegionInfo.h
//!    \brief Declares structure to store region information data
//!
//!    \authors Author:  Andrei Gheata <andrei.gheata@cern.ch>
//!

#pragma once

#ifndef RegionInfo_h
#define RegionInfo_h

#include <string>
#include <list>
#include <map>

namespace vgdml {

struct RegionCut {
  std::string name;
  double value;
  RegionCut(const std::string &n, double v) : name(n), value(v) {}
};

struct Region {
  std::string name;
  std::list<std::string> volNames;
  std::list<RegionCut> cuts;
};

} // namespace vgdml

#endif
