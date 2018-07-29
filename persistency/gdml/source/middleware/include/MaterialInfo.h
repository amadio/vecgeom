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

// TODO use std::optional and std::any

namespace vgdml {

struct Isotope
{
    std::array<std::string const, 6> possibleAttributes = {{"name", "Z", "N", "atomType", "atomValue", "state"}};
    std::vector<std::pair<std::string, std::string>> attributes;
};

struct Element
{
    std::array<std::string const, 4> possibleAttributes = {{"name", "formula", "Z", "atomValue"}};
    std::vector<std::pair<std::string, std::string>> attributes;
    std::vector<std::pair<std::string, std::string>> isotopeFractions;
};

struct Material
{
    std::array<std::string const, 4> const possibleAttributes = {{"name", "formula", "Z", "D"}};
    std::vector<std::pair<std::string, std::string>> attributes;
    std::vector<std::pair<std::string, std::string>> fractions;
    std::vector<std::pair<std::string, std::string>> composites;
};


} // namespace vgdml

#endif
