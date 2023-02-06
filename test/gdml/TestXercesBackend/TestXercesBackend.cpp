//!    \file TestXercesBackend.cpp
//!    \brief reads a gdml file to DOM and writes it back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include <iostream>
#include "Backend.h"
#include "xercesc/dom/DOMDocument.hpp"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestXercesBackend <filename>.gdml \n"
               "      will use TestXercesBackend.gdml by default\n"
            << std::endl;
}
} // namespace

int main(int argC, char *argV[])
{
  if (argC != 2) {
    usage();
  }
  auto const filename = std::string((argC > 1) ? argV[1] : "TestXercesBackend.gdml");
  auto aBackend       = vgdml::Backend(true);
  auto const loaded   = aBackend.Load(filename);
  aBackend.Save(loaded, "TestXercesBackend.out.gdml");
  return 0;
}
