//!    \file TestXercesMiddleware.cpp
//!    \brief reads a gdml file to DOM and writes it back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include <iostream>
#include "Backend.h"
#include "Middleware.h"
#include "xercesc/dom/DOMDocument.hpp"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestXercesMiddleware <filename>.gdml \n"
               "      will use TestXercesMiddleware.gdml by default\n"
            << std::endl;
}
} // namespace

int main(int argC, char *argV[])
{
  if (argC != 2) {
    usage();
  }
  auto const filename = std::string((argC > 1) ? argV[1] : "TestXercesMiddleware.gdml");
  auto aBackend       = vgdml::Backend();
  auto const aDOMDoc  = aBackend.Load(filename);
  aBackend.Save(aDOMDoc, "TestXercesMiddleware.out.gdml");
  auto aMiddleware      = vgdml::Middleware();
  auto loadedMiddleware = aMiddleware.Load(aDOMDoc);
  //  std::cout << loadedMiddleware << std::endl;
  return !loadedMiddleware;
}
