// this file is part of VecGeom
// a service that generates a cpp function/file which instantiates a given list of
// specialized navigators with the purpose to link them into a shared library
// the service also generates a CMakeLists.txt file to facilitate this process
// started 27.2.2016; sandro.wenzel@cern.ch

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

void GenerateHeaderIncludes(std::ostream &ss, std::vector<std::string> const &navigatornames)
{
  // include base stuff
  ss << "#include \"management/GeoManager.h\"\n";
  ss << "#include \"volumes/LogicalVolume.h\"\n";

  // include specific navigator headers ( must exist of course )
  for (auto &n : navigatornames) {
    ss << "#include \"" << n << ".h\"\n";
  }
}

void GenerateNavigatorInstantiationFunction(std::ostream &ss, std::vector<std::string> const &volumenames,
                                            std::vector<std::string> const &navigatornames)
{
  ss << "void InitSpecializedNavigators(){\n"; // start function
  ss << "std::cerr << \"---- SpecializedNavigator Initializer Called ---- \\n \";\n";
  ss << "int counter=0;\n";
  ss << "for( auto & lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap() ){\n";

  int counter = 0;
  for (auto &n : volumenames) {
    ss << " if(std::strcmp(lvol.second->GetName(),\"" << n << "\") == 0){\n";
    ss << " lvol.second->SetNavigator(vecgeom::" << navigatornames[counter] << "::Instance());\n";
    ss << " std::cerr << \"---- assigning specialized navigator \" "
       << "\"" << navigatornames[counter] << "\""
       << "\"\\n\"; \n";
    ss << " counter++;\n";
    ss << "}\n"; // end if
    counter++;
  }
  ss << "}\n"; // end loop
  ss << "std::cerr << \"overwrote \" << counter << \" navigators \\n\";\n";
  ss << "std::cerr << \"---- SpecializedNavigator Initializer Call Ends ---- \\n \";\n";
  ss << "}\n"; // end function
}

// to generate the CMakeFile in order to compile and link this
void GenerateCMakeFile(std::ostream &ss)
{
  ss << "cmake_minimum_required(VERSION 3.1.0)\n";
  ss << "find_package(VecGeom REQUIRED)\n";
  ss << "#it is allowed to set the compiler before project and language specification\n";
  ss << "set(CMAKE_C_COMPILER ${VECGEOM_C_COMPILER})\n";
  ss << "set(CMAKE_CXX_COMPILER ${VECGEOM_CXX_COMPILER})\n";
  ss << "project(navigatorlib)\n";

  ss << "enable_language(CXX)\n";
  ss << "set(CMAKE_CXX_STANDARD 17 CACHE STRING \"C++ ISO Standard\")\n";
  ss << "set(CMAKE_CXX_STANDARD_REQUIRED True)\n";

  ss << "if (NOT CMAKE_BUILD_TYPE)\n";
  ss << "  message(STATUS \"No build type selected, default to Release\")\n";
  ss << "  set(CMAKE_BUILD_TYPE \"Release\")\n";
  ss << "endif()\n";

  ss << "include_directories(${VECGEOM_INCLUDE_DIR})\n";
  ss << "# include stuff that was included by VecGeom during build\n";
  ss << "include_directories(${VECGEOM_EXTERNAL_INCLUDES})\n";
  ss << "message(STATUS \"COMPILING WITH ${CMAKE_CXX_FLAGS}\")\n";

  ss << "if(APPLE)\n";
  ss << "  # postpone final symbol resolution to plugin load-time (allow unresolved symbols now)\n";
  ss << "  # a measure that needs to be done on APPLE only\n";
  ss << "  set (CMAKE_MODULE_LINKER_FLAGS \"-Wl,-flat_namespace -Wl,-undefined,warning\")\n";
  ss << "endif()\n";
  ss << "add_library(GeneratedNavigators MODULE navigatorlib.cpp)\n";
}

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cerr << "usage : " << argv[0] << " LVolumeName [LVolumeName ...] \n";
    return 1;
  }

  std::vector<std::string> volumenames;
  std::vector<std::string> navigatornames;

  for (int i = 1; i < argc; ++i) {
    volumenames.push_back(argv[i]);
    navigatornames.push_back(std::string(argv[i]) + std::string("Navigator"));
  }

  std::ofstream cppoutputfile;
  cppoutputfile.open("navigatorlib.cpp");

  GenerateHeaderIncludes(cppoutputfile, navigatornames);

  GenerateNavigatorInstantiationFunction(cppoutputfile, volumenames, navigatornames);
  cppoutputfile.close();

  std::ofstream cmakeoutputfile;
  cmakeoutputfile.open("CMakeLists.txt");
  GenerateCMakeFile(cmakeoutputfile);
  cmakeoutputfile.close();

  return 0;
}
