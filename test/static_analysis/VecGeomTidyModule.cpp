#include "ClangTidy.h"
#include "ClangTidyModule.h"
#include "ClangTidyModuleRegistry.h"
#include "MaskedAssignCheck.h"

namespace clang {
namespace tidy {
namespace vecgeom {

class VecGeomTidyModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<MaskedAssignCheck>(
        "vecgeom-MaskedAssign");
  }
};

} // namespace vecgeom

// Register the  using this statically initialized variable.
static ClangTidyModuleRegistry::Add<vecgeom::VecGeomTidyModule> X("vecgeom-tidy-module", "Adds VecGeom specific checks");

// This anchor is used to force the linker to link in the generated object file
// and thus register the AliceO2Module.
volatile int VecGeomTidyModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// A function to execute upon load of shared library
// will register this plugin will clang-tidy
static int VecGeomTidyModule_sharedlib_init() {
  static clang::tidy::vecgeom::VecGeomTidyModule module;
  return clang::tidy::VecGeomTidyModuleAnchorSource;
}

#pragma clang diagnostic pop
