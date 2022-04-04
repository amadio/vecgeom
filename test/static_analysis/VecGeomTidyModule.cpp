#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/ClangTidyModuleRegistry.h>
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

// Register the  using this statically initialized variable.
static ClangTidyModuleRegistry::Add<VecGeomTidyModule> X("vecgeom-module", "Adds VecGeom specific checks.");

} // namespace vecgeom

// This anchor is used to force the linker to link in the generated object file
// and thus register the AliceO2Module.
volatile int VecGeomTidyModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
