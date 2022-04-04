#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_VECGEOM_MASKEDASSIGN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_VECGEOM_MASKEDASSIGN_H

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>

namespace clang {
namespace tidy {
namespace vecgeom {

/// A clang-tidy check checking for correct use
/// of vecCore::MaskedAssign inside VecGeom
///
class MaskedAssignCheck : public ClangTidyCheck {
public:
  MaskedAssignCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace vecgeom
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_VECGEOM_MASKEDASSIGN_H
