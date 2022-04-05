#include "MaskedAssignCheck.h"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <regex>
#include <iostream>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace vecgeom {

// taken from ClangTidyDiagnostic...cpp
// FIXME: move this someplace else
static bool LineIsMarkedWithNOLINT(SourceManager& SM, SourceLocation Loc) {
  bool Invalid;
  const char *CharacterData = SM.getCharacterData(Loc, &Invalid);
  if (!Invalid) {
    const char *P = CharacterData;
    while (*P != '\0' && *P != '\r' && *P != '\n')
      ++P;
    StringRef RestOfLine(CharacterData, P - CharacterData + 1);
    if (RestOfLine.find("NOLINT") != StringRef::npos) {
      return true;
    }
  }
  return false;
}


void MaskedAssignCheck::registerMatchers(MatchFinder *Finder) {
  // find instances of MaskedAssign call expr
  // for the moment very simple filter filtering on 3 arguments
  Finder->addMatcher(callExpr(argumentCountIs(3)).bind("call"), this);
}

void MaskedAssignCheck::check(const MatchFinder::MatchResult &Result) {
  const CallExpr *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("call");

  // check for presence of NOLINT ... in which case we will not perform the check
  if (LineIsMarkedWithNOLINT(*Result.SourceManager, MatchedCall->getExprLoc())) {
    return;
  }

  const std::regex Regex("[vecCore::]?MaskedAssign");
  auto Callee = MatchedCall->getDirectCallee();
  if (Callee) {
    auto str = MatchedCall->getDirectCallee()->getNameAsString();
    if (std::regex_match(str, Regex)) {
      Expr const *expr = MatchedCall->getArg(2)
        ->IgnoreConversionOperatorSingleStep()
        ->IgnoreParenLValueCasts()
        ->IgnoreParenImpCasts()
        ->IgnoreParenCasts();

      // third argument is simple variable reference; ok
      if (dyn_cast<DeclRefExpr>(expr))
        return;

      // third argument is simple variable passed by value; ok
      if (dyn_cast<StmtExpr>(expr))
        return;

      // third argument is simple integer literal; ok
      if (dyn_cast<IntegerLiteral>(expr))
        return;

      // third argument is simple floating point literal; ok
      if (dyn_cast<FloatingLiteral>(expr))
        return;

      // third argument is a boolean literal; ok
      if (dyn_cast<CXXBoolLiteralExpr>(expr))
        return;

      // third argument is simple operator call expr (usually inlined); ok
      if (dyn_cast<CXXOperatorCallExpr>(expr))
        return;

      // third argument is simple CXX constructor expression (usually inlined); ok
      if (dyn_cast<CXXConstructExpr>(expr))
        return;

      // third argument doesn't have non-trivial function call
      if (!expr->hasNonTrivialCall(*Result.Context))
        return;

      // check if this is due a macro expansion
      // FIXME: make sure this comes from vecCore__MaskedAssignFunc
      if (Result.SourceManager->isMacroBodyExpansion(MatchedCall->getExprLoc()))
        return;

      diag(expr->getExprLoc(), "please use vecCore__MaskedAssignFunc"
          " when third argument is non-trivial", DiagnosticIDs::Error);
    }
  }
}

} // namespace vecgeom
} // namespace tidy
} // namespace clang
