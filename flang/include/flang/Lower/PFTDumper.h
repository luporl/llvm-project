// PFTDumper.h //

#ifndef FORTRAN_LOWER_PFTDUMPER_H
#define FORTRAN_LOWER_PFTDUMPER_H

#include "llvm/Support/raw_ostream.h"
#include "flang/Evaluate/expression.h"
#include "flang/Lower/PFTDefs.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::ll {

void dumpEval(llvm::raw_ostream &OS, const lower::pft::Evaluation &Eval);

void dumpPAssign(llvm::raw_ostream &OS, const parser::AssignmentStmt &PAssign);
void dumpPExpr(llvm::raw_ostream &OS, const parser::Expr &PExpr);

void dumpEAssign(llvm::raw_ostream &OS, const evaluate::Assignment &EAssign);
void dumpEExpr(llvm::raw_ostream &OS,
               const evaluate::Expr<evaluate::SomeType> &EExpr);

} // namespace Fortran::ll

#endif
