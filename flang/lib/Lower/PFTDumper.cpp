// PFTDumper.cpp //

#include "flang/Lower/PFTDumper.h"
#include "flang/Lower/PFTBuilder.h"

// Macros
#define STR2(x)   #x
#define STR(x)    STR2(x)

// DeRef args - used to expand parenthesized args
//#define DR(...)   __VA_ARGS__

// NOTE: The macros bellow receive a T arg, that holds parenthesized args,
//       to avoid issues with template types with commas.
// DumpUnion
#define DU(...) [&](const __VA_ARGS__ &x) { dumpU(x); }
// DumpType
//#define DT(...) [&](const __VA_ARGS__ &x) { dumpT(x); }

// Type Name function generator
#define TYPENAME(...)                               \
static const char *typeName(const __VA_ARGS__ &X)   \
{                                                   \
  return STR((__VA_ARGS__));                        \
}

namespace Fortran::ll {

// common
TYPENAME(common::Indirection<parser::CharLiteralConstantSubstring>)
TYPENAME(common::Indirection<parser::Designator>)
TYPENAME(common::Indirection<parser::Expr>)
TYPENAME(common::Indirection<parser::FunctionReference>)
TYPENAME(common::Indirection<parser::SubstringInquiry>)

// parser
TYPENAME(Fortran::common::Indirection<Fortran::parser::AcImpliedDo, false>)
TYPENAME(parser::AcValue)
TYPENAME(parser::AcValue::Triplet)
TYPENAME(parser::ArrayConstructor)
TYPENAME(parser::Expr)
TYPENAME(parser::Expr::AND)
TYPENAME(parser::Expr::Add)
TYPENAME(parser::Expr::ComplexConstructor)
TYPENAME(parser::Expr::Concat)
TYPENAME(parser::Expr::DefinedBinary)
TYPENAME(parser::Expr::DefinedUnary)
TYPENAME(parser::Expr::Divide)
TYPENAME(parser::Expr::EQ)
TYPENAME(parser::Expr::EQV)
TYPENAME(parser::Expr::GE)
TYPENAME(parser::Expr::GT)
TYPENAME(parser::Expr::LE)
TYPENAME(parser::Expr::LT)
TYPENAME(parser::Expr::Multiply)
TYPENAME(parser::Expr::NE)
TYPENAME(parser::Expr::NEQV)
TYPENAME(parser::Expr::NOT)
TYPENAME(parser::Expr::Negate)
TYPENAME(parser::Expr::OR)
TYPENAME(parser::Expr::Parentheses)
TYPENAME(parser::Expr::PercentLoc)
TYPENAME(parser::Expr::Power)
TYPENAME(parser::Expr::Subtract)
TYPENAME(parser::Expr::UnaryPlus)
TYPENAME(parser::LiteralConstant)
TYPENAME(parser::StructureConstructor)


// evaluate
TYPENAME(evaluate::Add<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::ArrayConstructor<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::ArrayConstructorValue<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Constant<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Convert<evaluate::Type<common::TypeCategory::Integer, 4>, common::TypeCategory::Integer>)
TYPENAME(evaluate::Convert<evaluate::Type<common::TypeCategory::Integer, 4>, common::TypeCategory::Real>)
TYPENAME(evaluate::Designator<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Divide<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Expr<evaluate::SomeInteger>)
TYPENAME(evaluate::Expr<evaluate::SomeKind<common::TypeCategory::Character>>)
TYPENAME(evaluate::Expr<evaluate::SomeKind<common::TypeCategory::Complex>>)
TYPENAME(evaluate::Expr<evaluate::SomeKind<common::TypeCategory::Derived>>)
TYPENAME(evaluate::Expr<evaluate::SomeKind<common::TypeCategory::Logical>>)
TYPENAME(evaluate::Expr<evaluate::SomeKind<common::TypeCategory::Real>>)
//TYPENAME(evaluate::Expr<evaluate::SomeType>)
TYPENAME(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 16>>)
TYPENAME(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 1>>)
TYPENAME(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 2>>)
TYPENAME(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 8>>)
TYPENAME(evaluate::Extremum<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::FunctionRef<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::ImpliedDo<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Multiply<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Negate<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::NullPointer)
TYPENAME(evaluate::Parentheses<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::Power<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::ProcedureDesignator)
TYPENAME(evaluate::ProcedureRef)
TYPENAME(evaluate::Subtract<evaluate::Type<common::TypeCategory::Integer, 4>>)
TYPENAME(evaluate::value::Integer<128>)


class ExprDumper
{
public:
  ExprDumper(llvm::raw_ostream &OS) : OS(OS) {}

  template <typename T>
  void visit(const T &X)
  {
    common::visit(common::visitors{
      // parser
      [&](const common::Indirection<parser::Expr> &X) {
        printT0(X);
        dumpU(X.value());
        printDone();
      },

      [&](const parser::ArrayConstructor &X) {
        printT0(X);
        for (const auto &I : X.v.values)
          dumpU(I);
        printDone();
      },

      [&](const parser::Expr::Add &X) {
        printT0(X);
        dumpU(std::get<0>(X.t).value());
        dumpU(std::get<1>(X.t).value());
        printDone();
      },

      // evaluate
      DU(evaluate::Expr<evaluate::SomeInteger>),
      DU(evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 4>>),

      [&](const evaluate::ArrayConstructor<evaluate::Type<common::TypeCategory::Integer, 4>> &X) {
        printT0(X);
        for (const auto &I : X)
          dumpU(I);
        printDone();
      },

      [&](const evaluate::Add<evaluate::Type<common::TypeCategory::Integer, 4>> &X) {
        printT0(X);
        dumpU(X.left());
        dumpU(X.right());
        printDone();
      },

      // default
      [&](const auto &Other) { dumpT(Other); }
      }, X.u);
  }

private:
  void indent()
  {
    for (int I = 0; I < IndentN; I++)
      OS << "  ";
  }

  template <typename T>
  void printT(const T &X)
  {
      indent();
      OS << typeName(X) << "\n";
  }

  template <typename T>
  void printT0(const T &X)
  {
    IndentN++;
    printT(X);
  }

  void printDone()
  {
      IndentN--;
  }

  template <typename T>
  void dumpT(const T &X)
  {
      printT0(X);
      printDone();
  }

  template <typename T>
  void dumpU(const T &X)
  {
      printT0(X);
      visit(X);
      printDone();
  }

  llvm::raw_ostream &OS;
  int IndentN = 0;
};

void dumpPExpr(llvm::raw_ostream &OS, const parser::Expr &PExpr)
{
  ExprDumper ED(OS);
  ED.visit(PExpr);
}

void dumpPAssign(llvm::raw_ostream &OS, const parser::AssignmentStmt &PAssign)
{
  const parser::Expr &PExpr = std::get<1>(PAssign.t);

  OS << ">>> parser::AssignmentStmt: <var> = parser::Expr\n";
  dumpPExpr(OS, PExpr);
}

void dumpEExpr(llvm::raw_ostream &OS,
               const evaluate::Expr<evaluate::SomeType> &EExpr)
{
  ExprDumper ED(OS);
  ED.visit(EExpr);
}

void dumpEAssign(llvm::raw_ostream &OS, const evaluate::Assignment &EAssign)
{
  const evaluate::Expr<evaluate::SomeType> &EExpr = EAssign.rhs;

  OS << ">>> evaluate::Assignment: <var> = evaluate::Expr<evaluate::SomeType>\n";
  dumpEExpr(OS, EExpr);
}

void dumpEval(llvm::raw_ostream &OS, const lower::pft::Evaluation &Eval)
{
  if (!Eval.isA<parser::AssignmentStmt>())
    return;

  // Dump parser sub-tree
  const parser::AssignmentStmt &PAssign = Eval.get<parser::AssignmentStmt>();
  dumpPAssign(OS, PAssign);

  // Dump eval sub-tree
  if (!PAssign.typedAssignment->v)
    return;

  const evaluate::Assignment &EAssign = *PAssign.typedAssignment->v;
  dumpEAssign(OS, EAssign);
}

} // namespace Fortran::ll
