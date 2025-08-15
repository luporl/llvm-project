#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Runtime/entry-names.h"

namespace hlfir {
#define GEN_PASS_DEF_EXPRESSIONSIMPLIFICATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "expression-simplification"

/*
 * TODOs
 * - document simplifyCharCompare()
 * - reduce debug info and use LDBG() instead
 *   (llvm/Support/DebugLog.h)
 * - remove extra includes
 * - invert args nest and parentUses
 * - review style
 * - del delLastUser
 * - review TrimRemover name
 * - FIXME adapt logic for -O3
 * - TODO test case when str0 == str1
 * - test other comparisons: </>/<=/>=
 */

#define NL "\n"
static bool outEnabled = true;
static bool delOutEnabled = false;

static llvm::raw_ostream &out()
{
    if (outEnabled)
      return llvm::errs() << "LLL: simplifyCharCompare: ";
    return llvm::nulls() << "";
}

static llvm::raw_ostream &delOut(int nest)
{
  std::string tabs;

  while (nest--)
    tabs += "  ";
  return (delOutEnabled ? llvm::errs() : llvm::nulls()) << tabs;
}

static void delOpOperands(mlir::Operation *op, int nest);

static void delOp(mlir::Operation *op, int nest, int parentUses = 0) {
  int uses = std::distance(op->getUses().begin(), op->getUses().end());

  if (uses <= parentUses) {
    delOut(nest) << "delOp(): deleting " << *op << NL;
    delOpOperands(op, nest);
    op->dropAllReferences();
    op->dropAllUses();
    op->erase();
  } else {
    delOut(nest) << "delOp(): not deleting " << *op
      << ". Uses: " << std::distance(op->getUses().begin(), op->getUses().end()) << NL;
  }
}

static void delOpOperands(mlir::Operation *op, int nest)
{
  for (mlir::Value operand : op->getOperands()) {
    delOut(nest) << "delOpOperands(): operand: " << operand << NL;
    if (!operand)
      // already deleted
      continue;
    mlir::Operation *operandOp = operand.getDefiningOp();
    if (operandOp) {
      int uses = 0;
      for (auto &use : operandOp->getUses())
        if (use.getOwner() == op)
          ++uses;
      delOp(operandOp, nest + 1, uses);
    }
  }
}

template <typename Op>
static void delOpAndOperands(Op op)
{
  mlir::Operation *o = op.getOperation();
  if (!o)
    return;
  delOp(o, 0);
  delOut(0) << NL;
}

template <typename UserOp, typename Op>
static UserOp getFirstUser(Op o)
{
  mlir::Operation *op = o.getOperation();
  if (!op)
    return {};

  auto it = op->user_begin(), end = op->user_end(), prev = it;
  for (; it != end; prev = it++)
    ;
  if (prev != end)
    if (auto userOp = mlir::dyn_cast<UserOp>(**prev))
      return userOp;
  return {};
}

template <typename UserOp, typename Op>
static UserOp getLastUser(Op o)
{
  if (mlir::Operation *op = o.getOperation()) {
    if (!op->getUses().empty()) {
      if (auto userOp = mlir::dyn_cast<UserOp>(op->use_begin()->getOwner()))
        return userOp;
    }
  }
  return {};
}

template <typename UserOp, typename Op1, typename Op2>
static UserOp getPreviousUser(Op1 o, Op2 curUser)
{
  mlir::Operation *op = o.getOperation();
  if (!op)
    return {};

  for (auto user = op->user_begin(), end = op->user_end(); user != end; ++user) {
    if (*user == curUser) {
      if (++user == end)
        break;
      if (auto userOp = mlir::dyn_cast<UserOp>(*user))
        return userOp;
      break;
    }
  }
  return {};
}

template <typename UserOp, typename Op>
static void delLastUser(Op o)
{
  if (mlir::Operation *op = o.getOperation()) {
    if (!op->getUses().empty()) {
      if (auto userOp = mlir::dyn_cast<UserOp>(op->use_begin()->getOwner())) {
        delOpAndOperands(userOp);
      }
    }
  }
}

template <typename Op>
static Op expectOp(mlir::Value val) {
  if (Op op = mlir::dyn_cast_or_null<Op>(val.getDefiningOp()))
    return op;
  return nullptr;
}

template <typename Op>
static mlir::Value findDefSingle(fir::ConvertOp op) {
  if (auto defOp = expectOp<Op>(op->getOperand(0))) {
    return defOp.getResult();
  }
  return {};
}

template <typename... Ops>
static mlir::Value findDef(fir::ConvertOp op) {
  mlir::Value defOp;
  // Loop over the operation types given to see if any match, exiting once
  // a match is found. Cast to void is needed to avoid compiler complaining
  // that the result of expression is unused
  (void)((defOp = findDefSingle<Ops>(op), (defOp)) || ...);
  return defOp;
}

static mlir::Value findBoxDef(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    return findDef<fir::EmboxOp, fir::ReboxOp>(op);
  }
  return {};
}

template <typename Op>
static Op expOp(mlir::Value val, const char *id)
{
  auto op = expectOp<Op>(val);
  if (op)
    out() << id << ": " << op << NL;
  return op;
}

namespace {

class TrimRemover {
public:
  TrimRemover(fir::FirOpBuilder &builder, mlir::Value charVal, mlir::Value charLenVal)
    : builder(builder), charVal(charVal), charLenVal(charLenVal) {}
  TrimRemover(const TrimRemover &) = delete;

  bool charWasTrimmed();
  void removeTrim();

private:
  // input
  fir::FirOpBuilder &builder;
  mlir::Value charVal;
  mlir::Value charLenVal;

  // state

  // Needed for trim removal
  fir::ConvertOp charCvtOp;     // May be replaced by charVal
  fir::ConvertOp charLenCvtOp;  // May be replaced by charLenVal
  hlfir::DeclareOp charDeclOp;  // Replaces trim result
  fir::CallOp trimCallOp;

  hlfir::EndAssociateOp endAssocOp;   // The other assocOp user
  hlfir::DestroyOp destroyExprOp;     // The other asExprOp user
  // allocaOp can only be removed after all of its users and associated
  // storeOp are removed.
  fir::AllocaOp allocaOp;
};

template <typename Op>
static bool expectUses(Op &op, int expUses)
{
  mlir::Operation *o = op.getOperation();
  if (!o)
    return false;
  int uses = std::distance(o->use_begin(), o->use_end());
  if (uses != expUses) {
    out() << "expectUses: expected " << expUses << ", got " << uses << NL;
    for (auto user : o->getUsers())
      out() << "\t" << *user << NL;
  }
  return uses == expUses;
}

bool TrimRemover::charWasTrimmed() {
  out() << "arg: " << charVal << NL;

  // Get CharacterCompareScalar args
  charCvtOp = expOp<fir::ConvertOp>(charVal, "cvt");
  charLenCvtOp = expOp<fir::ConvertOp>(charLenVal, "cvtLen");
  if (!charCvtOp || !charLenCvtOp)
    return false;

  // Get decl and expression associated to 'charVal'
  auto assocOp = expOp<hlfir::AssociateOp>(charCvtOp.getOperand(), "assoc");
  // end_associate uses it twice
  if (!assocOp || !expectUses(assocOp, 3))
    return false;
  endAssocOp = getLastUser<hlfir::EndAssociateOp>(assocOp);
  if (!endAssocOp)
    return false;
  auto asExprOp = expOp<hlfir::AsExprOp>(assocOp.getOperand(0), "expr");
  if (!asExprOp || !expectUses(asExprOp, 2))
    return false;
  destroyExprOp = getLastUser<hlfir::DestroyOp>(asExprOp);
  if (!destroyExprOp)
    return false;
  auto declOp = expOp<hlfir::DeclareOp>(asExprOp.getOperand(0), "decl");
  if (!declOp || !expectUses(declOp, 1))
    return false;

  // Get associated box and alloca
  auto boxAddrOp = expOp<fir::BoxAddrOp>(declOp.getMemref(), "boxAddr");
  if (!boxAddrOp || !expectUses(boxAddrOp, 1))
    return false;
  auto loadOp = expOp<fir::LoadOp>(boxAddrOp.getOperand(), "load");
  if (!loadOp || !getFirstUser<fir::BoxEleSizeOp>(loadOp) || !expectUses(loadOp, 2))
    return false;
  allocaOp = expOp<fir::AllocaOp>(loadOp.getMemref(), "alloca");
  if (!allocaOp ||
      !getFirstUser<fir::StoreOp>(allocaOp) ||  // initialization
                                                // load
                                                // convert, used by trim
      !expectUses(allocaOp, 3))
    return false;

  // Get previous user of allocaOp and its user (trim call)
  if (auto userOp = getPreviousUser<fir::ConvertOp>(allocaOp, loadOp)) {
    out() << "prevUser: " << userOp << NL;
    // Check if the previous user is the only user one and a CallOp
    if (userOp.getOperation() && userOp->hasOneUse())
      trimCallOp = mlir::dyn_cast<fir::CallOp>(*userOp->user_begin());
  }

  if (!trimCallOp)
    return false;
  out() << "call: " << trimCallOp << NL;
  llvm::StringRef calleeName = trimCallOp.getCalleeAttr().getLeafReference().getValue();
  out() << "callee: " << calleeName << NL;
  if (calleeName != RTNAME_STRING(Trim))
    return false;

  // Get source char
  auto chrEmboxOp = expOp<fir::EmboxOp>(findBoxDef(trimCallOp.getOperand(1)), "chrEmbox");
  if (!chrEmboxOp)
    return false;
  charDeclOp = expOp<hlfir::DeclareOp>(chrEmboxOp.getMemref(), "charDecl");
  if (!charDeclOp)
    return false;

  // Found everything as expected.
  return true;
}

void TrimRemover::removeTrim()
{
  // Replace trim output char with its input
  mlir::Location loc = charVal.getLoc();
  auto cvtOp = fir::ConvertOp::create(builder, loc, charCvtOp.getType(),
      charDeclOp.getOriginalBase());
  charCvtOp.replaceAllUsesWith(cvtOp.getResult());

  // Replace trim output length with its input
  mlir::Value strLen = charDeclOp.getTypeparams().back();
  auto cvtLenOp = fir::ConvertOp::create(builder, loc, charLenCvtOp.getType(), strLen);
  charLenCvtOp.replaceAllUsesWith(cvtLenOp.getResult());

  // delelte old converts
  delOpAndOperands(charCvtOp);
  delOpAndOperands(charLenCvtOp);
  // delete trim call
  delOpAndOperands(trimCallOp);

  // delete associate (assocOp, endAssocOp)
  delOpAndOperands(endAssocOp);
  // delete expr (destroyOp, asExprOp, declOp)
  delOpAndOperands(destroyExprOp);
  // delete initialzation store and alloca
  delLastUser<fir::StoreOp>(allocaOp);
}

} // namespace

namespace {

class ExpressionSimplification
    : public hlfir::impl::ExpressionSimplificationBase<ExpressionSimplification> {
public:
  using ExpressionSimplificationBase<ExpressionSimplification
      >::ExpressionSimplificationBase;

  void runOnOperation() override;

private:
  void simplifyCharCompare(fir::CallOp call, const fir::KindMapping &kindMap);
};

void ExpressionSimplification::simplifyCharCompare(fir::CallOp call,
                                        const fir::KindMapping &kindMap) {
  fir::FirOpBuilder builder{call, kindMap};
  mlir::Operation::operand_range args = call.getArgs();
  TrimRemover lhsTrimRem(builder, args[0], args[2]);
  TrimRemover rhsTrimRem(builder, args[1], args[3]);

  outEnabled = true;
  if (lhsTrimRem.charWasTrimmed()) {
    delOut(0) << "\nLLL: *** DEL trim0 ***\n";
    lhsTrimRem.removeTrim();
  }
  if (rhsTrimRem.charWasTrimmed()) {
    delOut(0) << "\nLLL: *** DEL trim1 ***\n";
    rhsTrimRem.removeTrim();
  }
}

void ExpressionSimplification::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  fir::KindMapping kindMap = fir::getKindMapping(module);
  module.walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      if (mlir::SymbolRefAttr callee = call.getCalleeAttr()) {
        mlir::StringRef funcName = callee.getLeafReference().getValue();
        if (funcName.starts_with(RTNAME_STRING(CharacterCompareScalar))) {
          simplifyCharCompare(call, kindMap);
        }
      }
    }
  });
}

} // namespace
