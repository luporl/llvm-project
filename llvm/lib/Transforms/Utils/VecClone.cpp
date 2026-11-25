//===- VecClone.cpp - Synthesize OpenMP declare simd vector variants ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype port of https://reviews.llvm.org/D22792 to current LLVM.
//
// For each function definition with `_ZGV*` attributes (emitted by the
// OpenMPIRBuilder for `declare simd`), a definition of every vector variant is
// synthesized. Conceptually:
//
//   float foo(float x, int k)       // declare simd linear(k)
//
// becomes (for _ZGVnN4vl_foo):
//
//   <4 x float> _ZGVnN4vl_foo(<4 x float> x, int k) {
//     float xs[4] = x; float rs[4];
//     for (i = 0; i < 4; ++i)          // !llvm.loop vectorize.enable, width 4
//       rs[i] = <body of foo>(xs[i], k + i * step);
//     return rs;
//   }
//
// Masked variants only execute the body for lanes whose mask bit is set.
// The loop is later vectorized by LoopVectorize (at -O1 and above); at -O0 the
// variant is still a correct scalar implementation of the vector ABI.
//
// Limitations of the prototype: fixed-width VF only (no SVE); parameters must
// be vector, uniform or linear with a constant step. Unsupported variants are
// skipped.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/VecClone.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/VFABIDemangler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "vec-clone"

STATISTIC(NumVariantsCreated, "Number of vector variants synthesized");
STATISTIC(NumVariantsSkipped, "Number of unsupported vector variants skipped");

/// Type used to hold one lane in memory. i1 is widened to i8 so that each lane
/// is byte addressable.
static Type *getLaneMemType(Type *Ty) {
  return Ty->isIntegerTy(1) ? Type::getInt8Ty(Ty->getContext()) : Ty;
}

static const char *checkSupported(const Function &F, const VFInfo &Info) {
  if (Info.Shape.VF.isScalable())
    return "scalable VF";
  if (Info.ISA == VFISAKind::SVE)
    return "SVE";
  for (const VFParameter &P : Info.Shape.Parameters) {
    if (P.ParamKind == VFParamKind::GlobalPredicate)
      continue;
    Type *Ty = F.getArg(P.ParamPos)->getType();
    switch (P.ParamKind) {
    case VFParamKind::Vector:
    case VFParamKind::OMP_Uniform:
      break;
    case VFParamKind::OMP_Linear:
    case VFParamKind::OMP_LinearRef:
      if (!Ty->isIntegerTy() && !Ty->isPointerTy())
        return "linear parameter of unsupported type";
      break;
    case VFParamKind::OMP_LinearVal:
    case VFParamKind::OMP_LinearUVal:
      if (!Ty->isIntegerTy())
        return "linear val/uval on a reference";
      break;
    default:
      return "unsupported parameter kind";
    }
  }
  return nullptr;
}

/// Store vector \p V into a new entry-block alloca, one lane per element.
static AllocaInst *spillVector(IRBuilder<> &B, Value *V, unsigned VF,
                               const DataLayout &DL, const Twine &Name) {
  Type *EltTy = getLaneMemType(V->getType()->getScalarType());
  auto *MemVecTy = FixedVectorType::get(EltTy, VF);
  AllocaInst *A = B.CreateAlloca(MemVecTy, DL.getAllocaAddrSpace(), nullptr,
                                 Name);
  if (MemVecTy != V->getType())
    V = B.CreateZExt(V, MemVecTy);
  B.CreateStore(V, A);
  return A;
}

/// Load lane \p Index of the vector stored in \p A, converting it back to
/// \p ScalarTy.
static Value *loadLane(IRBuilder<> &B, AllocaInst *A, Value *Index,
                       Type *ScalarTy, const Twine &Name) {
  Type *EltTy = getLaneMemType(ScalarTy);
  Value *Ptr = B.CreateInBoundsGEP(EltTy, A, Index, Name + ".gep");
  Value *V = B.CreateLoad(EltTy, Ptr, Name);
  if (EltTy != ScalarTy)
    V = B.CreateTrunc(V, ScalarTy);
  return V;
}

static void stripIncompatibleAttrs(Function *Clone) {
  Clone->removeRetAttrs(AttributeFuncs::typeIncompatible(
      Clone->getReturnType(), Clone->getAttributes().getRetAttrs()));
  for (Argument &Arg : Clone->args())
    Arg.removeAttrs(
        AttributeFuncs::typeIncompatible(Arg.getType(), Arg.getAttributes()));
  // Do not propagate the vector variant attributes to the clone.
  SmallVector<StringRef, 8> ToRemove;
  for (const Attribute &A : Clone->getAttributes().getFnAttrs())
    if (A.isStringAttribute() && A.getKindAsString().starts_with("_ZGV"))
      ToRemove.push_back(A.getKindAsString());
  for (StringRef K : ToRemove)
    Clone->removeFnAttr(K);
  Clone->removeFnAttr(VFABI::MappingsAttrName);
}

static Function *createVariant(Function &F, const VFInfo &Info) {
  Module &M = *F.getParent();
  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = M.getDataLayout();
  unsigned VF = Info.Shape.VF.getFixedValue();
  FunctionType *VecFTy = VFABI::createFunctionType(Info, F.getFunctionType());

  Function *Clone = M.getFunction(Info.VectorName);
  if (Clone) {
    // A declaration may already exist (e.g. created by
    // InjectOpenMPVFABIMappings); give it a body.
    if (!Clone->isDeclaration() || Clone->getFunctionType() != VecFTy) {
      LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": " << Info.VectorName
                        << " already exists, skipping\n");
      return nullptr;
    }
  } else {
    Clone = Function::Create(VecFTy, GlobalValue::ExternalLinkage,
                             Info.VectorName, &M);
  }

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Clone);
  BasicBlock *Header = BasicBlock::Create(Ctx, "simd.loop.header", Clone);
  BasicBlock *Latch = BasicBlock::Create(Ctx, "simd.loop.latch", Clone);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "simd.end", Clone);

  IRBuilder<> EB(Entry);
  IRBuilder<> HB(Header);
  Type *I32 = Type::getInt32Ty(Ctx);
  PHINode *Index = HB.CreatePHI(I32, 2, "index");

  // Map each scalar argument to its per-lane value, computed in the header.
  ValueToValueMapTy VMap;
  for (const VFParameter &P : Info.Shape.Parameters) {
    Argument *VArg = Clone->getArg(P.ParamPos);
    if (P.ParamKind == VFParamKind::GlobalPredicate) {
      VArg->setName("mask");
      continue;
    }
    Argument *SArg = F.getArg(P.ParamPos);
    Type *Ty = SArg->getType();
    VArg->setName(SArg->getName());
    Value *Lane = nullptr;
    switch (P.ParamKind) {
    case VFParamKind::OMP_Uniform:
      Lane = VArg;
      break;
    case VFParamKind::Vector: {
      AllocaInst *A = spillVector(EB, VArg, VF, DL, VArg->getName() + ".vec");
      Lane = loadLane(HB, A, Index, Ty, VArg->getName() + ".lane");
      break;
    }
    default: { // Linear kinds with a constant step.
      int64_t Step = P.LinearStepOrPos;
      if (Ty->isPointerTy()) {
        Type *IdxTy = DL.getIndexType(Ty);
        Value *Off = HB.CreateMul(HB.CreateZExt(Index, IdxTy),
                                  ConstantInt::getSigned(IdxTy, Step));
        Lane = HB.CreateGEP(HB.getInt8Ty(), VArg, Off,
                            VArg->getName() + ".linear");
      } else {
        Value *Off = HB.CreateMul(HB.CreateZExtOrTrunc(Index, Ty),
                                  ConstantInt::getSigned(Ty, Step));
        Lane = HB.CreateAdd(VArg, Off, VArg->getName() + ".linear");
      }
      break;
    }
    }
    VMap[SArg] = Lane;
  }

  // Return value storage.
  AllocaInst *RetAlloca = nullptr;
  Type *ScalarRetTy = F.getReturnType();
  if (!ScalarRetTy->isVoidTy()) {
    auto *MemVecTy = FixedVectorType::get(getLaneMemType(ScalarRetTy), VF);
    RetAlloca = EB.CreateAlloca(MemVecTy, DL.getAllocaAddrSpace(), nullptr,
                                "ret.vec");
  }

  // Mask: execute the body only for active lanes.
  Value *MaskCond = nullptr;
  if (Info.isMasked()) {
    Argument *MaskArg = Clone->getArg(Clone->arg_size() - 1);
    AllocaInst *A = spillVector(EB, MaskArg, VF, DL, "mask.vec");
    MaskCond = loadLane(HB, A, Index, Type::getInt1Ty(Ctx), "mask.lane");
  }

  // Clone the scalar body. Its blocks are appended after Exit.
  SmallVector<ReturnInst *, 8> Returns;
  CallingConv::ID CC = Clone->getCallingConv();
  CloneFunctionInto(Clone, &F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  Clone->setCallingConv(CC);
  auto *BodyEntry = cast<BasicBlock>(VMap[&F.getEntryBlock()]);

  // Hoist static allocas of the body out of the loop.
  for (Instruction &I : make_early_inc_range(*BodyEntry))
    if (auto *AI = dyn_cast<AllocaInst>(&I);
        AI && isa<Constant>(AI->getArraySize()))
      AI->moveBefore(*Entry, Entry->end());
  EB.SetInsertPoint(Entry);
  EB.CreateBr(Header);

  if (MaskCond)
    HB.CreateCondBr(MaskCond, BodyEntry, Latch);
  else
    HB.CreateBr(BodyEntry);

  // Each return continues the loop; the returned values are merged in the
  // latch and stored unconditionally to the current lane (masked-off lanes are
  // undefined), which avoids predicated stores after vectorization.
  Latch->moveAfter(&Clone->back());
  Exit->moveAfter(Latch);
  IRBuilder<> LB(Latch);
  PHINode *RetPhi = nullptr;
  if (RetAlloca)
    RetPhi = LB.CreatePHI(ScalarRetTy, Returns.size() + 1, "ret.lane");
  for (ReturnInst *RI : Returns) {
    if (RetPhi)
      RetPhi->addIncoming(RI->getReturnValue(), RI->getParent());
    IRBuilder<>(RI).CreateBr(Latch);
    RI->eraseFromParent();
  }
  if (RetPhi && MaskCond)
    RetPhi->addIncoming(PoisonValue::get(ScalarRetTy), Header);
  if (RetPhi) {
    Type *EltTy = getLaneMemType(ScalarRetTy);
    Value *RV = RetPhi;
    if (EltTy != ScalarRetTy)
      RV = LB.CreateZExt(RV, EltTy);
    LB.CreateStore(RV,
                   LB.CreateInBoundsGEP(EltTy, RetAlloca, Index, "ret.gep"));
  }
  Value *Next = LB.CreateNUWAdd(Index, ConstantInt::get(I32, 1), "index.next");
  Value *Cond = LB.CreateICmpULT(Next, ConstantInt::get(I32, VF), "vl.cond");
  Instruction *BackEdge = LB.CreateCondBr(Cond, Header, Exit);
  Index->addIncoming(ConstantInt::get(I32, 0), Entry);
  Index->addIncoming(Next, Latch);

  Metadata *MDs[] = {
      nullptr,
      MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.vectorize.enable")}),
      MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.vectorize.width"),
                        ConstantAsMetadata::get(ConstantInt::get(I32, VF))})};
  MDNode *LoopID = MDNode::getDistinct(Ctx, MDs);
  LoopID->replaceOperandWith(0, LoopID);
  BackEdge->setMetadata(LLVMContext::MD_loop, LoopID);

  IRBuilder<> XB(Exit);
  if (RetAlloca) {
    Value *V = XB.CreateLoad(RetAlloca->getAllocatedType(), RetAlloca, "ret");
    if (V->getType() != VecFTy->getReturnType())
      V = XB.CreateTrunc(V, VecFTy->getReturnType());
    XB.CreateRet(V);
  } else {
    XB.CreateRetVoid();
  }

  // Attributes, calling convention and linkage.
  stripIncompatibleAttrs(Clone);
  Clone->setLinkage(GlobalValue::ExternalLinkage);
  Clone->setVisibility(F.getVisibility());
  if (Info.ISA == VFISAKind::AdvancedSIMD)
    Clone->setCallingConv(CallingConv::AArch64_VectorCall);

  return Clone;
}

PreservedAnalyses VecClonePass::run(Module &M, ModuleAnalysisManager &) {
  SmallVector<Function *, 8> Worklist;
  for (Function &F : M)
    if (!F.isDeclaration())
      Worklist.push_back(&F);

  SmallVector<GlobalValue *, 8> NewFuncs;
  for (Function *F : Worklist) {
    SmallVector<std::string, 8> Names;
    for (const Attribute &A : F->getAttributes().getFnAttrs())
      if (A.isStringAttribute() && A.getKindAsString().starts_with("_ZGV"))
        Names.push_back(A.getKindAsString().str());

    for (const std::string &Name : Names) {
      std::optional<VFInfo> Info =
          VFABI::tryDemangleForVFABI(Name, F->getFunctionType());
      if (!Info)
        continue;
      if (const char *Why = checkSupported(*F, *Info)) {
        LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": skipping " << Name << ": " << Why
                          << "\n");
        ++NumVariantsSkipped;
        continue;
      }
      if (Function *Clone = createVariant(*F, *Info)) {
        LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": created " << Clone->getName()
                          << "\n");
        NewFuncs.push_back(Clone);
        ++NumVariantsCreated;
      }
    }
  }

  if (NewFuncs.empty())
    return PreservedAnalyses::all();
  // Vector variants are part of the ABI: keep them even if unused here.
  appendToCompilerUsed(M, NewFuncs);
  return PreservedAnalyses::none();
}
