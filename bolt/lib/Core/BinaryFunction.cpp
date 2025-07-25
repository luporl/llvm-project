//===- bolt/Core/BinaryFunction.cpp - Low-level function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinaryFunction class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/DynoStats.h"
#include "bolt/Core/HashUtilities.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/NameResolver.h"
#include "bolt/Utils/NameShortener.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GenericLoopInfoImpl.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include <functional>
#include <limits>
#include <numeric>
#include <stack>
#include <string>

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> EnableBAT;
extern cl::opt<bool> Instrument;
extern cl::opt<bool> StrictMode;
extern cl::opt<bool> UpdateDebugSections;
extern cl::opt<unsigned> Verbosity;

extern bool BinaryAnalysisMode;
extern HeatmapModeKind HeatmapMode;
extern bool processAllFunctions();

static cl::opt<bool> CheckEncoding(
    "check-encoding",
    cl::desc("perform verification of LLVM instruction encoding/decoding. "
             "Every instruction in the input is decoded and re-encoded. "
             "If the resulting bytes do not match the input, a warning message "
             "is printed."),
    cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool> DotToolTipCode(
    "dot-tooltip-code",
    cl::desc("add basic block instructions as tool tips on nodes"), cl::Hidden,
    cl::cat(BoltCategory));

cl::opt<JumpTableSupportLevel>
JumpTables("jump-tables",
  cl::desc("jump tables support (default=basic)"),
  cl::init(JTS_BASIC),
  cl::values(
      clEnumValN(JTS_NONE, "none",
                 "do not optimize functions with jump tables"),
      clEnumValN(JTS_BASIC, "basic",
                 "optimize functions with jump tables"),
      clEnumValN(JTS_MOVE, "move",
                 "move jump tables to a separate section"),
      clEnumValN(JTS_SPLIT, "split",
                 "split jump tables section into hot and cold based on "
                 "function execution frequency"),
      clEnumValN(JTS_AGGRESSIVE, "aggressive",
                 "aggressively split jump tables section based on usage "
                 "of the tables")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool> NoScan(
    "no-scan",
    cl::desc(
        "do not scan cold functions for external references (may result in "
        "slower binary)"),
    cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool>
    PreserveBlocksAlignment("preserve-blocks-alignment",
                            cl::desc("try to preserve basic block alignment"),
                            cl::cat(BoltOptCategory));

static cl::opt<bool> PrintOutputAddressRange(
    "print-output-address-range",
    cl::desc(
        "print output address range for each basic block in the function when"
        "BinaryFunction::print is called"),
    cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool>
PrintDynoStats("dyno-stats",
  cl::desc("print execution info based on profile"),
  cl::cat(BoltCategory));

static cl::opt<bool>
PrintDynoStatsOnly("print-dyno-stats-only",
  cl::desc("while printing functions output dyno-stats and skip instructions"),
  cl::init(false),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string>
PrintOnly("print-only",
  cl::CommaSeparated,
  cl::desc("list of functions to print"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<bool>
    TimeBuild("time-build",
              cl::desc("print time spent constructing binary functions"),
              cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool> TrapOnAVX512(
    "trap-avx512",
    cl::desc("in relocation mode trap upon entry to any function that uses "
             "AVX-512 instructions"),
    cl::init(false), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltCategory));

bool shouldPrint(const BinaryFunction &Function) {
  if (Function.isIgnored())
    return false;

  if (PrintOnly.empty())
    return true;

  for (std::string &Name : opts::PrintOnly) {
    if (Function.hasNameRegex(Name)) {
      return true;
    }
  }

  std::optional<StringRef> Origin = Function.getOriginSectionName();
  return Origin && llvm::is_contained(opts::PrintOnly, *Origin);
}

} // namespace opts

namespace llvm {
namespace bolt {

template <typename R> static bool emptyRange(const R &Range) {
  return Range.begin() == Range.end();
}

/// Gets debug line information for the instruction located at the given
/// address in the original binary. The SMLoc's pointer is used
/// to point to this information, which is represented by a
/// DebugLineTableRowRef. The returned pointer is null if no debug line
/// information for this instruction was found.
static SMLoc findDebugLineInformationForInstructionAt(
    uint64_t Address, DWARFUnit *Unit,
    const DWARFDebugLine::LineTable *LineTable) {
  // We use the pointer in SMLoc to store an instance of DebugLineTableRowRef,
  // which occupies 64 bits. Thus, we can only proceed if the struct fits into
  // the pointer itself.
  static_assert(
      sizeof(decltype(SMLoc().getPointer())) >= sizeof(DebugLineTableRowRef),
      "Cannot fit instruction debug line information into SMLoc's pointer");

  SMLoc NullResult = DebugLineTableRowRef::NULL_ROW.toSMLoc();
  uint32_t RowIndex = LineTable->lookupAddress(
      {Address, object::SectionedAddress::UndefSection});
  if (RowIndex == LineTable->UnknownRowIndex)
    return NullResult;

  assert(RowIndex < LineTable->Rows.size() &&
         "Line Table lookup returned invalid index.");

  decltype(SMLoc().getPointer()) Ptr;
  DebugLineTableRowRef *InstructionLocation =
      reinterpret_cast<DebugLineTableRowRef *>(&Ptr);

  InstructionLocation->DwCompileUnitIndex = Unit->getOffset();
  InstructionLocation->RowIndex = RowIndex + 1;

  return SMLoc::getFromPointer(Ptr);
}

static std::string buildSectionName(StringRef Prefix, StringRef Name,
                                    const BinaryContext &BC) {
  if (BC.isELF())
    return (Prefix + Name).str();
  static NameShortener NS;
  return (Prefix + Twine(NS.getID(Name))).str();
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const BinaryFunction::State State) {
  switch (State) {
  case BinaryFunction::State::Empty:         OS << "empty"; break;
  case BinaryFunction::State::Disassembled:  OS << "disassembled"; break;
  case BinaryFunction::State::CFG:           OS << "CFG constructed"; break;
  case BinaryFunction::State::CFG_Finalized: OS << "CFG finalized"; break;
  case BinaryFunction::State::EmittedCFG:    OS << "emitted with CFG"; break;
  case BinaryFunction::State::Emitted:       OS << "emitted"; break;
  }

  return OS;
}

std::string BinaryFunction::buildCodeSectionName(StringRef Name,
                                                 const BinaryContext &BC) {
  return buildSectionName(BC.isELF() ? ".local.text." : ".l.text.", Name, BC);
}

std::string BinaryFunction::buildColdCodeSectionName(StringRef Name,
                                                     const BinaryContext &BC) {
  return buildSectionName(BC.isELF() ? ".local.cold.text." : ".l.c.text.", Name,
                          BC);
}

uint64_t BinaryFunction::Count = 0;

std::optional<StringRef>
BinaryFunction::hasNameRegex(const StringRef Name) const {
  const std::string RegexName = (Twine("^") + StringRef(Name) + "$").str();
  Regex MatchName(RegexName);
  return forEachName(
      [&MatchName](StringRef Name) { return MatchName.match(Name); });
}

std::optional<StringRef>
BinaryFunction::hasRestoredNameRegex(const StringRef Name) const {
  const std::string RegexName = (Twine("^") + StringRef(Name) + "$").str();
  Regex MatchName(RegexName);
  return forEachName([&MatchName](StringRef Name) {
    return MatchName.match(NameResolver::restore(Name));
  });
}

std::string BinaryFunction::getDemangledName() const {
  StringRef MangledName = NameResolver::restore(getOneName());
  return demangle(MangledName.str());
}

BinaryBasicBlock *
BinaryFunction::getBasicBlockContainingOffset(uint64_t Offset) {
  if (Offset > Size)
    return nullptr;

  if (BasicBlockOffsets.empty())
    return nullptr;

  /*
   * This is commented out because it makes BOLT too slow.
   * assert(std::is_sorted(BasicBlockOffsets.begin(),
   *                       BasicBlockOffsets.end(),
   *                       CompareBasicBlockOffsets())));
   */
  auto I =
      llvm::upper_bound(BasicBlockOffsets, BasicBlockOffset(Offset, nullptr),
                        CompareBasicBlockOffsets());
  assert(I != BasicBlockOffsets.begin() && "first basic block not at offset 0");
  --I;
  BinaryBasicBlock *BB = I->second;
  return (Offset < BB->getOffset() + BB->getOriginalSize()) ? BB : nullptr;
}

void BinaryFunction::markUnreachableBlocks() {
  std::stack<BinaryBasicBlock *> Stack;

  for (BinaryBasicBlock &BB : blocks())
    BB.markValid(false);

  // Add all entries and landing pads as roots.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    if (isEntryPoint(*BB) || BB->isLandingPad()) {
      Stack.push(BB);
      BB->markValid(true);
      continue;
    }
    // FIXME:
    // Also mark BBs with indirect jumps as reachable, since we do not
    // support removing unused jump tables yet (GH-issue20).
    for (const MCInst &Inst : *BB) {
      if (BC.MIB->getJumpTable(Inst)) {
        Stack.push(BB);
        BB->markValid(true);
        break;
      }
    }
  }

  // Determine reachable BBs from the entry point
  while (!Stack.empty()) {
    BinaryBasicBlock *BB = Stack.top();
    Stack.pop();
    for (BinaryBasicBlock *Succ : BB->successors()) {
      if (Succ->isValid())
        continue;
      Succ->markValid(true);
      Stack.push(Succ);
    }
  }
}

// Any unnecessary fallthrough jumps revealed after calling eraseInvalidBBs
// will be cleaned up by fixBranches().
std::pair<unsigned, uint64_t>
BinaryFunction::eraseInvalidBBs(const MCCodeEmitter *Emitter) {
  DenseSet<const BinaryBasicBlock *> InvalidBBs;
  unsigned Count = 0;
  uint64_t Bytes = 0;
  for (BinaryBasicBlock *const BB : BasicBlocks) {
    if (!BB->isValid()) {
      assert(!isEntryPoint(*BB) && "all entry blocks must be valid");
      InvalidBBs.insert(BB);
      ++Count;
      Bytes += BC.computeCodeSize(BB->begin(), BB->end(), Emitter);
    }
  }

  Layout.eraseBasicBlocks(InvalidBBs);

  BasicBlockListType NewBasicBlocks;
  for (auto I = BasicBlocks.begin(), E = BasicBlocks.end(); I != E; ++I) {
    BinaryBasicBlock *BB = *I;
    if (InvalidBBs.contains(BB)) {
      // Make sure the block is removed from the list of predecessors.
      BB->removeAllSuccessors();
      DeletedBasicBlocks.push_back(BB);
    } else {
      NewBasicBlocks.push_back(BB);
    }
  }
  BasicBlocks = std::move(NewBasicBlocks);

  assert(BasicBlocks.size() == Layout.block_size());

  // Update CFG state if needed
  if (Count > 0)
    recomputeLandingPads();

  return std::make_pair(Count, Bytes);
}

bool BinaryFunction::isForwardCall(const MCSymbol *CalleeSymbol) const {
  // This function should work properly before and after function reordering.
  // In order to accomplish this, we use the function index (if it is valid).
  // If the function indices are not valid, we fall back to the original
  // addresses.  This should be ok because the functions without valid indices
  // should have been ordered with a stable sort.
  const BinaryFunction *CalleeBF = BC.getFunctionForSymbol(CalleeSymbol);
  if (CalleeBF) {
    if (CalleeBF->isInjected())
      return true;
    return compareBinaryFunctionByIndex(this, CalleeBF);
  } else {
    // Absolute symbol.
    ErrorOr<uint64_t> CalleeAddressOrError = BC.getSymbolValue(*CalleeSymbol);
    assert(CalleeAddressOrError && "unregistered symbol found");
    return *CalleeAddressOrError > getAddress();
  }
}

void BinaryFunction::dump() const {
  // getDynoStats calls FunctionLayout::updateLayoutIndices and
  // BasicBlock::analyzeBranch. The former cannot be const, but should be
  // removed, the latter should be made const, but seems to require refactoring.
  // Forcing all callers to have a non-const reference to BinaryFunction to call
  // dump non-const however is not ideal either. Adding this const_cast is right
  // now the best solution. It is safe, because BinaryFunction itself is not
  // modified. Only BinaryBasicBlocks are actually modified (if it all) and we
  // have mutable pointers to those regardless whether this function is
  // const-qualified or not.
  const_cast<BinaryFunction &>(*this).print(dbgs(), "");
}

void BinaryFunction::print(raw_ostream &OS, std::string Annotation) {
  if (!opts::shouldPrint(*this))
    return;

  StringRef SectionName =
      OriginSection ? OriginSection->getName() : "<no origin section>";
  OS << "Binary Function \"" << *this << "\" " << Annotation << " {";
  std::vector<StringRef> AllNames = getNames();
  if (AllNames.size() > 1) {
    OS << "\n  All names   : ";
    const char *Sep = "";
    for (const StringRef &Name : AllNames) {
      OS << Sep << Name;
      Sep = "\n                ";
    }
  }
  OS << "\n  Number      : " << FunctionNumber;
  OS << "\n  State       : " << CurrentState;
  OS << "\n  Address     : 0x" << Twine::utohexstr(Address);
  OS << "\n  Size        : 0x" << Twine::utohexstr(Size);
  OS << "\n  MaxSize     : 0x" << Twine::utohexstr(MaxSize);
  OS << "\n  Offset      : 0x" << Twine::utohexstr(getFileOffset());
  OS << "\n  Section     : " << SectionName;
  OS << "\n  Orc Section : " << getCodeSectionName();
  OS << "\n  LSDA        : 0x" << Twine::utohexstr(getLSDAAddress());
  OS << "\n  IsSimple    : " << IsSimple;
  OS << "\n  IsMultiEntry: " << isMultiEntry();
  OS << "\n  IsSplit     : " << isSplit();
  OS << "\n  BB Count    : " << size();

  if (HasUnknownControlFlow)
    OS << "\n  Unknown CF  : true";
  if (getPersonalityFunction())
    OS << "\n  Personality : " << getPersonalityFunction()->getName();
  if (IsFragment)
    OS << "\n  IsFragment  : true";
  if (isFolded())
    OS << "\n  FoldedInto  : " << *getFoldedIntoFunction();
  for (BinaryFunction *ParentFragment : ParentFragments)
    OS << "\n  Parent      : " << *ParentFragment;
  if (!Fragments.empty()) {
    OS << "\n  Fragments   : ";
    ListSeparator LS;
    for (BinaryFunction *Frag : Fragments)
      OS << LS << *Frag;
  }
  if (hasCFG())
    OS << "\n  Hash        : " << Twine::utohexstr(computeHash());
  if (isMultiEntry()) {
    OS << "\n  Secondary Entry Points : ";
    ListSeparator LS;
    for (const auto &KV : SecondaryEntryPoints)
      OS << LS << KV.second->getName();
  }
  if (FrameInstructions.size())
    OS << "\n  CFI Instrs  : " << FrameInstructions.size();
  if (!Layout.block_empty()) {
    OS << "\n  BB Layout   : ";
    ListSeparator LS;
    for (const BinaryBasicBlock *BB : Layout.blocks())
      OS << LS << BB->getName();
  }
  if (getImageAddress())
    OS << "\n  Image       : 0x" << Twine::utohexstr(getImageAddress());
  if (ExecutionCount != COUNT_NO_PROFILE) {
    OS << "\n  Exec Count  : " << ExecutionCount;
    OS << "\n  Sample Count: " << RawSampleCount;
    OS << "\n  Profile Acc : " << format("%.1f%%", ProfileMatchRatio * 100.0f);
  }
  if (ExternEntryCount)
    OS << "\n  Extern Entry Count: " << ExternEntryCount;

  if (opts::PrintDynoStats && !getLayout().block_empty()) {
    OS << '\n';
    DynoStats dynoStats = getDynoStats(*this);
    OS << dynoStats;
  }

  OS << "\n}\n";

  if (opts::PrintDynoStatsOnly || !BC.InstPrinter)
    return;

  // Offset of the instruction in function.
  uint64_t Offset = 0;

  auto printConstantIslandInRange = [&](uint64_t Start, uint64_t End) {
    assert(Start <= End && "Invalid range");
    std::optional<uint64_t> IslandOffset = getIslandInRange(Start, End);

    if (!IslandOffset)
      return;

    // Print label if it exists at this offset.
    if (const BinaryData *BD =
            BC.getBinaryDataAtAddress(getAddress() + *IslandOffset))
      OS << BD->getName() << ":\n";

    const size_t IslandSize = getSizeOfDataInCodeAt(*IslandOffset);
    BC.printData(OS, BC.extractData(getAddress() + *IslandOffset, IslandSize),
                 *IslandOffset);
  };

  if (BasicBlocks.empty() && !Instructions.empty()) {
    // Print before CFG was built.
    uint64_t PrevOffset = 0;
    for (const std::pair<const uint32_t, MCInst> &II : Instructions) {
      Offset = II.first;

      // Print any constant islands inbeetween the instructions.
      printConstantIslandInRange(PrevOffset, Offset);

      // Print label if exists at this offset.
      auto LI = Labels.find(Offset);
      if (LI != Labels.end()) {
        if (const MCSymbol *EntrySymbol =
                getSecondaryEntryPointSymbol(LI->second))
          OS << EntrySymbol->getName() << " (Entry Point):\n";
        OS << LI->second->getName() << ":\n";
      }

      BC.printInstruction(OS, II.second, Offset, this);

      PrevOffset = Offset;
    }

    // Print any data at the end of the function.
    printConstantIslandInRange(PrevOffset, getMaxSize());
  }

  StringRef SplitPointMsg = "";
  for (const FunctionFragment &FF : Layout.fragments()) {
    OS << SplitPointMsg;
    SplitPointMsg = "-------   HOT-COLD SPLIT POINT   -------\n\n";
    for (const BinaryBasicBlock *BB : FF) {
      OS << BB->getName() << " (" << BB->size()
         << " instructions, align : " << BB->getAlignment() << ")\n";

      if (opts::PrintOutputAddressRange)
        OS << formatv("  Output Address Range: [{0:x}, {1:x}) ({2} bytes)\n",
                      BB->getOutputAddressRange().first,
                      BB->getOutputAddressRange().second, BB->getOutputSize());

      if (isEntryPoint(*BB)) {
        if (MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(*BB))
          OS << "  Secondary Entry Point: " << EntrySymbol->getName() << '\n';
        else
          OS << "  Entry Point\n";
      }

      if (BB->isLandingPad())
        OS << "  Landing Pad\n";

      uint64_t BBExecCount = BB->getExecutionCount();
      if (hasValidProfile()) {
        OS << "  Exec Count : ";
        if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE)
          OS << BBExecCount << '\n';
        else
          OS << "<unknown>\n";
      }
      if (hasCFI())
        OS << "  CFI State : " << BB->getCFIState() << '\n';
      if (opts::EnableBAT) {
        OS << "  Input offset: 0x" << Twine::utohexstr(BB->getInputOffset())
           << "\n";
      }
      if (!BB->pred_empty()) {
        OS << "  Predecessors: ";
        ListSeparator LS;
        for (BinaryBasicBlock *Pred : BB->predecessors())
          OS << LS << Pred->getName();
        OS << '\n';
      }
      if (!BB->throw_empty()) {
        OS << "  Throwers: ";
        ListSeparator LS;
        for (BinaryBasicBlock *Throw : BB->throwers())
          OS << LS << Throw->getName();
        OS << '\n';
      }

      Offset = alignTo(Offset, BB->getAlignment());

      // Note: offsets are imprecise since this is happening prior to
      // relaxation.
      Offset = BC.printInstructions(OS, BB->begin(), BB->end(), Offset, this);

      if (!BB->succ_empty()) {
        OS << "  Successors: ";
        // For more than 2 successors, sort them based on frequency.
        std::vector<uint64_t> Indices(BB->succ_size());
        std::iota(Indices.begin(), Indices.end(), 0);
        if (BB->succ_size() > 2 && BB->getKnownExecutionCount()) {
          llvm::stable_sort(Indices, [&](const uint64_t A, const uint64_t B) {
            return BB->BranchInfo[B] < BB->BranchInfo[A];
          });
        }
        ListSeparator LS;
        for (unsigned I = 0; I < Indices.size(); ++I) {
          BinaryBasicBlock *Succ = BB->Successors[Indices[I]];
          const BinaryBasicBlock::BinaryBranchInfo &BI =
              BB->BranchInfo[Indices[I]];
          OS << LS << Succ->getName();
          if (ExecutionCount != COUNT_NO_PROFILE &&
              BI.MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
            OS << " (mispreds: " << BI.MispredictedCount
               << ", count: " << BI.Count << ")";
          } else if (ExecutionCount != COUNT_NO_PROFILE &&
                     BI.Count != BinaryBasicBlock::COUNT_NO_PROFILE) {
            OS << " (inferred count: " << BI.Count << ")";
          }
        }
        OS << '\n';
      }

      if (!BB->lp_empty()) {
        OS << "  Landing Pads: ";
        ListSeparator LS;
        for (BinaryBasicBlock *LP : BB->landing_pads()) {
          OS << LS << LP->getName();
          if (ExecutionCount != COUNT_NO_PROFILE) {
            OS << " (count: " << LP->getExecutionCount() << ")";
          }
        }
        OS << '\n';
      }

      // In CFG_Finalized state we can miscalculate CFI state at exit.
      if (CurrentState == State::CFG && hasCFI()) {
        const int32_t CFIStateAtExit = BB->getCFIStateAtExit();
        if (CFIStateAtExit >= 0)
          OS << "  CFI State: " << CFIStateAtExit << '\n';
      }

      OS << '\n';
    }
  }

  // Dump new exception ranges for the function.
  if (!CallSites.empty()) {
    OS << "EH table:\n";
    for (const FunctionFragment &FF : getLayout().fragments()) {
      for (const auto &FCSI : getCallSites(FF.getFragmentNum())) {
        const CallSite &CSI = FCSI.second;
        OS << "  [" << *CSI.Start << ", " << *CSI.End << ") landing pad : ";
        if (CSI.LP)
          OS << *CSI.LP;
        else
          OS << "0";
        OS << ", action : " << CSI.Action << '\n';
      }
    }
    OS << '\n';
  }

  // Print all jump tables.
  for (const std::pair<const uint64_t, JumpTable *> &JTI : JumpTables)
    JTI.second->print(OS);

  OS << "DWARF CFI Instructions:\n";
  if (OffsetToCFI.size()) {
    // Pre-buildCFG information
    for (const std::pair<const uint32_t, uint32_t> &Elmt : OffsetToCFI) {
      OS << format("    %08x:\t", Elmt.first);
      assert(Elmt.second < FrameInstructions.size() && "Incorrect CFI offset");
      BinaryContext::printCFI(OS, FrameInstructions[Elmt.second]);
      OS << "\n";
    }
  } else {
    // Post-buildCFG information
    for (uint32_t I = 0, E = FrameInstructions.size(); I != E; ++I) {
      const MCCFIInstruction &CFI = FrameInstructions[I];
      OS << format("    %d:\t", I);
      BinaryContext::printCFI(OS, CFI);
      OS << "\n";
    }
  }
  if (FrameInstructions.empty())
    OS << "    <empty>\n";

  OS << "End of Function \"" << *this << "\"\n\n";
}

void BinaryFunction::printRelocations(raw_ostream &OS, uint64_t Offset,
                                      uint64_t Size) const {
  const char *Sep = " # Relocs: ";

  auto RI = Relocations.lower_bound(Offset);
  while (RI != Relocations.end() && RI->first < Offset + Size) {
    OS << Sep << "(R: " << RI->second << ")";
    Sep = ", ";
    ++RI;
  }
}

static std::string mutateDWARFExpressionTargetReg(const MCCFIInstruction &Instr,
                                                  MCPhysReg NewReg) {
  StringRef ExprBytes = Instr.getValues();
  assert(ExprBytes.size() > 1 && "DWARF expression CFI is too short");
  uint8_t Opcode = ExprBytes[0];
  assert((Opcode == dwarf::DW_CFA_expression ||
          Opcode == dwarf::DW_CFA_val_expression) &&
         "invalid DWARF expression CFI");
  (void)Opcode;
  const uint8_t *const Start =
      reinterpret_cast<const uint8_t *>(ExprBytes.drop_front(1).data());
  const uint8_t *const End =
      reinterpret_cast<const uint8_t *>(Start + ExprBytes.size() - 1);
  unsigned Size = 0;
  decodeULEB128(Start, &Size, End);
  assert(Size > 0 && "Invalid reg encoding for DWARF expression CFI");
  SmallString<8> Tmp;
  raw_svector_ostream OSE(Tmp);
  encodeULEB128(NewReg, OSE);
  return Twine(ExprBytes.slice(0, 1))
      .concat(OSE.str())
      .concat(ExprBytes.drop_front(1 + Size))
      .str();
}

void BinaryFunction::mutateCFIRegisterFor(const MCInst &Instr,
                                          MCPhysReg NewReg) {
  const MCCFIInstruction *OldCFI = getCFIFor(Instr);
  assert(OldCFI && "invalid CFI instr");
  switch (OldCFI->getOperation()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case MCCFIInstruction::OpDefCfa:
    setCFIFor(Instr, MCCFIInstruction::cfiDefCfa(nullptr, NewReg,
                                                 OldCFI->getOffset()));
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    setCFIFor(Instr, MCCFIInstruction::createDefCfaRegister(nullptr, NewReg));
    break;
  case MCCFIInstruction::OpOffset:
    setCFIFor(Instr, MCCFIInstruction::createOffset(nullptr, NewReg,
                                                    OldCFI->getOffset()));
    break;
  case MCCFIInstruction::OpRegister:
    setCFIFor(Instr, MCCFIInstruction::createRegister(nullptr, NewReg,
                                                      OldCFI->getRegister2()));
    break;
  case MCCFIInstruction::OpSameValue:
    setCFIFor(Instr, MCCFIInstruction::createSameValue(nullptr, NewReg));
    break;
  case MCCFIInstruction::OpEscape:
    setCFIFor(Instr,
              MCCFIInstruction::createEscape(
                  nullptr,
                  StringRef(mutateDWARFExpressionTargetReg(*OldCFI, NewReg))));
    break;
  case MCCFIInstruction::OpRestore:
    setCFIFor(Instr, MCCFIInstruction::createRestore(nullptr, NewReg));
    break;
  case MCCFIInstruction::OpUndefined:
    setCFIFor(Instr, MCCFIInstruction::createUndefined(nullptr, NewReg));
    break;
  }
}

const MCCFIInstruction *BinaryFunction::mutateCFIOffsetFor(const MCInst &Instr,
                                                           int64_t NewOffset) {
  const MCCFIInstruction *OldCFI = getCFIFor(Instr);
  assert(OldCFI && "invalid CFI instr");
  switch (OldCFI->getOperation()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case MCCFIInstruction::OpDefCfaOffset:
    setCFIFor(Instr, MCCFIInstruction::cfiDefCfaOffset(nullptr, NewOffset));
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    setCFIFor(Instr,
              MCCFIInstruction::createAdjustCfaOffset(nullptr, NewOffset));
    break;
  case MCCFIInstruction::OpDefCfa:
    setCFIFor(Instr, MCCFIInstruction::cfiDefCfa(nullptr, OldCFI->getRegister(),
                                                 NewOffset));
    break;
  case MCCFIInstruction::OpOffset:
    setCFIFor(Instr, MCCFIInstruction::createOffset(
                         nullptr, OldCFI->getRegister(), NewOffset));
    break;
  }
  return getCFIFor(Instr);
}

IndirectBranchType
BinaryFunction::processIndirectBranch(MCInst &Instruction, unsigned Size,
                                      uint64_t Offset,
                                      uint64_t &TargetAddress) {
  const unsigned PtrSize = BC.AsmInfo->getCodePointerSize();

  // The instruction referencing memory used by the branch instruction.
  // It could be the branch instruction itself or one of the instructions
  // setting the value of the register used by the branch.
  MCInst *MemLocInstr;

  // The instruction loading the fixed PIC jump table entry value.
  MCInst *FixedEntryLoadInstr;

  // Address of the table referenced by MemLocInstr. Could be either an
  // array of function pointers, or a jump table.
  uint64_t ArrayStart = 0;

  unsigned BaseRegNum, IndexRegNum;
  int64_t DispValue;
  const MCExpr *DispExpr;

  // In AArch, identify the instruction adding the PC-relative offset to
  // jump table entries to correctly decode it.
  MCInst *PCRelBaseInstr;
  uint64_t PCRelAddr = 0;

  auto Begin = Instructions.begin();
  if (BC.isAArch64()) {
    // Start at the last label as an approximation of the current basic block.
    // This is a heuristic, since the full set of labels have yet to be
    // determined
    for (const uint32_t Offset :
         llvm::make_first_range(llvm::reverse(Labels))) {
      auto II = Instructions.find(Offset);
      if (II != Instructions.end()) {
        Begin = II;
        break;
      }
    }
  }

  IndirectBranchType BranchType = BC.MIB->analyzeIndirectBranch(
      Instruction, Begin, Instructions.end(), PtrSize, MemLocInstr, BaseRegNum,
      IndexRegNum, DispValue, DispExpr, PCRelBaseInstr, FixedEntryLoadInstr);

  if (BranchType == IndirectBranchType::UNKNOWN && !MemLocInstr)
    return BranchType;

  if (MemLocInstr != &Instruction)
    IndexRegNum = BC.MIB->getNoRegister();

  if (BC.isAArch64()) {
    const MCSymbol *Sym = BC.MIB->getTargetSymbol(*PCRelBaseInstr, 1);
    assert(Sym && "Symbol extraction failed");
    ErrorOr<uint64_t> SymValueOrError = BC.getSymbolValue(*Sym);
    if (SymValueOrError) {
      PCRelAddr = *SymValueOrError;
    } else {
      for (std::pair<const uint32_t, MCSymbol *> &Elmt : Labels) {
        if (Elmt.second == Sym) {
          PCRelAddr = Elmt.first + getAddress();
          break;
        }
      }
    }
    uint64_t InstrAddr = 0;
    for (auto II = Instructions.rbegin(); II != Instructions.rend(); ++II) {
      if (&II->second == PCRelBaseInstr) {
        InstrAddr = II->first + getAddress();
        break;
      }
    }
    assert(InstrAddr != 0 && "instruction not found");
    // We do this to avoid spurious references to code locations outside this
    // function (for example, if the indirect jump lives in the last basic
    // block of the function, it will create a reference to the next function).
    // This replaces a symbol reference with an immediate.
    BC.MIB->replaceMemOperandDisp(*PCRelBaseInstr,
                                  MCOperand::createImm(PCRelAddr - InstrAddr));
    // FIXME: Disable full jump table processing for AArch64 until we have a
    // proper way of determining the jump table limits.
    return IndirectBranchType::UNKNOWN;
  }

  auto getExprValue = [&](const MCExpr *Expr) {
    const MCSymbol *TargetSym;
    uint64_t TargetOffset;
    std::tie(TargetSym, TargetOffset) = BC.MIB->getTargetSymbolInfo(Expr);
    ErrorOr<uint64_t> SymValueOrError = BC.getSymbolValue(*TargetSym);
    assert(SymValueOrError && "Global symbol needs a value");
    return *SymValueOrError + TargetOffset;
  };

  // RIP-relative addressing should be converted to symbol form by now
  // in processed instructions (but not in jump).
  if (DispExpr) {
    ArrayStart = getExprValue(DispExpr);
    BaseRegNum = BC.MIB->getNoRegister();
    if (BC.isAArch64()) {
      ArrayStart &= ~0xFFFULL;
      ArrayStart += DispValue & 0xFFFULL;
    }
  } else {
    ArrayStart = static_cast<uint64_t>(DispValue);
  }

  if (BaseRegNum == BC.MRI->getProgramCounter())
    ArrayStart += getAddress() + Offset + Size;

  if (FixedEntryLoadInstr) {
    assert(BranchType == IndirectBranchType::POSSIBLE_PIC_FIXED_BRANCH &&
           "Invalid IndirectBranch type");
    MCInst::iterator FixedEntryDispOperand =
        BC.MIB->getMemOperandDisp(*FixedEntryLoadInstr);
    assert(FixedEntryDispOperand != FixedEntryLoadInstr->end() &&
           "Invalid memory instruction");
    const MCExpr *FixedEntryDispExpr = FixedEntryDispOperand->getExpr();
    const uint64_t EntryAddress = getExprValue(FixedEntryDispExpr);
    uint64_t EntrySize = BC.getJumpTableEntrySize(JumpTable::JTT_PIC);
    ErrorOr<int64_t> Value =
        BC.getSignedValueAtAddress(EntryAddress, EntrySize);
    if (!Value)
      return IndirectBranchType::UNKNOWN;

    BC.outs() << "BOLT-INFO: fixed PIC indirect branch detected in " << *this
              << " at 0x" << Twine::utohexstr(getAddress() + Offset)
              << " referencing data at 0x" << Twine::utohexstr(EntryAddress)
              << " the destination value is 0x"
              << Twine::utohexstr(ArrayStart + *Value) << '\n';

    TargetAddress = ArrayStart + *Value;

    // Remove spurious JumpTable at EntryAddress caused by PIC reference from
    // the load instruction.
    BC.deleteJumpTable(EntryAddress);

    // Replace FixedEntryDispExpr used in target address calculation with outer
    // jump table reference.
    JumpTable *JT = BC.getJumpTableContainingAddress(ArrayStart);
    assert(JT && "Must have a containing jump table for PIC fixed branch");
    BC.MIB->replaceMemOperandDisp(*FixedEntryLoadInstr, JT->getFirstLabel(),
                                  EntryAddress - ArrayStart, &*BC.Ctx);

    return BranchType;
  }

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: addressed memory is 0x"
                    << Twine::utohexstr(ArrayStart) << '\n');

  ErrorOr<BinarySection &> Section = BC.getSectionForAddress(ArrayStart);
  if (!Section) {
    // No section - possibly an absolute address. Since we don't allow
    // internal function addresses to escape the function scope - we
    // consider it a tail call.
    if (opts::Verbosity >= 1) {
      BC.errs() << "BOLT-WARNING: no section for address 0x"
                << Twine::utohexstr(ArrayStart) << " referenced from function "
                << *this << '\n';
    }
    return IndirectBranchType::POSSIBLE_TAIL_CALL;
  }
  if (Section->isVirtual()) {
    // The contents are filled at runtime.
    return IndirectBranchType::POSSIBLE_TAIL_CALL;
  }

  if (BranchType == IndirectBranchType::POSSIBLE_FIXED_BRANCH) {
    ErrorOr<uint64_t> Value = BC.getPointerAtAddress(ArrayStart);
    if (!Value)
      return IndirectBranchType::UNKNOWN;

    if (BC.getSectionForAddress(ArrayStart)->isWritable())
      return IndirectBranchType::UNKNOWN;

    BC.outs() << "BOLT-INFO: fixed indirect branch detected in " << *this
              << " at 0x" << Twine::utohexstr(getAddress() + Offset)
              << " referencing data at 0x" << Twine::utohexstr(ArrayStart)
              << " the destination value is 0x" << Twine::utohexstr(*Value)
              << '\n';

    TargetAddress = *Value;
    return BranchType;
  }

  // Check if there's already a jump table registered at this address.
  MemoryContentsType MemType;
  if (JumpTable *JT = BC.getJumpTableContainingAddress(ArrayStart)) {
    switch (JT->Type) {
    case JumpTable::JTT_NORMAL:
      MemType = MemoryContentsType::POSSIBLE_JUMP_TABLE;
      break;
    case JumpTable::JTT_PIC:
      MemType = MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE;
      break;
    }
  } else {
    MemType = BC.analyzeMemoryAt(ArrayStart, *this);
  }

  // Check that jump table type in instruction pattern matches memory contents.
  JumpTable::JumpTableType JTType;
  if (BranchType == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE) {
    if (MemType != MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE)
      return IndirectBranchType::UNKNOWN;
    JTType = JumpTable::JTT_PIC;
  } else {
    if (MemType == MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE)
      return IndirectBranchType::UNKNOWN;

    if (MemType == MemoryContentsType::UNKNOWN)
      return IndirectBranchType::POSSIBLE_TAIL_CALL;

    BranchType = IndirectBranchType::POSSIBLE_JUMP_TABLE;
    JTType = JumpTable::JTT_NORMAL;
  }

  // Convert the instruction into jump table branch.
  const MCSymbol *JTLabel = BC.getOrCreateJumpTable(*this, ArrayStart, JTType);
  BC.MIB->replaceMemOperandDisp(*MemLocInstr, JTLabel, BC.Ctx.get());
  BC.MIB->setJumpTable(Instruction, ArrayStart, IndexRegNum);

  JTSites.emplace_back(Offset, ArrayStart);

  return BranchType;
}

MCSymbol *BinaryFunction::getOrCreateLocalLabel(uint64_t Address,
                                                bool CreatePastEnd) {
  const uint64_t Offset = Address - getAddress();

  if ((Offset == getSize()) && CreatePastEnd)
    return getFunctionEndLabel();

  auto LI = Labels.find(Offset);
  if (LI != Labels.end())
    return LI->second;

  // For AArch64, check if this address is part of a constant island.
  if (BC.isAArch64()) {
    if (MCSymbol *IslandSym = getOrCreateIslandAccess(Address))
      return IslandSym;
  }

  MCSymbol *Label = BC.Ctx->createNamedTempSymbol();
  Labels[Offset] = Label;

  return Label;
}

ErrorOr<ArrayRef<uint8_t>> BinaryFunction::getData() const {
  BinarySection &Section = *getOriginSection();
  assert(Section.containsRange(getAddress(), getMaxSize()) &&
         "wrong section for function");

  if (!Section.isText() || Section.isVirtual() || !Section.getSize())
    return std::make_error_code(std::errc::bad_address);

  StringRef SectionContents = Section.getContents();

  assert(SectionContents.size() == Section.getSize() &&
         "section size mismatch");

  // Function offset from the section start.
  uint64_t Offset = getAddress() - Section.getAddress();
  auto *Bytes = reinterpret_cast<const uint8_t *>(SectionContents.data());
  return ArrayRef<uint8_t>(Bytes + Offset, getMaxSize());
}

size_t BinaryFunction::getSizeOfDataInCodeAt(uint64_t Offset) const {
  if (!Islands)
    return 0;

  if (!llvm::is_contained(Islands->DataOffsets, Offset))
    return 0;

  auto Iter = Islands->CodeOffsets.upper_bound(Offset);
  if (Iter != Islands->CodeOffsets.end())
    return *Iter - Offset;
  return getMaxSize() - Offset;
}

std::optional<uint64_t>
BinaryFunction::getIslandInRange(uint64_t StartOffset,
                                 uint64_t EndOffset) const {
  if (!Islands)
    return std::nullopt;

  auto Iter = llvm::lower_bound(Islands->DataOffsets, StartOffset);
  if (Iter != Islands->DataOffsets.end() && *Iter < EndOffset)
    return *Iter;

  return std::nullopt;
}

bool BinaryFunction::isZeroPaddingAt(uint64_t Offset) const {
  ArrayRef<uint8_t> FunctionData = *getData();
  uint64_t EndOfCode = getSize();
  if (Islands) {
    auto Iter = Islands->DataOffsets.upper_bound(Offset);
    if (Iter != Islands->DataOffsets.end())
      EndOfCode = *Iter;
  }
  for (uint64_t I = Offset; I < EndOfCode; ++I)
    if (FunctionData[I] != 0)
      return false;

  return true;
}

Error BinaryFunction::handlePCRelOperand(MCInst &Instruction, uint64_t Address,
                                         uint64_t Size) {
  auto &MIB = BC.MIB;
  uint64_t TargetAddress = 0;
  if (!MIB->evaluateMemOperandTarget(Instruction, TargetAddress, Address,
                                     Size)) {
    std::string Msg;
    raw_string_ostream SS(Msg);
    SS << "BOLT-ERROR: PC-relative operand can't be evaluated:\n";
    BC.InstPrinter->printInst(&Instruction, 0, "", *BC.STI, SS);
    SS << '\n';
    Instruction.dump_pretty(SS, BC.InstPrinter.get());
    SS << '\n';
    SS << "BOLT-ERROR: cannot handle PC-relative operand at 0x"
       << Twine::utohexstr(Address) << ". Skipping function " << *this << ".\n";
    if (BC.HasRelocations)
      return createFatalBOLTError(Msg);
    IsSimple = false;
    return createNonFatalBOLTError(Msg);
  }
  if (TargetAddress == 0 && opts::Verbosity >= 1) {
    BC.outs() << "BOLT-INFO: PC-relative operand is zero in function " << *this
              << '\n';
  }

  const MCSymbol *TargetSymbol;
  uint64_t TargetOffset;
  std::tie(TargetSymbol, TargetOffset) =
      BC.handleAddressRef(TargetAddress, *this, /*IsPCRel*/ true);

  bool ReplaceSuccess = MIB->replaceMemOperandDisp(
      Instruction, TargetSymbol, static_cast<int64_t>(TargetOffset), &*BC.Ctx);
  (void)ReplaceSuccess;
  assert(ReplaceSuccess && "Failed to replace mem operand with symbol+off.");
  return Error::success();
}

MCSymbol *BinaryFunction::handleExternalReference(MCInst &Instruction,
                                                  uint64_t Size,
                                                  uint64_t Offset,
                                                  uint64_t TargetAddress,
                                                  bool &IsCall) {
  auto &MIB = BC.MIB;

  const uint64_t AbsoluteInstrAddr = getAddress() + Offset;
  BC.addInterproceduralReference(this, TargetAddress);
  if (opts::Verbosity >= 2 && !IsCall && Size == 2 && !BC.HasRelocations) {
    BC.errs() << "BOLT-WARNING: relaxed tail call detected at 0x"
              << Twine::utohexstr(AbsoluteInstrAddr) << " in function " << *this
              << ". Code size will be increased.\n";
  }

  assert(!MIB->isTailCall(Instruction) &&
         "synthetic tail call instruction found");

  // This is a call regardless of the opcode.
  // Assign proper opcode for tail calls, so that they could be
  // treated as calls.
  if (!IsCall) {
    if (!MIB->convertJmpToTailCall(Instruction)) {
      assert(MIB->isConditionalBranch(Instruction) &&
             "unknown tail call instruction");
      if (opts::Verbosity >= 2) {
        BC.errs() << "BOLT-WARNING: conditional tail call detected in "
                  << "function " << *this << " at 0x"
                  << Twine::utohexstr(AbsoluteInstrAddr) << ".\n";
      }
    }
    IsCall = true;
  }

  if (opts::Verbosity >= 2 && TargetAddress == 0) {
    // We actually see calls to address 0 in presence of weak
    // symbols originating from libraries. This code is never meant
    // to be executed.
    BC.outs() << "BOLT-INFO: Function " << *this
              << " has a call to address zero.\n";
  }

  return BC.getOrCreateGlobalSymbol(TargetAddress, "FUNCat");
}

void BinaryFunction::handleIndirectBranch(MCInst &Instruction, uint64_t Size,
                                          uint64_t Offset) {
  auto &MIB = BC.MIB;
  uint64_t IndirectTarget = 0;
  IndirectBranchType Result =
      processIndirectBranch(Instruction, Size, Offset, IndirectTarget);
  switch (Result) {
  default:
    llvm_unreachable("unexpected result");
  case IndirectBranchType::POSSIBLE_TAIL_CALL: {
    bool Result = MIB->convertJmpToTailCall(Instruction);
    (void)Result;
    assert(Result);
    break;
  }
  case IndirectBranchType::POSSIBLE_JUMP_TABLE:
  case IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE:
  case IndirectBranchType::POSSIBLE_PIC_FIXED_BRANCH:
    if (opts::JumpTables == JTS_NONE)
      IsSimple = false;
    break;
  case IndirectBranchType::POSSIBLE_FIXED_BRANCH: {
    if (containsAddress(IndirectTarget)) {
      const MCSymbol *TargetSymbol = getOrCreateLocalLabel(IndirectTarget);
      Instruction.clear();
      MIB->createUncondBranch(Instruction, TargetSymbol, BC.Ctx.get());
      TakenBranches.emplace_back(Offset, IndirectTarget - getAddress());
      addEntryPointAtOffset(IndirectTarget - getAddress());
    } else {
      MIB->convertJmpToTailCall(Instruction);
      BC.addInterproceduralReference(this, IndirectTarget);
    }
    break;
  }
  case IndirectBranchType::UNKNOWN:
    // Keep processing. We'll do more checks and fixes in
    // postProcessIndirectBranches().
    if (opts::Verbosity > 2) {
      outs() << "BOLT-WARNING: failed to match indirect branch, "
             << getPrintName() << " at 0x" << Twine::utohexstr(Offset)
             << " offset\n";
    }
    UnknownIndirectBranchOffsets.emplace(Offset);
    break;
  }
}

void BinaryFunction::handleAArch64IndirectCall(MCInst &Instruction,
                                               const uint64_t Offset) {
  auto &MIB = BC.MIB;
  const uint64_t AbsoluteInstrAddr = getAddress() + Offset;
  MCInst *TargetHiBits, *TargetLowBits;
  uint64_t TargetAddress, Count;
  Count = MIB->matchLinkerVeneer(Instructions.begin(), Instructions.end(),
                                 AbsoluteInstrAddr, Instruction, TargetHiBits,
                                 TargetLowBits, TargetAddress);
  if (Count) {
    MIB->addAnnotation(Instruction, "AArch64Veneer", true);
    --Count;
    for (auto It = std::prev(Instructions.end()); Count != 0;
         It = std::prev(It), --Count) {
      MIB->addAnnotation(It->second, "AArch64Veneer", true);
    }

    BC.addAdrpAddRelocAArch64(*this, *TargetLowBits, *TargetHiBits,
                              TargetAddress);
  }
}

std::optional<MCInst>
BinaryFunction::disassembleInstructionAtOffset(uint64_t Offset) const {
  assert(CurrentState == State::Empty && "Function should not be disassembled");
  assert(Offset < MaxSize && "Invalid offset");
  ErrorOr<ArrayRef<unsigned char>> FunctionData = getData();
  assert(FunctionData && "Cannot get function as data");
  MCInst Instr;
  uint64_t InstrSize = 0;
  const uint64_t InstrAddress = getAddress() + Offset;
  if (BC.DisAsm->getInstruction(Instr, InstrSize, FunctionData->slice(Offset),
                                InstrAddress, nulls()))
    return Instr;
  return std::nullopt;
}

Error BinaryFunction::disassemble() {
  NamedRegionTimer T("disassemble", "Disassemble function", "buildfuncs",
                     "Build Binary Functions", opts::TimeBuild);
  ErrorOr<ArrayRef<uint8_t>> ErrorOrFunctionData = getData();
  assert(ErrorOrFunctionData && "function data is not available");
  ArrayRef<uint8_t> FunctionData = *ErrorOrFunctionData;
  assert(FunctionData.size() == getMaxSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIB = BC.MIB;

  BC.SymbolicDisAsm->setSymbolizer(MIB->createTargetSymbolizer(*this));

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createNamedTempSymbol("BB0");

  // Map offsets in the function to a label that should always point to the
  // corresponding instruction. This is used for labels that shouldn't point to
  // the start of a basic block but always to a specific instruction. This is
  // used, for example, on RISC-V where %pcrel_lo relocations point to the
  // corresponding %pcrel_hi.
  LabelsMapType InstructionLabels;

  uint64_t Size = 0; // instruction size
  for (uint64_t Offset = 0; Offset < getSize(); Offset += Size) {
    MCInst Instruction;
    const uint64_t AbsoluteInstrAddr = getAddress() + Offset;

    // Check for data inside code and ignore it
    if (const size_t DataInCodeSize = getSizeOfDataInCodeAt(Offset)) {
      Size = DataInCodeSize;
      continue;
    }

    if (!BC.SymbolicDisAsm->getInstruction(Instruction, Size,
                                           FunctionData.slice(Offset),
                                           AbsoluteInstrAddr, nulls())) {
      // Functions with "soft" boundaries, e.g. coming from assembly source,
      // can have 0-byte padding at the end.
      if (isZeroPaddingAt(Offset))
        break;

      BC.errs()
          << "BOLT-WARNING: unable to disassemble instruction at offset 0x"
          << Twine::utohexstr(Offset) << " (address 0x"
          << Twine::utohexstr(AbsoluteInstrAddr) << ") in function " << *this
          << '\n';
      // Some AVX-512 instructions could not be disassembled at all.
      if (BC.HasRelocations && opts::TrapOnAVX512 && BC.isX86()) {
        setTrapOnEntry();
        BC.TrappedFunctions.push_back(this);
      } else {
        setIgnored();
      }

      break;
    }

    // Check integrity of LLVM assembler/disassembler.
    if (opts::CheckEncoding && !BC.MIB->isBranch(Instruction) &&
        !BC.MIB->isCall(Instruction) && !BC.MIB->isNoop(Instruction)) {
      if (!BC.validateInstructionEncoding(FunctionData.slice(Offset, Size))) {
        BC.errs() << "BOLT-WARNING: mismatching LLVM encoding detected in "
                  << "function " << *this << " for instruction :\n";
        BC.printInstruction(BC.errs(), Instruction, AbsoluteInstrAddr);
        BC.errs() << '\n';
      }

      // Verify that we've symbolized an operand if the instruction has a
      // relocation against it.
      if (getRelocationInRange(Offset, Offset + Size)) {
        bool HasSymbolicOp = false;
        for (MCOperand &Op : Instruction) {
          if (Op.isExpr()) {
            HasSymbolicOp = true;
            break;
          }
        }
        if (!HasSymbolicOp)
          return createFatalBOLTError(
              "expected symbolized operand for instruction at 0x" +
              Twine::utohexstr(AbsoluteInstrAddr));
      }
    }

    // Special handling for AVX-512 instructions.
    if (MIB->hasEVEXEncoding(Instruction)) {
      if (BC.HasRelocations && opts::TrapOnAVX512) {
        setTrapOnEntry();
        BC.TrappedFunctions.push_back(this);
        break;
      }

      if (!BC.validateInstructionEncoding(FunctionData.slice(Offset, Size))) {
        BC.errs() << "BOLT-WARNING: internal assembler/disassembler error "
                     "detected for AVX512 instruction:\n";
        BC.printInstruction(BC.errs(), Instruction, AbsoluteInstrAddr);
        BC.errs() << " in function " << *this << '\n';
        setIgnored();
        break;
      }
    }

    bool IsUnsupported = BC.MIB->isUnsupportedInstruction(Instruction);
    if (IsUnsupported)
      setIgnored();

    if (MIB->isBranch(Instruction) || MIB->isCall(Instruction)) {
      uint64_t TargetAddress = 0;
      if (MIB->evaluateBranch(Instruction, AbsoluteInstrAddr, Size,
                              TargetAddress)) {
        // Check if the target is within the same function. Otherwise it's
        // a call, possibly a tail call.
        //
        // If the target *is* the function address it could be either a branch
        // or a recursive call.
        bool IsCall = MIB->isCall(Instruction);
        const bool IsCondBranch = MIB->isConditionalBranch(Instruction);
        MCSymbol *TargetSymbol = nullptr;

        if (IsUnsupported)
          if (auto *TargetFunc =
                  BC.getBinaryFunctionContainingAddress(TargetAddress))
            TargetFunc->setIgnored();

        if (IsCall && TargetAddress == getAddress()) {
          // A recursive call. Calls to internal blocks are handled by
          // ValidateInternalCalls pass.
          TargetSymbol = getSymbol();
        }

        if (!TargetSymbol) {
          // Create either local label or external symbol.
          if (containsAddress(TargetAddress)) {
            TargetSymbol = getOrCreateLocalLabel(TargetAddress);
          } else {
            if (TargetAddress == getAddress() + getSize() &&
                TargetAddress < getAddress() + getMaxSize() &&
                !(BC.isAArch64() &&
                  BC.handleAArch64Veneer(TargetAddress, /*MatchOnly*/ true))) {
              // Result of __builtin_unreachable().
              LLVM_DEBUG(dbgs() << "BOLT-DEBUG: jump past end detected at 0x"
                                << Twine::utohexstr(AbsoluteInstrAddr)
                                << " in function " << *this
                                << " : replacing with nop.\n");
              BC.MIB->createNoop(Instruction);
              if (IsCondBranch) {
                // Register branch offset for profile validation.
                IgnoredBranches.emplace_back(Offset, Offset + Size);
              }
              goto add_instruction;
            }
            // May update Instruction and IsCall
            TargetSymbol = handleExternalReference(Instruction, Size, Offset,
                                                   TargetAddress, IsCall);
          }
        }

        if (!IsCall) {
          // Add taken branch info.
          TakenBranches.emplace_back(Offset, TargetAddress - getAddress());
        }
        BC.MIB->replaceBranchTarget(Instruction, TargetSymbol, &*Ctx);

        // Mark CTC.
        if (IsCondBranch && IsCall)
          MIB->setConditionalTailCall(Instruction, TargetAddress);
      } else {
        // Could not evaluate branch. Should be an indirect call or an
        // indirect branch. Bail out on the latter case.
        if (MIB->isIndirectBranch(Instruction))
          handleIndirectBranch(Instruction, Size, Offset);
        // Indirect call. We only need to fix it if the operand is RIP-relative.
        if (IsSimple && MIB->hasPCRelOperand(Instruction)) {
          if (auto NewE = handleErrors(
                  handlePCRelOperand(Instruction, AbsoluteInstrAddr, Size),
                  [&](const BOLTError &E) -> Error {
                    if (E.isFatal())
                      return Error(std::make_unique<BOLTError>(std::move(E)));
                    if (!E.getMessage().empty())
                      E.log(BC.errs());
                    return Error::success();
                  })) {
            return Error(std::move(NewE));
          }
        }

        if (BC.isAArch64())
          handleAArch64IndirectCall(Instruction, Offset);
      }
    } else if (BC.isRISCV()) {
      // Check if there's a relocation associated with this instruction.
      for (auto Itr = Relocations.lower_bound(Offset),
                ItrE = Relocations.lower_bound(Offset + Size);
           Itr != ItrE; ++Itr) {
        const Relocation &Relocation = Itr->second;
        MCSymbol *Symbol = Relocation.Symbol;

        if (Relocation::isInstructionReference(Relocation.Type)) {
          uint64_t RefOffset = Relocation.Value - getAddress();
          LabelsMapType::iterator LI = InstructionLabels.find(RefOffset);

          if (LI == InstructionLabels.end()) {
            Symbol = BC.Ctx->createNamedTempSymbol();
            InstructionLabels.emplace(RefOffset, Symbol);
          } else {
            Symbol = LI->second;
          }
        }

        uint64_t Addend = Relocation.Addend;

        // For GOT relocations, create a reference against GOT entry ignoring
        // the relocation symbol.
        if (Relocation::isGOT(Relocation.Type)) {
          assert(Relocation::isPCRelative(Relocation.Type) &&
                 "GOT relocation must be PC-relative on RISC-V");
          Symbol = BC.registerNameAtAddress("__BOLT_got_zero", 0, 0, 0);
          Addend = Relocation.Value + Relocation.Offset + getAddress();
        }
        int64_t Value = Relocation.Value;
        const bool Result = BC.MIB->replaceImmWithSymbolRef(
            Instruction, Symbol, Addend, Ctx.get(), Value, Relocation.Type);
        (void)Result;
        assert(Result && "cannot replace immediate with relocation");
      }
    }

add_instruction:
    if (getDWARFLineTable()) {
      Instruction.setLoc(findDebugLineInformationForInstructionAt(
          AbsoluteInstrAddr, getDWARFUnit(), getDWARFLineTable()));
    }

    // Record offset of the instruction for profile matching.
    if (BC.keepOffsetForInstruction(Instruction))
      MIB->setOffset(Instruction, static_cast<uint32_t>(Offset));

    if (BC.isX86() && BC.MIB->isNoop(Instruction)) {
      // NOTE: disassembly loses the correct size information for noops on x86.
      //       E.g. nopw 0x0(%rax,%rax,1) is 9 bytes, but re-encoded it's only
      //       5 bytes. Preserve the size info using annotations.
      MIB->setSize(Instruction, Size);
    }

    addInstruction(Offset, std::move(Instruction));
  }

  for (auto [Offset, Label] : InstructionLabels) {
    InstrMapType::iterator II = Instructions.find(Offset);
    assert(II != Instructions.end() && "reference to non-existing instruction");

    BC.MIB->setInstLabel(II->second, Label);
  }

  // Reset symbolizer for the disassembler.
  BC.SymbolicDisAsm->setSymbolizer(nullptr);

  if (uint64_t Offset = getFirstInstructionOffset())
    Labels[Offset] = BC.Ctx->createNamedTempSymbol();

  if (!IsSimple) {
    clearList(Instructions);
    return createNonFatalBOLTError("");
  }

  updateState(State::Disassembled);

  return Error::success();
}

MCSymbol *BinaryFunction::registerBranch(uint64_t Src, uint64_t Dst) {
  assert(CurrentState == State::Disassembled &&
         "Cannot register branch unless function is in disassembled state.");
  assert(containsAddress(Src) && containsAddress(Dst) &&
         "Cannot register external branch.");
  MCSymbol *Target = getOrCreateLocalLabel(Dst);
  TakenBranches.emplace_back(Src - getAddress(), Dst - getAddress());
  return Target;
}

void BinaryFunction::analyzeInstructionForFuncReference(const MCInst &Inst) {
  for (unsigned OpNum = 0; OpNum < MCPlus::getNumPrimeOperands(Inst); ++OpNum) {
    const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Inst, OpNum);
    if (!Symbol)
      continue;
    if (BinaryFunction *BF = BC.getFunctionForSymbol(Symbol))
      BF->setHasAddressTaken(true);
  }
}

bool BinaryFunction::scanExternalRefs() {
  bool Success = true;
  bool DisassemblyFailed = false;

  // Ignore pseudo functions.
  if (isPseudo())
    return Success;

  if (opts::NoScan) {
    clearList(Relocations);
    clearList(ExternallyReferencedOffsets);

    return false;
  }

  // List of external references for this function.
  std::vector<Relocation> FunctionRelocations;

  static BinaryContext::IndependentCodeEmitter Emitter =
      BC.createIndependentMCCodeEmitter();

  ErrorOr<ArrayRef<uint8_t>> ErrorOrFunctionData = getData();
  assert(ErrorOrFunctionData && "function data is not available");
  ArrayRef<uint8_t> FunctionData = *ErrorOrFunctionData;
  assert(FunctionData.size() == getMaxSize() &&
         "function size does not match raw data size");

  BC.SymbolicDisAsm->setSymbolizer(
      BC.MIB->createTargetSymbolizer(*this, /*CreateSymbols*/ false));

  // A list of patches for this function.
  using PatchTy = std::pair<uint64_t, MCInst>;
  std::vector<PatchTy> InstructionPatches;

  // Disassemble contents of the function. Detect code entry points and create
  // relocations for references to code that will be moved.
  uint64_t Size = 0; // instruction size
  MCInst Instruction;
  MCInst PrevInstruction;
  for (uint64_t Offset = 0; Offset < getSize(); Offset += Size) {
    // Check for data inside code and ignore it
    if (const size_t DataInCodeSize = getSizeOfDataInCodeAt(Offset)) {
      Size = DataInCodeSize;
      continue;
    }

    const uint64_t AbsoluteInstrAddr = getAddress() + Offset;
    PrevInstruction = Instruction;
    if (!BC.SymbolicDisAsm->getInstruction(Instruction, Size,
                                           FunctionData.slice(Offset),
                                           AbsoluteInstrAddr, nulls())) {
      if (opts::Verbosity >= 1 && !isZeroPaddingAt(Offset)) {
        BC.errs()
            << "BOLT-WARNING: unable to disassemble instruction at offset 0x"
            << Twine::utohexstr(Offset) << " (address 0x"
            << Twine::utohexstr(AbsoluteInstrAddr) << ") in function " << *this
            << '\n';
      }
      Success = false;
      DisassemblyFailed = true;
      break;
    }

    // Return true if we can skip handling the Target function reference.
    auto ignoreFunctionRef = [&](const BinaryFunction &Target) {
      if (&Target == this)
        return true;

      // Note that later we may decide not to emit Target function. In that
      // case, we conservatively create references that will be ignored or
      // resolved to the same function.
      if (!BC.shouldEmit(Target))
        return true;

      return false;
    };

    // Return true if we can ignore reference to the symbol.
    auto ignoreReference = [&](const MCSymbol *TargetSymbol) {
      if (!TargetSymbol)
        return true;

      if (BC.forceSymbolRelocations(TargetSymbol->getName()))
        return false;

      BinaryFunction *TargetFunction = BC.getFunctionForSymbol(TargetSymbol);
      if (!TargetFunction)
        return true;

      return ignoreFunctionRef(*TargetFunction);
    };

    // Handle calls and branches separately as symbolization doesn't work for
    // them yet.
    MCSymbol *BranchTargetSymbol = nullptr;
    if (BC.MIB->isCall(Instruction) || BC.MIB->isBranch(Instruction)) {
      uint64_t TargetAddress = 0;
      BC.MIB->evaluateBranch(Instruction, AbsoluteInstrAddr, Size,
                             TargetAddress);

      // Create an entry point at reference address if needed.
      BinaryFunction *TargetFunction =
          BC.getBinaryFunctionContainingAddress(TargetAddress);

      if (!TargetFunction || ignoreFunctionRef(*TargetFunction))
        continue;

      const uint64_t FunctionOffset =
          TargetAddress - TargetFunction->getAddress();
      BranchTargetSymbol =
          FunctionOffset ? TargetFunction->addEntryPointAtOffset(FunctionOffset)
                         : TargetFunction->getSymbol();
    }

    // Can't find more references. Not creating relocations since we are not
    // moving code.
    if (!BC.HasRelocations)
      continue;

    if (BranchTargetSymbol) {
      BC.MIB->replaceBranchTarget(Instruction, BranchTargetSymbol,
                                  Emitter.LocalCtx.get());
    } else {
      analyzeInstructionForFuncReference(Instruction);
      const bool NeedsPatch = llvm::any_of(
          MCPlus::primeOperands(Instruction), [&](const MCOperand &Op) {
            return Op.isExpr() &&
                   !ignoreReference(BC.MIB->getTargetSymbol(Op.getExpr()));
          });
      if (!NeedsPatch)
        continue;
    }

    // For AArch64, we need to undo relaxation done by the linker if the target
    // of the instruction is a function that we plan to move.
    //
    // Linker relaxation is documented at:
    // https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst
    // under #relocation-optimization.
    if (const Relocation *Rel;
        BC.isAArch64() && (Rel = getRelocationAt(Offset))) {
      // NOP+ADR sequence can originate from either ADRP+ADD or ADRP+LDR.
      // In either case, we convert it into ADRP+ADD.
      if (BC.MIB->isADR(Instruction) &&
          (Rel->Type == ELF::R_AARCH64_ADD_ABS_LO12_NC ||
           Rel->Type == ELF::R_AARCH64_LD64_GOT_LO12_NC)) {
        if (!BC.MIB->isNoop(PrevInstruction)) {
          // In case of unexpected conversion from the linker, skip target
          // optimization.
          const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Instruction);
          BC.errs() << "BOLT-WARNING: cannot undo linker relaxation for "
                       "instruction at 0x"
                    << Twine::utohexstr(AbsoluteInstrAddr) << " referencing "
                    << Symbol->getName() << '\n';
          if (BinaryFunction *TargetBF = BC.getFunctionForSymbol(Symbol))
            TargetBF->setIgnored();
          continue;
        }

        InstructionListType AdrpAdd =
            BC.MIB->undoAdrpAddRelaxation(Instruction, BC.Ctx.get());
        assert(AdrpAdd.size() == 2 && "Two instructions expected");
        LLVM_DEBUG({
          dbgs() << "BOLT-DEBUG: linker relaxation undone for instruction "
                    "at 0x"
                 << Twine::utohexstr(AbsoluteInstrAddr) << '\n';
        });
        InstructionPatches.push_back({AbsoluteInstrAddr - 4, AdrpAdd[0]});
        InstructionPatches.push_back({AbsoluteInstrAddr, AdrpAdd[1]});
        continue;
      }

      // If ADR was emitted by the compiler/assembler to reference a nearby
      // local function, we cannot move away that function due to ADR address
      // span limitation. Hence, we skip the optimization.
      if (BC.MIB->isADR(Instruction) &&
          Rel->Type == ELF::R_AARCH64_ADR_PREL_LO21) {
        BC.errs() << "BOLT-WARNING: unable to convert ADR that references "
                  << Rel->Symbol->getName()
                  << ". Will not optimize the target\n";
        if (BinaryFunction *TargetBF = BC.getFunctionForSymbol(Rel->Symbol))
          TargetBF->setIgnored();
        continue;
      }

      // In the case of GOT load, ADRP+LDR can also be converted into ADRP+ADD.
      // When this happens, it's not always possible to properly symbolize ADRP
      // operand and we might have to adjust the operand based on the next
      // instruction.
      if (BC.MIB->isAddXri(Instruction) &&
          Rel->Type == ELF::R_AARCH64_LD64_GOT_LO12_NC) {
        if (!BC.MIB->matchAdrpAddPair(PrevInstruction, Instruction)) {
          BC.errs() << "BOLT-ERROR: cannot find matching ADRP for relaxed LDR "
                       "instruction at 0x"
                    << Twine::utohexstr(AbsoluteInstrAddr) << '\n';
          exit(1);
        }

        // Check if ADRP was already patched. If not, add a new patch for it.
        if (InstructionPatches.empty() ||
            InstructionPatches.back().first != AbsoluteInstrAddr - 4)
          InstructionPatches.push_back(
              {AbsoluteInstrAddr - 4, PrevInstruction});

        // Adjust the operand for ADRP from the patch.
        MCInst &ADRPInst = InstructionPatches.back().second;
        const MCSymbol *ADRPSymbol = BC.MIB->getTargetSymbol(ADRPInst);
        const MCSymbol *ADDSymbol = BC.MIB->getTargetSymbol(Instruction);
        if (ADRPSymbol != ADDSymbol) {
          const int64_t Addend = BC.MIB->getTargetAddend(Instruction);
          BC.MIB->setOperandToSymbolRef(ADRPInst, /*OpNum*/ 1, ADDSymbol,
                                        Addend, BC.Ctx.get(),
                                        ELF::R_AARCH64_NONE);
        }
      }
    }

    // On AArch64, we use instruction patches for fixing references. We make an
    // exception for branch instructions since they require optional
    // relocations.
    if (BC.isAArch64()) {
      if (!BranchTargetSymbol) {
        LLVM_DEBUG(BC.printInstruction(dbgs(), Instruction, AbsoluteInstrAddr));
        InstructionPatches.push_back({AbsoluteInstrAddr, Instruction});
        continue;
      }

      // Conditional tail calls require new relocation types that are currently
      // not supported. https://github.com/llvm/llvm-project/issues/138264
      if (BC.MIB->isConditionalBranch(Instruction)) {
        if (BinaryFunction *TargetBF =
                BC.getFunctionForSymbol(BranchTargetSymbol)) {
          TargetBF->setNeedsPatch(true);
          continue;
        }
      }
    }

    // Emit the instruction using temp emitter and generate relocations.
    SmallString<256> Code;
    SmallVector<MCFixup, 4> Fixups;
    Emitter.MCE->encodeInstruction(Instruction, Code, Fixups, *BC.STI);

    // Create relocation for every fixup.
    for (const MCFixup &Fixup : Fixups) {
      std::optional<Relocation> Rel = BC.MIB->createRelocation(Fixup, *BC.MAB);
      if (!Rel) {
        Success = false;
        continue;
      }

      if (ignoreReference(Rel->Symbol))
        continue;

      if (Relocation::getSizeForType(Rel->Type) < 4) {
        // If the instruction uses a short form, then we might not be able
        // to handle the rewrite without relaxation, and hence cannot reliably
        // create an external reference relocation.
        Success = false;
        continue;
      }

      if (BC.isAArch64()) {
        // Allow the relocation to be skipped in case of the overflow during the
        // relocation value encoding.
        Rel->setOptional();

        if (!opts::CompactCodeModel)
          if (BinaryFunction *TargetBF = BC.getFunctionForSymbol(Rel->Symbol))
            TargetBF->setNeedsPatch(true);
      }

      Rel->Offset += getAddress() - getOriginSection()->getAddress() + Offset;
      FunctionRelocations.push_back(*Rel);
    }

    if (!Success)
      break;
  }

  // Reset symbolizer for the disassembler.
  BC.SymbolicDisAsm->setSymbolizer(nullptr);

  // Add relocations unless disassembly failed for this function.
  if (!DisassemblyFailed)
    for (Relocation &Rel : FunctionRelocations)
      getOriginSection()->addPendingRelocation(Rel);

  // Add patches grouping them together.
  if (!InstructionPatches.empty()) {
    uint64_t PatchGroupAddress;
    InstructionListType PatchGroup;
    for (auto PI = InstructionPatches.begin(), PE = InstructionPatches.end();
         PI != PE; ++PI) {
      auto &Patch = *PI;
      if (PatchGroup.empty())
        PatchGroupAddress = Patch.first;
      PatchGroup.push_back(Patch.second);
      if (std::next(PI) == PE || std::next(PI)->first != Patch.first + 4) {
        BC.createInstructionPatch(PatchGroupAddress, PatchGroup);
        PatchGroup.clear();
      }
    }
  }

  // Inform BinaryContext that this function symbols will not be defined and
  // relocations should not be created against them.
  if (BC.HasRelocations) {
    for (std::pair<const uint32_t, MCSymbol *> &LI : Labels)
      BC.UndefinedSymbols.insert(LI.second);
    for (MCSymbol *const EndLabel : FunctionEndLabels)
      if (EndLabel)
        BC.UndefinedSymbols.insert(EndLabel);
  }

  clearList(Relocations);
  clearList(ExternallyReferencedOffsets);

  if (Success && BC.HasRelocations)
    HasExternalRefRelocations = true;

  if (opts::Verbosity >= 1 && !Success)
    BC.outs() << "BOLT-INFO: failed to scan refs for  " << *this << '\n';

  return Success;
}

void BinaryFunction::postProcessEntryPoints() {
  if (!isSimple())
    return;

  for (auto &KV : Labels) {
    MCSymbol *Label = KV.second;
    if (!getSecondaryEntryPointSymbol(Label))
      continue;

    // In non-relocation mode there's potentially an external undetectable
    // reference to the entry point and hence we cannot move this entry
    // point. Optimizing without moving could be difficult.
    // In aggregation, register any known entry points for CFG construction.
    if (!BC.HasRelocations && !opts::AggregateOnly)
      setSimple(false);

    const uint32_t Offset = KV.first;

    // If we are at Offset 0 and there is no instruction associated with it,
    // this means this is an empty function. Just ignore. If we find an
    // instruction at this offset, this entry point is valid.
    if (!Offset || getInstructionAtOffset(Offset))
      continue;

    // On AArch64 there are legitimate reasons to have references past the
    // end of the function, e.g. jump tables.
    if (BC.isAArch64() && Offset == getSize())
      continue;

    // If we have grabbed a wrong code label which actually points to some
    // constant island inside the function, ignore this label and remove it
    // from the secondary entry point map.
    if (isStartOfConstantIsland(Offset)) {
      BC.SymbolToFunctionMap.erase(Label);
      removeSymbolFromSecondaryEntryPointMap(Label);
      continue;
    }

    BC.errs() << "BOLT-WARNING: reference in the middle of instruction "
                 "detected in function "
              << *this << " at offset 0x" << Twine::utohexstr(Offset) << '\n';
    if (BC.HasRelocations)
      setIgnored();
    setSimple(false);
    return;
  }
}

void BinaryFunction::postProcessJumpTables() {
  // Create labels for all entries.
  for (auto &JTI : JumpTables) {
    JumpTable &JT = *JTI.second;
    if (JT.Type == JumpTable::JTT_PIC && opts::JumpTables == JTS_BASIC) {
      opts::JumpTables = JTS_MOVE;
      BC.outs() << "BOLT-INFO: forcing -jump-tables=move as PIC jump table was "
                   "detected in function "
                << *this << '\n';
    }
    const uint64_t BDSize =
        BC.getBinaryDataAtAddress(JT.getAddress())->getSize();
    if (!BDSize) {
      BC.setBinaryDataSize(JT.getAddress(), JT.getSize());
    } else {
      assert(BDSize >= JT.getSize() &&
             "jump table cannot be larger than the containing object");
    }
    if (!JT.Entries.empty())
      continue;

    bool HasOneParent = (JT.Parents.size() == 1);
    for (uint64_t EntryAddress : JT.EntriesAsAddress) {
      // builtin_unreachable does not belong to any function
      // Need to handle separately
      bool IsBuiltinUnreachable =
          llvm::any_of(JT.Parents, [&](const BinaryFunction *Parent) {
            return EntryAddress == Parent->getAddress() + Parent->getSize();
          });
      if (IsBuiltinUnreachable) {
        MCSymbol *Label = getOrCreateLocalLabel(EntryAddress, true);
        JT.Entries.push_back(Label);
        continue;
      }
      // Create a local label for targets that cannot be reached by other
      // fragments. Otherwise, create a secondary entry point in the target
      // function.
      BinaryFunction *TargetBF =
          BC.getBinaryFunctionContainingAddress(EntryAddress);
      MCSymbol *Label;
      if (HasOneParent && TargetBF == this) {
        Label = getOrCreateLocalLabel(EntryAddress, true);
      } else {
        const uint64_t Offset = EntryAddress - TargetBF->getAddress();
        Label = Offset ? TargetBF->addEntryPointAtOffset(Offset)
                       : TargetBF->getSymbol();
      }
      JT.Entries.push_back(Label);
    }
  }

  // Add TakenBranches from JumpTables.
  //
  // We want to do it after initial processing since we don't know jump tables'
  // boundaries until we process them all.
  for (auto &JTSite : JTSites) {
    const uint64_t JTSiteOffset = JTSite.first;
    const uint64_t JTAddress = JTSite.second;
    const JumpTable *JT = getJumpTableContainingAddress(JTAddress);
    assert(JT && "cannot find jump table for address");

    uint64_t EntryOffset = JTAddress - JT->getAddress();
    while (EntryOffset < JT->getSize()) {
      uint64_t EntryAddress = JT->EntriesAsAddress[EntryOffset / JT->EntrySize];
      uint64_t TargetOffset = EntryAddress - getAddress();
      if (TargetOffset < getSize()) {
        TakenBranches.emplace_back(JTSiteOffset, TargetOffset);

        if (opts::StrictMode)
          registerReferencedOffset(TargetOffset);
      }

      EntryOffset += JT->EntrySize;

      // A label at the next entry means the end of this jump table.
      if (JT->Labels.count(EntryOffset))
        break;
    }
  }
  clearList(JTSites);

  // Conservatively populate all possible destinations for unknown indirect
  // branches.
  if (opts::StrictMode && hasInternalReference()) {
    for (uint64_t Offset : UnknownIndirectBranchOffsets) {
      for (uint64_t PossibleDestination : ExternallyReferencedOffsets) {
        // Ignore __builtin_unreachable().
        if (PossibleDestination == getSize())
          continue;
        TakenBranches.emplace_back(Offset, PossibleDestination);
      }
    }
  }
}

bool BinaryFunction::validateExternallyReferencedOffsets() {
  SmallPtrSet<MCSymbol *, 4> JTTargets;
  for (const JumpTable *JT : llvm::make_second_range(JumpTables))
    JTTargets.insert_range(JT->Entries);

  bool HasUnclaimedReference = false;
  for (uint64_t Destination : ExternallyReferencedOffsets) {
    // Ignore __builtin_unreachable().
    if (Destination == getSize())
      continue;
    // Ignore constant islands
    if (isInConstantIsland(Destination + getAddress()))
      continue;

    if (BinaryBasicBlock *BB = getBasicBlockAtOffset(Destination)) {
      // Check if the externally referenced offset is a recognized jump table
      // target.
      if (JTTargets.contains(BB->getLabel()))
        continue;

      if (opts::Verbosity >= 1) {
        BC.errs() << "BOLT-WARNING: unclaimed data to code reference (possibly "
                  << "an unrecognized jump table entry) to " << BB->getName()
                  << " in " << *this << "\n";
      }
      auto L = BC.scopeLock();
      addEntryPoint(*BB);
    } else {
      BC.errs() << "BOLT-WARNING: unknown data to code reference to offset "
                << Twine::utohexstr(Destination) << " in " << *this << "\n";
      setIgnored();
    }
    HasUnclaimedReference = true;
  }
  return !HasUnclaimedReference;
}

bool BinaryFunction::postProcessIndirectBranches(
    MCPlusBuilder::AllocatorIdTy AllocId) {
  auto addUnknownControlFlow = [&](BinaryBasicBlock &BB) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding unknown control flow in " << *this
                      << " for " << BB.getName() << "\n");
    HasUnknownControlFlow = true;
    BB.removeAllSuccessors();
    for (uint64_t PossibleDestination : ExternallyReferencedOffsets)
      if (BinaryBasicBlock *SuccBB = getBasicBlockAtOffset(PossibleDestination))
        BB.addSuccessor(SuccBB);
  };

  uint64_t NumIndirectJumps = 0;
  MCInst *LastIndirectJump = nullptr;
  BinaryBasicBlock *LastIndirectJumpBB = nullptr;
  uint64_t LastJT = 0;
  uint16_t LastJTIndexReg = BC.MIB->getNoRegister();
  for (BinaryBasicBlock &BB : blocks()) {
    for (BinaryBasicBlock::iterator II = BB.begin(); II != BB.end(); ++II) {
      MCInst &Instr = *II;
      if (!BC.MIB->isIndirectBranch(Instr))
        continue;

      // If there's an indirect branch in a single-block function -
      // it must be a tail call.
      if (BasicBlocks.size() == 1) {
        BC.MIB->convertJmpToTailCall(Instr);
        return true;
      }

      ++NumIndirectJumps;

      if (opts::StrictMode && !hasInternalReference()) {
        BC.MIB->convertJmpToTailCall(Instr);
        break;
      }

      // Validate the tail call or jump table assumptions now that we know
      // basic block boundaries.
      if (BC.MIB->isTailCall(Instr) || BC.MIB->getJumpTable(Instr)) {
        const unsigned PtrSize = BC.AsmInfo->getCodePointerSize();
        MCInst *MemLocInstr;
        unsigned BaseRegNum, IndexRegNum;
        int64_t DispValue;
        const MCExpr *DispExpr;
        MCInst *PCRelBaseInstr;
        MCInst *FixedEntryLoadInstr;
        IndirectBranchType Type = BC.MIB->analyzeIndirectBranch(
            Instr, BB.begin(), II, PtrSize, MemLocInstr, BaseRegNum,
            IndexRegNum, DispValue, DispExpr, PCRelBaseInstr,
            FixedEntryLoadInstr);
        if (Type != IndirectBranchType::UNKNOWN || MemLocInstr != nullptr)
          continue;

        if (!opts::StrictMode)
          return false;

        if (BC.MIB->isTailCall(Instr)) {
          BC.MIB->convertTailCallToJmp(Instr);
        } else {
          LastIndirectJump = &Instr;
          LastIndirectJumpBB = &BB;
          LastJT = BC.MIB->getJumpTable(Instr);
          LastJTIndexReg = BC.MIB->getJumpTableIndexReg(Instr);
          BC.MIB->unsetJumpTable(Instr);

          JumpTable *JT = BC.getJumpTableContainingAddress(LastJT);
          if (JT->Type == JumpTable::JTT_NORMAL) {
            // Invalidating the jump table may also invalidate other jump table
            // boundaries. Until we have/need a support for this, mark the
            // function as non-simple.
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG: rejected jump table reference"
                              << JT->getName() << " in " << *this << '\n');
            return false;
          }
        }

        addUnknownControlFlow(BB);
        continue;
      }

      // If this block contains an epilogue code and has an indirect branch,
      // then most likely it's a tail call. Otherwise, we cannot tell for sure
      // what it is and conservatively reject the function's CFG.
      bool IsEpilogue = llvm::any_of(BB, [&](const MCInst &Instr) {
        return BC.MIB->isLeave(Instr) || BC.MIB->isPop(Instr);
      });
      if (IsEpilogue) {
        BC.MIB->convertJmpToTailCall(Instr);
        BB.removeAllSuccessors();
        continue;
      }

      if (opts::Verbosity >= 2) {
        BC.outs() << "BOLT-INFO: rejected potential indirect tail call in "
                  << "function " << *this << " in basic block " << BB.getName()
                  << ".\n";
        LLVM_DEBUG(BC.printInstructions(dbgs(), BB.begin(), BB.end(),
                                        BB.getOffset(), this, true));
      }

      if (!opts::StrictMode)
        return false;

      addUnknownControlFlow(BB);
    }
  }

  if (HasInternalLabelReference)
    return false;

  // If there's only one jump table, and one indirect jump, and no other
  // references, then we should be able to derive the jump table even if we
  // fail to match the pattern.
  if (HasUnknownControlFlow && NumIndirectJumps == 1 &&
      JumpTables.size() == 1 && LastIndirectJump &&
      !BC.getJumpTableContainingAddress(LastJT)->IsSplit) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: unsetting unknown control flow in "
                      << *this << '\n');
    BC.MIB->setJumpTable(*LastIndirectJump, LastJT, LastJTIndexReg, AllocId);
    HasUnknownControlFlow = false;

    LastIndirectJumpBB->updateJumpTableSuccessors();
  }

  // Validate that all data references to function offsets are claimed by
  // recognized jump tables. Register externally referenced blocks as entry
  // points.
  if (!opts::StrictMode && hasInternalReference()) {
    if (!validateExternallyReferencedOffsets())
      return false;
  }

  if (HasUnknownControlFlow && !BC.HasRelocations)
    return false;

  return true;
}

void BinaryFunction::recomputeLandingPads() {
  updateBBIndices(0);

  for (BinaryBasicBlock *BB : BasicBlocks) {
    BB->LandingPads.clear();
    BB->Throwers.clear();
  }

  for (BinaryBasicBlock *BB : BasicBlocks) {
    std::unordered_set<const BinaryBasicBlock *> BBLandingPads;
    for (MCInst &Instr : *BB) {
      if (!BC.MIB->isInvoke(Instr))
        continue;

      const std::optional<MCPlus::MCLandingPad> EHInfo =
          BC.MIB->getEHInfo(Instr);
      if (!EHInfo || !EHInfo->first)
        continue;

      BinaryBasicBlock *LPBlock = getBasicBlockForLabel(EHInfo->first);
      if (!BBLandingPads.count(LPBlock)) {
        BBLandingPads.insert(LPBlock);
        BB->LandingPads.emplace_back(LPBlock);
        LPBlock->Throwers.emplace_back(BB);
      }
    }
  }
}

Error BinaryFunction::buildCFG(MCPlusBuilder::AllocatorIdTy AllocatorId) {
  auto &MIB = BC.MIB;

  if (!isSimple()) {
    assert(!BC.HasRelocations &&
           "cannot process file with non-simple function in relocs mode");
    return createNonFatalBOLTError("");
  }

  if (CurrentState != State::Disassembled)
    return createNonFatalBOLTError("");

  assert(BasicBlocks.empty() && "basic block list should be empty");
  assert((Labels.find(getFirstInstructionOffset()) != Labels.end()) &&
         "first instruction should always have a label");

  // Create basic blocks in the original layout order:
  //
  //  * Every instruction with associated label marks
  //    the beginning of a basic block.
  //  * Conditional instruction marks the end of a basic block,
  //    except when the following instruction is an
  //    unconditional branch, and the unconditional branch is not
  //    a destination of another branch. In the latter case, the
  //    basic block will consist of a single unconditional branch
  //    (missed "double-jump" optimization).
  //
  // Created basic blocks are sorted in layout order since they are
  // created in the same order as instructions, and instructions are
  // sorted by offsets.
  BinaryBasicBlock *InsertBB = nullptr;
  BinaryBasicBlock *PrevBB = nullptr;
  bool IsLastInstrNop = false;
  // Offset of the last non-nop instruction.
  uint64_t LastInstrOffset = 0;

  auto addCFIPlaceholders = [this](uint64_t CFIOffset,
                                   BinaryBasicBlock *InsertBB) {
    for (auto FI = OffsetToCFI.lower_bound(CFIOffset),
              FE = OffsetToCFI.upper_bound(CFIOffset);
         FI != FE; ++FI) {
      addCFIPseudo(InsertBB, InsertBB->end(), FI->second);
    }
  };

  // For profiling purposes we need to save the offset of the last instruction
  // in the basic block.
  // NOTE: nops always have an Offset annotation. Annotate the last non-nop as
  //       older profiles ignored nops.
  auto updateOffset = [&](uint64_t Offset) {
    assert(PrevBB && PrevBB != InsertBB && "invalid previous block");
    MCInst *LastNonNop = nullptr;
    for (BinaryBasicBlock::reverse_iterator RII = PrevBB->getLastNonPseudo(),
                                            E = PrevBB->rend();
         RII != E; ++RII) {
      if (!BC.MIB->isPseudo(*RII) && !BC.MIB->isNoop(*RII)) {
        LastNonNop = &*RII;
        break;
      }
    }
    if (LastNonNop && !MIB->getOffset(*LastNonNop))
      MIB->setOffset(*LastNonNop, static_cast<uint32_t>(Offset));
  };

  for (auto I = Instructions.begin(), E = Instructions.end(); I != E; ++I) {
    const uint32_t Offset = I->first;
    MCInst &Instr = I->second;

    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      // Always create new BB at branch destination.
      PrevBB = InsertBB ? InsertBB : PrevBB;
      InsertBB = addBasicBlockAt(LI->first, LI->second);
      if (opts::PreserveBlocksAlignment && IsLastInstrNop)
        InsertBB->setDerivedAlignment();

      if (PrevBB)
        updateOffset(LastInstrOffset);
    }

    // Mark all nops with Offset for profile tracking purposes.
    if (MIB->isNoop(Instr) && !MIB->getOffset(Instr)) {
      // If "Offset" annotation is not present, set it and mark the nop for
      // deletion.
      MIB->setOffset(Instr, static_cast<uint32_t>(Offset));
      // Annotate ordinary nops, so we can safely delete them if required.
      MIB->addAnnotation(Instr, "NOP", static_cast<uint32_t>(1), AllocatorId);
    }

    if (!InsertBB) {
      // It must be a fallthrough or unreachable code. Create a new block unless
      // we see an unconditional branch following a conditional one. The latter
      // should not be a conditional tail call.
      assert(PrevBB && "no previous basic block for a fall through");
      MCInst *PrevInstr = PrevBB->getLastNonPseudoInstr();
      assert(PrevInstr && "no previous instruction for a fall through");
      if (MIB->isUnconditionalBranch(Instr) &&
          !MIB->isIndirectBranch(*PrevInstr) &&
          !MIB->isUnconditionalBranch(*PrevInstr) &&
          !MIB->getConditionalTailCall(*PrevInstr) &&
          !MIB->isReturn(*PrevInstr)) {
        // Temporarily restore inserter basic block.
        InsertBB = PrevBB;
      } else {
        MCSymbol *Label;
        {
          auto L = BC.scopeLock();
          Label = BC.Ctx->createNamedTempSymbol("FT");
        }
        InsertBB = addBasicBlockAt(Offset, Label);
        if (opts::PreserveBlocksAlignment && IsLastInstrNop)
          InsertBB->setDerivedAlignment();
        updateOffset(LastInstrOffset);
      }
    }
    if (Offset == getFirstInstructionOffset()) {
      // Add associated CFI pseudos in the first offset
      addCFIPlaceholders(Offset, InsertBB);
    }

    const bool IsBlockEnd = MIB->isTerminator(Instr);
    IsLastInstrNop = MIB->isNoop(Instr);
    if (!IsLastInstrNop)
      LastInstrOffset = Offset;
    InsertBB->addInstruction(std::move(Instr));

    // Add associated CFI instrs. We always add the CFI instruction that is
    // located immediately after this instruction, since the next CFI
    // instruction reflects the change in state caused by this instruction.
    auto NextInstr = std::next(I);
    uint64_t CFIOffset;
    if (NextInstr != E)
      CFIOffset = NextInstr->first;
    else
      CFIOffset = getSize();

    // Note: this potentially invalidates instruction pointers/iterators.
    addCFIPlaceholders(CFIOffset, InsertBB);

    if (IsBlockEnd) {
      PrevBB = InsertBB;
      InsertBB = nullptr;
    }
  }

  if (BasicBlocks.empty()) {
    setSimple(false);
    return createNonFatalBOLTError("");
  }

  // Intermediate dump.
  LLVM_DEBUG(print(dbgs(), "after creating basic blocks"));

  // TODO: handle properly calls to no-return functions,
  // e.g. exit(3), etc. Otherwise we'll see a false fall-through
  // blocks.

  // Remove duplicates branches. We can get a bunch of them from jump tables.
  // Without doing jump table value profiling we don't have a use for extra
  // (duplicate) branches.
  llvm::sort(TakenBranches);
  auto NewEnd = llvm::unique(TakenBranches);
  TakenBranches.erase(NewEnd, TakenBranches.end());

  for (std::pair<uint32_t, uint32_t> &Branch : TakenBranches) {
    LLVM_DEBUG(dbgs() << "registering branch [0x"
                      << Twine::utohexstr(Branch.first) << "] -> [0x"
                      << Twine::utohexstr(Branch.second) << "]\n");
    BinaryBasicBlock *FromBB = getBasicBlockContainingOffset(Branch.first);
    BinaryBasicBlock *ToBB = getBasicBlockAtOffset(Branch.second);
    if (!FromBB || !ToBB) {
      if (!FromBB)
        BC.errs() << "BOLT-ERROR: cannot find BB containing the branch.\n";
      if (!ToBB)
        BC.errs()
            << "BOLT-ERROR: cannot find BB containing branch destination.\n";
      return createFatalBOLTError(BC.generateBugReportMessage(
          "disassembly failed - inconsistent branch found.", *this));
    }

    FromBB->addSuccessor(ToBB);
  }

  // Add fall-through branches.
  PrevBB = nullptr;
  bool IsPrevFT = false; // Is previous block a fall-through.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    if (IsPrevFT)
      PrevBB->addSuccessor(BB);

    if (BB->empty()) {
      IsPrevFT = true;
      PrevBB = BB;
      continue;
    }

    MCInst *LastInstr = BB->getLastNonPseudoInstr();
    assert(LastInstr &&
           "should have non-pseudo instruction in non-empty block");

    if (BB->succ_size() == 0) {
      // Since there's no existing successors, we know the last instruction is
      // not a conditional branch. Thus if it's a terminator, it shouldn't be a
      // fall-through.
      //
      // Conditional tail call is a special case since we don't add a taken
      // branch successor for it.
      IsPrevFT = !MIB->isTerminator(*LastInstr) ||
                 MIB->getConditionalTailCall(*LastInstr);
    } else if (BB->succ_size() == 1) {
      IsPrevFT = MIB->isConditionalBranch(*LastInstr);
    } else {
      IsPrevFT = false;
    }

    PrevBB = BB;
  }

  // Assign landing pads and throwers info.
  recomputeLandingPads();

  // Assign CFI information to each BB entry.
  annotateCFIState();

  // Annotate invoke instructions with GNU_args_size data.
  propagateGnuArgsSizeInfo(AllocatorId);

  // Set the basic block layout to the original order and set end offsets.
  PrevBB = nullptr;
  for (BinaryBasicBlock *BB : BasicBlocks) {
    Layout.addBasicBlock(BB);
    if (PrevBB)
      PrevBB->setEndOffset(BB->getOffset());
    PrevBB = BB;
  }
  PrevBB->setEndOffset(getSize());

  Layout.updateLayoutIndices();

  normalizeCFIState();

  // Clean-up memory taken by intermediate structures.
  //
  // NB: don't clear Labels list as we may need them if we mark the function
  //     as non-simple later in the process of discovering extra entry points.
  clearList(Instructions);
  clearList(OffsetToCFI);
  clearList(TakenBranches);

  // Update the state.
  CurrentState = State::CFG;

  // Make any necessary adjustments for indirect branches.
  if (!postProcessIndirectBranches(AllocatorId)) {
    if (opts::Verbosity) {
      BC.errs() << "BOLT-WARNING: failed to post-process indirect branches for "
                << *this << '\n';
    }

    if (BC.isAArch64())
      PreserveNops = BC.HasRelocations;

    // In relocation mode we want to keep processing the function but avoid
    // optimizing it.
    setSimple(false);
  }

  clearList(ExternallyReferencedOffsets);
  clearList(UnknownIndirectBranchOffsets);

  return Error::success();
}

void BinaryFunction::postProcessCFG() {
  if (isSimple() && !BasicBlocks.empty()) {
    // Convert conditional tail call branches to conditional branches that jump
    // to a tail call.
    removeConditionalTailCalls();

    postProcessProfile();

    // Eliminate inconsistencies between branch instructions and CFG.
    postProcessBranches();
  }

  // The final cleanup of intermediate structures.
  clearList(IgnoredBranches);

  // Remove "Offset" annotations, unless we need an address-translation table
  // later. This has no cost, since annotations are allocated by a bumpptr
  // allocator and won't be released anyway until late in the pipeline.
  if (!requiresAddressTranslation() && !opts::Instrument) {
    for (BinaryBasicBlock &BB : blocks())
      for (MCInst &Inst : BB)
        BC.MIB->clearOffset(Inst);
  }

  assert((!isSimple() || validateCFG()) &&
         "invalid CFG detected after post-processing");
}

void BinaryFunction::removeTagsFromProfile() {
  for (BinaryBasicBlock *BB : BasicBlocks) {
    if (BB->ExecutionCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB->ExecutionCount = 0;
    for (BinaryBasicBlock::BinaryBranchInfo &BI : BB->branch_info()) {
      if (BI.Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
          BI.MispredictedCount != BinaryBasicBlock::COUNT_NO_PROFILE)
        continue;
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}

void BinaryFunction::removeConditionalTailCalls() {
  // Blocks to be appended at the end.
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBlocks;

  for (auto BBI = begin(); BBI != end(); ++BBI) {
    BinaryBasicBlock &BB = *BBI;
    MCInst *CTCInstr = BB.getLastNonPseudoInstr();
    if (!CTCInstr)
      continue;

    std::optional<uint64_t> TargetAddressOrNone =
        BC.MIB->getConditionalTailCall(*CTCInstr);
    if (!TargetAddressOrNone)
      continue;

    // Gather all necessary information about CTC instruction before
    // annotations are destroyed.
    const int32_t CFIStateBeforeCTC = BB.getCFIStateAtInstr(CTCInstr);
    uint64_t CTCTakenCount = BinaryBasicBlock::COUNT_NO_PROFILE;
    uint64_t CTCMispredCount = BinaryBasicBlock::COUNT_NO_PROFILE;
    if (hasValidProfile()) {
      CTCTakenCount = BC.MIB->getAnnotationWithDefault<uint64_t>(
          *CTCInstr, "CTCTakenCount");
      CTCMispredCount = BC.MIB->getAnnotationWithDefault<uint64_t>(
          *CTCInstr, "CTCMispredCount");
    }

    // Assert that the tail call does not throw.
    assert(!BC.MIB->getEHInfo(*CTCInstr) &&
           "found tail call with associated landing pad");

    // Create a basic block with an unconditional tail call instruction using
    // the same destination.
    const MCSymbol *CTCTargetLabel = BC.MIB->getTargetSymbol(*CTCInstr);
    assert(CTCTargetLabel && "symbol expected for conditional tail call");
    MCInst TailCallInstr;
    BC.MIB->createTailCall(TailCallInstr, CTCTargetLabel, BC.Ctx.get());

    // Move offset from CTCInstr to TailCallInstr.
    if (const std::optional<uint32_t> Offset = BC.MIB->getOffset(*CTCInstr)) {
      BC.MIB->setOffset(TailCallInstr, *Offset);
      BC.MIB->clearOffset(*CTCInstr);
    }

    // Link new BBs to the original input offset of the BB where the CTC
    // is, so we can map samples recorded in new BBs back to the original BB
    // seem in the input binary (if using BAT)
    std::unique_ptr<BinaryBasicBlock> TailCallBB =
        createBasicBlock(BC.Ctx->createNamedTempSymbol("TC"));
    TailCallBB->setOffset(BB.getInputOffset());
    TailCallBB->addInstruction(TailCallInstr);
    TailCallBB->setCFIState(CFIStateBeforeCTC);

    // Add CFG edge with profile info from BB to TailCallBB.
    BB.addSuccessor(TailCallBB.get(), CTCTakenCount, CTCMispredCount);

    // Add execution count for the block.
    TailCallBB->setExecutionCount(CTCTakenCount);

    BC.MIB->convertTailCallToJmp(*CTCInstr);

    BC.MIB->replaceBranchTarget(*CTCInstr, TailCallBB->getLabel(),
                                BC.Ctx.get());

    // Add basic block to the list that will be added to the end.
    NewBlocks.emplace_back(std::move(TailCallBB));

    // Swap edges as the TailCallBB corresponds to the taken branch.
    BB.swapConditionalSuccessors();

    // This branch is no longer a conditional tail call.
    BC.MIB->unsetConditionalTailCall(*CTCInstr);
  }

  insertBasicBlocks(std::prev(end()), std::move(NewBlocks),
                    /* UpdateLayout */ true,
                    /* UpdateCFIState */ false);
}

uint64_t BinaryFunction::getFunctionScore() const {
  if (FunctionScore != -1)
    return FunctionScore;

  if (!isSimple() || !hasValidProfile()) {
    FunctionScore = 0;
    return FunctionScore;
  }

  uint64_t TotalScore = 0ULL;
  for (const BinaryBasicBlock &BB : blocks()) {
    uint64_t BBExecCount = BB.getExecutionCount();
    if (BBExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      continue;
    TotalScore += BBExecCount * BB.getNumNonPseudos();
  }
  FunctionScore = TotalScore;
  return FunctionScore;
}

void BinaryFunction::annotateCFIState() {
  assert(CurrentState == State::Disassembled && "unexpected function state");
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  // This is an index of the last processed CFI in FDE CFI program.
  uint32_t State = 0;

  // This is an index of RememberState CFI reflecting effective state right
  // after execution of RestoreState CFI.
  //
  // It differs from State iff the CFI at (State-1)
  // was RestoreState (modulo GNU_args_size CFIs, which are ignored).
  //
  // This allows us to generate shorter replay sequences when producing new
  // CFI programs.
  uint32_t EffectiveState = 0;

  // For tracking RememberState/RestoreState sequences.
  std::stack<uint32_t> StateStack;

  for (BinaryBasicBlock *BB : BasicBlocks) {
    BB->setCFIState(EffectiveState);

    for (const MCInst &Instr : *BB) {
      const MCCFIInstruction *CFI = getCFIFor(Instr);
      if (!CFI)
        continue;

      ++State;

      switch (CFI->getOperation()) {
      case MCCFIInstruction::OpRememberState:
        StateStack.push(EffectiveState);
        EffectiveState = State;
        break;
      case MCCFIInstruction::OpRestoreState:
        assert(!StateStack.empty() && "corrupt CFI stack");
        EffectiveState = StateStack.top();
        StateStack.pop();
        break;
      case MCCFIInstruction::OpGnuArgsSize:
        // OpGnuArgsSize CFIs do not affect the CFI state.
        break;
      default:
        // Any other CFI updates the state.
        EffectiveState = State;
        break;
      }
    }
  }

  if (opts::Verbosity >= 1 && !StateStack.empty()) {
    BC.errs() << "BOLT-WARNING: non-empty CFI stack at the end of " << *this
              << '\n';
  }
}

namespace {

/// Our full interpretation of a DWARF CFI machine state at a given point
struct CFISnapshot {
  /// CFA register number and offset defining the canonical frame at this
  /// point, or the number of a rule (CFI state) that computes it with a
  /// DWARF expression. This number will be negative if it refers to a CFI
  /// located in the CIE instead of the FDE.
  uint32_t CFAReg;
  int32_t CFAOffset;
  int32_t CFARule;
  /// Mapping of rules (CFI states) that define the location of each
  /// register. If absent, no rule defining the location of such register
  /// was ever read. This number will be negative if it refers to a CFI
  /// located in the CIE instead of the FDE.
  DenseMap<int32_t, int32_t> RegRule;

  /// References to CIE, FDE and expanded instructions after a restore state
  const BinaryFunction::CFIInstrMapType &CIE;
  const BinaryFunction::CFIInstrMapType &FDE;
  const DenseMap<int32_t, SmallVector<int32_t, 4>> &FrameRestoreEquivalents;

  /// Current FDE CFI number representing the state where the snapshot is at
  int32_t CurState;

  /// Used when we don't have information about which state/rule to apply
  /// to recover the location of either the CFA or a specific register
  constexpr static int32_t UNKNOWN = std::numeric_limits<int32_t>::min();

private:
  /// Update our snapshot by executing a single CFI
  void update(const MCCFIInstruction &Instr, int32_t RuleNumber) {
    switch (Instr.getOperation()) {
    case MCCFIInstruction::OpSameValue:
    case MCCFIInstruction::OpRelOffset:
    case MCCFIInstruction::OpOffset:
    case MCCFIInstruction::OpRestore:
    case MCCFIInstruction::OpUndefined:
    case MCCFIInstruction::OpRegister:
      RegRule[Instr.getRegister()] = RuleNumber;
      break;
    case MCCFIInstruction::OpDefCfaRegister:
      CFAReg = Instr.getRegister();
      CFARule = UNKNOWN;

      // This shouldn't happen according to the spec but GNU binutils on RISC-V
      // emits a DW_CFA_def_cfa_register in CIE's which leaves the offset
      // unspecified. Both readelf and llvm-dwarfdump interpret the offset as 0
      // in this case so let's do the same.
      if (CFAOffset == UNKNOWN)
        CFAOffset = 0;
      break;
    case MCCFIInstruction::OpDefCfaOffset:
      CFAOffset = Instr.getOffset();
      CFARule = UNKNOWN;
      break;
    case MCCFIInstruction::OpDefCfa:
      CFAReg = Instr.getRegister();
      CFAOffset = Instr.getOffset();
      CFARule = UNKNOWN;
      break;
    case MCCFIInstruction::OpEscape: {
      std::optional<uint8_t> Reg =
          readDWARFExpressionTargetReg(Instr.getValues());
      // Handle DW_CFA_def_cfa_expression
      if (!Reg) {
        CFARule = RuleNumber;
        break;
      }
      RegRule[*Reg] = RuleNumber;
      break;
    }
    case MCCFIInstruction::OpAdjustCfaOffset:
    case MCCFIInstruction::OpWindowSave:
    case MCCFIInstruction::OpNegateRAStateWithPC:
    case MCCFIInstruction::OpLLVMDefAspaceCfa:
    case MCCFIInstruction::OpLabel:
    case MCCFIInstruction::OpValOffset:
      llvm_unreachable("unsupported CFI opcode");
      break;
    case MCCFIInstruction::OpNegateRAState:
      if (!(opts::BinaryAnalysisMode || opts::HeatmapMode)) {
        llvm_unreachable("BOLT-ERROR: binaries using pac-ret hardening (e.g. "
                         "as produced by '-mbranch-protection=pac-ret') are "
                         "currently not supported by BOLT.");
      }
      break;
    case MCCFIInstruction::OpRememberState:
    case MCCFIInstruction::OpRestoreState:
    case MCCFIInstruction::OpGnuArgsSize:
      // do not affect CFI state
      break;
    }
  }

public:
  /// Advance state reading FDE CFI instructions up to State number
  void advanceTo(int32_t State) {
    for (int32_t I = CurState, E = State; I != E; ++I) {
      const MCCFIInstruction &Instr = FDE[I];
      if (Instr.getOperation() != MCCFIInstruction::OpRestoreState) {
        update(Instr, I);
        continue;
      }
      // If restore state instruction, fetch the equivalent CFIs that have
      // the same effect of this restore. This is used to ensure remember-
      // restore pairs are completely removed.
      auto Iter = FrameRestoreEquivalents.find(I);
      if (Iter == FrameRestoreEquivalents.end())
        continue;
      for (int32_t RuleNumber : Iter->second)
        update(FDE[RuleNumber], RuleNumber);
    }

    assert(((CFAReg != (uint32_t)UNKNOWN && CFAOffset != UNKNOWN) ||
            CFARule != UNKNOWN) &&
           "CIE did not define default CFA?");

    CurState = State;
  }

  /// Interpret all CIE and FDE instructions up until CFI State number and
  /// populate this snapshot
  CFISnapshot(
      const BinaryFunction::CFIInstrMapType &CIE,
      const BinaryFunction::CFIInstrMapType &FDE,
      const DenseMap<int32_t, SmallVector<int32_t, 4>> &FrameRestoreEquivalents,
      int32_t State)
      : CIE(CIE), FDE(FDE), FrameRestoreEquivalents(FrameRestoreEquivalents) {
    CFAReg = UNKNOWN;
    CFAOffset = UNKNOWN;
    CFARule = UNKNOWN;
    CurState = 0;

    for (int32_t I = 0, E = CIE.size(); I != E; ++I) {
      const MCCFIInstruction &Instr = CIE[I];
      update(Instr, -I);
    }

    advanceTo(State);
  }
};

/// A CFI snapshot with the capability of checking if incremental additions to
/// it are redundant. This is used to ensure we do not emit two CFI instructions
/// back-to-back that are doing the same state change, or to avoid emitting a
/// CFI at all when the state at that point would not be modified after that CFI
struct CFISnapshotDiff : public CFISnapshot {
  bool RestoredCFAReg{false};
  bool RestoredCFAOffset{false};
  DenseMap<int32_t, bool> RestoredRegs;

  CFISnapshotDiff(const CFISnapshot &S) : CFISnapshot(S) {}

  CFISnapshotDiff(
      const BinaryFunction::CFIInstrMapType &CIE,
      const BinaryFunction::CFIInstrMapType &FDE,
      const DenseMap<int32_t, SmallVector<int32_t, 4>> &FrameRestoreEquivalents,
      int32_t State)
      : CFISnapshot(CIE, FDE, FrameRestoreEquivalents, State) {}

  /// Return true if applying Instr to this state is redundant and can be
  /// dismissed.
  bool isRedundant(const MCCFIInstruction &Instr) {
    switch (Instr.getOperation()) {
    case MCCFIInstruction::OpSameValue:
    case MCCFIInstruction::OpRelOffset:
    case MCCFIInstruction::OpOffset:
    case MCCFIInstruction::OpRestore:
    case MCCFIInstruction::OpUndefined:
    case MCCFIInstruction::OpRegister:
    case MCCFIInstruction::OpEscape: {
      uint32_t Reg;
      if (Instr.getOperation() != MCCFIInstruction::OpEscape) {
        Reg = Instr.getRegister();
      } else {
        std::optional<uint8_t> R =
            readDWARFExpressionTargetReg(Instr.getValues());
        // Handle DW_CFA_def_cfa_expression
        if (!R) {
          if (RestoredCFAReg && RestoredCFAOffset)
            return true;
          RestoredCFAReg = true;
          RestoredCFAOffset = true;
          return false;
        }
        Reg = *R;
      }
      if (RestoredRegs[Reg])
        return true;
      RestoredRegs[Reg] = true;
      const int32_t CurRegRule = RegRule.contains(Reg) ? RegRule[Reg] : UNKNOWN;
      if (CurRegRule == UNKNOWN) {
        if (Instr.getOperation() == MCCFIInstruction::OpRestore ||
            Instr.getOperation() == MCCFIInstruction::OpSameValue)
          return true;
        return false;
      }
      const MCCFIInstruction &LastDef =
          CurRegRule < 0 ? CIE[-CurRegRule] : FDE[CurRegRule];
      return LastDef == Instr;
    }
    case MCCFIInstruction::OpDefCfaRegister:
      if (RestoredCFAReg)
        return true;
      RestoredCFAReg = true;
      return CFAReg == Instr.getRegister();
    case MCCFIInstruction::OpDefCfaOffset:
      if (RestoredCFAOffset)
        return true;
      RestoredCFAOffset = true;
      return CFAOffset == Instr.getOffset();
    case MCCFIInstruction::OpDefCfa:
      if (RestoredCFAReg && RestoredCFAOffset)
        return true;
      RestoredCFAReg = true;
      RestoredCFAOffset = true;
      return CFAReg == Instr.getRegister() && CFAOffset == Instr.getOffset();
    case MCCFIInstruction::OpAdjustCfaOffset:
    case MCCFIInstruction::OpWindowSave:
    case MCCFIInstruction::OpNegateRAStateWithPC:
    case MCCFIInstruction::OpLLVMDefAspaceCfa:
    case MCCFIInstruction::OpLabel:
    case MCCFIInstruction::OpValOffset:
      llvm_unreachable("unsupported CFI opcode");
      return false;
    case MCCFIInstruction::OpNegateRAState:
      if (!(opts::BinaryAnalysisMode || opts::HeatmapMode)) {
        llvm_unreachable("BOLT-ERROR: binaries using pac-ret hardening (e.g. "
                         "as produced by '-mbranch-protection=pac-ret') are "
                         "currently not supported by BOLT.");
      }
      break;
    case MCCFIInstruction::OpRememberState:
    case MCCFIInstruction::OpRestoreState:
    case MCCFIInstruction::OpGnuArgsSize:
      // do not affect CFI state
      return true;
    }
    return false;
  }
};

} // end anonymous namespace

bool BinaryFunction::replayCFIInstrs(int32_t FromState, int32_t ToState,
                                     BinaryBasicBlock *InBB,
                                     BinaryBasicBlock::iterator InsertIt) {
  if (FromState == ToState)
    return true;
  assert(FromState < ToState && "can only replay CFIs forward");

  CFISnapshotDiff CFIDiff(CIEFrameInstructions, FrameInstructions,
                          FrameRestoreEquivalents, FromState);

  std::vector<uint32_t> NewCFIs;
  for (int32_t CurState = FromState; CurState < ToState; ++CurState) {
    MCCFIInstruction *Instr = &FrameInstructions[CurState];
    if (Instr->getOperation() == MCCFIInstruction::OpRestoreState) {
      auto Iter = FrameRestoreEquivalents.find(CurState);
      assert(Iter != FrameRestoreEquivalents.end());
      NewCFIs.insert(NewCFIs.end(), Iter->second.begin(), Iter->second.end());
      // RestoreState / Remember will be filtered out later by CFISnapshotDiff,
      // so we might as well fall-through here.
    }
    NewCFIs.push_back(CurState);
  }

  // Replay instructions while avoiding duplicates
  for (int32_t State : llvm::reverse(NewCFIs)) {
    if (CFIDiff.isRedundant(FrameInstructions[State]))
      continue;
    InsertIt = addCFIPseudo(InBB, InsertIt, State);
  }

  return true;
}

SmallVector<int32_t, 4>
BinaryFunction::unwindCFIState(int32_t FromState, int32_t ToState,
                               BinaryBasicBlock *InBB,
                               BinaryBasicBlock::iterator &InsertIt) {
  SmallVector<int32_t, 4> NewStates;

  CFISnapshot ToCFITable(CIEFrameInstructions, FrameInstructions,
                         FrameRestoreEquivalents, ToState);
  CFISnapshotDiff FromCFITable(ToCFITable);
  FromCFITable.advanceTo(FromState);

  auto undoStateDefCfa = [&]() {
    if (ToCFITable.CFARule == CFISnapshot::UNKNOWN) {
      FrameInstructions.emplace_back(MCCFIInstruction::cfiDefCfa(
          nullptr, ToCFITable.CFAReg, ToCFITable.CFAOffset));
      if (FromCFITable.isRedundant(FrameInstructions.back())) {
        FrameInstructions.pop_back();
        return;
      }
      NewStates.push_back(FrameInstructions.size() - 1);
      InsertIt = addCFIPseudo(InBB, InsertIt, FrameInstructions.size() - 1);
      ++InsertIt;
    } else if (ToCFITable.CFARule < 0) {
      if (FromCFITable.isRedundant(CIEFrameInstructions[-ToCFITable.CFARule]))
        return;
      NewStates.push_back(FrameInstructions.size());
      InsertIt = addCFIPseudo(InBB, InsertIt, FrameInstructions.size());
      ++InsertIt;
      FrameInstructions.emplace_back(CIEFrameInstructions[-ToCFITable.CFARule]);
    } else if (!FromCFITable.isRedundant(
                   FrameInstructions[ToCFITable.CFARule])) {
      NewStates.push_back(ToCFITable.CFARule);
      InsertIt = addCFIPseudo(InBB, InsertIt, ToCFITable.CFARule);
      ++InsertIt;
    }
  };

  auto undoState = [&](const MCCFIInstruction &Instr) {
    switch (Instr.getOperation()) {
    case MCCFIInstruction::OpRememberState:
    case MCCFIInstruction::OpRestoreState:
      break;
    case MCCFIInstruction::OpSameValue:
    case MCCFIInstruction::OpRelOffset:
    case MCCFIInstruction::OpOffset:
    case MCCFIInstruction::OpRestore:
    case MCCFIInstruction::OpUndefined:
    case MCCFIInstruction::OpEscape:
    case MCCFIInstruction::OpRegister: {
      uint32_t Reg;
      if (Instr.getOperation() != MCCFIInstruction::OpEscape) {
        Reg = Instr.getRegister();
      } else {
        std::optional<uint8_t> R =
            readDWARFExpressionTargetReg(Instr.getValues());
        // Handle DW_CFA_def_cfa_expression
        if (!R) {
          undoStateDefCfa();
          return;
        }
        Reg = *R;
      }

      if (!ToCFITable.RegRule.contains(Reg)) {
        FrameInstructions.emplace_back(
            MCCFIInstruction::createRestore(nullptr, Reg));
        if (FromCFITable.isRedundant(FrameInstructions.back())) {
          FrameInstructions.pop_back();
          break;
        }
        NewStates.push_back(FrameInstructions.size() - 1);
        InsertIt = addCFIPseudo(InBB, InsertIt, FrameInstructions.size() - 1);
        ++InsertIt;
        break;
      }
      const int32_t Rule = ToCFITable.RegRule[Reg];
      if (Rule < 0) {
        if (FromCFITable.isRedundant(CIEFrameInstructions[-Rule]))
          break;
        NewStates.push_back(FrameInstructions.size());
        InsertIt = addCFIPseudo(InBB, InsertIt, FrameInstructions.size());
        ++InsertIt;
        FrameInstructions.emplace_back(CIEFrameInstructions[-Rule]);
        break;
      }
      if (FromCFITable.isRedundant(FrameInstructions[Rule]))
        break;
      NewStates.push_back(Rule);
      InsertIt = addCFIPseudo(InBB, InsertIt, Rule);
      ++InsertIt;
      break;
    }
    case MCCFIInstruction::OpDefCfaRegister:
    case MCCFIInstruction::OpDefCfaOffset:
    case MCCFIInstruction::OpDefCfa:
      undoStateDefCfa();
      break;
    case MCCFIInstruction::OpAdjustCfaOffset:
    case MCCFIInstruction::OpWindowSave:
    case MCCFIInstruction::OpNegateRAStateWithPC:
    case MCCFIInstruction::OpLLVMDefAspaceCfa:
    case MCCFIInstruction::OpLabel:
    case MCCFIInstruction::OpValOffset:
      llvm_unreachable("unsupported CFI opcode");
      break;
    case MCCFIInstruction::OpNegateRAState:
      if (!(opts::BinaryAnalysisMode || opts::HeatmapMode)) {
        llvm_unreachable("BOLT-ERROR: binaries using pac-ret hardening (e.g. "
                         "as produced by '-mbranch-protection=pac-ret') are "
                         "currently not supported by BOLT.");
      }
      break;
    case MCCFIInstruction::OpGnuArgsSize:
      // do not affect CFI state
      break;
    }
  };

  // Undo all modifications from ToState to FromState
  for (int32_t I = ToState, E = FromState; I != E; ++I) {
    const MCCFIInstruction &Instr = FrameInstructions[I];
    if (Instr.getOperation() != MCCFIInstruction::OpRestoreState) {
      undoState(Instr);
      continue;
    }
    auto Iter = FrameRestoreEquivalents.find(I);
    if (Iter == FrameRestoreEquivalents.end())
      continue;
    for (int32_t State : Iter->second)
      undoState(FrameInstructions[State]);
  }

  return NewStates;
}

void BinaryFunction::normalizeCFIState() {
  // Reordering blocks with remember-restore state instructions can be specially
  // tricky. When rewriting the CFI, we omit remember-restore state instructions
  // entirely. For restore state, we build a map expanding each restore to the
  // equivalent unwindCFIState sequence required at that point to achieve the
  // same effect of the restore. All remember state are then just ignored.
  std::stack<int32_t> Stack;
  for (BinaryBasicBlock *CurBB : Layout.blocks()) {
    for (auto II = CurBB->begin(); II != CurBB->end(); ++II) {
      if (const MCCFIInstruction *CFI = getCFIFor(*II)) {
        if (CFI->getOperation() == MCCFIInstruction::OpRememberState) {
          Stack.push(II->getOperand(0).getImm());
          continue;
        }
        if (CFI->getOperation() == MCCFIInstruction::OpRestoreState) {
          const int32_t RememberState = Stack.top();
          const int32_t CurState = II->getOperand(0).getImm();
          FrameRestoreEquivalents[CurState] =
              unwindCFIState(CurState, RememberState, CurBB, II);
          Stack.pop();
        }
      }
    }
  }
}

bool BinaryFunction::finalizeCFIState() {
  LLVM_DEBUG(
      dbgs() << "Trying to fix CFI states for each BB after reordering.\n");
  LLVM_DEBUG(dbgs() << "This is the list of CFI states for each BB of " << *this
                    << ": ");

  const char *Sep = "";
  (void)Sep;
  for (FunctionFragment &FF : Layout.fragments()) {
    // Hot-cold border: at start of each region (with a different FDE) we need
    // to reset the CFI state.
    int32_t State = 0;

    for (BinaryBasicBlock *BB : FF) {
      const int32_t CFIStateAtExit = BB->getCFIStateAtExit();

      // We need to recover the correct state if it doesn't match expected
      // state at BB entry point.
      if (BB->getCFIState() < State) {
        // In this case, State is currently higher than what this BB expect it
        // to be. To solve this, we need to insert CFI instructions to undo
        // the effect of all CFI from BB's state to current State.
        auto InsertIt = BB->begin();
        unwindCFIState(State, BB->getCFIState(), BB, InsertIt);
      } else if (BB->getCFIState() > State) {
        // If BB's CFI state is greater than State, it means we are behind in
        // the state. Just emit all instructions to reach this state at the
        // beginning of this BB. If this sequence of instructions involve
        // remember state or restore state, bail out.
        if (!replayCFIInstrs(State, BB->getCFIState(), BB, BB->begin()))
          return false;
      }

      State = CFIStateAtExit;
      LLVM_DEBUG(dbgs() << Sep << State; Sep = ", ");
    }
  }
  LLVM_DEBUG(dbgs() << "\n");

  for (BinaryBasicBlock &BB : blocks()) {
    for (auto II = BB.begin(); II != BB.end();) {
      const MCCFIInstruction *CFI = getCFIFor(*II);
      if (CFI && (CFI->getOperation() == MCCFIInstruction::OpRememberState ||
                  CFI->getOperation() == MCCFIInstruction::OpRestoreState)) {
        II = BB.eraseInstruction(II);
      } else {
        ++II;
      }
    }
  }

  return true;
}

bool BinaryFunction::requiresAddressTranslation() const {
  return opts::EnableBAT || hasSDTMarker() || hasPseudoProbe();
}

bool BinaryFunction::requiresAddressMap() const {
  if (isInjected())
    return false;

  return opts::UpdateDebugSections || isMultiEntry() ||
         requiresAddressTranslation();
}

uint64_t BinaryFunction::getInstructionCount() const {
  uint64_t Count = 0;
  for (const BinaryBasicBlock &BB : blocks())
    Count += BB.getNumNonPseudos();
  return Count;
}

void BinaryFunction::clearDisasmState() {
  clearList(Instructions);
  clearList(IgnoredBranches);
  clearList(TakenBranches);

  if (BC.HasRelocations) {
    for (std::pair<const uint32_t, MCSymbol *> &LI : Labels)
      BC.UndefinedSymbols.insert(LI.second);
    for (MCSymbol *const EndLabel : FunctionEndLabels)
      if (EndLabel)
        BC.UndefinedSymbols.insert(EndLabel);
  }
}

void BinaryFunction::setTrapOnEntry() {
  clearDisasmState();

  forEachEntryPoint([&](uint64_t Offset, const MCSymbol *Label) -> bool {
    MCInst TrapInstr;
    BC.MIB->createTrap(TrapInstr);
    addInstruction(Offset, std::move(TrapInstr));
    return true;
  });

  TrapsOnEntry = true;
}

void BinaryFunction::setIgnored() {
  IsIgnored = true;

  if (opts::processAllFunctions()) {
    // We can accept ignored functions before they've been disassembled.
    // In that case, they would still get disassembled and emitted, but not
    // optimized.
    if (CurrentState != State::Empty) {
      BC.errs() << "BOLT-ERROR: cannot ignore non-empty function " << *this
                << " in current mode\n";
      exit(1);
    }
    return;
  }

  IsSimple = false;
  LLVM_DEBUG(dbgs() << "Ignoring " << getPrintName() << '\n');

  if (CurrentState == State::Empty)
    return;

  clearDisasmState();

  // Clear CFG state too.
  if (hasCFG()) {
    releaseCFG();

    for (BinaryBasicBlock *BB : BasicBlocks)
      delete BB;
    clearList(BasicBlocks);

    for (BinaryBasicBlock *BB : DeletedBasicBlocks)
      delete BB;
    clearList(DeletedBasicBlocks);

    Layout.clear();
  }

  CurrentState = State::Empty;

  // Fix external references in the original function body.
  if (BC.HasRelocations) {
    LLVM_DEBUG(dbgs() << "Scanning refs in " << *this << '\n');
    scanExternalRefs();
  }
}

void BinaryFunction::duplicateConstantIslands() {
  assert(Islands && "function expected to have constant islands");

  for (BinaryBasicBlock *BB : getLayout().blocks()) {
    if (!BB->isCold())
      continue;

    for (MCInst &Inst : *BB) {
      int OpNum = 0;
      for (MCOperand &Operand : Inst) {
        if (!Operand.isExpr()) {
          ++OpNum;
          continue;
        }
        const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Inst, OpNum);
        // Check if this is an island symbol
        if (!Islands->Symbols.count(Symbol) &&
            !Islands->ProxySymbols.count(Symbol))
          continue;

        // Create cold symbol, if missing
        auto ISym = Islands->ColdSymbols.find(Symbol);
        MCSymbol *ColdSymbol;
        if (ISym != Islands->ColdSymbols.end()) {
          ColdSymbol = ISym->second;
        } else {
          ColdSymbol = BC.Ctx->getOrCreateSymbol(Symbol->getName() + ".cold");
          Islands->ColdSymbols[Symbol] = ColdSymbol;
          // Check if this is a proxy island symbol and update owner proxy map
          if (Islands->ProxySymbols.count(Symbol)) {
            BinaryFunction *Owner = Islands->ProxySymbols[Symbol];
            auto IProxiedSym = Owner->Islands->Proxies[this].find(Symbol);
            Owner->Islands->ColdProxies[this][IProxiedSym->second] = ColdSymbol;
          }
        }

        // Update instruction reference
        Operand = MCOperand::createExpr(BC.MIB->getTargetExprFor(
            Inst, MCSymbolRefExpr::create(ColdSymbol, *BC.Ctx), *BC.Ctx, 0));
        ++OpNum;
      }
    }
  }
}

#ifndef MAX_PATH
#define MAX_PATH 255
#endif

static std::string constructFilename(std::string Filename,
                                     std::string Annotation,
                                     std::string Suffix) {
  llvm::replace(Filename, '/', '-');
  if (!Annotation.empty())
    Annotation.insert(0, "-");
  if (Filename.size() + Annotation.size() + Suffix.size() > MAX_PATH) {
    assert(Suffix.size() + Annotation.size() <= MAX_PATH);
    Filename.resize(MAX_PATH - (Suffix.size() + Annotation.size()));
  }
  Filename += Annotation;
  Filename += Suffix;
  return Filename;
}

static std::string formatEscapes(const std::string &Str) {
  std::string Result;
  for (unsigned I = 0; I < Str.size(); ++I) {
    char C = Str[I];
    switch (C) {
    case '\n':
      Result += "&#13;";
      break;
    case '"':
      break;
    default:
      Result += C;
      break;
    }
  }
  return Result;
}

void BinaryFunction::dumpGraph(raw_ostream &OS) const {
  OS << "digraph \"" << getPrintName() << "\" {\n"
     << "node [fontname=courier, shape=box, style=filled, colorscheme=brbg9]\n";
  uint64_t Offset = Address;
  for (BinaryBasicBlock *BB : BasicBlocks) {
    auto LayoutPos = find(Layout.blocks(), BB);
    unsigned LayoutIndex = LayoutPos - Layout.block_begin();
    const char *ColdStr = BB->isCold() ? " (cold)" : "";
    std::vector<std::string> Attrs;
    // Bold box for entry points
    if (isEntryPoint(*BB))
      Attrs.push_back("penwidth=2");
    if (BLI && BLI->getLoopFor(BB)) {
      // Distinguish innermost loops
      const BinaryLoop *Loop = BLI->getLoopFor(BB);
      if (Loop->isInnermost())
        Attrs.push_back("fillcolor=6");
      else // some outer loop
        Attrs.push_back("fillcolor=4");
    } else { // non-loopy code
      Attrs.push_back("fillcolor=5");
    }
    ListSeparator LS;
    OS << "\"" << BB->getName() << "\" [";
    for (StringRef Attr : Attrs)
      OS << LS << Attr;
    OS << "]\n";
    OS << format("\"%s\" [label=\"%s%s\\n(C:%lu,O:%lu,I:%u,L:%u,CFI:%u)\\n",
                 BB->getName().data(), BB->getName().data(), ColdStr,
                 BB->getKnownExecutionCount(), BB->getOffset(), getIndex(BB),
                 LayoutIndex, BB->getCFIState());

    if (opts::DotToolTipCode) {
      std::string Str;
      raw_string_ostream CS(Str);
      Offset = BC.printInstructions(CS, BB->begin(), BB->end(), Offset, this,
                                    /* PrintMCInst = */ false,
                                    /* PrintMemData = */ false,
                                    /* PrintRelocations = */ false,
                                    /* Endl = */ R"(\\l)");
      OS << formatEscapes(CS.str()) << '\n';
    }
    OS << "\"]\n";

    // analyzeBranch is just used to get the names of the branch
    // opcodes.
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    const bool Success = BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch);

    const MCInst *LastInstr = BB->getLastNonPseudoInstr();
    const bool IsJumpTable = LastInstr && BC.MIB->getJumpTable(*LastInstr);

    auto BI = BB->branch_info_begin();
    for (BinaryBasicBlock *Succ : BB->successors()) {
      std::string Branch;
      if (Success) {
        if (Succ == BB->getConditionalSuccessor(true)) {
          Branch = CondBranch ? std::string(BC.InstPrinter->getOpcodeName(
                                    CondBranch->getOpcode()))
                              : "TB";
        } else if (Succ == BB->getConditionalSuccessor(false)) {
          Branch = UncondBranch ? std::string(BC.InstPrinter->getOpcodeName(
                                      UncondBranch->getOpcode()))
                                : "FB";
        } else {
          Branch = "FT";
        }
      }
      if (IsJumpTable)
        Branch = "JT";
      OS << format("\"%s\" -> \"%s\" [label=\"%s", BB->getName().data(),
                   Succ->getName().data(), Branch.c_str());

      if (BB->getExecutionCount() != COUNT_NO_PROFILE &&
          BI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        OS << "\\n(C:" << BI->Count << ",M:" << BI->MispredictedCount << ")";
      } else if (ExecutionCount != COUNT_NO_PROFILE &&
                 BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE) {
        OS << "\\n(IC:" << BI->Count << ")";
      }
      OS << "\"]\n";

      ++BI;
    }
    for (BinaryBasicBlock *LP : BB->landing_pads()) {
      OS << format("\"%s\" -> \"%s\" [constraint=false style=dashed]\n",
                   BB->getName().data(), LP->getName().data());
    }
  }
  OS << "}\n";
}

void BinaryFunction::viewGraph() const {
  SmallString<MAX_PATH> Filename;
  if (std::error_code EC =
          sys::fs::createTemporaryFile("bolt-cfg", "dot", Filename)) {
    BC.errs() << "BOLT-ERROR: " << EC.message() << ", unable to create "
              << " bolt-cfg-XXXXX.dot temporary file.\n";
    return;
  }
  dumpGraphToFile(std::string(Filename));
  if (DisplayGraph(Filename))
    BC.errs() << "BOLT-ERROR: Can't display " << Filename
              << " with graphviz.\n";
  if (std::error_code EC = sys::fs::remove(Filename)) {
    BC.errs() << "BOLT-WARNING: " << EC.message() << ", failed to remove "
              << Filename << "\n";
  }
}

void BinaryFunction::dumpGraphForPass(std::string Annotation) const {
  if (!opts::shouldPrint(*this))
    return;

  std::string Filename = constructFilename(getPrintName(), Annotation, ".dot");
  if (opts::Verbosity >= 1)
    BC.outs() << "BOLT-INFO: dumping CFG to " << Filename << "\n";
  dumpGraphToFile(Filename);
}

void BinaryFunction::dumpGraphToFile(std::string Filename) const {
  std::error_code EC;
  raw_fd_ostream of(Filename, EC, sys::fs::OF_None);
  if (EC) {
    if (opts::Verbosity >= 1) {
      BC.errs() << "BOLT-WARNING: " << EC.message() << ", unable to open "
                << Filename << " for output.\n";
    }
    return;
  }
  dumpGraph(of);
}

bool BinaryFunction::validateCFG() const {
  // Skip the validation of CFG after it is finalized
  if (CurrentState == State::CFG_Finalized)
    return true;

  for (BinaryBasicBlock *BB : BasicBlocks)
    if (!BB->validateSuccessorInvariants())
      return false;

  // Make sure all blocks in CFG are valid.
  auto validateBlock = [this](const BinaryBasicBlock *BB, StringRef Desc) {
    if (!BB->isValid()) {
      BC.errs() << "BOLT-ERROR: deleted " << Desc << " " << BB->getName()
                << " detected in:\n";
      this->dump();
      return false;
    }
    return true;
  };
  for (const BinaryBasicBlock *BB : BasicBlocks) {
    if (!validateBlock(BB, "block"))
      return false;
    for (const BinaryBasicBlock *PredBB : BB->predecessors())
      if (!validateBlock(PredBB, "predecessor"))
        return false;
    for (const BinaryBasicBlock *SuccBB : BB->successors())
      if (!validateBlock(SuccBB, "successor"))
        return false;
    for (const BinaryBasicBlock *LP : BB->landing_pads())
      if (!validateBlock(LP, "landing pad"))
        return false;
    for (const BinaryBasicBlock *Thrower : BB->throwers())
      if (!validateBlock(Thrower, "thrower"))
        return false;
  }

  for (const BinaryBasicBlock *BB : BasicBlocks) {
    std::unordered_set<const BinaryBasicBlock *> BBLandingPads;
    for (const BinaryBasicBlock *LP : BB->landing_pads()) {
      if (BBLandingPads.count(LP)) {
        BC.errs() << "BOLT-ERROR: duplicate landing pad detected in"
                  << BB->getName() << " in function " << *this << '\n';
        return false;
      }
      BBLandingPads.insert(LP);
    }

    std::unordered_set<const BinaryBasicBlock *> BBThrowers;
    for (const BinaryBasicBlock *Thrower : BB->throwers()) {
      if (BBThrowers.count(Thrower)) {
        BC.errs() << "BOLT-ERROR: duplicate thrower detected in"
                  << BB->getName() << " in function " << *this << '\n';
        return false;
      }
      BBThrowers.insert(Thrower);
    }

    for (const BinaryBasicBlock *LPBlock : BB->landing_pads()) {
      if (!llvm::is_contained(LPBlock->throwers(), BB)) {
        BC.errs() << "BOLT-ERROR: inconsistent landing pad detected in "
                  << *this << ": " << BB->getName()
                  << " is in LandingPads but not in " << LPBlock->getName()
                  << " Throwers\n";
        return false;
      }
    }
    for (const BinaryBasicBlock *Thrower : BB->throwers()) {
      if (!llvm::is_contained(Thrower->landing_pads(), BB)) {
        BC.errs() << "BOLT-ERROR: inconsistent thrower detected in " << *this
                  << ": " << BB->getName() << " is in Throwers list but not in "
                  << Thrower->getName() << " LandingPads\n";
        return false;
      }
    }
  }

  return true;
}

void BinaryFunction::fixBranches() {
  assert(isSimple() && "Expected function with valid CFG.");

  auto &MIB = BC.MIB;
  MCContext *Ctx = BC.Ctx.get();

  for (BinaryBasicBlock *BB : BasicBlocks) {
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch))
      continue;

    // We will create unconditional branch with correct destination if needed.
    if (UncondBranch)
      BB->eraseInstruction(BB->findInstruction(UncondBranch));

    // Basic block that follows the current one in the final layout.
    const BinaryBasicBlock *const NextBB =
        Layout.getBasicBlockAfter(BB, /*IgnoreSplits=*/false);

    if (BB->succ_size() == 1) {
      // __builtin_unreachable() could create a conditional branch that
      // falls-through into the next function - hence the block will have only
      // one valid successor. Since behaviour is undefined - we replace
      // the conditional branch with an unconditional if required.
      if (CondBranch)
        BB->eraseInstruction(BB->findInstruction(CondBranch));
      if (BB->getSuccessor() == NextBB)
        continue;
      BB->addBranchInstruction(BB->getSuccessor());
    } else if (BB->succ_size() == 2) {
      assert(CondBranch && "conditional branch expected");
      const BinaryBasicBlock *TSuccessor = BB->getConditionalSuccessor(true);
      const BinaryBasicBlock *FSuccessor = BB->getConditionalSuccessor(false);

      // Eliminate unnecessary conditional branch.
      if (TSuccessor == FSuccessor) {
        // FIXME: at the moment, we cannot safely remove static key branches.
        if (MIB->isDynamicBranch(*CondBranch)) {
          if (opts::Verbosity) {
            BC.outs()
                << "BOLT-INFO: unable to remove redundant dynamic branch in "
                << *this << '\n';
          }
          continue;
        }

        BB->removeDuplicateConditionalSuccessor(CondBranch);
        if (TSuccessor != NextBB)
          BB->addBranchInstruction(TSuccessor);
        continue;
      }

      // Reverse branch condition and swap successors.
      auto swapSuccessors = [&]() {
        if (!MIB->isReversibleBranch(*CondBranch)) {
          if (opts::Verbosity) {
            BC.outs() << "BOLT-INFO: unable to swap successors in " << *this
                      << '\n';
          }
          return false;
        }
        std::swap(TSuccessor, FSuccessor);
        BB->swapConditionalSuccessors();
        auto L = BC.scopeLock();
        MIB->reverseBranchCondition(*CondBranch, TSuccessor->getLabel(), Ctx);
        return true;
      };

      // Check whether the next block is a "taken" target and try to swap it
      // with a "fall-through" target.
      if (TSuccessor == NextBB && swapSuccessors())
        continue;

      // Update conditional branch destination if needed.
      if (MIB->getTargetSymbol(*CondBranch) != TSuccessor->getLabel()) {
        auto L = BC.scopeLock();
        MIB->replaceBranchTarget(*CondBranch, TSuccessor->getLabel(), Ctx);
      }

      // No need for the unconditional branch.
      if (FSuccessor == NextBB)
        continue;

      if (BC.isX86()) {
        // We are going to generate two branches. Check if their targets are in
        // the same fragment as this block. If only one target is in the same
        // fragment, make it the destination of the conditional branch. There
        // is a chance it will be a short branch which takes 4 bytes fewer than
        // a long conditional branch. For unconditional branch, the difference
        // is 3 bytes.
        if (BB->getFragmentNum() != TSuccessor->getFragmentNum() &&
            BB->getFragmentNum() == FSuccessor->getFragmentNum())
          swapSuccessors();
      }

      BB->addBranchInstruction(FSuccessor);
    }
    // Cases where the number of successors is 0 (block ends with a
    // terminator) or more than 2 (switch table) don't require branch
    // instruction adjustments.
  }
  assert((!isSimple() || validateCFG()) &&
         "Invalid CFG detected after fixing branches");
}

void BinaryFunction::propagateGnuArgsSizeInfo(
    MCPlusBuilder::AllocatorIdTy AllocId) {
  assert(CurrentState == State::Disassembled && "unexpected function state");

  if (!hasEHRanges() || !usesGnuArgsSize())
    return;

  // The current value of DW_CFA_GNU_args_size affects all following
  // invoke instructions until the next CFI overrides it.
  // It is important to iterate basic blocks in the original order when
  // assigning the value.
  uint64_t CurrentGnuArgsSize = 0;
  for (BinaryBasicBlock *BB : BasicBlocks) {
    for (auto II = BB->begin(); II != BB->end();) {
      MCInst &Instr = *II;
      if (BC.MIB->isCFI(Instr)) {
        const MCCFIInstruction *CFI = getCFIFor(Instr);
        if (CFI->getOperation() == MCCFIInstruction::OpGnuArgsSize) {
          CurrentGnuArgsSize = CFI->getOffset();
          // Delete DW_CFA_GNU_args_size instructions and only regenerate
          // during the final code emission. The information is embedded
          // inside call instructions.
          II = BB->erasePseudoInstruction(II);
          continue;
        }
      } else if (BC.MIB->isInvoke(Instr)) {
        // Add the value of GNU_args_size as an extra operand to invokes.
        BC.MIB->addGnuArgsSize(Instr, CurrentGnuArgsSize);
      }
      ++II;
    }
  }
}

void BinaryFunction::postProcessBranches() {
  if (!isSimple())
    return;
  for (BinaryBasicBlock &BB : blocks()) {
    auto LastInstrRI = BB.getLastNonPseudo();
    if (BB.succ_size() == 1) {
      if (LastInstrRI != BB.rend() &&
          BC.MIB->isConditionalBranch(*LastInstrRI)) {
        // __builtin_unreachable() could create a conditional branch that
        // falls-through into the next function - hence the block will have only
        // one valid successor. Such behaviour is undefined and thus we remove
        // the conditional branch while leaving a valid successor.
        BB.eraseInstruction(std::prev(LastInstrRI.base()));
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: erasing conditional branch in "
                          << BB.getName() << " in function " << *this << '\n');
      }
    } else if (BB.succ_size() == 0) {
      // Ignore unreachable basic blocks.
      if (BB.pred_size() == 0 || BB.isLandingPad())
        continue;

      // If it's the basic block that does not end up with a terminator - we
      // insert a return instruction unless it's a call instruction.
      if (LastInstrRI == BB.rend()) {
        LLVM_DEBUG(
            dbgs() << "BOLT-DEBUG: at least one instruction expected in BB "
                   << BB.getName() << " in function " << *this << '\n');
        continue;
      }
      if (!BC.MIB->isTerminator(*LastInstrRI) &&
          !BC.MIB->isCall(*LastInstrRI)) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding return to basic block "
                          << BB.getName() << " in function " << *this << '\n');
        MCInst ReturnInstr;
        BC.MIB->createReturn(ReturnInstr);
        BB.addInstruction(ReturnInstr);
      }
    }
  }
  assert(validateCFG() && "invalid CFG");
}

MCSymbol *BinaryFunction::addEntryPointAtOffset(uint64_t Offset) {
  assert(Offset && "cannot add primary entry point");

  const uint64_t EntryPointAddress = getAddress() + Offset;
  MCSymbol *LocalSymbol = getOrCreateLocalLabel(EntryPointAddress);

  MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(LocalSymbol);
  if (EntrySymbol)
    return EntrySymbol;

  assert(CurrentState == State::Empty || CurrentState == State::Disassembled);

  if (BinaryData *EntryBD = BC.getBinaryDataAtAddress(EntryPointAddress)) {
    EntrySymbol = EntryBD->getSymbol();
  } else {
    EntrySymbol = BC.getOrCreateGlobalSymbol(
        EntryPointAddress, Twine("__ENTRY_") + getOneName() + "@");
  }
  SecondaryEntryPoints[LocalSymbol] = EntrySymbol;

  BC.setSymbolToFunctionMap(EntrySymbol, this);

  return EntrySymbol;
}

MCSymbol *BinaryFunction::addEntryPoint(const BinaryBasicBlock &BB) {
  assert(CurrentState == State::CFG &&
         "basic block can be added as an entry only in a function with CFG");

  if (&BB == BasicBlocks.front())
    return getSymbol();

  MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(BB);
  if (EntrySymbol)
    return EntrySymbol;

  EntrySymbol =
      BC.Ctx->getOrCreateSymbol("__ENTRY_" + BB.getLabel()->getName());

  SecondaryEntryPoints[BB.getLabel()] = EntrySymbol;

  BC.setSymbolToFunctionMap(EntrySymbol, this);

  return EntrySymbol;
}

MCSymbol *BinaryFunction::getSymbolForEntryID(uint64_t EntryID) {
  if (EntryID == 0)
    return getSymbol();

  if (!isMultiEntry())
    return nullptr;

  uint64_t NumEntries = 1;
  if (hasCFG()) {
    for (BinaryBasicBlock *BB : BasicBlocks) {
      MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(*BB);
      if (!EntrySymbol)
        continue;
      if (NumEntries == EntryID)
        return EntrySymbol;
      ++NumEntries;
    }
  } else {
    for (std::pair<const uint32_t, MCSymbol *> &KV : Labels) {
      MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(KV.second);
      if (!EntrySymbol)
        continue;
      if (NumEntries == EntryID)
        return EntrySymbol;
      ++NumEntries;
    }
  }

  return nullptr;
}

uint64_t BinaryFunction::getEntryIDForSymbol(const MCSymbol *Symbol) const {
  if (!isMultiEntry())
    return 0;

  for (const MCSymbol *FunctionSymbol : getSymbols())
    if (FunctionSymbol == Symbol)
      return 0;

  // Check all secondary entries available as either basic blocks or lables.
  uint64_t NumEntries = 1;
  for (const BinaryBasicBlock *BB : BasicBlocks) {
    MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(*BB);
    if (!EntrySymbol)
      continue;
    if (EntrySymbol == Symbol)
      return NumEntries;
    ++NumEntries;
  }
  NumEntries = 1;
  for (const std::pair<const uint32_t, MCSymbol *> &KV : Labels) {
    MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(KV.second);
    if (!EntrySymbol)
      continue;
    if (EntrySymbol == Symbol)
      return NumEntries;
    ++NumEntries;
  }

  llvm_unreachable("symbol not found");
}

bool BinaryFunction::forEachEntryPoint(EntryPointCallbackTy Callback) const {
  bool Status = Callback(0, getSymbol());
  if (!isMultiEntry())
    return Status;

  for (const std::pair<const uint32_t, MCSymbol *> &KV : Labels) {
    if (!Status)
      break;

    MCSymbol *EntrySymbol = getSecondaryEntryPointSymbol(KV.second);
    if (!EntrySymbol)
      continue;

    Status = Callback(KV.first, EntrySymbol);
  }

  return Status;
}

BinaryFunction::BasicBlockListType BinaryFunction::dfs() const {
  BasicBlockListType DFS;
  std::stack<BinaryBasicBlock *> Stack;
  std::set<BinaryBasicBlock *> Visited;

  // Push entry points to the stack in reverse order.
  //
  // NB: we rely on the original order of entries to match.
  SmallVector<BinaryBasicBlock *> EntryPoints;
  llvm::copy_if(BasicBlocks, std::back_inserter(EntryPoints),
          [&](const BinaryBasicBlock *const BB) { return isEntryPoint(*BB); });
  // Sort entry points by their offset to make sure we got them in the right
  // order.
  llvm::stable_sort(EntryPoints, [](const BinaryBasicBlock *const A,
                              const BinaryBasicBlock *const B) {
    return A->getOffset() < B->getOffset();
  });
  for (BinaryBasicBlock *const BB : reverse(EntryPoints))
    Stack.push(BB);

  while (!Stack.empty()) {
    BinaryBasicBlock *BB = Stack.top();
    Stack.pop();

    if (!Visited.insert(BB).second)
      continue;
    DFS.push_back(BB);

    for (BinaryBasicBlock *SuccBB : BB->landing_pads()) {
      Stack.push(SuccBB);
    }

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch) && CondBranch &&
        BB->succ_size() == 2) {
      if (BC.MIB->getCanonicalBranchCondCode(BC.MIB->getCondCode(
              *CondBranch)) == BC.MIB->getCondCode(*CondBranch)) {
        Stack.push(BB->getConditionalSuccessor(true));
        Stack.push(BB->getConditionalSuccessor(false));
      } else {
        Stack.push(BB->getConditionalSuccessor(false));
        Stack.push(BB->getConditionalSuccessor(true));
      }
    } else {
      for (BinaryBasicBlock *SuccBB : BB->successors()) {
        Stack.push(SuccBB);
      }
    }
  }

  return DFS;
}

size_t BinaryFunction::computeHash(bool UseDFS, HashFunction HashFunction,
                                   OperandHashFuncTy OperandHashFunc) const {
  LLVM_DEBUG({
    dbgs() << "BOLT-DEBUG: computeHash " << getPrintName() << ' '
           << (UseDFS ? "dfs" : "bin") << " order "
           << (HashFunction == HashFunction::StdHash ? "std::hash" : "xxh3")
           << '\n';
  });

  if (size() == 0)
    return 0;

  assert(hasCFG() && "function is expected to have CFG");

  SmallVector<const BinaryBasicBlock *, 0> Order;
  if (UseDFS)
    llvm::copy(dfs(), std::back_inserter(Order));
  else
    llvm::copy(Layout.blocks(), std::back_inserter(Order));

  // The hash is computed by creating a string of all instruction opcodes and
  // possibly their operands and then hashing that string with std::hash.
  std::string HashString;
  for (const BinaryBasicBlock *BB : Order)
    HashString.append(hashBlock(BC, *BB, OperandHashFunc));

  switch (HashFunction) {
  case HashFunction::StdHash:
    return Hash = std::hash<std::string>{}(HashString);
  case HashFunction::XXH3:
    return Hash = llvm::xxh3_64bits(HashString);
  }
  llvm_unreachable("Unhandled HashFunction");
}

void BinaryFunction::insertBasicBlocks(
    BinaryBasicBlock *Start,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    const bool UpdateLayout, const bool UpdateCFIState,
    const bool RecomputeLandingPads) {
  const int64_t StartIndex = Start ? getIndex(Start) : -1LL;
  const size_t NumNewBlocks = NewBBs.size();

  BasicBlocks.insert(BasicBlocks.begin() + (StartIndex + 1), NumNewBlocks,
                     nullptr);

  int64_t I = StartIndex + 1;
  for (std::unique_ptr<BinaryBasicBlock> &BB : NewBBs) {
    assert(!BasicBlocks[I]);
    BasicBlocks[I++] = BB.release();
  }

  if (RecomputeLandingPads)
    recomputeLandingPads();
  else
    updateBBIndices(0);

  if (UpdateLayout)
    updateLayout(Start, NumNewBlocks);

  if (UpdateCFIState)
    updateCFIState(Start, NumNewBlocks);
}

BinaryFunction::iterator BinaryFunction::insertBasicBlocks(
    BinaryFunction::iterator StartBB,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    const bool UpdateLayout, const bool UpdateCFIState,
    const bool RecomputeLandingPads) {
  const unsigned StartIndex = getIndex(&*StartBB);
  const size_t NumNewBlocks = NewBBs.size();

  BasicBlocks.insert(BasicBlocks.begin() + StartIndex + 1, NumNewBlocks,
                     nullptr);
  auto RetIter = BasicBlocks.begin() + StartIndex + 1;

  unsigned I = StartIndex + 1;
  for (std::unique_ptr<BinaryBasicBlock> &BB : NewBBs) {
    assert(!BasicBlocks[I]);
    BasicBlocks[I++] = BB.release();
  }

  if (RecomputeLandingPads)
    recomputeLandingPads();
  else
    updateBBIndices(0);

  if (UpdateLayout)
    updateLayout(*std::prev(RetIter), NumNewBlocks);

  if (UpdateCFIState)
    updateCFIState(*std::prev(RetIter), NumNewBlocks);

  return RetIter;
}

void BinaryFunction::updateBBIndices(const unsigned StartIndex) {
  for (unsigned I = StartIndex; I < BasicBlocks.size(); ++I)
    BasicBlocks[I]->Index = I;
}

void BinaryFunction::updateCFIState(BinaryBasicBlock *Start,
                                    const unsigned NumNewBlocks) {
  const int32_t CFIState = Start->getCFIStateAtExit();
  const unsigned StartIndex = getIndex(Start) + 1;
  for (unsigned I = 0; I < NumNewBlocks; ++I)
    BasicBlocks[StartIndex + I]->setCFIState(CFIState);
}

void BinaryFunction::updateLayout(BinaryBasicBlock *Start,
                                  const unsigned NumNewBlocks) {
  BasicBlockListType::iterator Begin;
  BasicBlockListType::iterator End;

  // If start not provided copy new blocks from the beginning of BasicBlocks
  if (!Start) {
    Begin = BasicBlocks.begin();
    End = BasicBlocks.begin() + NumNewBlocks;
  } else {
    unsigned StartIndex = getIndex(Start);
    Begin = std::next(BasicBlocks.begin(), StartIndex + 1);
    End = std::next(BasicBlocks.begin(), StartIndex + NumNewBlocks + 1);
  }

  // Insert new blocks in the layout immediately after Start.
  Layout.insertBasicBlocks(Start, {Begin, End});
  Layout.updateLayoutIndices();
}

bool BinaryFunction::checkForAmbiguousJumpTables() {
  SmallSet<uint64_t, 4> JumpTables;
  for (BinaryBasicBlock *&BB : BasicBlocks) {
    for (MCInst &Inst : *BB) {
      if (!BC.MIB->isIndirectBranch(Inst))
        continue;
      uint64_t JTAddress = BC.MIB->getJumpTable(Inst);
      if (!JTAddress)
        continue;
      // This address can be inside another jump table, but we only consider
      // it ambiguous when the same start address is used, not the same JT
      // object.
      if (!JumpTables.count(JTAddress)) {
        JumpTables.insert(JTAddress);
        continue;
      }
      return true;
    }
  }
  return false;
}

void BinaryFunction::disambiguateJumpTables(
    MCPlusBuilder::AllocatorIdTy AllocId) {
  assert((opts::JumpTables != JTS_BASIC && isSimple()) || !BC.HasRelocations);
  SmallPtrSet<JumpTable *, 4> JumpTables;
  for (BinaryBasicBlock *&BB : BasicBlocks) {
    for (MCInst &Inst : *BB) {
      if (!BC.MIB->isIndirectBranch(Inst))
        continue;
      JumpTable *JT = getJumpTable(Inst);
      if (!JT)
        continue;
      if (JumpTables.insert(JT).second)
        continue;
      // This instruction is an indirect jump using a jump table, but it is
      // using the same jump table of another jump. Try all our tricks to
      // extract the jump table symbol and make it point to a new, duplicated JT
      MCPhysReg BaseReg1;
      uint64_t Scale;
      const MCSymbol *Target;
      // In case we match if our first matcher, first instruction is the one to
      // patch
      MCInst *JTLoadInst = &Inst;
      // Try a standard indirect jump matcher, scale 8
      std::unique_ptr<MCPlusBuilder::MCInstMatcher> IndJmpMatcher =
          BC.MIB->matchIndJmp(BC.MIB->matchReg(BaseReg1),
                              BC.MIB->matchImm(Scale), BC.MIB->matchReg(),
                              /*Offset=*/BC.MIB->matchSymbol(Target));
      if (!IndJmpMatcher->match(
              *BC.MRI, *BC.MIB,
              MutableArrayRef<MCInst>(&*BB->begin(), &Inst + 1), -1) ||
          BaseReg1 != BC.MIB->getNoRegister() || Scale != 8) {
        MCPhysReg BaseReg2;
        uint64_t Offset;
        // Standard JT matching failed. Trying now:
        //     movq  "jt.2397/1"(,%rax,8), %rax
        //     jmpq  *%rax
        std::unique_ptr<MCPlusBuilder::MCInstMatcher> LoadMatcherOwner =
            BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg1),
                              BC.MIB->matchImm(Scale), BC.MIB->matchReg(),
                              /*Offset=*/BC.MIB->matchSymbol(Target));
        MCPlusBuilder::MCInstMatcher *LoadMatcher = LoadMatcherOwner.get();
        std::unique_ptr<MCPlusBuilder::MCInstMatcher> IndJmpMatcher2 =
            BC.MIB->matchIndJmp(std::move(LoadMatcherOwner));
        if (!IndJmpMatcher2->match(
                *BC.MRI, *BC.MIB,
                MutableArrayRef<MCInst>(&*BB->begin(), &Inst + 1), -1) ||
            BaseReg1 != BC.MIB->getNoRegister() || Scale != 8) {
          // JT matching failed. Trying now:
          // PIC-style matcher, scale 4
          //    addq    %rdx, %rsi
          //    addq    %rdx, %rdi
          //    leaq    DATAat0x402450(%rip), %r11
          //    movslq  (%r11,%rdx,4), %rcx
          //    addq    %r11, %rcx
          //    jmpq    *%rcx # JUMPTABLE @0x402450
          std::unique_ptr<MCPlusBuilder::MCInstMatcher> PICIndJmpMatcher =
              BC.MIB->matchIndJmp(BC.MIB->matchAdd(
                  BC.MIB->matchReg(BaseReg1),
                  BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg2),
                                    BC.MIB->matchImm(Scale), BC.MIB->matchReg(),
                                    BC.MIB->matchImm(Offset))));
          std::unique_ptr<MCPlusBuilder::MCInstMatcher> LEAMatcherOwner =
              BC.MIB->matchLoadAddr(BC.MIB->matchSymbol(Target));
          MCPlusBuilder::MCInstMatcher *LEAMatcher = LEAMatcherOwner.get();
          std::unique_ptr<MCPlusBuilder::MCInstMatcher> PICBaseAddrMatcher =
              BC.MIB->matchIndJmp(BC.MIB->matchAdd(std::move(LEAMatcherOwner),
                                                   BC.MIB->matchAnyOperand()));
          if (!PICIndJmpMatcher->match(
                  *BC.MRI, *BC.MIB,
                  MutableArrayRef<MCInst>(&*BB->begin(), &Inst + 1), -1) ||
              Scale != 4 || BaseReg1 != BaseReg2 || Offset != 0 ||
              !PICBaseAddrMatcher->match(
                  *BC.MRI, *BC.MIB,
                  MutableArrayRef<MCInst>(&*BB->begin(), &Inst + 1), -1)) {
            llvm_unreachable("Failed to extract jump table base");
            continue;
          }
          // Matched PIC, identify the instruction with the reference to the JT
          JTLoadInst = LEAMatcher->CurInst;
        } else {
          // Matched non-PIC
          JTLoadInst = LoadMatcher->CurInst;
        }
      }

      uint64_t NewJumpTableID = 0;
      const MCSymbol *NewJTLabel;
      std::tie(NewJumpTableID, NewJTLabel) =
          BC.duplicateJumpTable(*this, JT, Target);
      {
        auto L = BC.scopeLock();
        BC.MIB->replaceMemOperandDisp(*JTLoadInst, NewJTLabel, BC.Ctx.get());
      }
      // We use a unique ID with the high bit set as address for this "injected"
      // jump table (not originally in the input binary).
      BC.MIB->setJumpTable(Inst, NewJumpTableID, 0, AllocId);
    }
  }
}

bool BinaryFunction::replaceJumpTableEntryIn(BinaryBasicBlock *BB,
                                             BinaryBasicBlock *OldDest,
                                             BinaryBasicBlock *NewDest) {
  MCInst *Instr = BB->getLastNonPseudoInstr();
  if (!Instr || !BC.MIB->isIndirectBranch(*Instr))
    return false;
  uint64_t JTAddress = BC.MIB->getJumpTable(*Instr);
  assert(JTAddress && "Invalid jump table address");
  JumpTable *JT = getJumpTableContainingAddress(JTAddress);
  assert(JT && "No jump table structure for this indirect branch");
  bool Patched = JT->replaceDestination(JTAddress, OldDest->getLabel(),
                                        NewDest->getLabel());
  (void)Patched;
  assert(Patched && "Invalid entry to be replaced in jump table");
  return true;
}

BinaryBasicBlock *BinaryFunction::splitEdge(BinaryBasicBlock *From,
                                            BinaryBasicBlock *To) {
  // Create intermediate BB
  MCSymbol *Tmp;
  {
    auto L = BC.scopeLock();
    Tmp = BC.Ctx->createNamedTempSymbol("SplitEdge");
  }
  // Link new BBs to the original input offset of the From BB, so we can map
  // samples recorded in new BBs back to the original BB seem in the input
  // binary (if using BAT)
  std::unique_ptr<BinaryBasicBlock> NewBB = createBasicBlock(Tmp);
  NewBB->setOffset(From->getInputOffset());
  BinaryBasicBlock *NewBBPtr = NewBB.get();

  // Update "From" BB
  auto I = From->succ_begin();
  auto BI = From->branch_info_begin();
  for (; I != From->succ_end(); ++I) {
    if (*I == To)
      break;
    ++BI;
  }
  assert(I != From->succ_end() && "Invalid CFG edge in splitEdge!");
  uint64_t OrigCount = BI->Count;
  uint64_t OrigMispreds = BI->MispredictedCount;
  replaceJumpTableEntryIn(From, To, NewBBPtr);
  From->replaceSuccessor(To, NewBBPtr, OrigCount, OrigMispreds);

  NewBB->addSuccessor(To, OrigCount, OrigMispreds);
  NewBB->setExecutionCount(OrigCount);
  NewBB->setIsCold(From->isCold());

  // Update CFI and BB layout with new intermediate BB
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  NewBBs.emplace_back(std::move(NewBB));
  insertBasicBlocks(From, std::move(NewBBs), true, true,
                    /*RecomputeLandingPads=*/false);
  return NewBBPtr;
}

void BinaryFunction::deleteConservativeEdges() {
  // Our goal is to aggressively remove edges from the CFG that we believe are
  // wrong. This is used for instrumentation, where it is safe to remove
  // fallthrough edges because we won't reorder blocks.
  for (auto I = BasicBlocks.begin(), E = BasicBlocks.end(); I != E; ++I) {
    BinaryBasicBlock *BB = *I;
    if (BB->succ_size() != 1 || BB->size() == 0)
      continue;

    auto NextBB = std::next(I);
    MCInst *Last = BB->getLastNonPseudoInstr();
    // Fallthrough is a landing pad? Delete this edge (as long as we don't
    // have a direct jump to it)
    if ((*BB->succ_begin())->isLandingPad() && NextBB != E &&
        *BB->succ_begin() == *NextBB && Last && !BC.MIB->isBranch(*Last)) {
      BB->removeAllSuccessors();
      continue;
    }

    // Look for suspicious calls at the end of BB where gcc may optimize it and
    // remove the jump to the epilogue when it knows the call won't return.
    if (!Last || !BC.MIB->isCall(*Last))
      continue;

    const MCSymbol *CalleeSymbol = BC.MIB->getTargetSymbol(*Last);
    if (!CalleeSymbol)
      continue;

    StringRef CalleeName = CalleeSymbol->getName();
    if (CalleeName != "__cxa_throw@PLT" && CalleeName != "_Unwind_Resume@PLT" &&
        CalleeName != "__cxa_rethrow@PLT" && CalleeName != "exit@PLT" &&
        CalleeName != "abort@PLT")
      continue;

    BB->removeAllSuccessors();
  }
}

bool BinaryFunction::isSymbolValidInScope(const SymbolRef &Symbol,
                                          uint64_t SymbolSize) const {
  // If this symbol is in a different section from the one where the
  // function symbol is, don't consider it as valid.
  if (!getOriginSection()->containsAddress(
          cantFail(Symbol.getAddress(), "cannot get symbol address")))
    return false;

  // Some symbols are tolerated inside function bodies, others are not.
  // The real function boundaries may not be known at this point.
  if (BC.isMarker(Symbol))
    return true;

  // It's okay to have a zero-sized symbol in the middle of non-zero-sized
  // function.
  if (SymbolSize == 0 && containsAddress(cantFail(Symbol.getAddress())))
    return true;

  if (cantFail(Symbol.getType()) != SymbolRef::ST_Unknown)
    return false;

  if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Global)
    return false;

  return true;
}

void BinaryFunction::adjustExecutionCount(uint64_t Count) {
  if (getKnownExecutionCount() == 0 || Count == 0)
    return;

  if (ExecutionCount < Count)
    Count = ExecutionCount;

  double AdjustmentRatio = ((double)ExecutionCount - Count) / ExecutionCount;
  if (AdjustmentRatio < 0.0)
    AdjustmentRatio = 0.0;

  for (BinaryBasicBlock &BB : blocks())
    BB.adjustExecutionCount(AdjustmentRatio);

  ExecutionCount -= Count;
}

BinaryFunction::~BinaryFunction() {
  for (BinaryBasicBlock *BB : BasicBlocks)
    delete BB;
  for (BinaryBasicBlock *BB : DeletedBasicBlocks)
    delete BB;
}

void BinaryFunction::constructDomTree() {
  BDT.reset(new BinaryDominatorTree);
  BDT->recalculate(*this);
}

void BinaryFunction::calculateLoopInfo() {
  if (!hasDomTree())
    constructDomTree();
  // Discover loops.
  BLI.reset(new BinaryLoopInfo());
  BLI->analyze(getDomTree());

  // Traverse discovered loops and add depth and profile information.
  std::stack<BinaryLoop *> St;
  for (auto I = BLI->begin(), E = BLI->end(); I != E; ++I) {
    St.push(*I);
    ++BLI->OuterLoops;
  }

  while (!St.empty()) {
    BinaryLoop *L = St.top();
    St.pop();
    ++BLI->TotalLoops;
    BLI->MaximumDepth = std::max(L->getLoopDepth(), BLI->MaximumDepth);

    // Add nested loops in the stack.
    for (BinaryLoop::iterator I = L->begin(), E = L->end(); I != E; ++I)
      St.push(*I);

    // Skip if no valid profile is found.
    if (!hasValidProfile()) {
      L->EntryCount = COUNT_NO_PROFILE;
      L->ExitCount = COUNT_NO_PROFILE;
      L->TotalBackEdgeCount = COUNT_NO_PROFILE;
      continue;
    }

    // Compute back edge count.
    SmallVector<BinaryBasicBlock *, 1> Latches;
    L->getLoopLatches(Latches);

    for (BinaryBasicBlock *Latch : Latches) {
      auto BI = Latch->branch_info_begin();
      for (BinaryBasicBlock *Succ : Latch->successors()) {
        if (Succ == L->getHeader()) {
          assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
                 "profile data not found");
          L->TotalBackEdgeCount += BI->Count;
        }
        ++BI;
      }
    }

    // Compute entry count.
    L->EntryCount = L->getHeader()->getExecutionCount() - L->TotalBackEdgeCount;

    // Compute exit count.
    SmallVector<BinaryLoop::Edge, 1> ExitEdges;
    L->getExitEdges(ExitEdges);
    for (BinaryLoop::Edge &Exit : ExitEdges) {
      const BinaryBasicBlock *Exiting = Exit.first;
      const BinaryBasicBlock *ExitTarget = Exit.second;
      auto BI = Exiting->branch_info_begin();
      for (BinaryBasicBlock *Succ : Exiting->successors()) {
        if (Succ == ExitTarget) {
          assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
                 "profile data not found");
          L->ExitCount += BI->Count;
        }
        ++BI;
      }
    }
  }
}

void BinaryFunction::updateOutputValues(const BOLTLinker &Linker) {
  if (!isEmitted()) {
    assert(!isInjected() && "injected function should be emitted");
    setOutputAddress(getAddress());
    setOutputSize(getSize());
    return;
  }

  const auto SymbolInfo = Linker.lookupSymbolInfo(getSymbol()->getName());
  assert(SymbolInfo && "Cannot find function entry symbol");
  setOutputAddress(SymbolInfo->Address);
  setOutputSize(SymbolInfo->Size);

  if (BC.HasRelocations || isInjected()) {
    if (hasConstantIsland()) {
      const auto IslandLabelSymInfo =
          Linker.lookupSymbolInfo(getFunctionConstantIslandLabel()->getName());
      assert(IslandLabelSymInfo && "Cannot find function CI symbol");
      setOutputDataAddress(IslandLabelSymInfo->Address);
      for (auto It : Islands->Offsets) {
        const uint64_t OldOffset = It.first;
        BinaryData *BD = BC.getBinaryDataAtAddress(getAddress() + OldOffset);
        if (!BD)
          continue;

        MCSymbol *Symbol = It.second;
        const auto SymInfo = Linker.lookupSymbolInfo(Symbol->getName());
        assert(SymInfo && "Cannot find CI symbol");
        auto &Section = *getCodeSection();
        const auto NewOffset = SymInfo->Address - Section.getOutputAddress();
        BD->setOutputLocation(Section, NewOffset);
      }
    }
    if (isSplit()) {
      for (FunctionFragment &FF : getLayout().getSplitFragments()) {
        ErrorOr<BinarySection &> ColdSection =
            getCodeSection(FF.getFragmentNum());
        // If fragment is empty, cold section might not exist
        if (FF.empty() && ColdSection.getError())
          continue;

        const MCSymbol *ColdStartSymbol = getSymbol(FF.getFragmentNum());
        // If fragment is empty, symbol might have not been emitted
        if (FF.empty() && (!ColdStartSymbol || !ColdStartSymbol->isDefined()) &&
            !hasConstantIsland())
          continue;
        assert(ColdStartSymbol && ColdStartSymbol->isDefined() &&
               "split function should have defined cold symbol");
        const auto ColdStartSymbolInfo =
            Linker.lookupSymbolInfo(ColdStartSymbol->getName());
        assert(ColdStartSymbolInfo && "Cannot find cold start symbol");
        FF.setAddress(ColdStartSymbolInfo->Address);
        FF.setImageSize(ColdStartSymbolInfo->Size);
        if (hasConstantIsland()) {
          const auto SymInfo = Linker.lookupSymbolInfo(
              getFunctionColdConstantIslandLabel()->getName());
          assert(SymInfo && "Cannot find cold CI symbol");
          setOutputColdDataAddress(SymInfo->Address);
        }
      }
    }
  }

  // Update basic block output ranges for the debug info, if we have
  // secondary entry points in the symbol table to update or if writing BAT.
  if (!requiresAddressMap())
    return;

  // AArch64 may have functions that only contains a constant island (no code).
  if (getLayout().block_empty())
    return;

  for (FunctionFragment &FF : getLayout().fragments()) {
    if (FF.empty())
      continue;

    const uint64_t FragmentBaseAddress =
        getCodeSection(isSimple() ? FF.getFragmentNum() : FragmentNum::main())
            ->getOutputAddress();

    BinaryBasicBlock *PrevBB = nullptr;
    for (BinaryBasicBlock *const BB : FF) {
      assert(BB->getLabel()->isDefined() && "symbol should be defined");
      if (!BC.HasRelocations) {
        if (BB->isSplit())
          assert(FragmentBaseAddress == FF.getAddress());
        else
          assert(FragmentBaseAddress == getOutputAddress());
        (void)FragmentBaseAddress;
      }

      // Injected functions likely will fail lookup, as they have no
      // input range. Just assign the BB the output address of the
      // function.
      auto MaybeBBAddress = BC.getIOAddressMap().lookup(BB->getLabel());
      const uint64_t BBAddress = MaybeBBAddress  ? *MaybeBBAddress
                                 : BB->isSplit() ? FF.getAddress()
                                                 : getOutputAddress();
      BB->setOutputStartAddress(BBAddress);

      if (PrevBB) {
        assert(PrevBB->getOutputAddressRange().first <= BBAddress &&
               "Bad output address for basic block.");
        assert((PrevBB->getOutputAddressRange().first != BBAddress ||
                !hasInstructions() || !PrevBB->getNumNonPseudos()) &&
               "Bad output address for basic block.");
        PrevBB->setOutputEndAddress(BBAddress);
      }
      PrevBB = BB;
    }

    PrevBB->setOutputEndAddress(PrevBB->isSplit()
                                    ? FF.getAddress() + FF.getImageSize()
                                    : getOutputAddress() + getOutputSize());
  }

  // Reset output addresses for deleted blocks.
  for (BinaryBasicBlock *BB : DeletedBasicBlocks) {
    BB->setOutputStartAddress(0);
    BB->setOutputEndAddress(0);
  }
}

DebugAddressRangesVector BinaryFunction::getOutputAddressRanges() const {
  DebugAddressRangesVector OutputRanges;

  if (isFolded())
    return OutputRanges;

  if (IsFragment)
    return OutputRanges;

  OutputRanges.emplace_back(getOutputAddress(),
                            getOutputAddress() + getOutputSize());
  if (isSplit()) {
    assert(isEmitted() && "split function should be emitted");
    for (const FunctionFragment &FF : getLayout().getSplitFragments())
      OutputRanges.emplace_back(FF.getAddress(),
                                FF.getAddress() + FF.getImageSize());
  }

  if (isSimple())
    return OutputRanges;

  for (BinaryFunction *Frag : Fragments) {
    assert(!Frag->isSimple() &&
           "fragment of non-simple function should also be non-simple");
    OutputRanges.emplace_back(Frag->getOutputAddress(),
                              Frag->getOutputAddress() + Frag->getOutputSize());
  }

  return OutputRanges;
}

uint64_t BinaryFunction::translateInputToOutputAddress(uint64_t Address) const {
  if (isFolded())
    return 0;

  // If the function hasn't changed return the same address.
  if (!isEmitted())
    return Address;

  if (Address < getAddress())
    return 0;

  // Check if the address is associated with an instruction that is tracked
  // by address translation.
  if (auto OutputAddress = BC.getIOAddressMap().lookup(Address))
    return *OutputAddress;

  // FIXME: #18950828 - we rely on relative offsets inside basic blocks to stay
  //        intact. Instead we can use pseudo instructions and/or annotations.
  const uint64_t Offset = Address - getAddress();
  const BinaryBasicBlock *BB = getBasicBlockContainingOffset(Offset);
  if (!BB) {
    // Special case for address immediately past the end of the function.
    if (Offset == getSize())
      return getOutputAddress() + getOutputSize();

    return 0;
  }

  return std::min(BB->getOutputAddressRange().first + Offset - BB->getOffset(),
                  BB->getOutputAddressRange().second);
}

DebugAddressRangesVector
BinaryFunction::translateInputToOutputRange(DebugAddressRange InRange) const {
  DebugAddressRangesVector OutRanges;

  // The function was removed from the output. Return an empty range.
  if (isFolded())
    return OutRanges;

  // If the function hasn't changed return the same range.
  if (!isEmitted()) {
    OutRanges.emplace_back(InRange);
    return OutRanges;
  }

  if (!containsAddress(InRange.LowPC))
    return OutRanges;

  // Special case of an empty range [X, X). Some tools expect X to be updated.
  if (InRange.LowPC == InRange.HighPC) {
    if (uint64_t NewPC = translateInputToOutputAddress(InRange.LowPC))
      OutRanges.push_back(DebugAddressRange{NewPC, NewPC});
    return OutRanges;
  }

  uint64_t InputOffset = InRange.LowPC - getAddress();
  const uint64_t InputEndOffset =
      std::min(InRange.HighPC - getAddress(), getSize());

  auto BBI = llvm::upper_bound(BasicBlockOffsets,
                               BasicBlockOffset(InputOffset, nullptr),
                               CompareBasicBlockOffsets());
  assert(BBI != BasicBlockOffsets.begin());

  // Iterate over blocks in the input order using BasicBlockOffsets.
  for (--BBI; InputOffset < InputEndOffset && BBI != BasicBlockOffsets.end();
       InputOffset = BBI->second->getEndOffset(), ++BBI) {
    const BinaryBasicBlock &BB = *BBI->second;
    if (InputOffset < BB.getOffset() || InputOffset >= BB.getEndOffset()) {
      LLVM_DEBUG(
          dbgs() << "BOLT-DEBUG: invalid debug address range detected for "
                 << *this << " : [0x" << Twine::utohexstr(InRange.LowPC)
                 << ", 0x" << Twine::utohexstr(InRange.HighPC) << "]\n");
      break;
    }

    // Skip the block if it wasn't emitted.
    if (!BB.getOutputAddressRange().first)
      continue;

    // Find output address for an instruction with an offset greater or equal
    // to /p Offset. The output address should fall within the same basic
    // block boundaries.
    auto translateBlockOffset = [&](const uint64_t Offset) {
      const uint64_t OutAddress = BB.getOutputAddressRange().first + Offset;
      return std::min(OutAddress, BB.getOutputAddressRange().second);
    };

    uint64_t OutLowPC = BB.getOutputAddressRange().first;
    if (InputOffset > BB.getOffset())
      OutLowPC = translateBlockOffset(InputOffset - BB.getOffset());

    uint64_t OutHighPC = BB.getOutputAddressRange().second;
    if (InputEndOffset < BB.getEndOffset()) {
      assert(InputEndOffset >= BB.getOffset());
      OutHighPC = translateBlockOffset(InputEndOffset - BB.getOffset());
    }

    // Check if we can expand the last translated range.
    if (!OutRanges.empty() && OutRanges.back().HighPC == OutLowPC)
      OutRanges.back().HighPC = std::max(OutRanges.back().HighPC, OutHighPC);
    else
      OutRanges.emplace_back(OutLowPC, std::max(OutLowPC, OutHighPC));
  }

  LLVM_DEBUG({
    dbgs() << "BOLT-DEBUG: translated address range " << InRange << " -> ";
    for (const DebugAddressRange &R : OutRanges)
      dbgs() << R << ' ';
    dbgs() << '\n';
  });

  return OutRanges;
}

MCInst *BinaryFunction::getInstructionAtOffset(uint64_t Offset) {
  if (CurrentState == State::Disassembled) {
    auto II = Instructions.find(Offset);
    return (II == Instructions.end()) ? nullptr : &II->second;
  } else if (CurrentState == State::CFG) {
    BinaryBasicBlock *BB = getBasicBlockContainingOffset(Offset);
    if (!BB)
      return nullptr;

    for (MCInst &Inst : *BB) {
      constexpr uint32_t InvalidOffset = std::numeric_limits<uint32_t>::max();
      if (Offset == BC.MIB->getOffsetWithDefault(Inst, InvalidOffset))
        return &Inst;
    }

    if (MCInst *LastInstr = BB->getLastNonPseudoInstr()) {
      if (std::optional<uint32_t> Size = BC.MIB->getSize(*LastInstr)) {
        if (BB->getEndOffset() - Offset == Size) {
          return LastInstr;
        }
      }
    }

    return nullptr;
  } else {
    llvm_unreachable("invalid CFG state to use getInstructionAtOffset()");
  }
}

MCInst *BinaryFunction::getInstructionContainingOffset(uint64_t Offset) {
  assert(CurrentState == State::Disassembled && "Wrong function state");

  if (Offset > Size)
    return nullptr;

  auto II = Instructions.upper_bound(Offset);
  assert(II != Instructions.begin() && "First instruction not at offset 0");
  --II;
  return &II->second;
}

void BinaryFunction::printLoopInfo(raw_ostream &OS) const {
  if (!opts::shouldPrint(*this))
    return;

  OS << "Loop Info for Function \"" << *this << "\"";
  if (hasValidProfile())
    OS << " (count: " << getExecutionCount() << ")";
  OS << "\n";

  std::stack<BinaryLoop *> St;
  for (BinaryLoop *L : *BLI)
    St.push(L);
  while (!St.empty()) {
    BinaryLoop *L = St.top();
    St.pop();

    for (BinaryLoop *Inner : *L)
      St.push(Inner);

    if (!hasValidProfile())
      continue;

    OS << (L->getLoopDepth() > 1 ? "Nested" : "Outer")
       << " loop header: " << L->getHeader()->getName();
    OS << "\n";
    OS << "Loop basic blocks: ";
    ListSeparator LS;
    for (BinaryBasicBlock *BB : L->blocks())
      OS << LS << BB->getName();
    OS << "\n";
    if (hasValidProfile()) {
      OS << "Total back edge count: " << L->TotalBackEdgeCount << "\n";
      OS << "Loop entry count: " << L->EntryCount << "\n";
      OS << "Loop exit count: " << L->ExitCount << "\n";
      if (L->EntryCount > 0) {
        OS << "Average iters per entry: "
           << format("%.4lf", (double)L->TotalBackEdgeCount / L->EntryCount)
           << "\n";
      }
    }
    OS << "----\n";
  }

  OS << "Total number of loops: " << BLI->TotalLoops << "\n";
  OS << "Number of outer loops: " << BLI->OuterLoops << "\n";
  OS << "Maximum nested loop depth: " << BLI->MaximumDepth << "\n\n";
}

bool BinaryFunction::isAArch64Veneer() const {
  if (empty() || hasIslandsInfo())
    return false;

  BinaryBasicBlock &BB = **BasicBlocks.begin();
  for (MCInst &Inst : BB)
    if (!BC.MIB->hasAnnotation(Inst, "AArch64Veneer"))
      return false;

  for (auto I = BasicBlocks.begin() + 1, E = BasicBlocks.end(); I != E; ++I) {
    for (MCInst &Inst : **I)
      if (!BC.MIB->isNoop(Inst))
        return false;
  }

  return true;
}

void BinaryFunction::addRelocation(uint64_t Address, MCSymbol *Symbol,
                                   uint32_t RelType, uint64_t Addend,
                                   uint64_t Value) {
  assert(Address >= getAddress() && Address < getAddress() + getMaxSize() &&
         "address is outside of the function");
  uint64_t Offset = Address - getAddress();
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: addRelocation in "
                    << formatv("{0}@{1:x} against {2}\n", *this, Offset,
                               (Symbol ? Symbol->getName() : "<undef>")));
  bool IsCI = BC.isAArch64() && isInConstantIsland(Address);
  std::map<uint64_t, Relocation> &Rels =
      IsCI ? Islands->Relocations : Relocations;
  if (BC.MIB->shouldRecordCodeRelocation(RelType))
    Rels[Offset] = Relocation{Offset, Symbol, RelType, Addend, Value};
}

} // namespace bolt
} // namespace llvm
