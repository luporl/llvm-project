//===-- AArch64TargetStreamer.h - AArch64 Target Streamer ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64TARGETSTREAMER_H
#define LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64TARGETSTREAMER_H

#include "AArch64MCAsmInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/AArch64BuildAttributes.h"
#include <cstdint>

namespace {
class AArch64ELFStreamer;
}

namespace llvm {

class AArch64TargetStreamer : public MCTargetStreamer {
public:
  AArch64TargetStreamer(MCStreamer &S);
  ~AArch64TargetStreamer() override;

  void finish() override;
  void emitConstantPools() override;

  /// Callback used to implement the ldr= pseudo.
  /// Add a new entry to the constant pool for the current section and return an
  /// MCExpr that can be used to refer to the constant pool location.
  const MCExpr *addConstantPoolEntry(const MCExpr *, unsigned Size, SMLoc Loc);

  /// Callback used to implement the .ltorg directive.
  /// Emit contents of constant pool for the current section.
  void emitCurrentConstantPool();

  /// Callback used to implement the .note.gnu.property section.
  void emitNoteSection(unsigned Flags, uint64_t PAuthABIPlatform = -1,
                       uint64_t PAuthABIVersion = -1);

  /// Callback used to emit AUTH expressions (e.g. signed
  /// personality function pointer).
  void emitAuthValue(const MCExpr *Expr, uint16_t Discriminator,
                     AArch64PACKey::ID Key, bool HasAddressDiversity);

  /// Callback used to implement the .inst directive.
  virtual void emitInst(uint32_t Inst);

  /// Callback used to implement the .variant_pcs directive.
  virtual void emitDirectiveVariantPCS(MCSymbol *Symbol) {};

  virtual void emitDirectiveArch(StringRef Name) {};
  virtual void emitDirectiveArchExtension(StringRef Name) {};

  virtual void emitARM64WinCFIAllocStack(unsigned Size) {}
  virtual void emitARM64WinCFISaveR19R20X(int Offset) {}
  virtual void emitARM64WinCFISaveFPLR(int Offset) {}
  virtual void emitARM64WinCFISaveFPLRX(int Offset) {}
  virtual void emitARM64WinCFISaveReg(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveRegX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveRegP(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveRegPX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveLRPair(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveFReg(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveFRegX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveFRegP(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveFRegPX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISetFP() {}
  virtual void emitARM64WinCFIAddFP(unsigned Size) {}
  virtual void emitARM64WinCFINop() {}
  virtual void emitARM64WinCFISaveNext() {}
  virtual void emitARM64WinCFIPrologEnd() {}
  virtual void emitARM64WinCFIEpilogStart() {}
  virtual void emitARM64WinCFIEpilogEnd() {}
  virtual void emitARM64WinCFITrapFrame() {}
  virtual void emitARM64WinCFIMachineFrame() {}
  virtual void emitARM64WinCFIContext() {}
  virtual void emitARM64WinCFIECContext() {}
  virtual void emitARM64WinCFIClearUnwoundToCall() {}
  virtual void emitARM64WinCFIPACSignLR() {}
  virtual void emitARM64WinCFISaveAnyRegI(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegIP(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegD(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegDP(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegQ(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegQP(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegIX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegIPX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegDX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegDPX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegQX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISaveAnyRegQPX(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFIAllocZ(int Offset) {}
  virtual void emitARM64WinCFISaveZReg(unsigned Reg, int Offset) {}
  virtual void emitARM64WinCFISavePReg(unsigned Reg, int Offset) {}

  /// Build attributes implementation
  virtual void
  emitAttributesSubsection(StringRef VendorName,
                          AArch64BuildAttributes::SubsectionOptional IsOptional,
                          AArch64BuildAttributes::SubsectionType ParameterType);
  virtual void emitAttribute(StringRef VendorName, unsigned Tag, unsigned Value,
                             std::string String);
  void activateAttributesSubsection(StringRef VendorName);
  std::unique_ptr<MCELFStreamer::AttributeSubSection>
  getActiveAttributesSubsection();
  std::unique_ptr<MCELFStreamer::AttributeSubSection>
  getAttributesSubsectionByName(StringRef Name);
  void
  insertAttributeInPlace(const MCELFStreamer::AttributeItem &Attr,
                         MCELFStreamer::AttributeSubSection &AttSubSection);

  SmallVector<MCELFStreamer::AttributeSubSection, 64> AttributeSubSections;

private:
  std::unique_ptr<AssemblerConstantPools> ConstantPools;
};

class AArch64TargetELFStreamer : public AArch64TargetStreamer {
private:
  AArch64ELFStreamer &getStreamer();

  MCSection *AttributeSection = nullptr;

  /// Build attributes implementation
  void emitAttributesSubsection(
      StringRef VendorName,
      AArch64BuildAttributes::SubsectionOptional IsOptional,
      AArch64BuildAttributes::SubsectionType ParameterType) override;
  void emitAttribute(StringRef VendorName, unsigned Tag, unsigned Value,
                     std::string String) override;
  void emitInst(uint32_t Inst) override;
  void emitDirectiveVariantPCS(MCSymbol *Symbol) override;
  void finish() override;

public:
  AArch64TargetELFStreamer(MCStreamer &S) : AArch64TargetStreamer(S) {}
};

class AArch64TargetWinCOFFStreamer : public llvm::AArch64TargetStreamer {
public:
  AArch64TargetWinCOFFStreamer(llvm::MCStreamer &S)
    : AArch64TargetStreamer(S) {}

  // The unwind codes on ARM64 Windows are documented at
  // https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
  void emitARM64WinCFIAllocStack(unsigned Size) override;
  void emitARM64WinCFISaveR19R20X(int Offset) override;
  void emitARM64WinCFISaveFPLR(int Offset) override;
  void emitARM64WinCFISaveFPLRX(int Offset) override;
  void emitARM64WinCFISaveReg(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveRegX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveRegP(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveRegPX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveLRPair(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveFReg(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveFRegX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveFRegP(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveFRegPX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISetFP() override;
  void emitARM64WinCFIAddFP(unsigned Size) override;
  void emitARM64WinCFINop() override;
  void emitARM64WinCFISaveNext() override;
  void emitARM64WinCFIPrologEnd() override;
  void emitARM64WinCFIEpilogStart() override;
  void emitARM64WinCFIEpilogEnd() override;
  void emitARM64WinCFITrapFrame() override;
  void emitARM64WinCFIMachineFrame() override;
  void emitARM64WinCFIContext() override;
  void emitARM64WinCFIECContext() override;
  void emitARM64WinCFIClearUnwoundToCall() override;
  void emitARM64WinCFIPACSignLR() override;
  void emitARM64WinCFISaveAnyRegI(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegIP(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegD(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegDP(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegQ(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegQP(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegIX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegIPX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegDX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegDPX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegQX(unsigned Reg, int Offset) override;
  void emitARM64WinCFISaveAnyRegQPX(unsigned Reg, int Offset) override;
  void emitARM64WinCFIAllocZ(int Offset) override;
  void emitARM64WinCFISaveZReg(unsigned Reg, int Offset) override;
  void emitARM64WinCFISavePReg(unsigned Reg, int Offset) override;

private:
  void emitARM64WinUnwindCode(unsigned UnwindCode, int Reg, int Offset);
};

MCTargetStreamer *
createAArch64ObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

MCTargetStreamer *createAArch64NullTargetStreamer(MCStreamer &S);

} // end namespace llvm

#endif
