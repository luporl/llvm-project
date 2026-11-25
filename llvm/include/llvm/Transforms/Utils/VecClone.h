//===- VecClone.h - Synthesize OpenMP declare simd vector variants -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For every function definition carrying `_ZGV*` (OpenMP declare simd) vector
// variant attributes, create a definition of each vector variant whose body is
// the scalar body wrapped in a VF-trip-count loop marked for vectorization.
//
// Based on https://reviews.llvm.org/D22792.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_VECCLONE_H
#define LLVM_TRANSFORMS_UTILS_VECCLONE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class VecClonePass : public RequiredPassInfoMixin<VecClonePass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_VECCLONE_H
