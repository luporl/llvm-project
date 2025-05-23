//===-- flang/lib/Semantics/openmp-dsa.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DSA_H_
#define FORTRAN_SEMANTICS_OPENMP_DSA_H_

#include "flang/Semantics/symbol.h"

static inline Fortran::semantics::Symbol::Flags GetSymbolDSA(
    const Fortran::semantics::Symbol &symbol) {
  using namespace Fortran::semantics;
  Symbol::Flags privateFlags{Symbol::Flag::OmpPrivate,
    Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate};
  Symbol::Flags dsaFlags{privateFlags |
    Symbol::Flags{Symbol::Flag::OmpShared, Symbol::Flag::OmpThreadprivate}};

  Symbol::Flags dsa{symbol.flags() & privateFlags};
  if (dsa.any()) {
    return dsa;
  }
  dsa = symbol.flags() & Symbol::Flags{Symbol::Flag::OmpShared};
  if (dsa.any()) {
    return dsa;
  }
  // If no DSA are set use those from the host associated symbol, if any.
  if ((symbol.flags() & dsaFlags).none()) {
    if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
      return GetSymbolDSA(details->symbol());
    }
  }
  return {};
}

#endif // FORTRAN_SEMANTICS_OPENMP_DSA_H_
