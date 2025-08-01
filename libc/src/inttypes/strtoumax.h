//===-- Implementation header for strtoumax ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_INTTYPES_STRTOUMAX_H
#define LLVM_LIBC_SRC_INTTYPES_STRTOUMAX_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

uintmax_t strtoumax(const char *__restrict str, char **__restrict str_end,
                    int base);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_INTTYPES_STRTOUMAX_H
