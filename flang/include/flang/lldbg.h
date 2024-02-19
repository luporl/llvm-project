#pragma once

#include "llvm/Support/raw_os_ostream.h"
#include <functional>

#define DBG(...) dbg([&] { __VA_ARGS__; })
#define LLDBG(on) LLDbg dbg(on, __func__)
#define NL '\n'

class LLDbg {
  bool _on;
  bool _first = true;
  const char *_func;

  void printHeader() {
    if (_on && _first) {
      llvm::errs() << "LLL: " << _func << ": ";
      _first = false;
    }
  }

public:
  LLDbg(int on, const char *func) : _on(on), _func(func) {}

  template <typename T> LLDbg &operator<<(T &&v) {
    printHeader();
    if (_on)
      llvm::errs() << v;
    return *this;
  }

  template <> LLDbg &operator<<(char &&c) {
    if (c == NL)
      _first = true;
    if (_on)
      llvm::errs() << c;
    return *this;
  }

  void operator()(std::function<void()> fn) {
    if (_on)
      fn();
  }
};
