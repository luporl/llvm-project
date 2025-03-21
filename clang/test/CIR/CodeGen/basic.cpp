// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @f1() -> !cir.int<s, 32>
// CHECK:    %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i"] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:    cir.return %[[I]] : !cir.int<s, 32>

int f2() {
  const int i = 2;
  return i;
}

// CHECK: cir.func @f2() -> !cir.int<s, 32>
// CHECK:    %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", init, const] {alignment = 4 : i64}
// CHECK:    %[[TWO:.*]] = cir.const #cir.int<2> : !cir.int<s, 32>
// CHECK:    cir.store %[[TWO]], %[[I_PTR]] : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:    cir.return %[[I]] : !cir.int<s, 32>

int f3(int i) {
  return i;
}

// CHECK: cir.func @f3(%[[ARG:.*]]: !cir.int<s, 32> loc({{.*}})) -> !cir.int<s, 32>
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", init] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:   cir.return %[[ARG_VAL]] : !cir.int<s, 32>

int f4(const int i) {
  return i;
}

// CHECK: cir.func @f4(%[[ARG:.*]]: !cir.int<s, 32> loc({{.*}})) -> !cir.int<s, 32>
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", init, const] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:   cir.return %[[ARG_VAL]] : !cir.int<s, 32>
