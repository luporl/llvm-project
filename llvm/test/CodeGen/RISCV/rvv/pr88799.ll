; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 4
; RUN: llc < %s -mtriple=riscv64-unknown-linux-gnu -mattr=+v | FileCheck %s

define i32 @main() vscale_range(2,2) {
; CHECK-LABEL: main:
; CHECK:       # %bb.0: # %vector.body
; CHECK-NEXT:    lui a0, 1040368
; CHECK-NEXT:    addi a0, a0, -144
; CHECK-NEXT:    vl2re16.v v8, (a0)
; CHECK-NEXT:    vs2r.v v8, (zero)
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    ret
vector.body:
  %0 = load <16 x i16>, ptr getelementptr ([3 x [23 x [23 x i16]]], ptr null, i64 -10593, i64 1, i64 22, i64 0), align 16
  store <16 x i16> %0, ptr null, align 2
  %wide.load = load <vscale x 8 x i16>, ptr getelementptr ([3 x [23 x [23 x i16]]], ptr null, i64 -10593, i64 1, i64 22, i64 0), align 16
  store <vscale x 8 x i16> %wide.load, ptr null, align 2
  ret i32 0
}
