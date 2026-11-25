; End-to-end: VecClone is part of the default pipelines and its loops are
; vectorized at O2.
; RUN: opt -enable-openmp-vfabi-mappings -passes='default<O2>' -S %s | FileCheck %s
; RUN: opt -enable-openmp-vfabi-mappings -passes='default<O0>' -S %s | FileCheck %s --check-prefix=O0

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define float @f(float %x, float %y) #0 {
entry:
  %m = fmul float %x, %y
  %a = fadd float %m, 1.0
  ret float %a
}

attributes #0 = { "_ZGVnN4vv_f" "_ZGVnM4vv_f" }

; Masked and unmasked variants are fully vectorized.
; CHECK-LABEL: define aarch64_vector_pcs <4 x float> @_ZGVnM4vv_f(<4 x float> %x, <4 x float> %y, <4 x i1> %mask)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[M:%.*]] = fmul <4 x float> %x, %y
; CHECK-NEXT:    [[A:%.*]] = fadd <4 x float> [[M]], splat (float 1.000000e+00)
; CHECK-NEXT:    ret <4 x float> [[A]]
; CHECK-LABEL: define aarch64_vector_pcs <4 x float> @_ZGVnN4vv_f(<4 x float> %x, <4 x float> %y)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[M:%.*]] = fmul <4 x float> %x, %y
; CHECK-NEXT:    [[A:%.*]] = fadd <4 x float> [[M]], splat (float 1.000000e+00)
; CHECK-NEXT:    ret <4 x float> [[A]]

; At O0 the variants are still defined (scalar loop).
; O0-DAG: define aarch64_vector_pcs <4 x float> @_ZGVnM4vv_f(
; O0-DAG: define aarch64_vector_pcs <4 x float> @_ZGVnN4vv_f(
