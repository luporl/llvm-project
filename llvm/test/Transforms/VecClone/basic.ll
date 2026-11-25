; RUN: opt -passes=vec-clone -S %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; float dowork(float *a, float b, int k)  declare simd uniform(a) linear(k)
define float @dowork(ptr %a, float %b, i32 %k) #0 {
entry:
  %idx = sext i32 %k to i64
  %p = getelementptr inbounds float, ptr %a, i64 %idx
  %v = load float, ptr %p, align 4
  %r = fadd float %v, %b
  ret float %r
}

; void scale(double *p, double s)  declare simd linear(p:1) uniform(s)
define void @scale(ptr %p, double %s) #1 {
entry:
  %v = load double, ptr %p, align 8
  %m = fmul double %v, %s
  store double %m, ptr %p, align 8
  ret void
}

attributes #0 = { "_ZGVnN4uvl_dowork" "_ZGVnM4uvl_dowork" }
attributes #1 = { "_ZGVnN2l8u_scale" }

; CHECK: @llvm.compiler.used = appending global [3 x ptr] [ptr @_ZGVnM4uvl_dowork, ptr @_ZGVnN4uvl_dowork, ptr @_ZGVnN2l8u_scale]

; CHECK-LABEL: define aarch64_vector_pcs <4 x float> @_ZGVnM4uvl_dowork(ptr %a, <4 x float> %b, i32 %k, <4 x i1> %mask) {
; CHECK:       entry:
; CHECK:         %b.vec = alloca <4 x float>
; CHECK:         store <4 x float> %b, ptr %b.vec
; CHECK:         %ret.vec = alloca <4 x float>
; CHECK:         %mask.vec = alloca <4 x i8>
; CHECK:         [[MASK8:%.*]] = zext <4 x i1> %mask to <4 x i8>
; CHECK:         store <4 x i8> [[MASK8]], ptr %mask.vec
; CHECK:       simd.loop.header:
; CHECK:         %index = phi i32 [ 0, %entry ], [ %index.next, %simd.loop.latch ]
; CHECK:         %b.lane = load float
; CHECK:         %k.linear = add i32 %k
; CHECK:         %mask.lane = load i8
; CHECK:         [[COND:%.*]] = trunc i8 %mask.lane to i1
; CHECK:         br i1 [[COND]], label %[[BODY:.*]], label %simd.loop.latch
; CHECK:       [[BODY]]:
; CHECK:         %idx = sext i32 %k.linear to i64
; CHECK:         %r = fadd float %v, %b.lane
; CHECK:         br label %simd.loop.latch
; CHECK:       simd.loop.latch:
; CHECK:         %ret.lane = phi float [ %r, %[[BODY]] ], [ poison, %simd.loop.header ]
; CHECK:         store float %ret.lane
; CHECK:         %index.next = add nuw i32 %index, 1
; CHECK:         %vl.cond = icmp ult i32 %index.next, 4
; CHECK:         br i1 %vl.cond, label %simd.loop.header, label %simd.end, !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       simd.end:
; CHECK:         %ret = load <4 x float>, ptr %ret.vec
; CHECK:         ret <4 x float> %ret

; CHECK-LABEL: define aarch64_vector_pcs <4 x float> @_ZGVnN4uvl_dowork(ptr %a, <4 x float> %b, i32 %k) {
; CHECK-NOT:     %mask
; CHECK:         br i1 %vl.cond, label %simd.loop.header, label %simd.end, !llvm.loop

; CHECK-LABEL: define aarch64_vector_pcs void @_ZGVnN2l8u_scale(ptr %p, double %s) {
; CHECK:         [[I64:%.*]] = zext i32 %index to i64
; CHECK:         [[OFF:%.*]] = mul i64 [[I64]], 8
; CHECK:         %p.linear = getelementptr i8, ptr %p, i64 [[OFF]]
; CHECK:         load double, ptr %p.linear
; CHECK:         store double %m, ptr %p.linear
; CHECK:         %vl.cond = icmp ult i32 %index.next, 2
; CHECK:       simd.end:
; CHECK-NEXT:    ret void

; CHECK: [[LOOP4]] = distinct !{[[LOOP4]], [[ENABLE:![0-9]+]], [[WIDTH4:![0-9]+]]}
; CHECK: [[ENABLE]] = !{!"llvm.loop.vectorize.enable"}
; CHECK: [[WIDTH4]] = !{!"llvm.loop.vectorize.width", i32 4}
