; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

@_ZTIi = external dso_local constant ptr
@_ZTIii= external dso_local constant ptr
declare void @foo(ptr)
declare void @vararg_foo(ptr, ...)
declare ptr @bar(ptr)
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @invokeLandingpad
define i32 @invokeLandingpad() personality ptr @__gxx_personality_v0 {
  ; CHECK: %[[a1:[0-9]+]] = llvm.mlir.addressof @_ZTIii : !llvm.ptr
  ; CHECK: %[[a3:[0-9]+]] = llvm.alloca %{{[0-9]+}} x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
  %1 = alloca i8
  ; CHECK: llvm.invoke @foo(%[[a3]]) to ^[[bb1:.*]] unwind ^[[bb4:.*]] : (!llvm.ptr) -> ()
  invoke void @foo(ptr %1) to label %bb1 unwind label %bb4

; CHECK: ^[[bb1]]:
bb1:
  ; CHECK: %{{[0-9]+}} = llvm.invoke @bar(%[[a3]]) to ^[[bb2:.*]] unwind ^[[bb4]] : (!llvm.ptr) -> !llvm.ptr
  %2 = invoke ptr @bar(ptr %1) to label %bb2 unwind label %bb4

; CHECK: ^[[bb2]]:
bb2:
  ; CHECK: llvm.invoke @vararg_foo(%[[a3]], %{{.*}}) to ^[[bb3:.*]] unwind ^[[bb4]] vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32) -> ()
  invoke void (ptr, ...) @vararg_foo(ptr %1, i32 0) to label %bb3 unwind label %bb4

; CHECK: ^[[bb3]]:
bb3:
  ; CHECK: llvm.invoke %{{.*}}(%[[a3]], %{{.*}}) to ^[[bb5:.*]] unwind ^[[bb4]] vararg(!llvm.func<void (ptr, ...)>) : !llvm.ptr, (!llvm.ptr, i32) -> ()
  invoke void (ptr, ...) undef(ptr %1, i32 0) to label %bb5 unwind label %bb4

; CHECK: ^[[bb4]]:
bb4:
  ; CHECK: %{{[0-9]+}} = llvm.landingpad (catch %{{[0-9]+}} : !llvm.ptr) (catch %[[a1]] : !llvm.ptr) (filter %{{[0-9]+}} : !llvm.array<1 x i1>) : !llvm.struct<(ptr, i32)>
  %3 = landingpad { ptr, i32 } catch ptr @_ZTIi catch ptr @_ZTIii
          filter [1 x i1] [i1 1]
  resume { ptr, i32 } %3

; CHECK: ^[[bb5]]:
bb5:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 1
}

declare i32 @foo2()

; CHECK-LABEL: @invokePhi
; CHECK-SAME:            (%[[cond:.*]]: i1) -> i32
define i32 @invokePhi(i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK: llvm.cond_br %[[cond]], ^[[bb1:.*]], ^[[bb2:.*]]
  br i1 %cond, label %call, label %nocall
; CHECK: ^[[bb1]]:
call:
  ; CHECK: %[[invoke:.*]] = llvm.invoke @foo2() to ^[[bb3:.*]] unwind ^[[bb5:.*]] : () -> i32
  %invoke = invoke i32 @foo2() to label %bb0 unwind label %bb1
; CHECK: ^[[bb2]]:
nocall:
  ; CHECK: llvm.br ^[[bb4:.*]](%[[c0]] : i32)
  br label %bb0
; CHECK: ^[[bb3]]:
  ; CHECK: llvm.br ^[[bb4]](%[[invoke]] : i32)
; CHECK: ^[[bb4]](%[[barg:.*]]: i32):
bb0:
  %ret = phi i32 [ 0, %nocall ], [ %invoke, %call ]
  ; CHECK: llvm.return %[[barg]] : i32
  ret i32 %ret
; CHECK: ^[[bb5]]:
bb1:
  ; CHECK: %[[lp:.*]] = llvm.landingpad cleanup : i32
  %resume = landingpad i32 cleanup
  ; CHECK: llvm.resume %[[lp]] : i32
  resume i32 %resume
}

; CHECK-LABEL: @invokePhiComplex
; CHECK-SAME:                   (%[[cond:.*]]: i1) -> i32
define i32 @invokePhiComplex(i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK: %[[c2:.*]] = llvm.mlir.constant(2 : i32) : i32
  ; CHECK: %[[c20:.*]] = llvm.mlir.constant(20 : i32) : i32
  ; CHECK: llvm.cond_br %[[cond]], ^[[bb1:.*]], ^[[bb2:.*]]
  br i1 %cond, label %call, label %nocall
; CHECK: ^[[bb1]]:
call:
  ; CHECK: %[[invoke:.*]] = llvm.invoke @foo2() to ^[[bb3:.*]] unwind ^[[bb5:.*]] : () -> i32
  %invoke = invoke i32 @foo2() to label %bb0 unwind label %bb1
; CHECK: ^[[bb2]]:
nocall:
  ; CHECK: llvm.br ^[[bb4:.*]](%[[c0]], %[[c1]], %[[c2]] : i32, i32, i32)
  br label %bb0
; CHECK: ^[[bb3]]:
  ; CHECK: llvm.br ^[[bb4]](%[[invoke]], %[[c20]], %[[invoke]] : i32, i32, i32)
; CHECK: ^[[bb4]](%[[barg0:.*]]: i32, %[[barg1:.*]]: i32, %[[barg2:.*]]: i32):
bb0:
  %a = phi i32 [ 0, %nocall ], [ %invoke, %call ]
  %b = phi i32 [ 1, %nocall ], [ 20, %call ]
  %c = phi i32 [ 2, %nocall ], [ %invoke, %call ]
  ; CHECK: %[[add0:.*]] = llvm.add %[[barg0]], %[[barg1]] : i32
  ; CHECK: %[[add1:.*]] = llvm.add %[[barg2]], %[[add0]] : i32
  %d = add i32 %a, %b
  %e = add i32 %c, %d
  ; CHECK: llvm.return %[[add1]] : i32
  ret i32 %e
; CHECK: ^[[bb5]]:
bb1:
  ; CHECK: %[[lp:.*]] = llvm.landingpad cleanup : i32
  %resume = landingpad i32 cleanup
  ; CHECK: llvm.resume %[[lp]] : i32
  resume i32 %resume
}

declare void @f0(ptr)
declare void @f1(i32)
declare void @f2({ptr, i32})

; CHECK-LABEL: @landingpad_dominance
define void @landingpad_dominance() personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK:    %[[null:.*]] = llvm.mlir.zero : !llvm.ptr
  ; CHECK:    %[[c1:.*]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK:    %[[undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i32)>
  ; CHECK:    %[[tmpstruct:.*]] = llvm.insertvalue %[[null]], %[[undef]][0] : !llvm.struct<(ptr, i32)>
  ; CHECK:    %[[struct:.*]] = llvm.insertvalue %[[c1]], %[[tmpstruct]][1] : !llvm.struct<(ptr, i32)>
  ; CHECK:    llvm.call @f0(%[[null]]) : (!llvm.ptr) -> ()
  call void @f0(ptr null)
  ; CHECK:    llvm.call @f1(%[[c1]]) : (i32) -> ()
  call void @f1(i32 0)
  ; CHECK:    llvm.invoke @f0(%[[null]]) to ^[[block2:.*]] unwind ^[[block1:.*]] : (!llvm.ptr) -> ()
  invoke void @f0(ptr null)
      to label %exit unwind label %catch

; CHECK:  ^[[block1]]:
catch:
  ; CHECK:    %[[lp:.*]] = llvm.landingpad (catch %[[null]] : !llvm.ptr) : !llvm.struct<(ptr, i32)>
  %lp = landingpad { ptr, i32 } catch ptr null
  ; CHECK:    llvm.call @f2(%[[struct]]) : (!llvm.struct<(ptr, i32)>) -> ()
  call void @f2({ptr, i32} {ptr null, i32 0})
  ; CHECK:    llvm.resume %[[lp]] : !llvm.struct<(ptr, i32)>
  resume {ptr, i32} %lp

; CHECK:  ^[[block2]]:
exit:
  ; CHECK:    llvm.return
  ret void
}

; // -----

declare i32 @__gxx_personality_v0(...)
declare void @foo(ptr)

; CHECK-LABEL: @invokeLandingpad
define i32 @invokeLandingpad(ptr %addr) personality ptr @__gxx_personality_v0 {
entry:
  %1 = alloca i8
  %zzz = load i32, ptr %addr, align 4
  invoke void @foo(ptr %1) to label %bb1 unwind label %bb2

bb1:
  %yyy = load i32, ptr %addr, align 4
  invoke void @foo(ptr %1) to label %bb3 unwind label %bb2

; CHECK: ^bb{{.*}}(%[[ARG:.*]]: i32):
bb2:
  %phi_var = phi i32 [ %zzz, %entry ], [ %yyy, %bb1 ], !dbg !5
  ; CHECK: llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  ; CHECK-NEXT: llvm.intr.dbg.value #di_local_variable #llvm.di_expression<[DW_OP_LLVM_fragment(64, 64)]> = %[[ARG]] : i32
  %3 = landingpad { ptr, i32 } cleanup, !dbg !5
  #dbg_value(i32 %phi_var, !8, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !7)
  br label %bb3

bb3:
  ; CHECK: llvm.return %{{[0-9]+}} : i32
  ret i32 1
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "landingpad.ll", directory: "/")
!3 = distinct !DISubprogram(name: "instruction_loc", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!4 = distinct !DISubprogram(name: "callee", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DILocation(line: 1, column: 2, scope: !3)
!6 = !DILocation(line: 2, column: 2, scope: !3)
!7 = !DILocation(line: 7, column: 4, scope: !4, inlinedAt: !6)
!8 = !DILocalVariable(scope: !4, name: "size")
