; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -global-isel=0                    -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=X64,SDAG-X64
; RUN: llc < %s -fast-isel -fast-isel-abort=1     -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=X64,FAST-X64
; RUN: llc < %s -global-isel -global-isel-abort=1 -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=X64,GISEL-X64
; RUN: llc < %s -global-isel=0                    -mtriple=i686-linux-gnu | FileCheck %s --check-prefixes=X86,DAG-X86,SDAG-X86
; RUN: llc < %s -fast-isel -fast-isel-abort=1     -mtriple=i686-linux-gnu | FileCheck %s --check-prefixes=X86,DAG-X86,FAST-X86
; RUN: llc < %s -global-isel -global-isel-abort=1 -mtriple=i686-linux-gnu | FileCheck %s --check-prefixes=X86,GISEL-X86

define i8 @test_srem_i8(i8 %arg1, i8 %arg2) nounwind {
; SDAG-X64-LABEL: test_srem_i8:
; SDAG-X64:       # %bb.0:
; SDAG-X64-NEXT:    movsbl %dil, %eax
; SDAG-X64-NEXT:    idivb %sil
; SDAG-X64-NEXT:    movsbl %ah, %eax
; SDAG-X64-NEXT:    # kill: def $al killed $al killed $eax
; SDAG-X64-NEXT:    retq
;
; FAST-X64-LABEL: test_srem_i8:
; FAST-X64:       # %bb.0:
; FAST-X64-NEXT:    movsbl %dil, %eax
; FAST-X64-NEXT:    idivb %sil
; FAST-X64-NEXT:    shrw $8, %ax
; FAST-X64-NEXT:    # kill: def $al killed $al killed $ax
; FAST-X64-NEXT:    retq
;
; GISEL-X64-LABEL: test_srem_i8:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    movsbl %dil, %eax
; GISEL-X64-NEXT:    idivb %sil
; GISEL-X64-NEXT:    shrw $8, %ax
; GISEL-X64-NEXT:    # kill: def $al killed $al killed $ax
; GISEL-X64-NEXT:    retq
;
; SDAG-X86-LABEL: test_srem_i8:
; SDAG-X86:       # %bb.0:
; SDAG-X86-NEXT:    movsbl {{[0-9]+}}(%esp), %eax
; SDAG-X86-NEXT:    idivb {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    movsbl %ah, %eax
; SDAG-X86-NEXT:    # kill: def $al killed $al killed $eax
; SDAG-X86-NEXT:    retl
;
; FAST-X86-LABEL: test_srem_i8:
; FAST-X86:       # %bb.0:
; FAST-X86-NEXT:    movsbl {{[0-9]+}}(%esp), %eax
; FAST-X86-NEXT:    idivb {{[0-9]+}}(%esp)
; FAST-X86-NEXT:    movb %ah, %al
; FAST-X86-NEXT:    retl
;
; GISEL-X86-LABEL: test_srem_i8:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    cbtw
; GISEL-X86-NEXT:    movzbl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    idivb %cl
; GISEL-X86-NEXT:    movb %ah, %al
; GISEL-X86-NEXT:    retl
  %ret = srem i8 %arg1, %arg2
  ret i8 %ret
}

define i16 @test_srem_i16(i16 %arg1, i16 %arg2) nounwind {
; X64-LABEL: test_srem_i16:
; X64:       # %bb.0:
; X64-NEXT:    movl %edi, %eax
; X64-NEXT:    # kill: def $ax killed $ax killed $eax
; X64-NEXT:    cwtd
; X64-NEXT:    idivw %si
; X64-NEXT:    movl %edx, %eax
; X64-NEXT:    retq
;
; DAG-X86-LABEL: test_srem_i16:
; DAG-X86:       # %bb.0:
; DAG-X86-NEXT:    movzwl {{[0-9]+}}(%esp), %eax
; DAG-X86-NEXT:    cwtd
; DAG-X86-NEXT:    idivw {{[0-9]+}}(%esp)
; DAG-X86-NEXT:    movl %edx, %eax
; DAG-X86-NEXT:    retl
;
; GISEL-X86-LABEL: test_srem_i16:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    movzwl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movzwl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    # kill: def $ax killed $ax killed $eax
; GISEL-X86-NEXT:    cwtd
; GISEL-X86-NEXT:    idivw %cx
; GISEL-X86-NEXT:    movl %edx, %eax
; GISEL-X86-NEXT:    retl
  %ret = srem i16 %arg1, %arg2
  ret i16 %ret
}

define i32 @test_srem_i32(i32 %arg1, i32 %arg2) nounwind {
; X64-LABEL: test_srem_i32:
; X64:       # %bb.0:
; X64-NEXT:    movl %edi, %eax
; X64-NEXT:    cltd
; X64-NEXT:    idivl %esi
; X64-NEXT:    movl %edx, %eax
; X64-NEXT:    retq
;
; X86-LABEL: test_srem_i32:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    cltd
; X86-NEXT:    idivl {{[0-9]+}}(%esp)
; X86-NEXT:    movl %edx, %eax
; X86-NEXT:    retl
  %ret = srem i32 %arg1, %arg2
  ret i32 %ret
}

define i64 @test_srem_i64(i64 %arg1, i64 %arg2) nounwind {
; X64-LABEL: test_srem_i64:
; X64:       # %bb.0:
; X64-NEXT:    movq %rdi, %rax
; X64-NEXT:    cqto
; X64-NEXT:    idivq %rsi
; X64-NEXT:    movq %rdx, %rax
; X64-NEXT:    retq
;
; DAG-X86-LABEL: test_srem_i64:
; DAG-X86:       # %bb.0:
; DAG-X86-NEXT:    subl $12, %esp
; DAG-X86-NEXT:    pushl {{[0-9]+}}(%esp)
; DAG-X86-NEXT:    pushl {{[0-9]+}}(%esp)
; DAG-X86-NEXT:    pushl {{[0-9]+}}(%esp)
; DAG-X86-NEXT:    pushl {{[0-9]+}}(%esp)
; DAG-X86-NEXT:    calll __moddi3
; DAG-X86-NEXT:    addl $28, %esp
; DAG-X86-NEXT:    retl
;
; GISEL-X86-LABEL: test_srem_i64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    pushl %esi
; GISEL-X86-NEXT:    subl $24, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %edx
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %esi
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    movl %edx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    movl %esi, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll __moddi3
; GISEL-X86-NEXT:    addl $24, %esp
; GISEL-X86-NEXT:    popl %esi
; GISEL-X86-NEXT:    retl
  %ret = srem i64 %arg1, %arg2
  ret i64 %ret
}
