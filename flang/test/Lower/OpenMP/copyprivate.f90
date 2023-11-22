! Test COPYPRIVATE.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_tp
!CHECK:       %[[SYNC_VAR_ADDR:.*]] = fir.alloca f32 {bindc_name = "a", pinned, uniq_name = "_QFtest_tpEa"}
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %[[SYNC_VAR_ADDR]] {uniq_name = "_QFtest_tpEa"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:       omp.single {
!CHECK:         hlfir.assign %{{.*}} to %[[SYNC_VAR]]#0 temporary_lhs : f32, !fir.ref<f32>
!CHECK-NEXT:    omp.terminator
!CHECK-NEXT:  }
!CHECK-NEXT:  %[[TMP:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<f32>
!CHECK-NEXT:  omp.barrier
!CHECK-NEXT:  hlfir.assign %[[TMP]] to %{{.*}}#1 temporary_lhs : f32, !fir.ref<f32>
!CHECK-NEXT:  omp.barrier
subroutine test_tp()
  real(4), save :: a = 2.5
  !$omp threadprivate(a)

  !$omp single
  a = 1.5
  !$omp end single copyprivate(a)
end subroutine

!CHECK-LABEL: func @_QPtest_priv
!CHECK:       %[[ORIG_VAR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_privEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_privEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.parallel {
!CHECK:         omp.single {
!CHECK:           hlfir.assign %{{.*}} to %[[SYNC_VAR]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK-NEXT:    %[[TMP:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    hlfir.assign %[[TMP]] to %{{.*}}#1 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK:       }
subroutine test_priv()
  integer :: i

  i = 11
  !$omp parallel firstprivate(i)
  !$omp single
  i = i + 1
  !$omp end single copyprivate(i)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_array
!CHECK:       %[[ORIG_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest_arrayEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!CHECK:       omp.parallel {
!CHECK:         omp.single {
!CHECK:           hlfir.assign %{{.*}}#1 to %[[SYNC_VAR]]#0 temporary_lhs : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK-NEXT:    hlfir.assign %[[SYNC_VAR]]#0 to %{{.*}}#1 temporary_lhs : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
!CHECK-NEXT:    omp.barrier
!CHECK:       }
subroutine test_array()
  integer :: a(10), i

  a = -1
  !$omp parallel firstprivate(a)
  !$omp single
  do i = 1, 5
    a(i) = i * 10
  end do
  !$omp end single copyprivate(a)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_type
!CHECK:       %[[ORIG_VAR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_typeEt"} : (!fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>)
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_typeEt"} : (!fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>)
!CHECK:       omp.parallel {
!CHECK:         omp.single {
!CHECK:           hlfir.assign %{{.*}}#1 to %[[SYNC_VAR]]#0 temporary_lhs : !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK-NEXT:    hlfir.assign %[[SYNC_VAR]]#0 to %{{.*}}#1 temporary_lhs : !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QFtest_typeTty{i:i32,r:f32,a:!fir.array<10xi32>}>>
!CHECK-NEXT:    omp.barrier
!CHECK:       }
subroutine test_type()
  type ty
    integer :: i
    real :: r
    integer, dimension(10) :: a
  end type

  integer :: i
  type(ty) :: t

  t%i = -1
  t%r = -1.5
  t%a = -1
  !$omp parallel firstprivate(t)
  !$omp single
  t%i = 42
  t%r = 3.14
  do i = 1, 5
    t%a(i) = i * 10
  end do
  !$omp end single copyprivate(t)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_multi
!CHECK:       %[[I_SYNC_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFtest_multiEi"}
!CHECK:       %[[I_SYNC:.*]]:2 = hlfir.declare %[[I_SYNC_ADDR]] {uniq_name = "_QFtest_multiEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[J_SYNC_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFtest_multiEj"}
!CHECK:       %[[J_SYNC:.*]]:2 = hlfir.declare %[[J_SYNC_ADDR]] {uniq_name = "_QFtest_multiEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[K_SYNC_ADDR:.*]] = fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFtest_multiEk"}
!CHECK:       %[[K_SYNC:.*]]:2 = hlfir.declare %[[K_SYNC_ADDR]] {uniq_name = "_QFtest_multiEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.parallel {
!CHECK:         omp.single {
!CHECK:           %[[I:.*]] = fir.load %{{.*}}#1 : !fir.ref<i32>
!CHECK:           hlfir.assign %[[I]] to %[[I_SYNC]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:           %[[J:.*]] = fir.load %{{.*}}#1 : !fir.ref<i32>
!CHECK:           hlfir.assign %[[J]] to %[[J_SYNC]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:           %[[K:.*]] = fir.load %{{.*}}#1 : !fir.ref<i32>
!CHECK:           hlfir.assign %[[K]] to %[[K_SYNC]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK-NEXT:    %[[K:.*]] = fir.load %[[K_SYNC]]#0 : !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    hlfir.assign %[[K]] to %{{.*}}#1 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    %[[J:.*]] = fir.load %[[J_SYNC]]#0 : !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    hlfir.assign %[[J]] to %{{.*}}#1 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    %[[I:.*]] = fir.load %[[I_SYNC]]#0 : !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK-NEXT:    hlfir.assign %[[I]] to %{{.*}}#1 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    omp.barrier
!CHECK:       }
subroutine test_multi()
  integer, save :: i, j, k
  !$omp threadprivate(i, j, k)

  i = 11
  j = 12
  k = 13
  !$omp parallel
  !$omp single
  i = 21
  j = 22
  k = 23
  !$omp end single copyprivate(i, j, k)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_alloc
!CHECK:       %[[SYNC_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", pinned, uniq_name = "_QFtest_allocEa"}
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %[[SYNC_ADDR]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK-NEXT:  omp.parallel {
!CHECK:         omp.single {
!CHECK:           fir.if %{{.*}} {
!CHECK:             %[[TMP0:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {fir.must_be_heap = true, uniq_name = "_QFtest_allocEa.alloc"}
!CHECK:             %[[TMP1:.*]] = fir.embox %[[TMP0]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:             fir.store %[[TMP1]] to %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           } else {
!CHECK:             %[[TMP2:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!CHECK:             %[[TMP3:.*]] = fir.embox %[[TMP2]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:             fir.store %[[TMP3]] to %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           }
!CHECK:           %[[TMP4:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           fir.if %{{.*}} {
!CHECK:             hlfir.assign %{{.*}} to %[[TMP4]] temporary_lhs : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:           }
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK:         fir.if %{{.*}} {
!CHECK:           %[[TMP5:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK-NEXT:      omp.barrier
!CHECK-NEXT:      hlfir.assign %[[TMP5]] to %{{.*}} temporary_lhs : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK-NEXT:      omp.barrier
!CHECK:         }
!CHECK:         fir.if %{{.*}} {
!CHECK:           %[[TMP6:.*]] = fir.load %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           %[[TMP7:.*]] = fir.box_addr %[[TMP6]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!CHECK:           fir.freemem %[[TMP7]] : !fir.heap<!fir.array<?xi32>>
!CHECK:         }
subroutine test_alloc()
  integer, allocatable :: a(:)
  integer :: i

  allocate(a(10))
  a = -1
  !$omp parallel firstprivate(a)
  !$omp single
  do i = 1, 5
    a(i) = i * 10
  end do
  !$omp end single copyprivate(a)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_alloc_tp
!CHECK:       %[[SYNC_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", pinned, uniq_name = "_QFtest_alloc_tpEa"}
!CHECK:       %[[SYNC_VAR:.*]]:2 = hlfir.declare %[[SYNC_ADDR]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_tpEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK-NEXT:  omp.parallel {
!CHECK:         omp.single {
!CHECK:           fir.if %{{.*}} {
!CHECK:             %[[TMP0:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {fir.must_be_heap = true, uniq_name = "_QFtest_alloc_tpEa.alloc"}
!CHECK:             %[[TMP1:.*]] = fir.embox %[[TMP0]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:             fir.store %[[TMP1]] to %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           } else {
!CHECK:             %[[TMP2:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!CHECK:             %[[TMP3:.*]] = fir.embox %[[TMP2]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:             fir.store %[[TMP3]] to %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           }
!CHECK:           %[[TMP4:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           fir.if %{{.*}} {
!CHECK:             hlfir.assign %{{.*}} to %[[TMP4]] temporary_lhs : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK:           }
!CHECK-NEXT:      omp.terminator
!CHECK-NEXT:    }
!CHECK:         fir.if %{{.*}} {
!CHECK:           %[[TMP5:.*]] = fir.load %[[SYNC_VAR]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK-NEXT:      omp.barrier
!CHECK-NEXT:      hlfir.assign %[[TMP5]] to %{{.*}} temporary_lhs : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK-NEXT:      omp.barrier
!CHECK:         }
!CHECK:         fir.if %{{.*}} {
!CHECK:           %[[TMP6:.*]] = fir.load %[[SYNC_VAR]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           %[[TMP7:.*]] = fir.box_addr %[[TMP6]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!CHECK:           fir.freemem %[[TMP7]] : !fir.heap<!fir.array<?xi32>>
!CHECK:         }
subroutine test_alloc_tp()
  integer, save, allocatable :: a(:)
  !$omp threadprivate(a)
  integer :: i

  !$omp parallel
  allocate(a(10))
  a = -1
  !$omp single
  do i = 1, 5
    a(i) = i * 10
  end do
  !$omp end single copyprivate(a)
  !$omp end parallel
end subroutine
