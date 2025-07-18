//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NEW_ALLOCATE_H
#define _LIBCPP___NEW_ALLOCATE_H

#include <__config>
#include <__cstddef/max_align_t.h>
#include <__cstddef/size_t.h>
#include <__new/align_val_t.h>
#include <__new/global_new_delete.h> // for _LIBCPP_HAS_SIZED_DEALLOCATION
#include <__type_traits/type_identity.h>
#include <__utility/element_count.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_CONSTEXPR inline _LIBCPP_HIDE_FROM_ABI bool __is_overaligned_for_new(size_t __align) _NOEXCEPT {
#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
  return __align > __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
  return __align > _LIBCPP_ALIGNOF(max_align_t);
#endif
}

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_CFI _Tp*
__libcpp_allocate(__element_count __n, [[__maybe_unused__]] size_t __align = _LIBCPP_ALIGNOF(_Tp)) {
  size_t __size = static_cast<size_t>(__n) * sizeof(_Tp);
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
    return static_cast<_Tp*>(__builtin_operator_new(__size, static_cast<align_val_t>(__align)));
#endif

  return static_cast<_Tp*>(__builtin_operator_new(__size));
}

#if _LIBCPP_HAS_SIZED_DEALLOCATION
#  define _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(...) __VA_ARGS__
#else
#  define _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(...) /* nothing */
#endif

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI void
__libcpp_deallocate(__type_identity_t<_Tp>* __ptr,
                    __element_count __n,
                    [[__maybe_unused__]] size_t __align = _LIBCPP_ALIGNOF(_Tp)) _NOEXCEPT {
  [[__maybe_unused__]] size_t __size = static_cast<size_t>(__n) * sizeof(_Tp);
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
    return __builtin_operator_delete(
        __ptr _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(, __size), static_cast<align_val_t>(__align));
#endif
  return __builtin_operator_delete(__ptr _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(, __size));
}

#undef _LIBCPP_ONLY_IF_SIZED_DEALLOCATION

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI void __libcpp_deallocate_unsized(
    __type_identity_t<_Tp>* __ptr, [[__maybe_unused__]] size_t __align = _LIBCPP_ALIGNOF(_Tp)) _NOEXCEPT {
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
    return __builtin_operator_delete(__ptr, static_cast<align_val_t>(__align));
#endif
  return __builtin_operator_delete(__ptr);
}
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NEW_ALLOCATE_H
