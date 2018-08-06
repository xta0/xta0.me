---
layout: post
list_title: C++ 源码不完全分析-1 | C++ Source code analysis
title: C++ 源码不完全分析-1
---

### _LIBCPP_TYPE_VIS 宏定义

经常看到C++ 源码中，类定义前面有这个宏，例如

```cpp
class _LIBCPP_TYPE_VIS __thread_struct
{
    ...
    void notify_all_at_thread_exit(condition_variable*, mutex*);
    void __make_ready_at_thread_exit(__assoc_sub_state*);
};



#ifndef _LIBCPP_TYPE_VIS
#  if !defined(_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
#    define _LIBCPP_TYPE_VIS __attribute__ ((__visibility__("default")))
#  else
#    define _LIBCPP_TYPE_VIS
#  endif
#endif
#define _LIBCPP_INLINE_VISIBILITY __attribute__ ((__visibility__("hidden"), __always_inline__))
#define _LIBCPP_HIDDEN __attribute__ ((__visibility__("hidden")))
```
这个宏使GCC用来修饰这个类的符号可见性，

与其对应的还有

