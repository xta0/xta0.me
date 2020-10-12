---
list_title: CppCon 2020 Notes
title: CppCon 2020 Notes
layout: post
categories: ["C++"]
---

## Back to Basics

### Move Semantics

- RValue References Guideline: **No rvalue reference to const type**
    - Use a non-const rvalue reference instead.
    - Most uses of rvalue references modify the object being referenced
    - Most rvalues are not const