---
list_title: Build a Bytecode Interpreter
title:   Build a Bytecode Interpreter
layout: post
categories: ["Programing Language"]
---

### Stack-based Machines

Use stack for operands and operators. The result is always on top of the stack.

```
// source code
x = 15
x + 10 -5
```

The bytecode can look something like this

```
// bytecode

push %15
set %0
push %0
push $10
add
push $5
sub
```

### Register-based Machines

Operates on a set of virtual registers, which are just some data storage. The convention is the result is always in the "accumulator" register.