---
layout: post
title: Stack Recursion
---

```python
def hanoi(n,src,mid,dst):
    print("move",n,"from",src,"to",dst)
    if n == 1:
        return
    else
        move(n-1,src,dst,mid)
        move(1,src,mid,dst)
        move(n-1,mid,src,dst)
```