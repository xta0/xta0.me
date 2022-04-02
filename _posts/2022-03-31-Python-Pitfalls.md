---
list_title: Python | Pitfalls
title: Coding Conventions
layout: post
categories: ["Python"]
---

### Mutable objects

如果concat两个包含mutable object的list，则结果

```python
x = [[1, 2]]
y = x + x
# y -> [[1, 2],[1, 2]] 
x[0] = [-1, -1]
# y -> [[-1, -1], [-1, -1]] 
```
实际上 `id(y[0]) == id (y0[1]) == id(x)`
