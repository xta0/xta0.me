---
layout: post
list_title: UIView的frame和bounds | Frame & Bounds in iOS
title: UIView的frame和bounds
categories: [iOS]
---

## Frame

<em> Update@2015/01/08 </em>

我们知道View的frame是针对其Superview坐标系的位置，关于frame的计算方法如下: 

```objective-c
view.frame.origin.x = center.x - 1/2 * bounds.size.width;   
view.frame.origin.y = center.y - 1/2 * bounds.size.height;  
view.frame.size.width = bounds.size.width;
view.frame.size.height = bounds.size.height;  
```

从上面的公式我们能看出，view的frame受到它的center和bounds的size的约束。此外还有两个影响frame的因素是transform和layer的`anchorPoint`但是我们这里不讨论它们带来的影响。


### 改变bounds的origin，对subview的影响：

我们往往忽略的一点是对`bounds.origin`的理解，因为通常情况下它的值都为`(0,0)`，但是如果这个值不为0会怎样呢?

``````objective-c
UIView* v  = [[UIView alloc]initWithFrame:CGRectMake(0, 0, 100, 100)];
v .backgroundColor = [UIColor yellowColor];
[self.view addSubview:v];
v .bounds = CGRectMake(-20, -20, 100,100);

UIView* subv = [[UIView alloc]initWithFrame:CGRectMake(0, 0, 50, 50)];
subv.backgroundColor = [UIColor redColor];
[v  addSubview:subv];
```

结果是这个样子的:

![](/assets/images/2011/06/bounds1.png)

- 由于`v`的`bounds.size`没发生变化，`center`也没法生变化，因此`v`的`frame`不会发生变化，它相对父类的位置不会变化。因此`bounds.origin`根本不影响View的`frame`
- 由于`v`的`bounds.origin`发生了变化，相当于`v`自身的坐标系改变了，原点变成了`(-20,-20)`，那么`(0,0)`点的位置就变了，那么它的subView:`subv`的位置也会发生变化。因此`bounds.origin`仅仅影响View内部的坐标系进而影响Subview的位置。


### 改变bounds的origin和size，对frame的影响：

我们如果同时改变bounds.size和bounds.origin会怎样呢?

``````objective-c
v .bounds = CGRectMake(-20, -20, 50,50);
```

结果是这个样子的:

![Alt text](/assets/images/2011/06/bounds2.png)

- 由于`v`的`center`没有发生变化，但是`v`的`bounds.size`缩小了1倍，因此`v`的`frame`发生了变化，根据公式，变成了`{25,25,50,50}`
- 由于`v`的`bounds.origin`不会影响frame，因此影响的还是`v`的subView：`subv`的位置。




