---
updated: "2016-12-20"
layout: post
title: 理解AutoLayout
categories: [iOS]
---


## AutoLayout

### AutoResizing

在`AutoLayout`技术出现之前，如果想要实现view大小和位置的相对变化，我们有两种有两种方式，一种是使用手动的计算的方式，或者使用`AutoResizing`，也叫`Spring & Struts`。这两种方式都有优点也都有局限性。手动布局的方式理论上可以应对各种复杂的布局变化，但缺点是代码复杂，不易维护和修改；`AutoResizing`适用于view的尺寸跟随父view变化的场景非常方便，但是对于需要动态调整view间距的场景则不够灵活（例如，两个并排的view，当一个view变宽时，可能会遮住另一个view，此时`AutoResizing`没有办法自动调整二者的间距）

### StackView

StackView是iOS 9引入的一种自动布局技术，它的用法类似于CSS中的FlexBox，通过容器来约束子View的位置。和FlexBox类似，StackView是一种弹性容器，具有水平和竖直两种布局方向。在布局方向上StackView可以通过`spacing`来控制子元素间距，通过`distribution`控制空白区域的分割方式；在垂直布局的方向上可以通过`Alignment`来控制元素的对齐方式，其属性值有`leading`，`center`和`Trailing`


- `Intrinsic Content Size`表示view自身大小，和view展示的内容相关
- `Content Hugging` 控制在布局方向上，当剩余空间充足时，哪个view会占用剩余空间，类似`FlexGrow`，在Autolayout中，这个值较低的view会grow，当该值相同时，在布局方向上的第一个view会grow

- `Compression Resistance` 控制在布局方向上，当剩余空间不足时，哪个view将被压缩，类似`FlexShrink`，在Autolayout中，这个值较低的view会被shrink，当该值相同时，在布局方向上的第一个view会shrink

和AutoLayout一样，StackView的优点是对于使用Storyboard的场景非常好用，如果使用代码布局则非常低效

### Autolayout Concept



### Introduce to AutoLayout 

### LayoutConstraint的公式

> item1.attribute1 = multiplier x item2.attribute2 + constant

如何描述：Button.centerX = Superview.centerX:

```objc
	
 NSLayoutConstraint* constraint1 = [NSLayoutConstraint constraintWithItem:self.button
                            attribute:NSLayoutAttributeCenterX
                            relatedBy:NSLayoutRelationEqual
                               toItem:self.button.superview
                            attribute:NSLayoutAttributeCenterX
                           multiplier:1.0
                             constant:0.0];
[self.view addConstraint:constraint1];
	
```
这个例子中，`item1`是`self.button`，`item2`是`self.button.superView`，`attribute1`是`button.centerX`,`attribuet2`是`self.button.superView.centerX`。整个的意思是将`self.button`横向居中。

同理，可以描述`self.button.paddingBottom`：

```objc

NSLayoutConstraint* constraint2 = [NSLayoutConstraint constraintWithItem:self.button
                                                    attribute:NSLayoutAttributeBottom
                                                    relatedBy:NSLayoutRelationEqual
                                                       toItem:self.button.superview
                                                    attribute:NSLayoutAttributeBottom
                                                   multiplier:1.0
                                                     constant:-100];
                                                     
[self.view addConstraint:constraint2];

```
`self.button`距离底部有100的padding

### 添加NSLayoutConstraint

- 如果两个view同级别，有相同的superView,那么constraint应该add到他们的superView

- 如果两个view同级别，但是有不同的superView，那么constrait应该add到他们superView共同的superView上

- 如果两个view不同级别，一个是另一个的superView，那么constraint应该add到superView上


### NSLayoutConstraint的优先级

- 默认NSLayoutConstraint的优先级为:`NSLayoutPriorityRequired = 1000`

- 如果layout的行为有冲突，根据优先级来决定先满足哪一条

- 常用的优先级有`250`,`750`, `1000`

### Layout Behind the Scenes 

1. Update Constraints: 从child到parent
2. Layout: 从parent到child
3. Display: 从parent到child

- `intrinsicContentSize`

返回系统计算的默认UIView高度，比如button的size，系统会将image和title长度等考虑进去，返回一个合适的高度

### The Visual Format Language

- 大小，优先级： `[wideView(>=60@700)]`: `wideView`的宽度至少为60，优先级为700

![Alt text](/blog/assets/images/2012/08/al-1.png)

- 垂直对齐，等高: `V:[redBox][yellowBox(==redBox)]`

![Alt text](/blog/assets/images/2012/08/al-2.png)

- 组合：`H:|-[Find]-[FindNext]-[FindField(>=20)]-|`

![Alt text](/blog/assets/images/2012/08/al-3.png)

### Debug

- debug方法:`lldb : po [[UIWindow keyWindow] _autolayoutTrace]`

### Compatibility

`@property(nonatomic) BOOL translatesAutoresizingMaskIntoConstraints` 默认为YES
使用AutoLayout的UIView需要将这个属性设为NO

## Best Practice for Mastering Auto Layout

### NSLayoutConstraint

- 一个新的类

- 可以表示View自身的属性，如`foo.width=100`

- 也可以表示不同View之间的关系，如`foo.width = bar.width`

- 可以描述表达式:`foo.width >= 100`

- 可以描述优先级

- 可以用InterfaceBuilder，VFL，基本的API三种方式描述

