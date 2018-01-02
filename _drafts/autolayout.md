#AutoLayout

##Introduce to AutoLayout 

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

![Alt text](/blog/images/2012/08/al-1.png)

- 垂直对齐，等高: `V:[redBox][yellowBox(==redBox)]`

![Alt text](/blog/images/2012/08/al-2.png)

- 组合：`H:|-[Find]-[FindNext]-[FindField(>=20)]-|`

![Alt text](/blog/images/2012/08/al-3.png)

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

