---
layout: post
list_title:  我们开源了FlexLayout | FlexLayout Open Sourced!
title: 我们开源了FlexLayout
categories: [iOS]
---

iOS开发中的UI位置计算是一直是一件令人头痛的事情，尤其是对于比较复杂的页面，一个UI元素的位置的计算需要依赖其它UI元素，而且布局规则复杂，边界条件很多，这种情况下写出来的代码不仅难以阅读，而且更难维护，相信很多iOS程序员都曾遇到过这种情况。

FlexLayout是我们团队研发的一套基于[flex模型](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)的相对布局系统，它可以有效的降低View位置计算的复杂度。在iOS上实现真正意义上的相对布局。

FlexLayout提供一组声明式(declarative)的API来描述一个UI组件(Node或Component)的结构，属性，相对位置等信息；通过一个pure function将数据映射为一组对数据的描述；开发者只需要像设计师一样描绘页面"长什么样"而不用思考具体的实现方式。

俗话说:"Talk is cheap, show me the code",我们以一个实际的应用场景来说明如何使用FlexLayout

![](/assets/images/2016/03/flex001.png)

如上图cell，要求高度根据内容动态变化，名字长度自适应，名字和时间两端对齐，但是评价星星始终跟在名字后面，同时又不能压缩它后面的时间label。

## 使用FlexLayout

### 像React一样思考

接下来我们讨论如何使用FlexLayout来描述这个cell，就像React中的Component，第一步是思考如何将cell切分成若干个Component，以上面的布局为例，我们首将其先切成几个大的node,如下图所示

![](/assets/images/2016/03/flex002.png)

### 用代码描述

接下来我们开始用代码描述每个node，习惯上，我们先从粒度最小的node开始，自底向上构建，以上面的cell为例，我们先从第一个红框开始

```cpp
using namespace o2o::flex;
- (FlexLayout )titleLayout:(NSString* )name Time:(NSString* )time Score:(float)score{
    return FlexLayout
        //children元素之间的间隔
        .direction = FlexDirection::Horizontal,
        .spacing = 5,
        .alignItems = FlexAlign::Center,//children元素垂直居中
        .children = {
            {
                //姓名
                .content = TextNode{
                    .text = name,
                    .font = [UIFont systemFontOfSize:14.0f],
                    .color = [UIColor blackColor],
                }
            },
            {
                //星星
                .viewBuilder = ^{

                    O2OStarView* starView  = [[O2OStarView alloc] initWithOrigin:CGPointMake(0, 0) viewType:O2OStarViewTypeForDisplay starWidth:14 starMargin:0 starNumber:5];
                    starView.score = score;
                    return starView;

                },
                .width = 70,
                .height = 12,
                .flexShrink = 0,
            },
            {
                //时间
                .content = TextNode{
                    .text = time,
                    .font = [UIFont systemFontOfSize:12.0f],
                    .color = [UIColor grayColor],
                },
                .flexShrink = 0,
                .marginLeft = Auto
            }
        }
    };
}
```

如果熟悉flex，那么上面的代码应该一目了然，无需过多解释。对于不熟悉flex模型的，上面代码做了这么几件事:

- 首先定义了一个容器Node，它的布局方向是水平方向(Horizontal),有3个子Node(children)，他们之间的间隔(spacing)是5，指定子元素垂直方向上的布局方式(alignItems)是居中(FlexAlign::Center)
- 第一个子元素是姓名，用在FlexLayout中用TextNode描述
- 第二个子元素是评价的星星，由于这个View不属于UIKit原生组件，是一个自定义view，这种情况FlexLayout提供一个ViewBuiler用来创建自定义View
- 第三个子元素是时间，同样使用TextNode


### 使用flexShrink

根据之前的约定，我们需要星星一直跟在名字的后面，当名字过长时，自身不被压缩，同时后面的时间node也不被压缩。实现这个规则，如果使用UIKit，计算会变得很复杂，要考虑名字的长度，星星的长度，时间的文字长度，以及他们之间的间隔，还要考虑最右边的padding，实现起来要定义一些局部变量来保存各种计算结果，还要通过复杂的过程性的语句来实现上面的规则，代码可读性和维护性都很糟糕。有了FlexLayout，只需要指定星星和时间的flexShrink为0，表示当总体宽度不够时，不压缩自身宽度，那么只有名字的长度会被压缩，如下图所示:

![Alt text](/assets/images/2016/03/flex003.png)

### 定义Pure Function

了解了FlexLayout的布局语法，我们需要一个Pure Function将数据映射为一个完整的FlexLayout：

```cpp
- (FlexLayout)layoutForModel:(CModel* )model{
    return FlexLayout {
        .backColor = "white",
        .padding = 10,
        .spacing = 10,
        .children = {
            {
                .content = ImageNode{.image = model.image},
                .width = 40,
                .height = 40,
                .cornerRadius = 20,
                .flexShrink = 0,
            },
            {
                .direction = FlexDirection::Vertical,
                .flexGrow = 1,
                .spacing = 6,
                .children = {
                    [self titleLayout:model.name Time:model.time Score:model.score],
                    {.content = TextNode{
                        .lines = 0,
                        .text = model.sign,
                        .font = [UIFont systemFontOfSize:14.0f],
                        .color = [UIColor lightGrayColor]
                    }}
                }
            }
        }
    };
}

```

至此，我们的工作就做完了，剩下的工作就完全交给FlexLayout了，FlexLayout会帮你创建view，绑定属性，计算位置。

> <strong>关于State与Side Effect</strong>：如果熟悉React，应该知道props和state的概念，对于FlexLayout来说，Props可以类比上面方法中的CModel*，而state在FlexLayout中没有对应的实现，原因是Pure Function没有办法真正的规避Side Effect，只能通过约定，这显然不是一个很好的方式，我们的[另一个项目](#react)会在设计上完善这种情况


## 声明式，没有计算

上面代码中，如果使用[传统方式](https://gist.github.com/vizlabxt/cc8764619f90866adeb2)，那么我们需要实现sizeThatFits:先算出view的高度,然后再通过layoutSubViews逐个计算出元素的位置，实际项目中这样的代码很常见也非常难以维护，而使用FlexLayout我们无需关心每个view的计算过程。Flexlayout的语法是declarative的，使整个页面的结构变的清晰易读，代码也容易维护。

## 结论

我们在支付宝口碑的业务中，很多地方都使用了FlexLayout，以商家中心券为例，对于如此复杂的页面，代码量比传统方式减少了20%，而且可读性和维护性都提升了，最为重要的是，作为程序员，看到这样的代码能让我们心情愉悦，高效率的coding

![Alt text](/assets/images/2016/03/flex004.png)

## 附录

### 关于Flexlayout的实现

FlexLayout的核心是实现Flex布局模型的算法，关于Flex布局模型的算法，Github上有很多种实现，对于客户端来说，C语言的版本是可以直接移植的，但是经过我们的多次试验，发现css_layout自身有一些局限性，因此我们参照FlexBox官网提供的说明，自行实现了一套标准的算法，和css_layout的区别可以[参考这里](#css)。

FlexLayout使用C++编写，里面除了使用了常用的C++集合类，模板类之外，还大量使用了C++ 11的一些新特性，例如unorderd_map，统一初始化函数{}，lambda表达式，右值引用等等。其中统一初始化函数( Aggregate initialization)给FlexLayout的API设计提供了极大的灵活性，C++的各类容器提供了严格的类型检查，不会发生运行时的类型错误，Template为struct提供了默认值的实现能力，等等

<h3 id="react"> 关于FlexLayout的未来 </h3>

仅仅创建一套声明式的相对布局系统并不是我们的终极目标，我们的另一个项目"FNode"在FlexLayout的基础上构建了一套Functional UI，像React一样通过构建一个个component完成页面的展现，同时引入单向数据流保持业务逻辑简单清晰，敬请期待

<h3 id="css"> FlexLayout和css_layout关于Flex模型实现的比较 </h3>

| 特性 | css-layout | VZFlexLayout | 备注 |
| --- | --- | --- | --- |
| [flex-direction](#direction)| 支持 | 支持 |  |
| [flex-wrap](#wrap) | 支持 | 支持 | 均不支持 wrap-reverse |
| [align-items](#alignItems), [align-self](#alignSelf), [justify-content](#justifyContent) | 支持 | 支持 |  |
| [align-content](#alignContent) | 部分支持（不支持 space-between 和 space-around） | 支持 |  |
| flex | 部分支持（合并 flex-grow 与 flex-shrink ，不支持修改 flex-basis） | 完整支持 [flex-basis](#flexBasis), [flex-grow](#flexGrow), [flex-shrink](#flexShrink) |  |
| [固定布局元素](#fixed) | 支持（使用 left, right, top, bottom 指定元素位置和大小） | 支持（使用 width, height, margin 控制位置和大小，可以自动计算宽高、自动居中，更为灵活） |  |
| RTL布局方向 | 支持 | 不支持 |  |
| [width](#width)/[height](#height) | 支持（有些情况下对 auto 的计算不正确） | 支持 |  |
| [min-width](#minWidth)/[max-width](#maxWidth), [min-height](#minHeight)/[max-height](#maxHeight)| 支持 | 支持 |  |
| [margin](#margin) | 支持 | 支持（支持设置为 auto） |  |
| [padding](#padding), [border-width](#borderWidth) | 支持 | 支持 |  |
| [spacing](#spacing), [lineSpacing](#lineSpacing) | 不支持 | 支持 | 非css特性 |


## 参考

- [A Complete Guide to Flexbox](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [CSS3 Flexible Box](http://www.w3schools.com/css/css3_flexbox.asp)
- [ComponentKit](http://componentkit.org/)
