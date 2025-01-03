---
layout: post
list_title:  谈谈AsyncDisplayKit | Introduce to AsyncDisplayKit
title: Facebook的AsyncDisplayKit
categories: iOS
---

### AsyncDisplayKit简介

[AsyncDisplayKit](https://github.com/facebook/AsyncDisplayKit)(后面简称AS)，是Facebook开源的一套异步绘制UI的framework，用来提高GUI的绘制效率。它最初和[POP](https://github.com/facebook/pop)一起被用在Facebook的Paper上，Paper在当时引起了强烈的反响，因为其引入了很多物理效果和流畅的动画表现。然后Facebook开源了它们的物理效果引擎[POP](https://github.com/facebook/pop)，同时也宣布会开源一套全新的UI绘制引擎，大概7个月前，Facebook宣布开源AsyncDisplayKit。


### 一段以前的故事

大概在2年多以前，我经历的一个项目是做一个SNS的APP，当时市面上的主流机型是iPhone4/4s，那时候我们面临的一个很大的技术问题是Timeline列表的UI性能问题，由于用户输入的内容都比较复杂，有文字，图片，表情，关键字，短连接等，因此在UI的呈现上也比较复杂。

我们当时使用的文字排版引擎是`CoreText`，`CoreText`虽然能解决各种元素混排的问题，但是其性能却非常一般，实时绘制的成本很高，直观感受就是列表滚动起来很卡。随后我们进行了一系列的优化，最后的结果是将整个绘制过程都分离到了另一个线程中，具体来说就是我们在一个后台线程开辟了一块内存，创建了一个`CGContextRef`，用这个context完成对`attributedString`的绘制(`CoreText` API是线程安全的)，生成一张bitmap，然后在主线程中将这个bitmap作为layer的backing store，直接显示出来。

这种方式最终极大的提升了UI性能，因为主线程不再做任何CPU相关的计算了，GPU面对的也是"a single texture"，节省了compositing的过程。

当我们将这种技术实践了一段时间后，我们开始想能否将它抽象一下，用到其它View的渲染上，原因是当时iPhone5已经出了，iPhone4开始变少，这意味着多核的时代来临了，我们可以充分利用CPU的性能完成更多并发的绘制。当我把这个想法说出来的时候，立刻得到了另一个同事的响应，于是我们找了个周末，在西溪的一个茶馆里，开始研究异步绘制，思路和我们上面提到的类似，只不过针对View，我们使用了`layer.renderInContext(ctx)`的方式，将所有的SubView绘制成一张bitmap。然后我们将绘制好的bitmap放到memory cache里。

这个方法居然work了，demo中我们极大的提高了UITableView的性能。当时我们均感到十分兴奋，因为我们相当于重新定义了一套UIKit的渲染方式。

但是我们很快就遇到了一些难解的问题，比如View上元素的事件无法响应了（因为全变成layer了）；对于单核CPU的设备，这种大量的，高并发的，后台CPU运算，速度实在太慢，跟不上主线程的显式；`renderInContext`有时候线程不安全；生成过多的bitmap，消耗过多的内存等等。就在我们准备仔细思考这些问题时，团队突然解散了，项目失败了，各种变动导致我们最终放弃了对异步绘制的研究。后面，我抽了点时间将上面提到的内容汇总了一篇文章在[这里](http://akadealloc.github.io/blog/2013/07/12/custom-drawing.html)。

直到前几个月，看到了[AsyncDisplayKit](https://github.com/facebook/AsyncDisplayKit)，一看名字就猜到它是干嘛的了, 然后反反复复看了几遍[NSLondon](http://vimeo.com/103589245)以及[Paper](https://www.youtube.com/watch?v=OiY1cheLpmI&list=PLb0IAmt7-GS2sh8saWW4z8x2vo7puwgnR)上的Tech Talk，发现他们的思路和我上面提到的是一样的，也是在另一个线程中绘制bitmap，只是他们没有使用`renderInContext`。但是Facebook就是Facebook，他们花了1年的时间解决了我们当时遇到的所有的问题。

相比POP, AS并没有引起人们很大的关注，原因一方面是现在的硬件设备强大了，无论是CPU的计算速度还是GPU的渲染能力都变强了，对于一些对帧率不敏感的App，不做优化也还算流畅。第二个原因是很多人不太明白它究竟在解决什么问题，是干什么的，只有在这方面吃过苦，有过优化经验的人才明白它的价值。

### Paper流畅的动画


在谈AS之前，不得不先聊一聊Paper，Paper是[Facebook Creative Labs](https://www.facebook.com/labs)的第一个App，Facebook完成Paper用了两年的时间，两年前（2012年初）的主流设备是iPhone4, 这种硬件(ARMV6+PowerVRSGX)对于实现Paper的物理效果是非常困难的(那时候还没有UIKitDynamics)，因此Paper的开发人员不得不先花一年的时间来重新构建一套全新的动画引擎，然后又花了一年的时间来做上层业务的开发。两年过去了，Paper惊艳的问世了，POP也开源了，但同时Apple也发布了UIKitDynamics，二者孰优孰劣，暂时还没有评判的标准，但是对于支持iOS 5.0的App，显然POP是更好的选择。

除了技术以外，让我感触比较深的是Facebook肯花两年的时间来打磨一个App，在技术上的投入就花了1年的时间，放眼望去国内的互联网公司，即使是BAT，有多少肯在一个项目上花超过1年的时间，尤其还是在没有产出的情况下。从这个角度说，Paper可以说是技术驱动的产品，没有POP和AS，就没有Paper，而技术驱动的产品，想要做出点样子，没有个一年半载的投入和潜心研究，成功的可能性是非常低的，显然这和国内强调敏捷，浮躁的技术氛围是背道而驰的。

### POP与CoreAnimation

为什么说两年前想做出Paper这种App比较难呢，在iOS7之前，像Paper这种，全部基于手势的App很少，多数都是以静态动画为主的App：

- 手势基本上以点击等非连续手势为主
- 动画基本上是静态的，非interactive的，动画执行的过程也是不可打断的（想象一下`UINavigationController`的`push`和`pop`动画）。

所有的老技术都具有局限性，而最大的局限性就是受当时硬件的制约，CoreAnimation也是这样，它最初的设计是：单核CPU+MainThread，这种设计对于Static Animation做了很多优化。具体来说是，对于静态动画，CoreAnimation是在另一个进程中渲染，然后保证该进程比当前进程有着更高的优先级，这样即使当前进程的MainThread被短暂阻塞，用户也能看到流畅的动画，而无法感知当前进程的情况。这种抢占式设计主要是为了弥补当时硬件设备性能上的劣势，提升帧率，增强用户体验。但是随着硬件条件的升级，CoreAnimation的设计却并没有改变。

这就有一个问题，我们看到Paper中的动画基本上没有静态动画，动画都是interactive的，都是由连续手势产生的，比如，手指拖拽一个小球从A点到B点，那么对于这个场景如果使用静态动画(`-[UIView animationWithDuration....]`)效果是什么样呢？显然小球的移动速度跟不上手指的移动速度。原因上面也提到了，静态动画使用的Render Server，是跨进程的，进程间通信本身就有时间的损耗。因此对于连续手势产生的动画，我们不能直接丢给CoreAnimation做动画。

既然我们不能使用静态动画，那我们就需要在当前进程内，自己控制对小球位置的移动，控制每一帧中小球的位置:

```objc
- (void)_renderTime:(CFTimeInterval)time items:(POPAnimtorItemList)items
{
    // begin transaction with actions disabled
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    ...
    std::vector<POPAnimatorItemRef> vector{ std::begin(items), std::end(items) };
    for (auto item : vector) {
        [self _renderTime:time item:item];
    }
   	...
   [CATransaction commit];
}
```

由于`CADisplayLink`跑在`MainRunloop`中，因此，这种做法对主线程的阻塞与否是非常敏感的。

>对于iOS7以前的系统，当非连续的手势(如touch，tap等)被检测到时，即使当前的MainThread被阻塞住，系统仍然可以缓存这些输入，等到主线程空闲时，再丢给runloop处理。但是对于连续的手势(如pan等)，显然系统是没有足够的buffer去缓存这些输入的，因此这些输入会被丢弃。

POP通过这种方式是实现了它的动画引擎，接下来的问题是如何高效的渲染每一帧


### AsyncDisplayKit的原理简介

上面我们已经看到连续手势面临的挑战就是需要主线程实时处理每一帧，如果按照60fps的标准，每一帧大概有16ms的时间，刨除GPU的耗时，CPU大概只有~5ms的时间，那在5ms的时间内CPU要完成对View的创建，layout，绘制等等，显然压力巨大。因此需要有一种手段来保证渲染的高效性。于是便有了AS，AS面对是CPU和GPU两方面的优化。

- CPU方面：AS认为从Apple A5开始，多核设备已经开始成为主流，那么可以考虑将一些耗时的CPU操作放到后台线程，这样就减少了主线程的工作，而多核又可以满足在后台线程的并发执行。

- GPU方面：减少GPU Compositing和Blending的一种方式就是将每一帧上的的内容都渲染成一张Texture，而这个工作可以再后台线程中完成。

AS基本上就是做了这两方面的优化，这些和我们2年前做的优化工作差不多，但是AS以一种系统性，抽象的方式将异步绘制应用到了整个UIKit中，这一点非常了不起。

![Alt text](/assets/images/2015/01/ASNode.png)

AS通过定义Node封装了UIView和CALayer，使用Node可以像使用UIView一样，但是Node是线程安全的，你可以在另一个线程中去创建，layout，绘制Node:

```objectivec

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        
        UIImage* img = [UIImage imageNamed:@"CATransaction.png"];
        ASImageNode* node = [[ASImageNode alloc]init];
        node.contentMode = UIViewContentModeScaleAspectFill;
        node.bounds = CGRectMake(0, 0, 320, 200);
        node.position = CGPointMake(self.view.bounds.size.width/2, 300);
        node.image = img;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            
            [self.view addSubview:node.view];
        });
    });

```
	
Node的异步绘制过程(伪代码)：

```objectivec

- (void)display
{
	dispatch_async(queue, ^{
	
		CGContextRef ctx = newContextOfSize(self.bounds.size);
		
		[self.node drawInContext:ctx];
	
		dispatch_async(main,^{self.contents = ctx})
	
	});

}

```
实际的绘制代码在:`ASDisplayNode`和`ASDisplayNode+AsyncDisplay.mm`中，感兴趣的可以自行慢慢解读，后面如果有时间我可能会写一篇分析AS源码的文章。

上手使用AS还是有一定的成本的，我推荐Ray上的这篇文章：[AsyncDisplayKit Tutorial](http://www.raywenderlich.com/86365/asyncdisplaykit-tutorial-achieving-60-fps-scrolling)。这是一篇从入门到高级玩法都讲到了的好文，耐心把它看完，实践完，仔细思考和体会过后，不知不觉间AS就上手了。


### 总结

最后总结一下，AS适用于对主线程阻塞很敏感的场景，比如，绘制很复杂的文本，以及UIScrollView, UITableView, UICollectionView这种对帧率要求较高的UI场景。对于其它的场景，使用AS也会带来性能的提升。接下来，准备在项目中逐步应用AS，发挥它的威力。
		
### Resources

- [Paper Tech Talk](https://www.youtube.com/watch?v=OiY1cheLpmI&list=PLb0IAmt7-GS2sh8saWW4z8x2vo7puwgnR)
- [NSLondon](http://vimeo.com/103589245)
- [Getting Started Guide](http://asyncdisplaykit.org/guide/)
- [AsyncDisplayKit Tutorial](http://www.raywenderlich.com/86365/asyncdisplaykit-tutorial-achieving-60-fps-scrolling)