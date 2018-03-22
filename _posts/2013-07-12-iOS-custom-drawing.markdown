---
title: 一些提高UI绘制性能的技巧
layout: post
tag: iOS
categories: 随笔

---

<em>所有文章均为作者原创，转载请注明出处</em>

今天讨论下UI绘制的性能问题，通常来说，如果熟悉下面列出的内容基本上能解决80%以上的性能问题了：

1. [UIView是如何渲染到屏幕上的](http://vizlabxt.github.io/blog/2012/10/22/UIView-Rendering/)

2. WWDC2011:Session 318 - iOS Performance in Depth

3. WWDC2012:Session 238 - iOS App Performance_ Graphics and Animations

4. 熟练Instrument的Time Profiler和Core Animation

接下来讨论三个我认为有价值的点：

1. Twitter的这篇关于tableview优化的文章，是对还是错？

2. CoreText最佳性能优化方案

3. 异步绘制


##Layer Trees v.s. Flat Drawing

基本上，如果去优化UITableview的滚动性能，都会读到<a href="https://github.com/kennethreitz/osx-gcc-installer/">Twitter的这篇文章</a>。这篇文章其实就说了一件事：将cell上复杂的UI层次结构，简化为一个Layer。

例如：要展示这样一个cell：

<a href="/assets/images/2013/07/cell.png"><img class="alignnone size-full wp-image-251" alt="cell" src="/assets/images/2013/07/cell.png" width="311" height="96" /></a>

###传统的做法：

传统的做法是定义一个`cell`，在`cell`的`contentView`上添加UI元素：三个label，一个imagView：

```objc

@implementation ETTableViewDemoAppStoreCell

- (id)initWithStyle:(UITableViewCellStyle)style reuseIdentifier:(NSString *)reuseIdentifier 
{
    if (self=[super initWithStyle:style reuseIdentifier:reuseIdentifier]) 
    {
        [self.contentView addSubview:self.nameLabel];
        [self.contentView addSubview:self.priceLabel];
        [self.contentView addSubview:self.summaryLabel];
        [self.contentView addSubview:self.imgView];
    }
    return self;
}  

- (void)setItem:(ETTableViewDemoAppStoreItem* )item
{
    self.nameLabel.text = item.appName;
    self.priceLabel.text = item.appPrice;
    self.summaryLabel.text = item.appSummary;
    [self.imgView setImageURL:item.appImageURL];
}

- (void)layoutSubviews
{
    [super layoutSubviews];

    self.imgView.frame = CGRectMake(10, 10, 80, 80);
    self.nameLabel.frame = CGRectMake(100, 10, 210, 18);
    self.priceLabel.frame = CGRectMake(100, 32, 100, 14);
    self.summaryLabel.frame = CGRectMake(100, 45, 210, 50);
}

@end

```

###Twitter的做法：

首先也是需要定义个cell：

```objc
@interface ETTableViewDemoAppStoreFlatCell : ETTableViewSingleImageCell

@end
```

这里面不定义任何UI元素，然后在cell的contentView上只add一个view：

```objc
- (id)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {

        _internalContentView = [[ETTableViewDemoAppStoreFlatContentView alloc]initWithFrame:CGRectZero];
        _internalContentView.backgroundColor = [UIColor clearColor];
        _internalContentView.contentMode = UIViewContentModeRedraw;
        [self.contentView addSubview:_internalContentView];

    }
    return self;
}
```

由于`internalContentView`的`contentMode`为`Redraw`，因此每当它`frame`改变的时候，都会重新绘制一次。

这个InternalContentView的定义如下：

```objc
@interface ETTableViewDemoAppStoreFlatContentView : UIView

@property(nonatomic,strong) ETTableViewDemoAppStoreItem* item;

@end

@implementation ETTableViewDemoAppStoreFlatContentView

- (void)drawRect:(CGRect)rect
{
    [super drawRect:rect];

    //draw image
    [self.item.image drawInRect:CGRectMake(10, 10, 80, 80)];

    //draw text
    [[UIColor blackColor]set];
    [self.item.appName drawInRect:CGRectMake(100, 10, 210, 18) withFont:[UIFont systemFontOfSize:16.0f] lineBreakMode:NSLineBreakByTruncatingTail];

    //draw price
    [[UIColor grayColor] set];
    [self.item.appPrice drawInRect:CGRectMake(100, 32, 100, 14) withFont:[UIFont systemFontOfSize:14.0f] lineBreakMode:NSLineBreakByTruncatingTail];

    //draw price
    [[UIColor grayColor] set];
    [self.item.appPrice drawInRect:CGRectMake(100, 45, 210, 50) withFont:[UIFont systemFontOfSize:14.0f] lineBreakMode:NSLineBreakByTruncatingTail];
}

@end
```

`InternalContent`通过实现它`drawRect`的方法，将数据绘制出来。

然后就是cell如何去触发internalContentView绘制：

```objc
- (void)prepareForReuse
{
    [super prepareForReuse];

    _internalContentView.frame = CGRectZero;
}
- (void)layoutSubviews
{
    [super layoutSubviews];

    _internalContentView.frame = CGRectMake(0, 0, CGRectGetWidth(self.frame), CGRectGetHeight(self.frame));
}
```
当tableview滚动的时候，由于`view`的`size`不断变化，`view`可以不断的重绘。

下面我们先来想下这两种方式背后的差别：

- 传统方式：当新的一帧到来时候，`cell`首先要将数据交给三个label去绘制，由于`cell`上的UI元素本身就是复用的，因此`label`不会重新创建，只需要重新update自己的backing store，将新的text数据写进去，生成bitmap。`imageView`同理，但是由于`imageview`不需要update backing store，它的`layer`直接指向了一块bitmap。到此CPU的工作做完了，GPU要将`cell`和这三个`label`加`imageview`组成的`layer tree`一起做compositing，最后渲染出来。

- twitter的方式：当一个新的runloop到来，由于`cell`上之只有一个`internalContentView`，它更新完自己的backing store后生成一块bitmap交给GPU，而GPU由于之渲染一个bitmap，composting的压力也不大。

从上面的过程来看，显然，传统方式GPU的工作多一些，当layer-tree很复杂的时候，GPU的耗时也会增加，但是CPU的压力不大。twitter这种方式，CPU的压力大一些，因为Core Graphic的api是CPU在执行，但是由于之渲染一个bitmap，GPU的压力小很多。

###结论：

所以，这是一个test - measure的过程。twitter最开始使用这种方式而获得性能上的成功要追溯到2008年，那时候的iPhone还是低清屏的3gs，到了retina的时代，这种方式还适不适合，<a href="http://floriankugler.com/blog/2013/5/24/layer-trees-vs-flat-drawing-graphics-performance-across-ios-device-generations">这篇文章</a>给出了量化的结论：在retina的时代里，使用Core Graphic绘制的代价远高于GPU渲染layer-tree的代价，在iPhone4及以后的平台上使用这种方式绘制cell，时间反而会变长，twitter的这种方式过时了！

但就我个人而言，我更喜欢使用这种方式，倒不是因为效率问题，而是这种写法很简单，项目里，一般不复杂的列表，没有性能问题的，都可以这么实现。但是如果很复杂的列表，有很多subview，使用这种方式就不太适合了，尤其在retina屏上，效率降低了。

##CoreText Optimization

这一节主要来讨论使用CoreText绘制文本的优化，场景是这样的：

对于社交类的app如微博，微信，都有用户内容的timeline（通常是一个tableview），在timeline中，用户会发一些表情，会有##话题，@某人，发起一个http连接等，一般这种场合在iOS中会用`CoreText`处理，假设我们绘制这样一段文本：

<a href="/blogimages/2013/07/coretext.png"><img class="alignnone size-full wp-image-257" alt="coretext" src="/assets/images/2013/07/coretext.png" width="308" height="161" /></a>

绘制这段文本的瓶颈都有什么呢？

1. 需要通过正则表达式匹配：[哈哈]...这种文本，然后替换成表情

2. 需要通过正则表达式匹配：@，#，http://这种关键字，然后高亮

3. 绘制AttributedString和表情图片

这些都要通过CPU来完成，而且在tableview滚动的时候，label虽然可以复用，但上面展示的内容却需要实时的计算和绘制，就以上述那段文本为例，上述3条花费的时间如下：

 <a href="/assets/images/2013/07/time.png"><img src="/assets/images/2013/07/time.png" alt="time" width="785" height="124"/></a>

这个结果能看出两件事情：

1. 在非retina的3gs上，CoreText绘制的时间低于retina屏的iPhone4，再次印证了上面的结论。

2. 即使是iphone5在不做优化的情况下，从检测到绘制这段文字也需要16ms左右。虽然实际情况中，不是每一段文本内容都这么复杂，但如果碰见一两条，就会出现明显的卡顿感出现，而且即使文本不复杂的情况，就算它需要8ms，那么如果不优化，留给其它UI元素的绘制时间就会变短，对于60fps，16ms/f的标准，是很难达到的。

下面我们讨论下优化方案：

1. 修改正则表达式，提高效率

2. 先把文本绘制出来，然后在另一个线程中计算正则表达式，匹配keywords和表情

3. 提前渲染好把它变成图片

第一种方案确有优化的空间，但是空间不大，即使优化到3ms以内，绘制仍然需要8-10ms，总体也超过了10ms，意义不大。

第二种方案可以，但会有视觉的突变感，文本的frame也会有突变，用户体验很差

第三种方案是最佳的，在展示文本前，先提前把它绘制成一幅图片保存好，显示的时候让layer的content指向它。

目前开源的有<a href="https://github.com/mattt/TTTAttributedLabel">TTTAttributeLabel</a>，320作者写的，也是AFNetworking的作者，它里面优化的方案是第二种：deferDetection，但实际的效果来看，并不理想。

我用第三种方案的思路实现了`ETAttributedLabel`，用法和`UILabel`一样。它最大的优势在于提供了一个`parser`，可以将计算，解析，绘制等耗时的操作和UI显示剥离开,而且是线程安全的。我们可以先使用`parser`在另一个线程中生成好`attributedString`然后丢给`label`直接显示，也可以使用parser将文本直接会制成图片，然后丢给`label.layer.content`：


```objc 

ETUIAttributeStringParser* parser = [ETUIAttributeStringParser new];
parser.constraintTextWidth = 300;
parser.lineHeight = 20;
parser.textColor = [UIColor blackColor];
parser.textFont = [UIFont systemFontOfSize:14.0f];
parser.linkColor = [UIColor orangeColor];
parser.backgroundColor = [UIColor clearColor];
parser.highlightedLinkBackgroundColor = [UIColor colorWithWhite:0.5 alpha:0.5];
parser.cleanText =@"#wwdc2013#针对 iOS 7 中增加的一个整体调整字体的支持的描述[smile]。从大小上和样式类型上，以及对 Accessbility (辅助功能) 上的支持，即针对一些存在视力或听力障碍的用户的特别适配[dizzle][dizzle]。针对文字排版相关的增强，推出了TextKit[smile][kiss]，关于这块的详细描述的相关的sessions一共有三个[crash][crash]，足以证明 @TextKit 的重要性[cry][cry]。https://developer.apple.com/wwdc/videos/2013";
[parser highlightKeywords:@[@"增强",@"视力或听力障碍"]];


[parser preRenderText:CGSizeMake(300, 100)];

_label = [[ETUIAttributeLabel alloc]initWithFrame:CGRectZero];
_label.backgroundColor = [UIColor clearColor];
_label.attributedString = parser.attributedString;
_label.highLightedKeywords = parser.highLightedKeywords;
_label.attributedImages = parser.attributeImages;
_label.usePreRenderedImage = YES;
_label.layer.frame = CGRectMake(10, 50, 300, 100);
_label.layer.contents = (id)parser.preRenderedImage.CGImage;

[self.view addSubview:_label];
``` 

假如我们一次请求服务器获得了10条数据，那么在`tableview`刷新前，我们使用`parser`将每条数据的text渲染成图片，渲染完成后，通知`tableview`刷新，然后在`cell`绘制的时候为`label`的`layer.content`赋值。

##Asynchronous Drawing

从刚才讨论的第一点我们能知道在retina屏幕上，使用Core Graphic绘制让人很失望

从刚才讨论的第二点我们能知道绘制UIView最快的方法就是把它当成imageview：

```objc
self.view.content.layer = (id)image.cgImage;
```

如果我们把第一点和第二点结合起来，我们把需要用Core Graphic绘制的代码放到另一个线程中去绘制，生成image后直接赋值给view。即一种异步绘制的技术，我的框架中，ETAsyncDrawingCache专门为此而设计，它提供了一个方法：

```objc
- (void)drawViewAsyncWithCacheKey:(NSString *)cacheKey
                             size:(CGSize)imageSize
                  backgroundColor:(UIColor *)backgroundColor
                       targetView:(UIView *)targetView
                  completionBlock:(ETAsyncDrawingCompletionBlock)completionBlock;
``` 

这个方法可以把一个view丢到另一个线程中渲染，然后得到这个view的image：

```objc
- (void)drawViewAsyncWithCacheKey:(NSString *)cacheKey
                             size:(CGSize)imageSize
                  backgroundColor:(UIColor *)backgroundColor
                       targetView:(UIView *)targetView
                  completionBlock:(ETAsyncDrawingCompletionBlock)completionBlock
{
    __block NSString* _cacheKey = cacheKey;
    UIImage *cachedImage = [_memCache objectForKey:cacheKey];
    
    if (cachedImage)
    {
        completionBlock(cachedImage,_cacheKey);
        return;
    }
    
    completionBlock = [completionBlock copy];
    
    dispatch_block_t loadImageBlock = ^{
        BOOL opaque = [self colorIsOpaque:backgroundColor];
        
        UIImage *resultImage = nil;
        
        UIGraphicsBeginImageContextWithOptions(imageSize, opaque, 0);
        {
            CGContextRef context = UIGraphicsGetCurrentContext();
            
            CGRect rectToDraw = (CGRect){.origin = CGPointZero, .size = imageSize};
            
            BOOL shouldDrawBackgroundColor = ![backgroundColor isEqual:[UIColor clearColor]];
            
            if (shouldDrawBackgroundColor)
            {
                CGContextSaveGState(context);
                {
                    CGContextSetFillColorWithColor(context, backgroundColor.CGColor);
                    CGContextFillRect(context, rectToDraw);
                }
                CGContextRestoreGState(context);
            }
            
            [targetView.layer renderInContext:context];
            
            resultImage = UIGraphicsGetImageFromCurrentImageContext();
            
            if (resultImage) {
                [_memCache setObject:resultImage forKey:cacheKey];
            }
            
        }
        UIGraphicsEndImageContext();
        
        //notify
        [[ETThread sharedInstance] enqueueOnMainThread:^{
            completionBlock(resultImage,_cacheKey);
        }];
        
    };
    
    //background drawing
    dispatch_async(_backgroundQueue, loadImageBlock);
    
}
```

通过这种绘制方式来绘制UITableViewCell，非常流畅，但是也有它的问题：

（1）消耗更多的内存

（2）view的点击事件需要自己处理

> 2014年3月14日 注：`CALayer`的`renderInContext:`方法存在线程安全问题，不建议使用

