---
list_title: 笔记 | 深度学习 | Object Detection - Mask R-CNN 
title: Object Detection - Mask R-CNN 
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

在Objective Detection上比较有名的model是R-CNN以它相关的变种。下面简略介绍一下其实现的思路

### R-CNN

R-CNN是Region-based Convolutional Neural Networks的缩写。其主要的思路是

1. 通过selective search为一张图片生成约2000个RoI
2. 由于生成RoI尺寸大小不同，我们需要将它们warp成一个固定大小的矩形，作为后面CNN网络的输入
3. 将warp后的RoI输入CNN进行分类
4. 同时对RoI进行bounding box regression。

整个过程如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-r-cnn-1.png">

R-CNN虽然能完成目标检测的任务，但是速度却非常的慢
- 有2000个region proposal, 训练需要84小时
- 如果用vGG16，一次inference需要47s

### Fast R-CNN



<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn.png">

我们需要首先为图像生成一系列RoIs(Region of Interests)，每一个RoI都是一个bounding box。以Fast R-CNN为例，我们需要为每张图片准备大概2000个。然后用卷积神经网络提取图片中的feature。论文中使用的VGG16。例如，一张`(1, 3, 512, 512)`的图片经过CNN网络后得到一组`(512, 3, 16, 16)`的feature map。

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-6.png">

接下来我们要对从feature maps中提取RoI，由于我们的feature map的大小已经从`(512, 512)`变成了`(16, 16)`, 我们的RoI区域也要等比例缩小，例如一个`(x:296, y:192, h:145, w:200)`的bounding box，在feature map中将变成`(x:9.25, y:6, h:4.53, w:6.25 )`如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-4.png">

此时我们发现bbox出现了小数，我们需要其quantize成整数，但是这会损失一部分数据，如下图三所示

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-1.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-2.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-3.png"></div>
</div>

接下来我们要对RoI中的像素进行RoI Pooling，其目的是将各种大小不一的bbox映射成长度统一的vector以便进行后面的FC。具体做法是使用max pooling，还是上面的例子，经过quantization后，我们的bbox变成了`4x6`的，接下来我们用max pooling把它映射成`3x3`的，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-fast-r-cnn-5.png" width="60%">

上图中我们发现最下面一行数据也被丢弃掉了。通过RoI Pooling我们可以得到一组`($roi,512,3,3`的feature map，这也是后面FC层的输入，最终通过两层FC我们得到了两个输出结果，一个是classification，表明RoI中的Object类别，另一个是RoI的bbox，用来标识Object

### Faster R-CNN

相比Fast R-CNN，Faster R-CNN的主要改变是将RoI的提取整合进了网络


### Mask R-CNN

Mask R-CNN是基于Faster R-CNN的架构，引入了Instant Segmentation。它除了输出目标物体的类型和bbox意外，还输出一个segmeation mask，其结构如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-mask-r-cnn.png">

由于Mask R-CNN需要生成像素级别的mask，前面提到的RoI Pooling由于损失太多data因此精度大大降低。为了解决这个问题Mask R-CNN对上面RoI pooling的改进，提出了RoI Align。我们下面来重点介绍这个算法

前面已经知道RoI Pooling的两次quantization损失了很多data，RoI Align通过使用双线性二次插值弥补了这一点。还是以前面例子来说明，下图是我们前面的bbox，我们的目标还是对其进行3x3的RoI Pooling操作

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-1.png">

和之前不同的是，我们此时不对x,y,w,h进行取整，而是应用双线性插值，具体做法是将RoI区域划分成3x3的格子，然后计算4个点的位置如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-2.png" width="60%">

接下来我们便可以使用下面公式对上面的4个点进行双线性插值计算，其中Q值为每个点对应的像素值

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-3.png">

下面我们以左上角`(9.44, 6.50)`为例，看一下双线性插值是如何计算的。公式里的`x,y`分别对应`(9.44, 6.50)`。其中`(x1, x2, y1, y2)`分别对应四个点邻接cell的中心值，以`(9.44, 6.50)`为例，和它最近的cell的中心点为`(9.50, 6.50)`，因此`x1`为`9.5`,`y1`为`6.50`; 左下角点邻接cell的中心值为`(9.50, 7.50)`，由于`x1`没有发生变化还是`9.5`，而`y2`变成了`7.5`; 以此类推，我们可以得到右上和右下两个cell的中心点分别为 `(10.50, 6.50)`和`(10.50, 7.50)`，此时`x1,x2,y1,y2`的值均已确定，我们可以套用上面公式计算出`(x,y)`点处的值为`0.14`。

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-4.png">

按照上述方式我们可以为剩余三个点计算其双线性插值结果，如下图所示

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-5.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-6.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-7.png"></div>
</div>

当我们结算完这个4个点的双线性插值结果，我们便可以用max pooling得到该区域的RoI Align的结果。

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-8.png">

重复上述步骤我们可以得到这RoI Region的结果，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-9.gif">


理解了RoI Align之后，我们便可以和Fast R-CNN一样对所有的feature map进行RoI Align的操作，得到一组`($roi,512,3,3)`的feature map，作为后续layer的输入

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-10.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-roi-align-11.png"></div>
</div>

和RoI Pooling相比，RoI Align利用了更多RoI Region周围的像素信息（上图中左边绿色部分），因此可以得到更准确的结果

## Resources

- [R-CNN]()
- [Fast R-CNN]()
- [Faster R-CNN]()
- [Mask R-CNN]()
- [Selective Search](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html#selective-search)