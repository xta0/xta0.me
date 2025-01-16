---
layout: post
list_title:  Machine Learning | 大数据学习 | Learning With Large Datasets
title: Learning With Large Datasets
meta: Coursera Stanford Machine Learning Cousre Note, Chapter9
categories: [Machine Learning,AI]
mathjax: true
---

当数据集很大时，进行梯度下降计算会带来较高的计算量。我们需要寻找一些其它的最优化算法来处理大数据的情况

### Stochastic gradient descent

以最开始提到的线性回归为例，目标函数为：

<math display="block">
<msub><mi>h</mi><mi>θ</mi></msub>
<mi>(x)</mi>
<mo>=</mo>
<munderover>
<mo>∑</mo>
<mrow>
  <mi>j</mi>
  <mo>=</mo>
  <mn>0</mn>
</mrow>
<mi>n</mi>
</munderover>
<msub><mi>θ</mi><mi>j</mi></msub>
<msub><mi>x</mi><mi>j</mi></msub>
</math>

代价函数为：

<math display="block">
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>θ</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
<mstyle>
  <mfrac>
  <mn>1</mn>
  <mrow>
  <mn>2</mn>
  <mi>m</mi>
  </mrow>
  </mfrac>
</mstyle>
    <mstyle>
      <munderover>
        <mo>∑</mo>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>i</mi>
          <mo>=</mo>
          <mn>1</mn>
        </mrow>
        <mi>m</mi>
      </munderover>
      <msup>
        <mfenced open="(" close=")">
          <mrow>
            <msub>
              <mi>h</mi>
              <mi>θ</mi>
            </msub>
            <mo stretchy="false">(</mo>
            <msup>
              <mi>x</mi>
              <mi>(i)</mi>
            </msup>
            <mo stretchy="false">)</mo>
            <mo>−</mo>
            <msup>
              <mi>y</mi>
              <mi>(i)</mi>
            </msup>
          </mrow>
        </mfenced>
        <mn>2</mn>
      </msup>
    </mstyle>
</math>

梯度下降公式为：

<math display="block">
<msub>
          <mi>θ</mi>
          <mn>j</mn>
        </msub>
        <mo>:=</mo>
        <msub>
          <mi>θ</mi>
          <mn>j</mn>
        </msub>
        <mo>−</mo>
        <mi>α</mi>
        <mfrac>
          <mn>1</mn>
          <mi>m</mi>
        </mfrac>
        <munderover>
          <mo movablelimits="false">∑</mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>θ</mi>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
        <mo>−</mo>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
        <mo>⋅</mo>
        <msubsup>
          <mi>x</mi>
          <mn>j</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
</math>

上述梯度下降公式也称作“Batch Gradient Descent”它在<math><mi>m</mi></math>非常大时（假设 m 为 3 亿，美国人口数量)计算求和的过程非常耗时，它需要把磁盘中所有 m 的数据读入内存，仅仅为了计算一个微分项，而计算机的内存无法一次性存储 3 亿条数据，因此只能批量读入，批量计算，才能完成一次梯度下降的计算。

![](/assets/images/2017/09/ml-11-1.png)

随机梯度下下降法实际上是扫描所有的训练样本，首先是计算第一组样本(x(1),y(1))，对它的代价函数计算一小步的梯度下降，然后把 θ 修改一点使其对第一个训练样本拟合变得好一点。在完成这个内部循环后（先遍历 j 再遍历 i），再转向第二个，第三个样本直到所有样本。外部 Repeat 循环会多次遍历整个训练集，确保算法收敛。需要注意几点：

* Repeat 的次数可以选择 1-10，当样本量足够大时，一次内循环就够了
* 随机梯度下降可以保证参数朝着全局最小值的方向被更新，但无法保证下降的顺序（收敛形式不同）和最终的梯度值不发生变化
* 随机梯度下降是在靠近全局最小值的区域内徘徊，而不是直接得到全局最小值并停留在那个点上
* Learning Rate α 的值通常为常量，如果希望 θ 完全收敛，可以让 α 随时间减小( <math><mi>α</mi><mo>=</mo><mfrac><mtext>const1</mtext><mrow><mtext>iterationNumber</mtext><mo>+</mo><mtext>const2</mtext></mrow></mfrac></math>)，通常来说不需要这么做

为了观察算法的学习情况，可以画出随机梯度下降的学习曲线，可能有如下几种情况

![](/assets/images/2017/09/ml-11-2.png)

### Mini-batch gradient descent

* Batch gradient descent： 每次迭代使用全部 m 个训练样本
* Stochasitc gradient descent： 每次迭代使用 1 个训练样本
* Mini-Batch gradient descent：每次迭代使用 b 个训练样本，b 的取值一般为 2-100

![](/assets/images/2017/09/ml-11-3.png)

使用 mini-batch 梯度下降的另一个好处是，可以并行处理，将数据分段后，使用向量化进行并行运算
