---
layout: post
title: Machine Learning - Chap8
meta: Coursera Stanford Machine Learning Cousre Note, Chapter8
categories: [ml-stanford,course]
mathjax: true
---

## Chapter8: Anomaly detection

异常检测主要是通过概率模型对输入的样本数据进行判定，判断是否为异常数据。比如在线购物网站识别用户的异常行为，飞机部件出厂检测，电脑状态检测等。以电脑监控为例，可以使用如下几个值很大或者很小的 feature：

* <math><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo></math> memeory use of the computer
* <math><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo></math> number of disk accesses / etc
* <math><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo></math> CPU load
* <math><msub><mi>x</mi><mn>4</mn></msub><mo>=</mo></math> network traffic
* <math><msub><mi>x</mi><mn>5</mn></msub><mo>=</mo></math> (CPU load) / (network traffic)
* <math><msub><mi>x</mi><mn>6</mn></msub><mo>=</mo></math> (CPU load)^2 / (network traffic)

我们的目标就是构建一个概率模型，对任意输入<math><mi>x</mi></math>判断其是否是异常数据。在介绍具体算法之前，先回顾一下高斯分布：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <msqrt>
        <mn>2</mn>
        <mi>&#x03C0;<!-- π --></mi>
      </msqrt>
      <mi>&#x03C3;<!-- σ --></mi>
    </mrow>
  </mfrac>
  <mi>exp</mi>
  <mfenced open="(" close=")">
    <mrow>
      <mo>&#x2212;<!-- − --></mo>
      <mfrac>
        <mrow>
          <mo stretchy="false">(</mo>
          <mi>x</mi>
          <mo>&#x2212;<!-- − --></mo>
          <mi>&#x03BC;<!-- μ --></mi>
          <msup>
            <mo stretchy="false">)</mo>
            <mn>2</mn>
          </msup>
        </mrow>
        <mrow>
          <mn>2</mn>
          <msup>
            <mi>&#x03C3;<!-- σ --></mi>
            <mn>2</mn>
          </msup>
        </mrow>
      </mfrac>
    </mrow>
  </mfenced>
</math>

其中：

* 期望：<math><mi>μ</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> <msup><mi>x</mi><mi>(i)</mi></msup> </math> 表示样本的均值，极大似然估计值
* 方差：<math><msup><mi>σ</mi><mn>2</mn></msup><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> <mo stretchy="false">(</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></math> 表示样本的与均值之间的平均偏差，方差越大，曲线越扁平；方差越小，曲线越陡峭
* 如果某种数据样本符合高斯分布，记作<math><mi>X</mi><mo>~</mo><mi>N</mi><mo stretchy="false">(</mo><mi>μ</mi><mo>,</mo><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo></math>，高斯分布也叫做正态分布(Normal Distribution)

给定训练集<math><mo>{</mo><msup><mi>x</mi><mi>(1)</mi></msup><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>}</mo></math>，<math><mi>x</mi></math>属于 n 维并服从高斯分布:<math><msub><mi>x</mi><mi>i</mi></msub><mo>~</mo><mi>N</mi><mo stretchy="false">(</mo><msub><mi>μ</mi><mi>i</mi></msub><mo>,</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo stretchy="faslse">)</mo></math>异常检测的概率模型如下：

<math display="block">
<mi>P</mi>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo stretchy="false">)</mo>
<mo>=</mo>
<mi>P</mi>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mn>1</mn></msub>
<mo>;</mo>
<msub><mi>μ</mi><mn>1</mn></msub>
<mo>,</mo>
<msubsup><mi>σ</mi><mn>1</mn><mn>2</mn></msubsup>
<mo stretchy="false">)</mo>
<mi>P</mi>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mn>2</mn></msub>
<mo>;</mo>
<msub><mi>μ</mi><mn>2</mn></msub>
<mo>,</mo>
<msubsup><mi>σ</mi><mn>2</mn><mn>2</mn></msubsup>
<mo stretchy="false">)</mo>
<mo>...</mo>
<mi>P</mi>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mi>n</mi></msub>
<mo>;</mo>
<msub><mi>μ</mi><mi>n</mi></msub>
<mo>,</mo>
<msubsup><mi>σ</mi><mi>n</mi><mn>2</mn></msubsup>
<mo stretchy="false">)</mo>
<mo>=</mo>
<munderover> <mo>Π</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover> 
<mi>P</mi>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mi>j</mi></msub>
<mo>;</mo>
<msub><mi>μ</mi><mi>j</mi></msub>
<mo>,</mo>
<msubsup><mi>σ</mi><mi>j</mi><mn>2</mn></msubsup>
<mo stretchy="false">)</mo>
</math>

检测步骤如下：

* 选取特征数据<math><msub><mi>x</mi><mi>j</mi></msub></math>
* 计算每个特征的<math> <msub> <mi>μ</mi> <mn>1</mn> </msub> <mo>...</mo> <msub> <mi>μ</mi> <mi>n</mi> </msub> <mo>,</mo> <msubsup> <mi>σ</mi> <mn>1</mn> <mn>2</mn> </msubsup> <mo>...</mo><msubsup><mi>σ</mi><mn>n</mn><mn>2</mn></msubsup></math>
  <math display="block"><msub><mi>μ</mi><mi>j</mi></msub><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> <msubsup><mi>x</mi><mi>j</mi><mi>(i)</mi></msubsup> </math>

<math display="block"><msubsup><mi>σ</mi><mi>j</mi><mn>2</mn></msubsup><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> <mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>j</mi><mi>(i)</mi></msubsup><mo>-</mo><msub><mi>μ</mi><mi>j</mi></msub><msup><mo stretchy="false">)</mo><mn>2</mn></msup></math>

Octave 代码如下:

```matlab
function [mu sigma2] = estimateGaussian(X)

[m, n] = size(X);
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

for i=1:n
    mu(i)  = (1/m)*sum(X(:,i));
    sigma2(i) =  (1/m)*sum((X(:,i)-mu(i)).^2);
end

end
```

* 对新的输入样本<math><mi>x</mi></math>,计算<math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math>，根据阈值判断是否为异常数据<math><mi>P</mi> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo><mo><</mo><mi>ε</mi></math>

<math display="block">
<mi>P</mi>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo stretchy="false">)</mo>
<mo>=</mo>
<munderover> <mo>Π</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover> 
<mi>P</mi>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mi>j</mi></msub>
<mo>;</mo>
<msub><mi>μ</mi><mi>j</mi></msub>
<mo>,</mo>
<msubsup><mi>σ</mi><mi>j</mi><mn>2</mn></msubsup>
<mo stretchy="false">)</mo>
<mo>=</mo>
<munderover> <mo>Π</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover>
<mfrac>
<mn>1</mn>
<mrow>
<msqrt>
<mn>2</mn>
<mi>π</mi>
</msqrt>
<msub><mi>σ</mi><mi>j</mi></msub>
</mrow>
</mfrac>
<mi>exp</mi>
<mfenced open="(" close=")">
<mrow>
<mo>−</mo>
<mfrac>
<mrow>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo>−</mo>
<msub><mi>μ</mi><mi>j</mi></msub>
<msup>
<mo stretchy="false">)</mo>
<mn>2</mn>
</msup>
</mrow>
<mrow>
<mn>2</mn>
<msubsup><mi>σ</mi><mi>j</mi><mn>2</mn></msubsup>
</mrow>
</mfrac>
</mrow>
</mfenced>
</math>

![](/assets/images/2017/09/ml-10-1.png)

接下来的问题是，对于不服从高斯分布的样本，我们怎么将其转化为高斯分布样本

![](/assets/images/2017/09/ml-10-2.png)

如上图所示，如果样本数据不服从高斯分布，可以使用<math><mo>log</mo></math>或指数来调整样本分布。另外，上述模型对某些异常数据可能无法区分，我们希望：

* 对于正常的样本数据<math><mi>x</mi></math>，<math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math>值很大
* 对于异常的样本数据<math><mi>x</mi></math>，<math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math>值很小

但是对于某些异常数据，<math> <mi>P</mi> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo></math>可能也很大，它们在各自样本分布曲线中均处于正常范畴，但是在二维高斯函数分布中却属于异常数据这时我们需要对异常数据进行分析，如下图所示

![](/assets/images/2017/09/ml-10-3.png)

因此我们要优化之前的模型，相比原来独立计算各维度的概率，优化后的模型使用协方差来表示各维度之间的关系：

<math display="block">
<mi>P</mi>
<mo stretchy="false">(</mo>
<mi>x</mi><mo>;</mo>
<mi>μ</mi><mo>,</mo>
<mi>∑</mi>
<mo stretchy="false">)</mo>
<mo>=</mo>
<mfrac>
<mn>1</mn>
<mrow>
<mo stretchy="false">(</mo>
<mn>2</mn>
<mi>π</mi>
<msup><mo stretchy="false">)</mo><mi>n/2</mi></msup>
<mo>|</mo><mo>∑</mo><msup><mo>|</mo><mn>1/2</mn></msup>
</mrow>
</mfrac>
<mi>exp</mi>
<mo>(</mo>
<mo>-</mo>
<mfrac>
<mrow><mn>1</mn></mrow>
<mrow><mn>2</mn></mrow>
</mfrac>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo>-</mo>
<mi>μ</mi>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup>
<mo>∑</mo>
<mn>-1</mn>
</msup>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo>-</mo>
<mi>μ</mi>
<mo stretchy="false">)</mo>
<mo>)</mo>
</math>

* 其中<math><mi>μ</mi></math>为 N 维矩阵，<math><mi>μ</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> <msup><mi>x</mi><mi>(i)</mi></msup> </math>
* <math><mo>∑</mo></math>为 NxN 协方差矩阵，<math><mo>∑</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><mi>μ</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mi>T</mi></msup></math>

多元高斯函数以及协方差的变化对其影响如下：

![](/assets/images/2017/09/ml-10-4.png)

![](/assets/images/2017/09/ml-10-5.png)

![](/assets/images/2017/09/ml-10-6.png)

![](/assets/images/2017/09/ml-10-7.png)

我们可以使用新的概率模型来替换上面式子，同样根据阈值判断是否为异常数据<math><mi>P</mi> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo><mo><</mo><mi>ε</mi></math>。和之前的公式相比，在数学上一元高斯函数相当于是多元高斯函数的一个特例（协方差矩阵为对角阵）

![](/assets/images/2017/09/ml-10-8.png)

和一元高斯函数相比，多元高斯函数计算复杂度更高，不利于大规模计算以及特征的扩展，当 N 很大时，计算 NxN 的<math><mo>∑</mo></math>矩阵的逆矩阵会很耗时。此外多元高斯模型还要求 m（训练集大小）远大于（至少 10 倍）n（特征数量），否则<math><mo>∑</mo></math>是奇异矩阵，不可逆。而一元模型及时在 m 很小的情况下也可以很好的预测，因此通常情况下我们使用一元模型

![](/assets/images/2017/09/ml-10-9.png)

如果发现计算多元高斯模型时，<math><mo>∑</mo></math>是奇异矩阵不可逆，通常有两种情况，一是 m 小于 n，另一种是包含冗余特征（特征之间线性相关），比如<math><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><msub><mi>x</mi><mn>2</mn></msub></math>,<math><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><msub><mi>x</mi><mn>4</mn></msub><mo>+</mo><msub><mi>x</mi><mn>5</mn></msub></math>等

### 基于内容的推荐系统

假设我们有如下数据，左边是电影名称，右边是用户给电影的评分，假设每部电影有两个特征，分别是爱情片和动作片。我们的目的是根据用户的打分习惯推测出用户未评分的电影分数。

![](/assets/images/2017/09/ml-10-10.png)

* 使用<math><msub><mi>n</mi><mi>u</mi></msub></math>表示用户数量，上面例子中<math><msub><mi>N</mi><mi>u</mi></msub><mo>=</mo><mn>4</mn></math>
* 使用<math><msub><mi>n</mi><mi>m</mi></msub></math>表示电影数量，上面例子中<math><msub><mi>N</mi><mi>m</mi></msub><mo>=</mo><mn>5</mn></math>
* 使用<math><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></math>表示用户<math><mi>j</mi></math>是否对电影<math><mi>i</mi></math>进行过评分（1 或者 0）
* 使用<math><msup><mi>y</mi><mi>(i,j)</mi></msup></math>表示用户<math><mi>j</mi></math>对电影<math><mi>i</mi></math>的评分
* 使用<math><msup><mi>x</mi><mi>(i)</mi></msup></math>表示电影<math><mi>i</mi></math>样本特征向量，其中<math><msub><mi>x</mi><mn>0</mn></msub><mo>=</mo><mn>1</mn></math>，以上面例子第一条数据为例：<math> <msup><mi>x</mi><mi>(1)</mi></msup> <mo>=</mo> <mo>[</mo> <mtable> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0.9</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> </mtable> <mo>]</mo></math>
* 使用<math><msup><mi>θ</mi><mi>(j)</mi></msup></math>表示用户<math><mi>j</mi></math>的参数向量，维度为(N+1)X1
* 则用户对<math><mi>j</mi></math>电影<math><mi>i</mi></math>的评分的**线性模型**为

<math display="block">
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo>=</mo>
<mo stretchy="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
</math>

上例中，以计算 Alice 对《Cute puppies of love》的评分为例：<math> <msup><mi>x</mi><mi>(3)</mi></msup> <mo>=</mo> <mo>[</mo> <mtable> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0.99</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> </mtable><mo>]</mo></math>，假设 θ 已知，<math><msup><mi>θ</mi><mi>(1)</mi></msup><mo>=</mo><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></math>，则<math><mo stretchy="false">(</mo><msup><mi>θ</mi><mi>(1)</mi></msup><msup><mo>)</mo><mi>T</mi></msup><msup><mi>x</mi><mi>(3)</mi></msup><mo>=</mo><mn>5</mn><mo> \* </mo><mn>0.99</mn><mo>=</mo><mn>4.95</mn></math>。抽象出来，对单用户<math><mi>j</mi></math>求解<math><mi>θ</mi></math>的代价函数以及优化方程为:

<math display="block">
<munder>
<mtext>min</mtext>
<mrow><msup><mi>θ</mi><mi>(j)</mi></msup></mrow>
</munder>
<mfrac><mn>1</mn><mn>2</mn></mfrac>
<munder>
<mo>∑</mo>
<mi>i:r(i,j)=1</mi>
</munder>
<mo>(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<msup><mo>)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
</math>

对所有用户，求解<math><msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><msub><mi>n</mi><mi>u</mi></msub></msup></math>:

<math display="block">
<munder>
<mtext>min</mtext>
<mrow><msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><mi>nu</mi></msup></mrow>
</munder>
<mfrac><mn>1</mn><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nu</mi> </mrow></munderover>
<munder><mo>∑</mo><mi>i:r(i,j)=1</mi></munder>
<mo>(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<msup><mo>)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nu</mi></mrow> </munderover>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi></mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
</math>

第一个求和项表示对所有用户 j，累加他们所有评过分的电影总和。对上述式子进行梯度下降，求解 θ 值

<math display="block">
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<munder><mo>∑</mo><mi>i:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mspace width="0.5em"/>
<mtext>(for k=0)</mtext>
</math>
<br>
<math display="block">
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<mo>(</mo>
<munder><mo>∑</mo><mi>i:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>+</mo>
<mi>λ</mi>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>)</mo>
<mspace width="0.5em"/>
<mtext>(for k ≠ 0)</mtext>
</math>

#### 协同过滤(Collaborative Filtering)

协同过滤是一种构建推荐系统的方法，它有一个特点是 feature learning，算法可以自行学习所要使用的特征。还是以上面电影评分为例，对于<math><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub></math>两个特征，我们假设有人会告诉我们他们的值，比如某个电影的浪漫指数是多少，动作指数是多少。但是要让每个人都看一遍这些电影然后收集这两个样本的数据是非常耗时的一件事情，有时也是不切实际的，此外，在除了这两个 feature 之外我们还想要获取更多的 feature，从哪里能得到这些 feature 呢。如果我们有一种算法可以在已知个人偏好（θ 值）的前提下，自行推算出 feature 值。如下图所示

![](/assets/images/2017/09/ml-10-11.png)

假设 Alice, Bob, Carol, Dave 提前告诉了我们他们的个人偏好，即<math><msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(3)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(4)</mi></msup></math>，根据这些特征向量我们可以推测对于第一部电影 Alice 和 Bob 喜欢，Carol 和 Dave 不喜欢，因此它可能是一部爱情片而不是动作片，即<math><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><mn>1</mn><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></math>。数学上看是需要找到<math><msup><mi>x</mi><mi>(1)</mi></msup></math>使<math><mo stretchy="false">(</mo> <msup><mi>θ</mi><mi>(1)</mi></msup> <msup> <mo stretchy="false">)</mo> <mi>T</mi> </msup> <msup><mi>x</mi> <mi>(1)</mi> </msup><mo>≈</mo><mn>5</mn><mo>,</mo><mo stretchy="false">(</mo> <msup><mi>θ</mi><mi>(2)</mi></msup> <msup> <mo stretchy="false">)</mo> <mi>T</mi> </msup> <msup><mi>x</mi> <mi>(2)</mi> </msup><mo>≈</mo><mn>5</mn><mo>,</mo><mo stretchy="false">(</mo> <msup><mi>θ</mi><mi>(3)</mi></msup> <msup> <mo stretchy="false">)</mo> <mi>T</mi> </msup> <msup><mi>x</mi> <mi>(3)</mi> </msup><mo>≈</mo><mn>0</mn><mo>,</mo><mo stretchy="false">(</mo> <msup><mi>θ</mi><mi>(4)</mi></msup> <msup> <mo stretchy="false">)</mo> <mi>T</mi> </msup> <msup><mi>x</mi> <mi>(4)</mi> </msup><mo>≈</mo><mn>0</mn></math>。类似的可以计算出<math><msup><mi>x</mi> <mi>(2)</mi> </msup></math>,<math><msup><mi>x</mi> <mi>(3)</mi> </msup></math>,<math><msup><mi>x</mi> <mi>(4)</mi> </msup></math>...

因此我们的优化目标为：给定<math><msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><msub><mi>n</mi><mi>u</mi></msub></msup></math>，对单个 feature <math><msup><mi>x</mi> <mi>(i)</mi> </msup></math> 有:

<math display="block">
<munder>
<mtext>min</mtext>
<mrow><msup><mi>x</mi><mi>(i)</mi></msup></mrow>
</munder>
<mfrac><mn>1</mn><mn>2</mn></mfrac>
<munder>
<mo>∑</mo><mi>j:r(i,j)=1</mi>
</munder>
<mo>(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<msup><mo>)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
</math>

对多个 feature，给定<math><msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>θ</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><msub><mi>n</mi><mi>u</mi></msub></msup></math>，求解 <math><msup><mi>x</mi> <mi>(1)</mi> </msup></math>，<math><msup><mi>x</mi> <mi>(2)</mi> </msup></math>，<math><msup><mi>x</mi> <mi>(3)</mi> </msup></math>...<math><msup><mi>x</mi> <mi>(n)</mi> </msup></math> 有：

<math display="block">
<munder>
<mtext>min</mtext>
<mrow><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(nm)</mi></msup></mrow>
</munder>
<mfrac><mn>1</mn><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nm</mi> </mrow></munderover>
<munder><mo>∑</mo><mi>i:r(i,j)=1</mi></munder>
<mo>(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<msup><mo>)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nm</mi> </mrow> </munderover>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(j)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
</math>

使用梯度下降法求解<math><msup><mi>x</mi> <mi>(i)</mi> </msup></math>

<math display="block">
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<munder><mo>∑</mo><mi>j:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mspace width="0.5em"/>
<mtext>(for k=0)</mtext>
</math>
<br>
<math display="block">
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<mo>(</mo>
<munder><mo>∑</mo><mi>j:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>+</mo>
<mi>λ</mi>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>)</mo>
<mspace width="0.5em"/>
<mtext>(for k ≠ 0)</mtext>
</math>

对比上一节的公式可以发现，上一节的公式是已知 x 求 θ，这节是已知 θ 求 x，那正确的求解顺序是怎样呢，一种做法是先 Guess 一组 θ 值然后求解 x，再求解 θ 再求解 x，循环直到算法收敛。这种方式可以得到最终的 θ 和 x 值，但是计算过于繁琐和低效，还有一种方式是将两个优化函数联合起来得到一个新的优化函数：

<math display="block">
<mi>J</mi>
<mo stretchy="false">(</mo>
<msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(nm)</mi></msup>
<mo>,</mo>
<msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><mi>(nu)</mi></msup>
<mo stretchy="false">)</mo>
<mo>=</mo>
<mfrac><mn>1</mn><mn>2</mn></mfrac>
<munder><mo>∑</mo><mi>(i,j):r(i,j)=1</mi></munder>
<mo>(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<msup><mo>)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nm</mi> </mrow> </munderover>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(j)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
<mo>+</mo>
<mfrac><mi>λ</mi><mn>2</mn></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>nu</mi></mrow> </munderover>
<munderover> <mo>∑</mo> <mrow> <mi>k</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi></mrow> </munderover>
<mo stretchy="false">(</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
</math>

这个式子分别对 x，θ 求导可以还原出上面式子，另外上述式子成立的前提是 x，θ 都为 n 维矩阵，不需要对<math><msub><mi>θ</mi><mn>0</mn></msub></math>和<math><msub><mi>x</mi><mn>0</mn></msub></math>做特殊处理。总结一下协同过滤算法：

* 初始化<math><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(nm)</mi></msup>
  <mo>,</mo>
  <msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><mi>(nu)</mi></msup>
  </math>为较小的随机数
* 使用低度下降或其它最优化方法求<math><mi>J</mi> <mo stretchy="false">(</mo> <msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(nm)</mi></msup> <mo>,</mo> <msup><mi>θ</mi><mi>(1)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>θ</mi><mi>(nu)</mi></msup> <mo stretchy="false">)</mo></math>的最小值，得到最优解 θ 和 x,不需要对<math><msub><mi>θ</mi><mn>0</mn></msub></math>和<math><msub><mi>x</mi><mn>0</mn></msub></math>做特殊处理.

<math display="block">
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<mo>(</mo>
<munder><mo>∑</mo><mi>j:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>+</mo>
<mi>λ</mi>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>)</mo>
</math>
<br>
<math display="block">
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>:=</mo>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>-</mo>
<mi>α</mi>
<mo>(</mo>
<munder><mo>∑</mo><mi>i:r(i,j)=1</mi></munder>
<mo stretchy="false">(</mo>
<mo stretchy ="false">(</mo>
<msup><mi>θ</mi><mi>(j)</mi></msup>
<msup>
<mo stretchy="false">)</mo>
<mi>T</mi>
</msup>
<msup><mi>x</mi>
<mi>(i)</mi>
</msup>
<mo>-</mo>
<msup><mi>y</mi><mi>(i,j)</mi></msup>
<mo stretchy="false">)</mo>
<msubsup><mi>x</mi><mi>k</mi><mi>(i)</mi></msubsup>
<mo>+</mo>
<mi>λ</mi>
<msubsup><mi>θ</mi><mi>k</mi><mi>(j)</mi></msubsup>
<mo>)</mo>
</math>

* 对用户的打分计算使用公式<math><msup><mi>θ</mi><mn>T</mn></msup><mi>x</mi></math>

#### 向量化实现

上述计算也可以通过向量化来表示：

![](/assets/images/2017/09/ml-10-12.png)

通过对 Y 矩阵的低秩分解求得 θ 和 x。

另一个问题是，我们同样可以通过协同过滤来找到和某个电影相似主题的电影，假设对于电影<math><mi>i</mi></math>，我们使用协同过滤的算法得到了一系列特征向量<math><msup><mi>x</mi><mi>(i)</mi></msup></math>，比如<math><msup><mi>x</mi><mi>(1)</mi></msup><mo>=</mo><mtext>romance</mtext></math>,<math><msup><mi>x</mi><mi>(2)</mi></msup><mo>=</mo><mtext>action</mtext></math>,<math><msup><mi>x</mi><mi>(3)</mi></msup><mo>=</mo><mtext>comdy</mtext></math>...。怎么找到另一部电影<math><mi>j</mi></math>和它类似？可以使用下面式子

<math display="block">
	<mtext>Find smallest</mtext>
	<mspace width="1em" />
	<mo>||</mo>
	<msup><mi>x</mi><mi>(i)</mi></msup>
	<mo>-</mo>
	<msup><mi>x</mi><mi>(j)</mi></msup>
	<mo>||</mo>
</math>

### 均值归一化

还是上述例子，假设用户 Eve 没有对任何电影评分过，使用上述模型进行计算，将会得到 0 分的结果，如下图所示

![](/assets/images/2017/09/ml-10-13.png)

解决方式是对于这种情况，使用均值来代替

![](/assets/images/2017/09/ml-10-14.png)
