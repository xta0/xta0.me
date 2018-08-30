---
layout: post
list_title: 机器学习 | Machine Learning | 分类 | Classification
title: 分类
meta: Coursera Stanford Machine Learning Cousre Note, Chapter3
categories: [Machine Learning,AI]
mathjax: true
---

### Classification

分类问题, Email: Spam/NotSpam, Tumor: 恶性/良性

<math display="block">
<mi>y</mi>
<mo>∈</mo>
<mo>{</mo>
<mn>0</mn>
<mo>,</mo>
<mn>1</mn>
<mo>}</mo>
</math>

* 0："Negative Class"(e.g., benign tumor)
* 1: "Positive Class"(e.g., malignant tumor)

对于分类场景，使用线性回归模型不适合，原因是 <math><msub><mi>h</mi><mi>θ</mi></msub></math> 的值区间不能保证在[0,1]之间，因此需要一种新的模型，叫做 logistic regression，逻辑回归。

### Logistic Regression Model

逻辑回归 wiki

在给出模型前，先不考虑 y 的取值是离散的，我们希望能使：<math><mn>0</mn><mo>≤</mo><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≤</mo><mn>1</mn></math>，可以将式子做一些变换：

* 线性函数：<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>X</mi></math>
* 做如下变换：<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>)</mo></math>, 另<math><mi>z</mi><mo>=</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>,</mo><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>

* 得到函数：<math><mi>g</mi><mo stretchy="false">(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mstyle></math>
  函数曲线如下:

![](/assets/images/2017/09/ml-5-1.png)

函数`g(z)`, 如上图将所有实数映射到了(0,1]空间内，这使他可以将任意一个 h(x)的值空间转化为适合分类器取值的空间, `g(z)`也叫做**Sigmoid Function**`hθ(x)`的输出是结果是`1`的概率，比如`hθ(x)=0.7`表示 70%的概率我们的输出结果为`1`，因此输出是`0`的概率则是 30%：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
   <msub>
     <mi>h</mi>
     <mi>θ</mi>
   </msub>
   <mo stretchy="false">(</mo>
   <mi>x</mi>
   <mo stretchy="false">)</mo>
   <mo>=</mo>
   <mi>P</mi>
   <mo stretchy="false">(</mo>
   <mi>y</mi>
   <mo>=</mo>
   <mn>1</mn>
   <mrow class="MJX-TeXAtom-ORD">
     <mo stretchy="false">|</mo>
   </mrow>
   <mi>x</mi>
   <mo>;</mo>
   <mi>θ</mi>
   <mo stretchy="false">)</mo>
   <mo>=</mo>
   <mn>1</mn>
   <mo>−</mo>
   <mi>P</mi>
   <mo stretchy="false">(</mo>
   <mi>y</mi>
   <mo>=</mo>
   <mn>0</mn>
   <mrow class="MJX-TeXAtom-ORD">
     <mo stretchy="false">|</mo>
   </mrow>
   <mi>x</mi>
   <mo>;</mo>
	 <mi>θ</mi>
   <mo stretchy="false">)</mo>
   <mo>,</mo>
   <mspace width="1em"></mspace>
	<mi>P</mi>
   <mo stretchy="false">(</mo>
   <mi>y</mi>
   <mo>=</mo>
   <mn>0</mn>
   <mrow class="MJX-TeXAtom-ORD">
     <mo stretchy="false">|</mo>
   </mrow>
   <mi>x</mi>
   <mo>;</mo>
   <mi>θ</mi>
   <mo stretchy="false">)</mo>
   <mo>+</mo>
   <mi>P</mi>
   <mo stretchy="false">(</mo>
   <mi>y</mi>
   <mo>=</mo>
   <mn>1</mn>
   <mrow class="MJX-TeXAtom-ORD">
     <mo stretchy="false">|</mo>
   </mrow>
   <mi>x</mi>
   <mo>;</mo>
   <mi>θ</mi>
   <mo stretchy="false">)</mo>
   <mo>=</mo>
   <mn>1</mn>
</math>

### Decision Boundary

对函数<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>)</mo></math>，假设：

* <math><mo>"</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>"</mo></math> if <math><math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>0.5</mn></math>

* <math><mo>"</mo><mi>y</mi><mo>=</mo><mn>0</mn><mo>"</mo></math> if <math><math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo><</mo><mn>0.5</mn></math>

通过观察函数<math><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>上一节的曲线可以发现，当`z`大于 0 的时候`g(z)≥0.5`，因此只需要<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo> > </mo><mn>0</mn></math>即可，即：

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <mi>x</mi>
        <mo>&#x2265;<!-- ≥ --></mo>
        <mn>0</mn>
        <mo stretchy="false">&#x21D2;<!-- ⇒ --></mo>
        <mi>y</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <mi>x</mi>
        <mo>&lt;</mo>
        <mn>0</mn>
        <mo stretchy="false">&#x21D2;<!-- ⇒ --></mo>
        <mi>y</mi>
        <mo>=</mo>
        <mn>0</mn>
      </mtd>
    </mtr>
  </mtable>
</math>

`g(z)`小于`0.5`的情况同理。

#### linear decision boundaries

举个例子，假设有一个线性预测函数<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo>g</mo><mo>(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>)</mo></math>，其中<math><mo>θ</mo></math>的值已经确定为`[-3,1,1]`，则问题变为求如果要`y=1`，那么需要<math><mi>h</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>-3</mn><mo>+</mo><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>≥</mo><mn>0</mn></math>，即找到<math><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub></math>满足<math><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>≥</mo><mn>3</mn></math>

如下图所示：

![](/assets/images/2017/09/ml-5-2.png)

由上图可看出：<math><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>3</mn></math>可作为**Boundary Function**，也叫**Decision Boundary**

#### Non-linear decision boundaries

对于非线性的预测函数，例如：

<math display="block">
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>g</mo>
  <mo>(</mo>
  <msub>
    <mi>θ</mi>
    <mn>0</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>θ</mi>
    <mn>2</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>2</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>θ</mi>
    <mn>3</mn>
  </msub>
  <msubsup>
    <mi>x</mi>
    <mn>1</mn>
    <mn>2</mn>
  </msubsup>
	<mo>+</mo>
	<msub>
    <mi>θ</mi>
    <mn>4</mn>
  </msub>
  <msubsup>
    <mi>x</mi>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo>)</mo>
</math>

假设`θ`值已经确定`[-1,0,0,1,1]`，同上，变为求如果要`y=1`，那么需要<math><mn>-1</mn><mo>+</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup><mo>≥</mo><mn>0</mn></math>，即找到<math><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub></math>满足<math><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup><mo>≥</mo><mn>0</mn></math>，则边界函数为<math><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup><mo>=</mo><mn>0</mn></math>，如下图所示

![](/assets/images/2017/09/ml-5-3.png)

则落在圈外的样本点，可以预测`y=1`

### Cost Function

由上面可知，给定：

* 训练集<math><mo>{</mo><mo>(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo>)</mo><mo>,</mo><mo>...</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo>)</mo><mo>}</mo></math>，m 个样本，其中:<math><mi>x</mi><mo>∈</mo><mo>[</mo><mtable> <mtr> <msub><mi>x</mi><mi>1</mi></msub> </mtr> <mtr> <msub><mi>x</mi><mi>2</mi></msub> </mtr> <mtr> <mtd><mo>...</mo></mtd> </mtr> <mtr> <msub><mi>x</mi><mi>n</mi></msub> </mtr></mtable><mo>]</mo><mspace width="1em"></mspace><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo><mi>y</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0,1</mn><mo stretchy="false">}</mo></math>

* 预测函数: <math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mstyle></math>

问题还是怎么求解<math><mo>θ</mo></math>，如果使用之前先行回归的 cost function，即 <math><mi>J</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover> <mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi> <mi>i</mi></msub><mo stretchy="false">)</mo><mo>−</mo><msub><mi>y</mi><mi>i</mi></msub> <msup><mo>)</mo><mn>2</mn></msup></math>

这时会出现`J(θ)`不是 convex function 的情况，原因是`h(x)`变成了复杂的非线性函数，因此梯度下降无法得到最小值（得到极小值的概率更高）。

* 逻辑回归的 Cost Function：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mstyle>
          <mfrac>
            <mn>1</mn>
            <mi>m</mi>
          </mfrac>
        </mstyle>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>m</mi>
        </munderover>
        <mrow class="MJX-TeXAtom-ORD">
          <mi mathvariant="normal">C</mi>
          <mi mathvariant="normal">o</mi>
          <mi mathvariant="normal">s</mi>
          <mi mathvariant="normal">t</mi>
        </mrow>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
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
        <mo>,</mo>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mrow class="MJX-TeXAtom-ORD">
          <mi mathvariant="normal">C</mi>
          <mi mathvariant="normal">o</mi>
          <mi mathvariant="normal">s</mi>
          <mi mathvariant="normal">t</mi>
        </mrow>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>,</mo>
        <mi>y</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mo>&#x2212;<!-- − --></mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo stretchy="false">)</mo>
        <mspace width="thickmathspace" />
      </mtd>
      <mtd>
        <mtext>if y = 1</mtext>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mrow class="MJX-TeXAtom-ORD">
          <mi mathvariant="normal">C</mi>
          <mi mathvariant="normal">o</mi>
          <mi mathvariant="normal">s</mi>
          <mi mathvariant="normal">t</mi>
        </mrow>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>,</mo>
        <mi>y</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mo>&#x2212;<!-- − --></mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mo stretchy="false">(</mo>
        <mn>1</mn>
        <mo>&#x2212;<!-- − --></mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo stretchy="false">)</mo>
        <mspace width="thickmathspace" />
      </mtd>
      <mtd>
        <mtext>if y = 0</mtext>
      </mtd>
    </mtr>
  </mtable>
</math>

当`y=1`的时候，`J(θ) = 0` -> `h(x)=1`；`J(θ) = ∞` -> `h(x)=0`，如下图所示

![](/assets/images/2017/09/ml-5-2.png)

当`y=0`的时候，`J(θ) =0` -> `h(x)=0`，`J(θ) = ∞` -> `h(x)=1`，如下图所示

![](/assets/images/2017/09/ml-5-3.png)

图上可以看出`J(θ)`有极值点，接下来的问题就是分别求解`h(x)=0`和`h(x)=1`两种情况下的`θ`值

### Simplifed Cost Function

上述 Cost Function 可以简化为一行：

<math display="block">
  <mrow class="MJX-TeXAtom-ORD">
    <mi mathvariant="normal">C</mi>
    <mi mathvariant="normal">o</mi>
    <mi mathvariant="normal">s</mi>
    <mi mathvariant="normal">t</mi>
  </mrow>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>h</mi>
    <mi>&#x03B8;<!-- θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>,</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mi>y</mi>
  <mspace width="thickmathspace" />
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>h</mi>
    <mi>&#x03B8;<!-- θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;<!-- − --></mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;<!-- − --></mo>
  <msub>
    <mi>h</mi>
    <mi>&#x03B8;<!-- θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
</math>

之所以将上述式子简化为一行，其目的是方便使用概率论中的最大似然估计求解，接下来还是通过梯度下降法求解<math><mi>min></mi><msub><mi>J</mi><mo>(θ)</mo></msub></math>。<math><msub><mi>J</mi><mo>(θ)</mo></msub></math>可完整写为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>&#x03B8;<!-- θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <mstyle displaystyle="true">
    <munderover>
      <mo>&#x2211;<!-- ∑ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mi>m</mi>
    </munderover>
    <mo stretchy="false">[</mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mo>+</mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mo stretchy="false">)</mo>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mo stretchy="false">]</mo>
  </mstyle>
</math>

向量化的实现为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>h</mi>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <mi>X</mi>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mfrac>
          <mn>1</mn>
          <mi>m</mi>
        </mfrac>
        <mo>&#x22C5;<!-- ⋅ --></mo>
        <mfenced open="(" close=")">
          <mrow>
            <mo>&#x2212;<!-- − --></mo>
            <msup>
              <mi>y</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>T</mi>
              </mrow>
            </msup>
            <mi>log</mi>
            <mo>&#x2061;<!-- ⁡ --></mo>
            <mo stretchy="false">(</mo>
            <mi>h</mi>
            <mo stretchy="false">)</mo>
            <mo>&#x2212;<!-- − --></mo>
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo>&#x2212;<!-- − --></mo>
            <mi>y</mi>
            <msup>
              <mo stretchy="false">)</mo>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>T</mi>
              </mrow>
            </msup>
            <mi>log</mi>
            <mo>&#x2061;<!-- ⁡ --></mo>
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo>&#x2212;<!-- − --></mo>
            <mi>h</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </mfenced>
      </mtd>
    </mtr>
  </mtable>
</math>

### Gradient Descent

前面可知梯度下降公式为:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>R</mi>
        <mi>e</mi>
        <mi>p</mi>
        <mi>e</mi>
        <mi>a</mi>
        <mi>t</mi>
        <mspace width="thickmathspace" />
        <mo fence="false" stretchy="false">{</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mspace width="thickmathspace" />
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>:=</mo>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03B1;<!-- α --></mi>
        <mstyle>
          <mfrac>
            <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
            <mrow>
              <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
              <msub>
                <mi>&#x03B8;<!-- θ --></mi>
                <mi>j</mi>
              </msub>
            </mrow>
          </mfrac>
        </mstyle>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

对<math><msub><mi>J</mi><mo>(θ)</mo></msub></math>求偏导，得到梯度下降公式：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>R</mi>
        <mi>e</mi>
        <mi>p</mi>
        <mi>e</mi>
        <mi>a</mi>
        <mi>t</mi>
        <mspace width="thickmathspace" />
        <mo fence="false" stretchy="false">{</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mspace width="thickmathspace" />
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>:=</mo>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mfrac>
          <mi>&#x03B1;<!-- α --></mi>
          <mi>m</mi>
        </mfrac>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>m</mi>
        </munderover>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
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
        <mo>&#x2212;<!-- − --></mo>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
        <msubsup>
          <mi>x</mi>
          <mi>j</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

注意到上述公式和线性回归使用的梯度下降公式相同，不同的是`h(θ)`，上述公式向量化表示为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x03B8;<!-- θ --></mi>
  <mo>:=</mo>
  <mi>&#x03B8;<!-- θ --></mi>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mi>&#x03B1;<!-- α --></mi>
    <mi>m</mi>
  </mfrac>
  <msup>
    <mi>X</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>T</mi>
    </mrow>
  </msup>
  <mo stretchy="false">(</mo>
  <mi>g</mi>
  <mo stretchy="false">(</mo>
  <mi>X</mi>
  <mi>&#x03B8;<!-- θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mover>
      <mi>y</mi>
      <mo stretchy="false">&#x2192;<!-- → --></mo>
    </mover>
  </mrow>
  <mo stretchy="false">)</mo>
</math>

### Advanced Optimization

对于求解<math><msub><mi>J</mi><mo>(θ)</mo></msub></math>的最小值，目前的做法是使用梯度下降法，对`θ`求偏导，除了梯度下降之外，还有其它几种优化算法：

* Conjugate gradient
* BFGS
* L-BFGS

这三种算法的优点是：

* 不需要人工选择`α`
* 比低度下降更快

缺点是：比较复杂

开发者不需要自己实现这些算法，在一般的数值计算库里都有相应的实现，例如 python，Octave 等。我们只需要关心两个问题，如何给出:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mstyle>
          <mfrac>
            <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
            <mrow>
              <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
              <msub>
                <mi>&#x03B8;<!-- θ --></mi>
                <mi>j</mi>
              </msub>
            </mrow>
          </mfrac>
        </mstyle>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

我们可以使用 Octave 写这样一个函数:

```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

然后使用 Octave 自带的`fminunc()`优化算法计算出`θ`的值：

```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

`fminunc()`函数接受三个参数：costFunction，初始的 θ 值（至少是 2x1 的向量），还有 options

### Multiclass classification

前面我们解决的问题都是针对两个场景进行分类，即 y = {0,1} 实际生活中，常常有多个场景需要归类，例如

* Email foldering/tagging：Work，Friends，Family，Hobby
* Weather：Sunny，Cloudy，Rain，Snow

即分类结果不只是 0 或 1，而是多个 y = {0,1...n}，解决多个分类我们使用 one-vs-all 的方式，即选取某一个场景进行归类时，将其他场景合并为起对立的场景。如图：

![](/assets/images/2017/09/ml-5-4.png)

上图可知我们先取一个 class 进行计算，将其他的归类为另一个 class，这样就可以使用前面提到的 binary regression model 进行计算，即

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>y</mi>
        <mo>&#x2208;<!-- ∈ --></mo>
        <mo fence="false" stretchy="false">{</mo>
        <mn>0</mn>
        <mo>,</mo>
        <mn>1</mn>
        <mo>.</mo>
        <mo>.</mo>
        <mo>.</mo>
        <mi>n</mi>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msubsup>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>0</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mi>P</mi>
        <mo stretchy="false">(</mo>
        <mi>y</mi>
        <mo>=</mo>
        <mn>0</mn>
        <mrow class="MJX-TeXAtom-ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <mi>x</mi>
        <mo>;</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msubsup>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mi>P</mi>
        <mo stretchy="false">(</mo>
        <mi>y</mi>
        <mo>=</mo>
        <mn>1</mn>
        <mrow class="MJX-TeXAtom-ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <mi>x</mi>
        <mo>;</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mo>&#x22EF;<!-- ⋯ --></mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msubsup>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>n</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mi>P</mi>
        <mo stretchy="false">(</mo>
        <mi>y</mi>
        <mo>=</mo>
        <mi>n</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <mi>x</mi>
        <mo>;</mo>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mrow class="MJX-TeXAtom-ORD">
          <mi mathvariant="normal">p</mi>
          <mi mathvariant="normal">r</mi>
          <mi mathvariant="normal">e</mi>
          <mi mathvariant="normal">d</mi>
          <mi mathvariant="normal">i</mi>
          <mi mathvariant="normal">c</mi>
          <mi mathvariant="normal">t</mi>
          <mi mathvariant="normal">i</mi>
          <mi mathvariant="normal">o</mi>
          <mi mathvariant="normal">n</mi>
        </mrow>
        <mo>=</mo>
        <munder>
          <mo movablelimits="true">max</mo>
          <mi>i</mi>
        </munder>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

* One-vs-all(one-vs-rest):

Train a logistic regression classifier<math><msubsup><mi>h</mi><mi>θ</mi><mi>(i)</mi></msubsup><mi>(x)</mi></math> for each class <math><mi>i</mi></math> to predict the probability that <math><mi>y</mi><mo>=</mo><mi>i</mi></math>.

On a new input <math><mi>x</mi></math>, to make a prediction, pick the class <math><mi>i</mi></math> that maximizes

* 总结一下就是对每种分类先计算他的<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi></math>，当有一个新的`x`需要分类时，选一个让<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi></math>值最大的分类器。

### 附录：Regularization

### The problem of overfitting

以线性回归的预测房价为例，如下图所示：

![](/assets/images/2017/09/ml-5-5.png)

可以看到:

* 通过线性函数预测房价不够精确(underfit)，术语叫做"High Bias"，其原因主要是样本的 feature 太少了。
* 通过二次函数拟合出来的曲线刚好可以贯穿大部分数据样本，术语叫做"Just Right"
* 通过四阶多项式拟合出来的曲线虽然能贯穿所有数据样本，但是曲线本身不够规则，当有新样本出现时不能很好的预测。这种情况我们叫做**Over Fitting（过度拟合）**，术语叫做"High variance"。If we have too many features, the learned hypothesis may fit the training set very well（`J(θ)=0`）, but fail to generalize to new examples(predict prices on new examples) Over Fitting 的问题在样本少，feature 多的时候很明显

* Addressing overfitting: - Reduce number of features - Manually select which features to keep. - Model selection algorithm - Regularization - Keep all features, but reduce magnitude/values of parameters <math><msub><mi>θ</mi><mi>j</mi></msub></math> - Works well when we have a lot of features, each of which contributes a bit to predicting<math><mi>y</mi></math>


### Regularization Cost Function

如果要减少 overfitting 的情况，我们可以降低一些参数的权重，假设我们想要让下面的函数变成二次方程：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>0</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>1</mn>
  </msub>
  <mi>x</mi>
  <mo>+</mo>
  <msub>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>2</mn>
  </msub>
  <msup>
    <mi>x</mi>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msub>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>3</mn>
  </msub>
  <msup>
    <mi>x</mi>
    <mn>3</mn>
  </msup>
  <mo>+</mo>
  <msub>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>4</mn>
  </msub>
  <msup>
    <mi>x</mi>
    <mn>4</mn>
  </msup>
</math>

我们想要在不去掉 θ3 和 θ4 的前提下，降低<math><msub><mi>θ</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup></math>和<math><msub><mi>θ</mi><mn>4</mn></msub><msup><mi>x</mi><mn>4</mn></msup></math>的影响，我们可以修改 cost function 为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>m</mi>
  <mi>i</mi>
  <msub>
    <mi>n</mi>
    <mi>&#x03B8;<!-- θ --></mi>
  </msub>
  <mtext>&#xA0;</mtext>
  <mstyle displaystyle="true">
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <mi>m</mi>
      </mrow>
    </mfrac>
  </mstyle>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
  </munderover>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>h</mi>
    <mi>&#x03B8;<!-- θ --></mi>
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
  <mo>&#x2212;<!-- − --></mo>
  <msup>
    <mi>y</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <mn>1000</mn>
  <mo>&#x22C5;<!-- ⋅ --></mo>
  <msubsup>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>3</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <mn>1000</mn>
  <mo>&#x22C5;<!-- ⋅ --></mo>
  <msubsup>
    <mi>&#x03B8;<!-- θ --></mi>
    <mn>4</mn>
    <mn>2</mn>
  </msubsup>
</math>

我们在原来的 cost function 后面增加了两项，为了让 cost function 接近 0，我们需要让<math><msub><mi>θ</mi><mn>3</mn></msub></math>,<math><msub><mi>θ</mi><mn>4</mn></msub></math>近似为 0，这样即会极大的减少<math><msub><mi>θ</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup></math>和<math><msub><mi>θ</mi><mn>4</mn></msub><msup><mi>x</mi><mn>4</mn></msup></math>的值，减少后的曲线如下图粉色曲线，更接近二次函数：

![](/assets/images/2017/09/ml-5-6.png)

Small values for parameters <math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>n</mn></msub></math>

* "Simpler" hypothesis，选取更小的`θ`值能得到更简单的预测函数，例如上面的例子，如果将 <math><msub><mi>θ</mi><mn>3</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>4</mn></msub></math>近似为 0 的话，那么上述函数将变为二次方程，更贴近合理的假设函数
* Housing example: - Feature: <math><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo><msub><mi>x</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>100</mn></msub></math> - Parameters: <math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>100</mn></msub></math>

100 个 feature，如何有效的选取这些`θ`呢，改变 cost function：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mstyle displaystyle="true">
    <mi>J</mi>
    <mo stretchy="false">(</mo>
    <mi>&#x03B8;<!-- θ --></mi>
    <mo stretchy="false">)</mo>
    <mo>=</mo>
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <mi>m</mi>
      </mrow>
    </mfrac>
    <mfenced open="[" close="]">
      <mrow>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>m</mi>
        </munderover>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
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
        <mo>&#x2212;<!-- − --></mo>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
        <mo>+</mo>
        <mi>&#x03BB;<!-- λ --></mi>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>n</mi>
        </munderover>
        <msubsup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
          <mn>2</mn>
        </msubsup>
      </mrow>
    </mfenced>
  </mstyle>
</math>

参数 λ 叫做**regularization parameter**，它的作用是既要保证曲线的拟合程度够高同时又要确保 θ 的值尽量小。如果 λ 的值选取过大，例如<math><mi>λ</mi><mo>=</mo><msup><mi>10</mi><mi>10</mi></msup></math>，会导致计算出的所有的 θ 值都接近 0，从而使<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi><mo>=</mo><msub><mi>θ</mi><mi>0</mi></msub></math> 即产生"Under fitting"

### Regularized linear regression

有了上面的式子，我们可以将它应用到线性回归：

* 修改梯度下降公式为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mtext>Repeat</mtext>
        <mtext>&#xA0;</mtext>
        <mo fence="false" stretchy="false">{</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>0</mn>
        </msub>
        <mo>:=</mo>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mn>0</mn>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03B1;<!-- α --></mi>
        <mtext>&#xA0;</mtext>
        <mfrac>
          <mn>1</mn>
          <mi>m</mi>
        </mfrac>
        <mtext>&#xA0;</mtext>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>m</mi>
        </munderover>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>h</mi>
          <mi>&#x03B8;<!-- θ --></mi>
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
        <mo>&#x2212;<!-- − --></mo>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
        <msubsup>
          <mi>x</mi>
          <mn>0</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>:=</mo>
        <msub>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03B1;<!-- α --></mi>
        <mtext>&#xA0;</mtext>
        <mfenced open="[" close="]">
          <mrow>
            <mfenced open="(" close=")">
              <mrow>
                <mfrac>
                  <mn>1</mn>
                  <mi>m</mi>
                </mfrac>
                <mtext>&#xA0;</mtext>
                <munderover>
                  <mo>&#x2211;<!-- ∑ --></mo>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mi>i</mi>
                    <mo>=</mo>
                    <mn>1</mn>
                  </mrow>
                  <mi>m</mi>
                </munderover>
                <mo stretchy="false">(</mo>
                <msub>
                  <mi>h</mi>
                  <mi>&#x03B8;<!-- θ --></mi>
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
                <mo>&#x2212;<!-- − --></mo>
                <msup>
                  <mi>y</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo stretchy="false">)</mo>
                <msubsup>
                  <mi>x</mi>
                  <mi>j</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mrow>
            </mfenced>
            <mo>+</mo>
            <mfrac>
              <mi>&#x03BB;<!-- λ --></mi>
              <mi>m</mi>
            </mfrac>
            <msub>
              <mi>&#x03B8;<!-- θ --></mi>
              <mi>j</mi>
            </msub>
          </mrow>
        </mfenced>
      </mtd>
      <mtd>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mi>j</mi>
        <mo>&#x2208;<!-- ∈ --></mo>
        <mo fence="false" stretchy="false">{</mo>
        <mn>1</mn>
        <mo>,</mo>
        <mn>2...</mn>
        <mi>n</mi>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

将<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mi>λ</mi><mi>m</mi></mfrac><msub><mi>θ</mi><mi>j</mi></msub></math>提出来，得到:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>:=</mo>
  <msub>
    <mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>−</mo>
  <mi>α</mi>
  <mfrac>
    <mi>λ</mi>
    <mi>m</mi>
  </mfrac>
  <mo stretchy="false">)</mo>
  <mo>−</mo>
  <mi>α</mi>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munderover>
    <mo>∑</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
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
  <msubsup>
    <mi>x</mi>
    <mi>j</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
</math>

上述式子中<math xmlns="http://www.w3.org/1998/Math/MathML"><mn>1</mn><mo>−</mo><mi>α</mi><mfrac><mi>λ</mi><mi>m</mi></mfrac></math>必须小于 1，这样就减小了<math><msub><mi>θ</mi><mi>j</mi></msub></math>的值，后面的式子和之前梯度下降的式子相同

* 应用到 Normal Equation

和之前的公式相比，增加了一项：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <mi>&#x03B8;<!-- θ --></mi>
        <mo>=</mo>
        <msup>
          <mfenced open="(" close=")">
            <mrow>
              <msup>
                <mi>X</mi>
                <mi>T</mi>
              </msup>
              <mi>X</mi>
              <mo>+</mo>
              <mi>&#x03BB;<!-- λ --></mi>
              <mo>&#x22C5;<!-- ⋅ --></mo>
              <mi>L</mi>
            </mrow>
          </mfenced>
          <mrow class="MJX-TeXAtom-ORD">
            <mo>&#x2212;<!-- − --></mo>
            <mn>1</mn>
          </mrow>
        </msup>
        <msup>
          <mi>X</mi>
          <mi>T</mi>
        </msup>
        <mi>y</mi>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mtext>where</mtext>
        <mtext>&#xA0;</mtext>
        <mtext>&#xA0;</mtext>
        <mi>L</mi>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mn>0</mn>
              </mtd>
              <mtd />
              <mtd />
              <mtd />
              <mtd />
            </mtr>
            <mtr>
              <mtd />
              <mtd>
                <mn>1</mn>
              </mtd>
              <mtd />
              <mtd />
              <mtd />
            </mtr>
            <mtr>
              <mtd />
              <mtd />
              <mtd>
                <mn>1</mn>
              </mtd>
              <mtd />
              <mtd />
            </mtr>
            <mtr>
              <mtd />
              <mtd />
              <mtd />
              <mtd>
                <mo>&#x22F1;<!-- ⋱ --></mo>
              </mtd>
              <mtd />
            </mtr>
            <mtr>
              <mtd />
              <mtd />
              <mtd />
              <mtd />
              <mtd>
                <mn>1</mn>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
    </mtr>
  </mtable>
</math>

L 是一个（n+1)x(n+1)的对单位阵，第一项是 0。在引入 λ.L 之后<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>+</mo><mi>λ</mi><mi>L</mi></math>保证可逆

### Regularized logistic regression

逻辑回归也有 overfitting 的问题，如图所示

![](/assets/images/2017/09/ml-5-7.png)

处理方式和线性回归相同，之前知道逻辑回归的 cost function 如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>&#x03B8;<!-- θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
  </munderover>
  <mstyle mathsize="1.2em">
    <mo stretchy="false">[</mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mtext>&#xA0;</mtext>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mo>+</mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mo stretchy="false">)</mo>
    <mtext>&#xA0;</mtext>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mstyle>
      <mo stretchy="false">]</mo>
    </mstyle>
  </mstyle>
</math>

我们可以在最后加一项来 regularize 这个函数：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>&#x03B8;<!-- θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
  </munderover>
  <mstyle mathsize="1.2em">
    <mo stretchy="false">[</mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mtext>&#xA0;</mtext>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mo>+</mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msup>
      <mi>y</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mo stretchy="false">)</mo>
    <mtext>&#xA0;</mtext>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
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
    <mo stretchy="false">)</mo>
    <mstyle>
      <mo stretchy="false">]</mo>
      <mo>+</mo>
      <mfrac>
        <mi>λ</mi>
        <mrow>
          <mn>2</mn>
          <mi>m</mi>
        </mrow>
      </mfrac>
      <munderover>
        <mo>∑</mo>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>j</mi>
          <mo>=</mo>
          <mn>1</mn>
        </mrow>
        <mi>n</mi>
      </munderover>
      <msubsup>
        <mi>θ</mi>
        <mi>j</mi>
        <mn>2</mn>
      </msubsup>
    </mstyle>
  </mstyle>
</math>

第二项：<math xmlns="http://www.w3.org/1998/Math/MathML"><munderover><mo>∑</mo><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>θ</mi><mi>j</mi><mn>2</mn></msubsup></math>是排除了<math><msub><mi>θ</mi><mi>0</mi></msub></math>的，因此，计算梯度下降要对<math><msub><mi>θ</mi><mi>0</mi></msub></math>单独计算:

![](/assets/images/2017/09/ml-5-8.png)

* Octave Demo

```matlab
function [J, grad] = lrCostFunction(theta, X, y, lambda{

%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));



h = sigmoid(X*theta); %  X:118x28, theta:28x1 h:118x1

%向量化实现
J = 1/m * (-y'*log(h)-(1-y)'*log(1-h)) + 0.5*lambda/m * sum(theta([2:end]).^2);
%代数形式实现
%J = 1/m * sum((-y).*log(h) - (1-y).*log(1-h)) + 0.5*lambda/m * sum(theta([2:end]).^2);

grad = 1/m * X'*(h-y);

r = lambda/m .* theta;
r(1) = 0; %skip theta(0)
grad = grad + r;


% =============================================================

grad = grad(:);

}

function g = sigmoid(z)
	g = 1.0 ./ (1.0 + exp(-z));
end
```
