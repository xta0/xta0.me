---
layout: post
title: Machine Learning - Chap5
meta: Coursera Stanford Machine Learning Cousre Note, Chapter5
categories: [ml-stanford,course]
---

## Chapter 5 机器学习的应用

### 调试预测函数

假设我们设计了一个线性回归公式来预测房价，但是当我们用这个公式预测房价的时候结果却有很大的偏差，这时候我们该怎么做，可以尝试：

- 增加训练样本 
- 减少样本特征集合
- 增加feature
- 在已有feature基础上，增加多项式作为新的feature项(<math><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>,</mo><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup><mo>,</mo><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>,</mo></math>etc.)
- 尝试减小<math><mi>λ</mi></math> 
- 尝试增大<math><mi>λ</mi></math> 

具体该怎么做呢？显然满目的随机选取一种方法去优化是不合理的，一种合理的办法是实现diagnostic机制帮助我们找到出错点或者排除一些无效的优化方法

### 评估预测函数

当我们拿到训练样本时，可以把它分为两部分，一部分是训练集(70%)，一部分是测试集(30%)

- 训练集用<math><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo stretchy="false">)</mo><mo>...</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo stretchy="false">)</mo></math>表示
- 测试集用<math> <mo stretchy="false">(</mo> <msubsup><mi>x</mi><mi>test</mi><mi>(1)</mi></msubsup> <mo>,</mo> <msubsup><mi>y</mi><mi>test</mi><mi>(1)</mi></msubsup> <mo stretchy="false">)</mo> <mo>,</mo> <mo stretchy="false">(</mo> <msubsup><mi>x</mi><mi>test</mi><mi>(2)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mi>(2)</mi></msubsup><mo stretchy="false">)</mo><mo>...</mo><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>test</mi><mi>(mtest)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mi>(mtest)</mi></msubsup><mo stretchy="false">)</mo></math>表示

使用这部分数据集的方法是：

1. 使用训练集，求<math> <msub> <mi>J</mi> <mrow> <mi>t</mi> <mi>r</mi> <mi>a</mi> <mi>i</mi> <mi>n</mi> </mrow> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>的最小值，得到<math><mi>Θ</mi></math>

2. 使用测试集，计算<math> <msub> <mi>J</mi><mi>test</mi></msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>

- 对于**线性回归**函数，计算<math> <msub> <mi>J</mi><mi>test</mi></msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>的公式为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>J</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>t</mi>
      <mi>e</mi>
      <mi>s</mi>
      <mi>t</mi>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <msub>
          <mi>m</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>t</mi>
            <mi>e</mi>
            <mi>s</mi>
            <mi>t</mi>
          </mrow>
        </msub>
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
    <mrow class="MJX-TeXAtom-ORD">
      <msub>
        <mi>m</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mi>t</mi>
          <mi>e</mi>
          <mi>s</mi>
          <mi>t</mi>
        </mrow>
      </msub>
    </mrow>
  </munderover>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>h</mi>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <msubsup>
    <mi>x</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>t</mi>
      <mi>e</mi>
      <mi>s</mi>
      <mi>t</mi>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <msubsup>
    <mi>y</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>t</mi>
      <mi>e</mi>
      <mi>s</mi>
      <mi>t</mi>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
</math>

- 对于**逻辑回归**函数，计算<math> <msub> <mi>J</mi><mi>test</mi></msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
	<msub>
    <mi>J</mi>
	<mi>test</mi>
  </msub>
  <mo stretchy="false">(</mo><mi>Θ</mi><mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>−</mo>
  <mfrac>
    <mn>1</mn>
    <msub><mi>m</mi><mi>test</mi></msub>
  </mfrac>
  <mstyle displaystyle="true">
    <munderover>
      <mo>∑</mo>
      <mrow>
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mi>m</mi>
    </munderover>
    <mo stretchy="false">[</mo>
    <msubsup>
      <mi>y</mi>
      <mi>test</mi>
  		<mi>(i)</mi>
    </msubsup>
    <mi>log</mi>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>h</mi>
      <mi>&#x03B8;<!-- θ --></mi>
    </msub>
    <mo stretchy="false">(</mo>
    <msubsup>
      <mi>x</mi>
      <mi>test</mi>
      <mrow>
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msubsup>
    <mo stretchy="false">)</mo>
    <mo stretchy="false">)</mo>
    <mo>+</mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>&#x2212;<!-- − --></mo>
    <msubsup>
      <mi>y</mi>
      <mi>test</mi>
      <mrow class="MJX-TeXAtom-ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msubsup>
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
    <msubsup>
      <mi>x</mi>
      <mi>test</mi>
      <mrow>
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msubsup>
    <mo stretchy="false">)</mo>
    <mo stretchy="false">)</mo>
    <mo stretchy="false">]</mo>
  </mstyle>
</math>


- 对于**分类/多重分类**问题，先给出错误分类的预测结果为

	<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"> <mi>e</mi> <mi>r</mi> <mi>r</mi> <mo stretchy="false">(</mo> <msub> <mi>h</mi> <mi mathvariant="normal">&#x0398;<!-- Θ --></mi> </msub> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo> <mo>,</mo> <mi>y</mi> <mo stretchy="false">)</mo> <mo>=</mo> <mtable rowspacing="4pt" columnspacing="1em"> <mtr> <mtd> <mn>1</mn> </mtd> <mtd> <mtext>if&#xA0;</mtext> <msub> <mi>h</mi> <mi mathvariant="normal">&#x0398;<!-- Θ --></mi> </msub> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo> <mo>&#x2265;<!-- ≥ --></mo> <mn>0.5</mn> <mtext>&#xA0;</mtext> <mi>a</mi> <mi>n</mi> <mi>d</mi> <mtext>&#xA0;</mtext> <mi>y</mi> <mo>=</mo> <mn>0</mn> <mtext>&#xA0;</mtext> <mi>o</mi> <mi>r</mi> <mtext>&#xA0;</mtext> <msub> <mi>h</mi> <mi mathvariant="normal">&#x0398;<!-- Θ --></mi> </msub> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo> <mo>&lt;</mo> <mn>0.5</mn> <mtext>&#xA0;</mtext> <mi>a</mi> <mi>n</mi> <mi>d</mi> <mtext>&#xA0;</mtext> <mi>y</mi> <mo>=</mo> <mn>1</mn> </mtd> </mtr> <mtr> <mtd> <mn>0</mn> </mtd> <mtd> <mtext>o</mtext> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>r</mi> <mi>w</mi> <mi>i</mi> <mi>s</mi> <mi>e</mi> </mtd> </mtr> </mtable> </math>

	计算<math> <msub> <mi>J</mi><mi>test</mi></msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>的公式为：

	<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"> <mtext>Test Error</mtext> <mo>=</mo> <mstyle displaystyle="true"> <mfrac> <mn>1</mn> <msub> <mi>m</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>t</mi> <mi>e</mi> <mi>s</mi> <mi>t</mi> </mrow> </msub> </mfrac> </mstyle> <munderover> <mo>&#x2211;<!-- ∑ --></mo> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <msub> <mi>m</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>t</mi> <mi>e</mi> <mi>s</mi> <mi>t</mi> </mrow> </msub> </mrow> </munderover> <mi>e</mi> <mi>r</mi> <mi>r</mi> <mo stretchy="false">(</mo> <msub> <mi>h</mi> <mi mathvariant="normal">&#x0398;<!-- Θ --></mi> </msub> <mo stretchy="false">(</mo> <msubsup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>t</mi> <mi>e</mi> <mi>s</mi> <mi>t</mi> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo stretchy="false">)</mo> <mo>,</mo> <msubsup> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>t</mi> <mi>e</mi> <mi>s</mi> <mi>t</mi> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo stretchy="false">)</mo> </math>

### 模型选择和训练样本重新划分

假设预测能后很好的fit我们的训练样本，也不能代表这个预测函数就是最好的，对于训练集以外的数据，预测函数的预测错误率可能高于训练样本的平均错误率。

在模型选择上，我们可能会纠结于如下几种模型：

1. <math><msub><mi>h</mi><mi>(θ)</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mi></mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mi></mn></msub><mi>x</mi></math> （d=1）
2.  <math><msub><mi>h</mi><mi>(θ)</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mi></mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mi></mn></msub><mi>x</mi><mo>+</mo><msub><mi>θ</mi><mn>2</mi></mn></msub><msup><mi>x</mi><mn>2</mn></msup></math>
3. <math> <msub><mi>h</mi><mi>(θ)</mi></msub> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>=</mo> <msub><mi>θ</mi><mn>0</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>1</mn></msub> <mi>x</mi><mo>+</mo> <msub><mi>θ</mi><mn>2</mn></msub> <msup><mi>x</mi><mn>2</mn></msup> <mo>+</mo> <msub><mi>θ</mi><mn></mn></msub> <msup><mi>x</mi><mn>3</mn></msup></math>
4. ...
5. <math><msub><mi>h</mi><mi>(θ)</mi></msub> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>=</mo> <msub><mi>θ</mi><mn>0</mn></msub> <mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub> <mi>x</mi><mo>+</mo> <msub><mi>θ</mi><mn>2</mn></msub> <msup><mi>x</mi><mn>2</mn></msup> <mo>+</mo> <msub><mi>θ</mi><mn>3</mn></msub> <msup><mi>x</mi><mn>3</mn></msup> <mo>+</mo> <mo>...</mo> <mo>+</mo> <msub><mi>θ</mi><mn>10</mn></msub> <msup><mi>x</mi><mn>10</mn></msup></math>
  
因此我们在选模型的时候需要考虑多项式的维度，即<math><mi>d</mi></math>为多少，此外我们还需要对原始的训练样本做进一步划分：

- 训练样本<math><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo stretchy="false">)</mo><mo>...</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo stretchy="false">)</mo></math> 占60%
- 交叉验证样本<math><mo stretchy="false">(</mo> <msubsup><mi>x</mi><mi>cv</mi><mi>(1)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>cv</mi><mi>(1)</mi></msubsup><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo> <msubsup><mi>x</mi><mi>cv</mi><mi>(2)</mi></msubsup> <mo>,</mo><msubsup><mi>y</mi><mi>cv</mi><mi>(2)</mi></msubsup> <mo stretchy="false">)</mo> <mo>...</mo> <mo stretchy="false">(</mo> <msubsup><mi>x</mi><mi>cv</mi><mi>(mvc)</mi></msubsup> <mo>,</mo><msubsup><mi>y</mi><mi>cv</mi><mi>(mcv)</mi></msubsup><mo stretchy="false">)</mo></math>占比20%
- 测试集<math><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>test</mi><mi>(1)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mi>(1)</mi></msubsup><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>test</mi><mi>(2)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mi>(2)</mi></msubsup><mo stretchy="false">)</mo><mo>...</mo><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>test</mi><mi>(mtest)</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mi>(mtest)</mi></msubsup><mo stretchy="false">)</mo></math>占20%

然后我们可以按照下面步骤来对上面几种model进行评估：

1. 使用训练集，计算各自的<math><mi>Θ</mi></math>值，用<math><msup><mi>Θ</mi><mi>(d)</mi></msup></math>表示
2. 使用验证集，计算各自的<math> <msub> <mi>J</mi><mi>cv</mi></msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>值，用<math> <msub> <mi>J</mi><mi>cv</mi></msub> <mo stretchy="false">(</mo> <msup><mi>Θ</mi><mi>(d)</mi></msup> <mo stretchy="false">)</mo> </math>，找到最小值
3. 将上一步得到最小值的Θ用测试集验证，得到通用的错误估计值<math> <msub> <mi>J</mi><mi>test</mi> </msub> <mo stretchy="false">(</mo> <msup> <mi mathvariant="normal">Θ</mi> <mrow> <mo stretchy="false">(</mo> <mi>d</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> </math>

### 观察Bias和Variance

这一小节讨论模型多项式的维度d和Bias以及Variance的关系，用来观察模型是否有overfitting的问题。回一下bias和variance的概念, High bias会造成模型的Underfit，原因往往是多项式维度d过低，High variance会造成模型的Overfit，原因可能是多项式维度d过高。我们需要在两者间找到一个合适的d值。

- Training error,也就是<math><mi>J</mi><mi>(θ)</mi></math>,由前面可知，公式为

<math display="block"> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo> <mo>=</mo> <mstyle> <mfrac> <mn>1</mn> <mrow> <mn>2</mn> <mi>m</mi> </mrow> </mfrac> </mstyle> <mstyle> <munderover> <mo>∑</mo> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mi>m</mi> </munderover> <msup> <mfenced open="(" close=")"> <mrow> <msub> <mi>h</mi> <mi>θ</mi> </msub> <mo stretchy="false">(</mo> <msub> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> </mrow> </msub> <mo stretchy="false">)</mo> <mo>−</mo> <msub> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> </mrow> </msub> </mrow> </mfenced> <mn>2</mn> </msup> </mstyle> </math>

- Cross Validation error 的公式为

<math display="block"> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo> <mo>=</mo> <mstyle displaystyle="true"> <mfrac> <mn>1</mn> <mrow> <mn>2</mn> <msub> <mi>m</mi> <mi>cv</mi> </msub> </mrow> </mfrac> </mstyle> <munderover> <mo>&#x2211;<!-- ∑ --></mo> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <msub> <mi>m</mi> <mi>cv</mi> </msub> </mrow> </munderover> <mo stretchy="false">(</mo> <msub> <mi>h</mi> <mi mathvariant="normal">θ</mi> </msub> <mo stretchy="false">(</mo> <msubsup> <mi>x</mi> <mi>cv</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo stretchy="false">)</mo> <mo>−</mo> <msubsup> <mi>y</mi> <mi>cv</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <msup> <mo stretchy="false">)</mo> <mn>2</mn> </msup> </math>

二者的区别在于输入的数据集不同，以及<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo></math>输入样本少。我们建立一个坐标系，横轴是多项式维度d的值，从0到无穷大，纵轴是预测函数的错误率cost of J(θ)，二者的关系是：

- 对于<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>（Training Error），当多项式维度d增大的时候，错误率逐渐降低
- 对于<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo></math>（Cross Validation Error），当多项式维度d增大到某个值的时候<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo></math>的值将逐渐偏离并大于<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>

如下图所示：

![](/images/2017/09/ml-7-1.png)

总结一下：

- **High bias(Underfitting)**：<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo></math>的值和<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>的值都很大，有<math> <msub> <mi>J</mi> <mi>cv</mi></msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> <mo>≈</mo> <msub> <mi>J</mi> <mi>train</mi></msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>
- **High variance(Overfitting)**：<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>θ</mi> <mo stretchy="false">)</mo></math>的值会小于<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>，在超过某个d值后会大于<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi mathvariant="normal">θ</mi> <mo stretchy="false">)</mo> </math>的值


### Regularization项

参考之前对Regularization的介绍，λ的值对预测函数的影响如下图所示

![](/images/2017/09/ml-7-2.png)

为了找到最适合的λ值，可以按如下步骤

1. 先列出一个λ数组： (i.e. λ∈{ 0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24 })
2. 创建一系列不同degree的预测函数
3. 遍历λ数组，对每一个λ带入所有model，求出<math> <mi>Θ</mi></math>
4. 将上一步得到的<math> <mi>Θ</mi></math>，使用交叉验正数据集计算<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>，注意<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>是不带regularization项的，或者λ=0
5. 选出上面步骤中使<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>得到最小值的<math> <mi>Θ</mi></math>和λ
6. 将上一步选出的<math> <mi>Θ</mi></math>和λ使用test数据集进行验证，看预测结果是否符合预期

### 绘制学习曲线

学习曲线是指横坐标为训练样本个数，纵坐标为错误率的曲线。错误率会随着训练样本变大，当样本数量超过某个值后，错误率增长会变慢，类似log曲线

- High bias
	- 样本数少：<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> 值很低，<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>值很高
	- 样本数大：<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> 和 <math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>都很高，当样本数超过某个值时，有<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> ≈ <math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>
	- 如果某个学习算法有较高的bias，仅通过增加训练样本是没用的
	- 学习曲线图如下所示
	
	![](/images/2017/09/ml-7-3.png)
	
- high variance
	- 样本数少：<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> 值很低，<math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>值很高
	- 样本数大：<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> 升高，但还是保持在一个较低的水平。 <math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>则持续降低，但还是保持在一个较高的水平，因此有<math> <msub> <mi>J</mi> <mi>train</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math> < <math> <msub> <mi>J</mi> <mi>cv</mi> </msub> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </math>。两者的差值很大
	- 	如果某个学习算法有较高的variance，通过增加训练样本有可能有帮助
	-  学习曲线图如下所示
	
	![](/images/2017/09/ml-7-4.png)
	

### 调试预测函数(Revisit)

呼应这一章第一节的内容，调试预测函数有如下几种办法：

- 增加训练样本数量
	- 解决 high variance问题
- 减少样本特征数量
	- 解决 high variance问题
- 增加样本特征
	- 解决 high bias问题
- 增加多项式维度
	- 解决 high bias问题
- 减小λ值
	- 解决 high bias问题
- 增大λ值
	- 解决 high variance问题

#### 调试神经网络  	 

- 一个层数少，每层unit个数少的神经网络，容易underfitting，计算成本很低
- 一个层数多，每层unit个数也很多的神经网络，容易overfitting，计算成本也很高，这种情况可加入regularization项（增大λ）来尝试解决

对于神经网络，可以默认使用一层来求出Θ，然后构建几个多层神经网络使用交叉数据样本进行验证，找到最合适的神经网络结构


## Machine Learning System Design

这一章比较独立，以垃圾邮件分类器为例讲Machine Learning的系统设计。

对于垃圾邮件，通常来说会包含一些关键字，比如"discount"，"deal"等。我们用一个nx1的向量表示每一封邮件，向量中的每一项代表某个关键词是否出现，即如果这封email出现了某个关键词，那么对应向量中的该项的值为1，否则为0。向量长度为1000<n<5000，如下图所示：

![](/images/2017/09/ml-7-5.png)

为了训练出准确的分类器模型，有哪些挑战

- 如何收集足够多的高质量训练数据，广告邮件，垃圾邮件等，这些训练数据的来源该去哪里获得
- 选取哪些作为关键的特征，比如邮件中出现"discount","deal"等，但是这些可能会误杀一些邮件，出了这些是否还是其它的信息可作为feature
- 如何设计出全面严谨的模型，比如有些垃圾邮件会故意把某些单词拼错(e.g. m0rtage, med1cine, w4tches等)来绕过反垃圾检查，训练模型要能识别出这些错误的拼写

当要实现一个Machine Learning系统时，推荐按照下列做法一步步做

1. 先从一个简单的算法开始，快速实现，在验证样本集中进行验证
2. 绘制学习曲线，观察是否需要增加训练样本或者增加特征
3. 进行误差分析，例如，在验证集中找出那些分类不准的email，查看出错原因，观察能否找到一些共性或者系统性的错误

错误分析的的一个很重要的点是要使用的优化方法进行量化，比如使用stemming和未使用stemming的错误率对比

- Error Metrics for Skewed CLasses

所谓Skewed Class指的是分类问题中，对于某些结果出现的可能性很小，比如在患癌症的诊断中，癌症的样本占比很少，非癌症的训练样本很多，因此训练出来的模型，在预测结果上可能99.5%都趋向于一个结果，这时我们怎么去衡量模型预测的准确率，需要引入"准确率"与"召回率"的概念：

![](/images/2017/09/ml-7-6.png)

如上图所示

- 准确率(Precision)：precision = #(true positives)/#(predicted positives) 
- 召回率(Recall): recall = #(true positives)/#(actual positives)

有时候准确率和召回率两者是矛盾的，较高的准确率可能会带来较低的召回率，反之亦然。

回忆之前逻辑回归的sigmoid公式，如果想要y=1，需要<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi> <mo> >= </mo> <mn>0.5</mn></math>，如果要追求较高的预测准确率，则需要调高阈值，比如我们希望在预测足够准确的情况下再通知病人他是否患有癌症，这时阈值就要设的较高，比如<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi> <mo> >= </mo> <mn>0.9</mn></math>，但这会带来较低的召回率。同理，如果我们追求较高的召回率，则需要降低阈值，还是癌症的例子，我们希望尽量遗漏癌症的case，哪怕有30%的可能性也要提早通知病人治疗，这时阈值就要设的低一些，比如<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi> <mo> >= </mo> <mn>0.3</mn></math>，但这又会带来较低的预测准确率

<math display="block"><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi> <mo> >= </mo> <mi>threshold</mi></math>

有什么办法可以帮助我们判断threshold的值到底好不好呢，可以使用F分数

<math display="block">
	<msub><mi>F</mi><mn>1</mn></msub><mi> Score</mi>
	<mo>:</mo>
	<mn>2</mn>
	<mfrac>
		<mrow><mi>P</mi><mi>R</mi></mrow>
		<mrow><mi>P</mi><mo>+</mo><mi>R</mi></mrow>
	</mfrac>
</math>

F分数越大说明threshold计算出的值越合理，通常我们使用交叉验证样本集算出一个较好的threshold再使用测试集验证结果
