---
layout: post
title: Machine Learning - Chap6
meta: Coursera Stanford Machine Learning Cousre Note, Chapter6
categories: [ml-stanford,course]
---

## Chapter6 SVM

支持向量机SVM（support vector machine）是另一种监督学习的算法，它主要用解决**分类**问题（二分类）和**回归分析**中。SVM和前面几种机器学习算法相比，在处理复杂的非线性方程（不规则分类问题）时效果很好。在介绍SVM之前，先回顾一下logistic regression，在逻辑回归中，我们的预测函数为：

<math display="block"><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>)</mo><mspace width="2em"></mspace><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>


- 如果要<math><mi>y</mi><mo>=</mo><mn>1</mn></math>，需要<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mn>1</mn></math>，即<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi> <mo>>></mo><mn>0</mn></math>

- 如果要<math><mi>y</mi><mo>=</mo><mn>0</mn></math>，需要<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mn>0</mn></math>，即<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi> <mo><<</mo><mn>0</mn></math>

![](/images/2017/09/ml-8-1.png)

预测函数在某点的代价函数为

<math display="block">
  <mrow class="MJX-TeXAtom-ORD">
	<mi> Cost of example </mi>
  </mrow>
  <mo>:</mo>
  <mo>−</mo>
  <mi>y</mi>
  <mspace width="thickmathspace" />
  <mi>log</mi>
  <mo> ⁡ </mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
  <mo>−</mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>−</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mi>log</mi>
  <mo> ⁡ </mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>−</mo>
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
</math>

另<math><mi>z</mi><mo>=</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></math>, 带入上面式子得到

<math display="block">
  <mrow class="MJX-TeXAtom-ORD">
	<mi> Cost of example </mi>
  </mrow>
  <mo>:</mo>
  <mo>-</mo>
  <mi>y</mi>
  <mi>log</mi>
  <mo stretchy="false">(</mo>
  <mfrac>
  	<mn>1</mn>
  	<mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow>
  </mfrac>
  <mo stretchy="false">)</mo>
  <mo>-</mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn><mo>-</mo><mi>y</mi>
  <mo stretchy="false">)</mo>
  <mi>log</mi>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>-</mo>
  <mfrac>
  	<mn>1</mn>
  	<mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>

- 当<math><mi>y</mi><mo>=</mo><mn>1</mn></math>时，后一项为零，上述式子为：<math><mo>-</mo><mi>log</mi> <mo stretchy="false">(</mo> <mfrac> <mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow> </mfrac> <mo stretchy="false">)</mo></math> 

- 当<math><mi>y</mi><mo>=</mo><mn>0</mn></math>时，后一项为零，上述式子为： <math><mo>-</mo><mi>log</mi> <mo stretchy="false">(</mo> <mn>1</mn> <mo>-</mo> <mfrac> <mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow> </mfrac> <mo stretchy="false">)</mo></math>

对应的函数曲线为：

![](/images/2017/09/ml-8-2.png)


对于SVM，我们使用粉色的函数来近似cost function，分别表示为<math><msub><mi>cost</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math>和<math><msub><mi>cost</mi><mn>0</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math>，有了这两项，我们再回头看一下逻辑回归完整的代价函数：

<math display="block">
    <munder>
      <mrow class="MJX-TeXAtom-OP">
        <mtext>min</mtext>
      </mrow>
      <mi>θ</mi>
    </munder>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <mo>[</mo>
  <munderover>
    <mo>∑</mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
  </munderover>
  <mstyle>
    <msup>
      <mi>y</mi>
       <mi>(i)</mi>
    </msup>
    <mo stretchy="false">(</mo>
    <mo>-</mo>
    <mi>log</mi>
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
    <mo stretchy="false">)</mo>
    <mo stretchy="false">)</mo>
    <mo>+</mo>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>−</mo>
    <msup>
      <mi>y</mi>
      <mi>(i)</mi>
    </msup>
    <mo stretchy="false">)</mo>
    <mo stretchy="false">(</mo>
    <mo stretchy="false">(</mo>
    <mo>-</mo>
    <mi>log</mi>
    <mo stretchy="false">(</mo>
    <mn>1</mn>
    <mo>−</mo>
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
    <mo stretchy="false">)</mo>
    <mo stretchy="false">)</mo>
    <mstyle>
      <mo>]</mo>
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

用<math><msub><mi>cost</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math>和<math><msub><mi>cost</mi><mn>0</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math> 替换中括号中的第一项和第二项，得到：

<math display="block">
  <mstyle displaystyle="true">
    <munder>
      <mrow class="MJX-TeXAtom-OP">
        <mtext>min</mtext>
      </mrow>
      <mi>θ</mi>
    </munder>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mtext>&#xA0;</mtext>
    <mfrac>
      <mn>1</mn>
      <mi>m</mi>
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
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <msub>
          <mtext>cost</mtext>
          <mn>1</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
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
        <msub>
          <mtext>cost</mtext>
          <mn>0</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
      </mrow>
    </mfenced>
    <mo>+</mo>
    <mfrac>
      <mi>&#x03BB;<!-- λ --></mi>
      <mrow>
        <mn>2</mn>
        <mi>m</mi>
      </mrow>
    </mfrac>
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
  </mstyle>
</math>

去掉常量m，将上述式子的表现形式上作如下变换

<math display="block"><mi>A</mi><mo>+</mo><mi>λB</mi><mo>-></mo> <mi>CA</mi><mo>+</mo><mi>B</mi></math>

对于逻辑回归，可以通过增大λ值来达到提升B的权重，从而控制过度拟合。在SVM中，可以通过减小C的值来提升B的权重，可以认为 <math><mi>C</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>λ</mi></mrow></mfrac></math>，但是过大的C可能也会带来过度拟合的问题

SVM的Cost函数为:

<math display="block">
  <mstyle displaystyle="true">
    <munder>
      <mrow class="MJX-TeXAtom-OP">
        <mtext>min</mtext>
      </mrow>
      <mi>&#x03B8;<!-- θ --></mi>
    </munder>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mtext>&#xA0;</mtext>
    <mi>C</mi>
    <mfenced open="[" close="]">
      <mrow>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <msub>
          <mtext>cost</mtext>
          <mn>1</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
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
        <msub>
          <mtext>cost</mtext>
          <mn>0</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
      </mrow>
    </mfenced>
    <mo>+</mo>
    <mfrac>
      <mn>1</mn>
      <mn>2</mn>
    </mfrac>
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
  </mstyle>
</math>

SVM的预测函数为

![](/images/2017/09/ml-8-3.png)

对于<math><msub><mi>cost</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math> 和 <math><msub><mi>cost</mi><mn>0</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math> ，它们的函数曲线如下图所示:

![](/images/2017/09/ml-8-4.png)

当C的值很大时，如果让cost函数的值最小，那么我们希望中括号里的值为0，这样我们的优化目标就变成了：

- 如果要<math><mi>y</mi><mo>=</mo><mn>1</mn></math>，我们希望<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>>=</mo><mn>1</mn></math>，此时<math><msub><mi>cost</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math>的值为0

- 如果要<math><mi>y</mi><mo>=</mo><mn>0</mn></math>，我们希望<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo><=</mo><mn>-1</mn></math>，此时<math><msub><mi>cost</mi><mn>0</mn></msub><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></math>的值为0

先忽略数学求解，按照这个优化目标最后得出的SVM预测函数可以参考下图做直观的理解：

![](/images/2017/09/ml-8-5.png)

上图中，如果我们想要用一条曲线做为Decision Boundary来划分正负样本，可以有很多选择，比如绿线，粉线和黑线，其中黑线代表的预测函数为SVM分类模型，它的一个特点是样本到这条线的距离较其它预测函数比较大，因此SVM分类模型也叫做 **Large Margin Classifier**。

假设有向量 <math><mi>u</mi><mo>=</mo><mfenced open="[" close="]"> <mtable> <mtr> <mtd> <msub> <mi>u</mi> <mn>1</mn> </msub> </mtd> </mtr> <mtr> <mtd> <msub> <mi>u</mi> <mn>2</mn> </msub> </mtd> </mtr></mtable> </mfenced></math> 和 <math><mi>v</mi><mo>=</mo><mfenced open="[" close="]"> <mtable> <mtr> <mtd> <msub> <mi>v</mi> <mn>1</mn> </msub> </mtd> </mtr> <mtr> <mtd> <msub> <mi>v</mi> <mn>2</mn> </msub> </mtd> </mtr></mtable> </mfenced></math>，其中<math><mi>u</mi></math>的长度为<math><mo>||</mo><mi>u</mi><mo>||</mo><mo>=</mo><msqrt><msup><mi>u1</mi><mn>2</mn></msup><mo>+</mo><msup><mi>u2</mi><mn>2</mn></msup></msqrt></math>，我们想要求解向量u和v的內积：<math><msup><mi>u</mi><mi>T</mi></msup><mi>v</mi></math>

![](/images/2017/09/ml-8-6.png)

参考上图，定义p为向量v在u上的投影，有如下等式

<math display="block">
	<msup><mi>u</mi><mi>T</mi></msup><mi>v</mi>
	<mo>=</mo>
	<mi>p</mi><mo>||</mo><mi>u</mi><mo>||</mo>
</math>

其中p是有符号的，当uv夹角大于90度时，v在u的投影p为负数。接下来我们用这个式子来求解SVM模型。

SVM代价函数的优化目标为：

<math display="block">
	<munder>
      <mrow>
        <mtext>min</mtext>
      </mrow>
      <mi>θ</mi>
    </munder>
    <mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac>
    <munderover>
      <mo>∑</mo>
      <mrow>
        <mi>j</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow>
        <mi>n</mi>
      </mrow>
    </munderover>
	<msubsup>
		<mi>θ</mi>
		<mi>j</mi>
		<mn>2</mn>
	</msubsup>
</math>
<math display="block">
	<msup><mi>θ</mi><mi>T</mi></msup><msup><mi>x</mi><mi>(i)</mi></msup>
	<mo>>=</mo>
	<mn>1</mn>
	<mspace width="2em"/>
	<mtext>if</mtext>
	<mspace width ="1em"/>
	<msup><mi>y</mi><mi>(i)</mi></msup>
	<mo>=</mo>
	<mn>1</mn>
</math>
<math display="block">
	<msup><mi>θ</mi><mi>T</mi></msup><msup><mi>x</mi><mi>(i)</mi></msup>
	<mo><=</mo>
	<mn>-1</mn>
	<mspace width="2em"/>
	<mtext>if</mtext>
	<mspace width ="1em"/>
	<msup><mi>y</mi><mi>(i)</mi></msup>
	<mo>=</mo>
	<mn>0</mn>
</math>


为了简化问题，我们令<math><msub><mi>θ</mi><mn>0</mn></msub></math>=0，令n=2，上述优化方程简化为为：

<math display="block">
	<munder>
      <mrow>
        <mtext>min</mtext>
      </mrow>
      <mi>θ</mi>
    </munder>
    <mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac>
    <munderover>
      <mo>∑</mo>
      <mrow>
        <mi>j</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow>
        <mi>n</mi>
      </mrow>
    </munderover>
	<msubsup>
		<mi>θ</mi>
		<mi>j</mi>
		<mn>2</mn>
	</msubsup>
	<mo>=</mo>
	<mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac>
	<mo stretchy="false">(</mo>
	<msubsup><mi>θ</mi><mn>1</mn><mn>2</mn></msubsup>
	<mo>+</mo>
	<msubsup><mi>θ</mi><mn>2</mn><mn>2</mn></msubsup>
	<mo stretchy="false">)</mo>
	<mo>=</mo>
	<mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac>
	<mo>||</mo>
	<mi>θ</mi>
	<msup><mo>||</mo><mn>2</mn></msup>
</math>


![](/images/2017/09/ml-8-7.png)


接下来考虑<math><msup><mi>θ</mi><mi>T</mi></msup><msup><mi>x</mi><mi>(i)</mi></msup></math>的替换，由前面可知，<math><msup><mi>θ</mi><mi>T</mi></msup><msup><mi>x</mi><mi>(i)</mi></msup></math>的内积可等价于向量x在向量θ
上的投影p乘以θ的范数，如上图所示，即：

<math display="block">
	<msup><mi>θ</mi><mi>T</mi></msup><msup><mi>x</mi><mi>(i)</mi></msup>
	<mo>=</mo>
	<msup><mi>p</mi><mi>(i)</mi></msup>
	<mo>||</mo><mi>θ</mi><mo>||</mo>
	<mo>=</mo>
	<msub><mi>θ</mi><mn>1</mn></msub><msubsup><mi>x</mi><mn>1</mn><mi>(i)</mi></msubsup>
	<mo>+</mo>
	<msub><mi>θ</mi><mn>2</mn></msub><msubsup><mi>x</mi><mn>2</mn><mi>(i)</mi></msubsup>
</math>

进而上面的优化目标变为：

<math display="block">
	<munder>
      <mrow>
        <mtext>min</mtext>
      </mrow>
      <mi>θ</mi>
    </munder>
    <mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac>
    <munderover>
      <mo>∑</mo>
      <mrow>
        <mi>j</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow>
        <mi>n</mi>
      </mrow>
    </munderover>
	<msubsup>
		<mi>θ</mi>
		<mi>j</mi>
		<mn>2</mn>
	</msubsup>
</math>
<math display="block">
	<msup><mi>p</mi><mi>(i)</mi></msup>
	<mo>||</mo><mi>θ</mi><mo>||</mo>
	<mo>>=</mo>
	<mn>1</mn>
	<mspace width="2em"/>
	<mtext>if</mtext>
	<mspace width ="1em"/>
	<msup><mi>y</mi><mi>(i)</mi></msup>
	<mo>=</mo>
	<mn>1</mn>
</math>
<math display="block">
	<msup><mi>p</mi><mi>(i)</mi></msup>
	<mo>||</mo><mi>θ</mi><mo>||</mo>
	<mo><=</mo>
	<mn>-1</mn>
	<mspace width="2em"/>
	<mtext>if</mtext>
	<mspace width ="1em"/>
	<msup><mi>y</mi><mi>(i)</mi></msup>
	<mo>=</mo>
	<mn>0</mn>
</math>

接着我们用上面的视角再重新看待分类问题，如下图


![](/images/2017/09/ml-8-8.png)

同样的样本数据，左图是用绿线做Decision Boundary，正负样本x在θ上的投影p长度很小，对于<math><mi>y</mi><mo>=</mo><mn>1</mn></math>的结果，要求<math><msup><mi>p</mi><mi>(i)</mi></msup><mo>||</mo><mi>θ</mi><mo>||</mo><mo>>=</mo><mn>1</mn></math>那么则需要<math><mo>||</mo><mi>θ</mi><mo>||</mo></math>很大，而<math><mo>||</mo><mi>θ</mi><mo>||</mo></math>显然会使优化函数<math> <munder> <mrow> <mtext>min</mtext> </mrow> <mi>θ</mi> </munder> <mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac> <munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>n</mi> </mrow> </munderover> <msubsup> <mi>θ</mi> <mi>j</mi> <mn>2</mn> </msubsup> </math>的值变大。因此这样的Decision Boundary是我们不想要的。右图的绿线是SVM模型的Decision Boundary，按照上面的推理可以看出，SVM模型得到的<math><mo>||</mo><mi>θ</mi><mo>||</mo></math>比左边的要小，进而使优化函数能得出更优解。注意的是，<math><msub><mi>θ</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn></math>可以使Decision Boundary穿过原点，即使<math><msub><mi>θ</mi><mn>0</mn></msub><mo>≠</mo><mn>0</mn></math>，结论也依旧成立。

上面解释了SVM也叫做 **Large Margin Classifier** 的原因。接下来讨论如何使用SVM解决复杂非线性分类方程，也叫做求解 **Kernels** 。

假设有一系列非线性样本如下图：


![](/images/2017/09/ml-8-9.png)

我们使用一个非线性方程来描述Decision Boundary（上图蓝线）：

<math display="block">
<mi>h</mi>
<mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
<mo>=</mo>
<mo>{</mo>
<mtable>
<mtr>
	<mtd><mn>1</mn></mtd>
	<mtd><mtext>if</mtext></mtd>
	<mtd><msub><mi>θ</mi><mn>0</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>3</mn></msub><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>4</mn></msub><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup> <mo>+</mo> <msub><mi>θ</mi><mn>5</mn></msub><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup> <mo>+</mo> <mo>...</mo><mo>>=</mo><mn>0</mn></mtd>
</mtr>
<mtr>
	<mtd><mn>0</mn></mtd>
	<mtd><mtext>otherwise</mtext></mtd>
</mtr>
</mtable>
</math>

用<math><mi>f</mi></math>替换多项式中的<math><mi>x</mi></math>，有

<math display="block"> <mtable> <mtr> <mtd><msub><mi>θ</mi><mn>0</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>1</mn></msub><msub><mi>f</mi><mn>1</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>2</mn></msub><msub><mi>f</mi><mn>2</mn></msub> <mo>+</mo> <msub><mi>θ</mi><mn>3</mn></msub><msub><mi>f</mi><mn>3</mn></msub><mo>+...</mo></mtd> </mtr> </mtable> </math>

其中：
<math display="block"> <mtable> <mtr> <mtd><msub><mi>f</mi><mn>1</mn></msub><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub></mtd> </mtr> <mtr> <mtd><msub><mi>f</mi><mn>2</mn></msub><mo>=</mo><msub><mi>x</mi><mn>2</mn></msub></mtd> </mtr> <mtr> <mtd><msub><mi>f</mi><mn>3</mn></msub><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub></mtd> </mtr> <mtr> <mtd><msub><mi>f</mi><mn>4</mn></msub><mo>=...</mo></mtd> </mtr> </mtable> </math>

显然这种构造方式是由于样本特征数量有限，我们需要使用样本的一系列高阶组合来当做新的feature从而产生高阶多项式。但是对于特征的产生，即<math><mi>f</mi></math>的取值有没有更好的选择呢？接下来我们使用kernal来产生新的feature <math><msub><mi>f</mi><mn>1</mn></msub><mo>,</mo><msub><mi>f</mi><mn>2</mn></msub><mo>,</mo><msub><mi>f</mi><mn>3</mn></msub></math>

给定任一个样本值<math><mi>x</mi></math>，定义<math><mi>f</mi></math>为：

<math display="block">
	<msub><mi>f</mi><mi>i</mi></msub>
	<mo>=</mo>
	<mtext>similarity</mtext>
	<mo stretchy="false">(</mo>
	<mi>x</mi>
	<mo>,</mo>
	<msup><mi>l</mi><mi>(i)</mi></msup>
	<mo stretchy="false">)</mo>
	<mo>=</mo>
	<mtext>exp</mtext>
	<mo>(</mo>
	<mo>-</mo>
	<mfrac>
		<mrow><mo>||</mo><mi>x</mi><mo>-</mo>
		<msup><mi>l</mi><mi>(i)</mi></msup>
		<msup><mo>||</mo><mn>2</mn></msup>
		</mrow>
		<mrow><mn>2</mn>
		<msup><mi>σ</mi><mn>2</mn></msup></mrow>
	</mfrac>
	<mo>)</mo>
	<mo>=</mo>
	<mtext>exp</mtext>
	<mo>(</mo>
	<mo>-</mo>
	<mfrac>
		<mrow>
			<munderover> <mo>∑</mo> 
				<mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow>
				<mrow> <mi>n</mi> </mrow>
			</munderover>
		<mo stretchy="false">(</mo>
		<msub><mi>x</mi><mi>j</mi></msub>
		<mo>-</mo>
		<msubsup><mi>l</mi><mi>j</mi><mi>(i)</mi></msubsup>
		<msup><mo stretchy="false">)</mo><mn>2</mn></msup>
		</mrow>
		<mrow>
			<mn>2</mn>
			<msup><mi>σ</mi><mn>2</mn></msup>
		</mrow>
	</mfrac>
	<mo>)</mo>
</math>


其中:

- <math><msup><mi>l</mi><mi>(i)</mi></msup></math>成为 **Landmark** ，每个标记点会定义一个新的feature变量，选取方式在后面将会介绍
- **similarity** 函数称为 **Kernel** 函数，Kernel函数有多种，在上述公式中，kernel函数为高斯函数，有时也记作：<math><mi>K</mi> <mo stretchy="false">(</mo> <mi>x</mi> <mo>,</mo> <msup><mi>l</mi><mi>(i)</mi></msup> <mo stretchy="false">)</mo> </math>
- 假设<math><mi>x</mi><mo>≈</mo><msup><mi>l</mi><mi>(i)</mi></msup></math>，则有<math> <msub><mi>f</mi><mi>i</mi></msub> <mo>≈</mo> <mtext>exp</mtext> <mo>(</mo><mo>-</mo> <mfrac> <mrow><mn>0</mn> </mrow> <mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow> </mfrac> <mo>)</mo> <mo>≈</mo><mn>1</mn> </math>
- 假设<math><mi>x</mi></math>和landmark，即<math><msup><mi>l</mi><mi>(i)</mi></msup></math>很远，则有<math><msub><mi>f</mi><mi>i</mi></msub> <mo>≈</mo><mtext>exp</mtext><mo>(</mo><mo>-</mo> <mfrac> <mrow><mtext>(large number)</mtext> </mrow> <mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow> </mfrac><mo>)</mo><mo>≈</mo><mn>0</mn> </math>
- 给定一个新的样本<math><mi>x</mi></math>，我们可以计算<math><msub><mi>f</mi><mn>1</mn></msub><mo>,</mo><msub><mi>f</mi><mn>2</mn></msub><mo>,</mo><msub><mi>f</mi><mn>3</mn></msub></math>的值

- 高斯核函数Octave实现

```matlab

function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
x1 = x1(:); x2 = x2(:);

sim = e^(-(sum((x1-x2).^2)/(2*sigma^2)));

% =============================================================
    
end



```


假如我们有一个landmark：<math> <msup><mi>l</mi><mn>(1)</mn></msup> <mo>=</mo><mfenced open="[" close="]"> <mtable> <mtr> <mtd> <mn>3</mn> </mtd> </mtr> <mtr> <mtd> <mn>5</mn> </mtd> </mtr> </mtable> </mfenced> </math>，<math> <msub><mi>f</mi><mn>1</mn></msub> <mo>=</mo><mtext>exp</mtext> <mo>(</mo> <mo>-</mo> <mfrac> <mrow><mo>||</mo><mi>x</mi><mo>-</mo><msup><mi>l</mi><mn>(1)</mn></msup><msup><mo>||</mo><mn>2</mn></msup></mrow> <mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow> </mfrac> <mo>)</mo> </math>，令<math><msup><mi>σ</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn></math>，下图是Kernel函数的可视化呈现：


![](/images/2017/09/ml-8-10.png)


当给定的样本点 <math><mi>x</mi></math> 更靠近 <math><mfenced open="[" close="]"> <mtable> <mtr> <mtd> <mn>3</mn> </mtd> </mtr> <mtr> <mtd> <mn>5</mn> </mtd> </mtr></mtable></mfenced></math> 时，<math><msub><mi>f</mi><mn>1</mn></msub></math>更接近最大值1。同理，离得更远则 <math><msub><mi>f</mi><mn>1</mn></msub></math> 更接近于0。接下来我们观察<math><msup><mi>σ</mi><mn>2</mn></msup></math>对f曲线的影响

![](/images/2017/09/ml-8-11.png)


- 上图可以看到当<math><msup><mi>σ</mi><mn>2</mn></msup></math>越小时，f下降到0的速度越快，反之则越慢


当我们计算出<math><msub><mi>f</mi><mn>1</mn></msub><mo>,</mo><msub><mi>f</mi><mn>2</mn></msub><mo>,</mo><msub><mi>f</mi><mn>3</mn></msub></math>的值之后，给定θ值，我们就能绘制预测函数了，如下图：

![](/images/2017/09/ml-8-12.png)


假设我们有一个样本<math><mi>x</mi></math>（粉色的点），显然它更靠近<math><msup><mi>l</mi><mn>(1)</mn></msup></math>，因此有：<math><msub><mi>f</mi><mn>1</mn></msub> <mo>≈</mo> <mn>1</mn><mo>,</mo><msub><mi>f</mi><mn>2</mn></msub> <mo>≈</mo><mn>0</mn><mo>,</mo><msub><mi>f</mi><mn>3</mn></msub> <mo>≈</mo> <mn>0</mn><mo></math>，对应的预测函数为：

<math display="block">
<msub><mi>θ</mi><mn>0</mn></msub>
<mo>+</mo>
<msub><mi>θ</mi><mn>1</mn></msub><mo> * </mo><mn>1</mn>
<mo>+</mo>
<msub><mi>θ</mi><mn>2</mn></msub><mo> * </mo><mn>0</mn>
<mo>+</mo>
<msub><mi>θ</mi><mn>3</mn></msub><mo> * </mo><mn>0</mn>
<mo>=</mo>
<mn>-0.5</mn><mo>+</mo><mn>1</mn>
<mo>=</mo>
<mn>0.5</mn>
<mo>>=</mo>
<mn>0</mn>
</math>

因此这个样本点<math><mi>x</mi></math>的分类结果为<math><mi>y</mi><mo>=</mo><mn>1</mn></math>。
同理，对于另一个样本点<math><mi>x</mi></math>（绿色的点），它距离三个landmark都很远，因此有<math><msub><mi>f</mi><mn>1</mn></msub> <mo>≈</mo> <mn>0</mn><mo>,</mo><msub><mi>f</mi><mn>2</mn></msub> <mo>≈</mo><mn>0</mn><mo>,</mo><msub><mi>f</mi><mn>3</mn></msub> <mo>≈</mo> <mn>0</mn><mo></math>，预测函数为：

<math display="block">
<msub><mi>θ</mi><mn>0</mn></msub>
<mo>+</mo>
<msub><mi>θ</mi><mn>1</mn></msub><mo> * </mo><mn>0</mn>
<mo>+</mo>
<msub><mi>θ</mi><mn>2</mn></msub><mo> * </mo><mn>0</mn>
<mo>+</mo>
<msub><mi>θ</mi><mn>3</mn></msub><mo> * </mo><mn>0</mn>
<mo>=</mo>
<mn>-0.5</mn><mo>+</mo><mn>0</mn>
<mo>=</mo>
<mn>-0.5</mn>
<mo><</mo>
<mn>0</mn>
</math>

因此这个样本点<math><mi>x</mi></math>的分类结果为<math><mi>y</mi><mo>=</mo><mn>0</mn></math>。接下来我们讨论如何选择则Landmark。

通常来说在给定一组训练样本之后，我们把每个样本点标记为一个landmark，如图所示


![](/images/2017/09/ml-8-13.png)


具体来说，给定训练样本：<math><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo stretchy="false">)</mo><mo>...</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo stretchy="false">)</mo></math> ，使 <math><msup><mi>l</mi><mi>(1)</mi></msup><mo>=</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>l</mi><mi>(2)</mi></msup><mo>=</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>...</mo><msup><mi>l</mi><mi>(m)</mi></msup><mo>=</mo><msup><mi>x</mi><mi>(m)</mi></msup></math>，对于任意训练样本<math><mi>x</mi></math>，计算<math><mi>f</mi></math>向量：

<math display="block">
<mi>f</mi>
<mo>=</mo>
<mo>[</mo>
<mtable>
<mtr>
	<mtd><msub><mi>f</mi><mn>0</mn></msub> <mo>=</mo><mn>1</mn></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>f</mi><mn>1</mn></msub> <mo>=</mo> <mtext>similarity</mtext> <mo stretchy="false">(</mo> <mi>x</mi> <mo>,</mo> <msup><mi>l</mi><mn>(1)</mn></msup> <mo stretchy="false">)</mo></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>f</mi><mn>2</mn></msub> <mo>=</mo> <mtext>similarity</mtext> <mo stretchy="false">(</mo> <mi>x</mi> <mo>,</mo> <msup><mi>l</mi><mn>(2)</mn></msup> <mo stretchy="false">)</mo></mtd>
</mtr>
<mtr>
	<mtd><mo>...</mo></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>f</mi><mi>m</mi></msub> <mo>=</mo> <mtext>similarity</mtext> <mo stretchy="false">(</mo> <mi>x</mi> <mo>,</mo> <msup><mi>l</mi><mi>(m)</mi></msup> <mo stretchy="false">)</mo></mtd>
</mtr>
</mtable>
<mo>]</mo>
</math>

![](/images/2017/09/ml-8-14.png)

在得到<math><mi>f</mi></math>向量后，我们可以构建预测函数

<math display="block">
<mi>h</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
<mo>=</mo>
<mo>{</mo>
<mtable>
	<mtr>
		<mtd><mn>1</mn></mtd>
		<mtd><mtext>if</mtext></mtd>
		<mtd><msup><mi>θ</mi><mi>T</mi></msup><mi>f</mi><mo>>=</mo><mn>0</mn></mtd>
	</mtr>
	<mtr>
		<mtd><mn>0</mn></mtd>
		<mtd><mtext>otherwise</mtext></mtd>
	</mtr>
</mtable>
</math>

带入cost函数，通过最优化解法，计算出θ值。注意下面给出的cost函数和上面给出的cost函数有一点不同，这里使用第<math><mi>1</mi></math>的kernal函数的结果<math><msup><mi>f</mi><mi>(i)</mi></msup></math>代替原先在<math><mi>1</mi></math>的样本值<math><msup><mi>x</mi><mi>(i)</mi></msup></math>。SVM的最优化算法实际上是凸优化问题，好的SVM软件包可以求出最小值，不需要担心局部最优解的问题

<math display="block">
  <mstyle displaystyle="true">
    <munder>
      <mrow class="MJX-TeXAtom-OP">
        <mtext>min</mtext>
      </mrow>
      <mi>&#x03B8;<!-- θ --></mi>
    </munder>
    <mo>&#x2061;<!-- ⁡ --></mo>
    <mtext>&#xA0;</mtext>
    <mi>C</mi>
    <mfenced open="[" close="]">
      <mrow>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>m</mi>
          </mrow>
        </munderover>
        <msup>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <msub>
          <mtext>cost</mtext>
          <mn>1</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>f</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
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
        <msub>
          <mtext>cost</mtext>
          <mn>0</mn>
        </msub>
        <mo stretchy="false">(</mo>
        <msup>
          <mi>&#x03B8;<!-- θ --></mi>
          <mi>T</mi>
        </msup>
        <msup>
          <mi>f</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
      </mrow>
    </mfenced>
    <mo>+</mo>
    <mfrac>
      <mn>1</mn>
      <mn>2</mn>
    </mfrac>
    <munderover>
      <mo>&#x2211;<!-- ∑ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>j</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mi>m</mi>
    </munderover>
    <msubsup>
      <mi>&#x03B8;<!-- θ --></mi>
      <mi>j</mi>
      <mn>2</mn>
    </msubsup>
  </mstyle>
</math>


最后还要说明几点：

- 某些SVM算法，在Regularization一项会有不同的表示方式，有的将<math><mfrac> <mn>1</mn> <mn>2</mn> </mfrac> <munderover> <mo>∑</mo> <mrow> <mi>j</mi> <mo>=</mo> <mn>1</mn> </mrow> <mi>m</mi> </munderover> <msubsup> <mi>θ</mi> <mi>j</mi> <mn>2</mn> </msubsup></math> 表示为 <math><msup><mi>θ</mi><mi>T</mi></msup><mi>θ</mi></math>，数学上是等价的。也有些算法为了计算效率将其表示为<math><msup><mi>θ</mi><mi>T</mi></msup><mi>M</mi><mi>θ</mi></math>，<math><mi>M</mi></math>为样本数量，暂不关心其中的细节。对于最优化函数的解法，这个课程不讨论，调用现有的函数即可。

- SVM参数
	- 选择<math><mi>C</mi><mo stretchy="false">(</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>λ</mi></mrow></mfrac><mo stretchy="false">)</mo></math>的值。<math><mi>C</mi></math>值很大，会造成较低偏差（bias），但会带来较高的方差(variance)，有overfit的趋势；<math><mi>C</mi></math>值很小，会造成较高的偏差和较低的方差，有underfit的趋势。

	- 选择<math><msup><mi>σ</mi><mn>2</mn></msup></math>的值，<math><msup><mi>σ</mi><mn>2</mn></msup></math>越大，<math><mi>f</mi></math>下降的越平滑，会导致样本点<math><msup><mi>x</mi><mi>(i)</mi></msup></math>据landmark点<math><msup><mi>l</mi><mi>(i)</mi></msup></math>的距离偏大，从而带来较大的偏差和较低的方差，有underfit的趋势。<math><msup><mi>σ</mi><mn>2</mn></msup></math>越小，<math><mi>f</mi></math>下降速度越快，越陡峭，会带来较低的偏差和较高的方差，有overfit的趋势。
	
- 参数选择的Octave实现思路

```matlab

function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [ 0.01, 0.03, 0,1, 0.3, 1, 3, 10, 30 ];

max_error = Inf;

for c = values;
    for s = values;
        model       = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        predictions = svmPredict(model, Xval); 
        error = mean(double(predictions ~= yval));
        if error < max_error;
            max_error = error;
            C=c;
            sigma=s;
        end;
    end
end; 

% =========================================================================

end

```

### SVM编程实现思路
	
接下来讨论如何编程实现SVM，前面提到，SVM给出了一个优化目标和求解最优解问题，最优解的求解可以通过使用一些库来完成，比如`liblinear`，`libsvm`等。我们只需要

- 给出参数<math><mi>C</mi></math>的值
- 给出Kernel的核函数
	- 如果样本特征<math><mi>n</mi></math>很大，而样本数量<math><mi>m</mi></math>很少的的情况下使用
，可以选择"No kernel"，所谓"No kernel"实际上就是"Linear kernel"，使用线性方程 <math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>=</mo><msub> <mi>θ</mi> <mn>0</mn> </msub> <mo>+</mo> <msub> <mi>θ</mi> <mn>1</mn> </msub> <msub> <mi>x</mi> <mn>1</mn> </msub> <mo>+</mo> <msub> <mi>θ</mi> <mn>2</mn> </msub> <msub> <mi>x</mi> <mn>2</mn> </msub> <mo>+</mo> <msub> <mi>θ</mi> <mn>3</mn> </msub> <msub> <mi>x</mi> <mn>3</mn> </msub> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msub> <mi>θ</mi> <mi>n</mi> </msub> <msub> <mi>x</mi> <mi>n</mi> </msub></math> 来判定<math><mi>y</mi></math>值	- 如果样本特征<math><mi>n</mi></math>很少，而样本数量<math><mi>m</mi></math>很大，可以选择高斯kernel，使用高斯kernel要注意几点
		- 给出合适的<math><msup><mi>σ</mi><mn>2</mn></msup></math>的值。
		- 对样本进行feature scaling，不同feature间的数量级可能不同，避免范数计算产生较大偏差，对feature要进行归一化
	- 如果要使用其它的核函数，要确保该核函数满足"Mercer's Theorem"，这样SVM库才能对核函数进行通用的最优化求解
	- 对于多个核函数，选择在交叉验证数据集表现最好的核函数

对于多个分类结果的场景，许多SVM库提供了函数可直接调用。或者采用之前提到的one-vs-all的方式

![](/images/2017/09/ml-8-15.png)

### Logistic regression vs. SVM

我们用<math><mi>n</mi></math>表示样本特征数量，用<math><mi>m</mi></math>表示样本数量

- 如果 <math><mi>n</mi><mo>></mo><mi>m</mi></math>(n=10,000,m=10~1000)，比如Spam邮件过滤，有几千个关键词作为feature，而邮件数量即样本数量要明显少于特征数量，这时可以使用逻辑回归或者SVM线性核函数
- 如果 <math><mi>n</mi></math>很小(1 ~ 1000)，<math><mi>m</mi></math>处于中等数量级（10 ~ 10000），这时可以使用SVM高斯核函数
- 如果 <math><mi>n</mi></math>很小(1 ~ 1000)，<math><mi>m</mi></math>很大（几十万），这种情况下使用SVM高斯核函数会很慢，这时需要增加更多的feature，然后使用逻辑回归或者线性SVM
- 对于神经网络来说没有这些限制，但是训练起来比较慢
