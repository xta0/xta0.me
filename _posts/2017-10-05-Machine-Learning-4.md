---
layout: post
list_title: 机器学习 | Machine Learning | 神经网络 | Neural Networks
title: 神经网络
meta: Coursera Stanford Machine Learning Cousre Note, Chapter4
categories: [Machine Learning,AI]
mathjax: true
---

## Non-linear hypotheses

神经网络是一个很老的概念，在机器学习领域目前还是主流，之前已经介绍了linear regression和logistics regression，为什么还要学习神经网络？以non-linear classification为例，如下图所示

![](/assets/images/2017/09/ml-6-1.png)

我们可以通过构建对feature的多项式$g_\theta$来确定预测函数，但这中方法适用于feature较少的情况，比如上图中，只有两个feature: x1和x2。

当feature多的时候，产生多项式就会变得很麻烦，还是预测房价的例子，可能的feature有很多，比如：`x1=size`,`x2=#bedrooms`,`x3=#floors`,`x4=age`,...`x100=#schools`等等，假设`n=100`，有几种做法：

- 构建二阶多项式
	- 如`x1^2,x1x2,x1x3,x1x4,...,x1x100,x2^2,x2x3...`有约为5000项(n^2/2)，计算的代价非常高。
	- 取一个子集，比如`x1^2, x2^2,x3^2...x100^2`，这样就只有100个feature，但是100个feature会导致结果误差很高

- 构建三阶多项式
	- 如`x1x2x3, x1^2x2, x1^2x3...x10x11x17...`, 有约为n^3量级的组合情况，约为170,000个

另一个例子是图像识别与分类，例如识别一辆汽车，对于图像来说，是通过对像素值进行学习（车共有的像素特征vs非汽车特征），那么feature就是图片的像素值，如下图所示，假如有两个像素点是车图片都有的：

![](/assets/images/2017/09/ml-6-2.png)

假设图片大小为50x50，总共2500个像素点，即2500个feature（灰度图像，RGB乘以三），如果使用二次方程，那么有接近300万个feature，显然图像这种场景使用non linear regression不合适，需要探索新的学习方式

	
## Model Representation

所谓神经元(Neuron)就是一种计算单元，它的输入是一组特征信息$x_1$...$x_n$，输出为预测函数的结果。

![](/assets/images/2017/09/ml-6-4.png)

如上图，是一个单层的神经网络，其中：

- 输入端$x_0$默认为1，也叫做"bias unit."
- $h_\theta(x)$ 和逻辑回归预测方程一样，用$\frac{1}{1+e^{-\theta^Tx}}$表示，也叫做**activation**函数
- θ矩阵在神经网络里也被叫做权重weight

一个多层神经网络如下图

![](/assets/images/2017/09/ml-6-5.png)

- 第一层是叫"Input Layer"，最后一层叫"Output Layer"，中间叫"Hidden Layer"，上面例子中，对于hidden layer我们使用$a_0^2...a_n^2$表示，他们也叫做"activation units."

- $a_i^{(j)} = "activition" \thinspace of \thinspace unit \thinspace {i} \thinspace in \thinspace \thinspace layer \thinspace{j}$
- $\theta^{j} = \thinspace matrix \thinspace of \thinspace weights \thinspace controlling \thinspace function \thinspace mapping \thinspace from \thinspace layer \thinspace j \thinspace to \thinspace {j+1}$

$$
\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
\end{bmatrix}
\to
\begin{bmatrix}
a_1^(2) \\
a_2^(2) \\
a_3^(2) \\
\end{bmatrix}
\to
h_\theta(x)
$$

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mfenced open="[" close="]">
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mn>0</mn>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mn>1</mn>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mn>2</mn>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mn>3</mn>
          </msub>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
  <mo stretchy="false">→</mo>
  <mfenced open="[" close="]">
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <msubsup>
            <mi>a</mi>
            <mn>1</mn>
            <mrow class="MJX-TeXAtom-ORD">
              <mo stretchy="false">(</mo>
              <mn>2</mn>
              <mo stretchy="false">)</mo>
            </mrow>
          </msubsup>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msubsup>
            <mi>a</mi>
            <mn>2</mn>
            <mrow class="MJX-TeXAtom-ORD">
              <mo stretchy="false">(</mo>
              <mn>2</mn>
              <mo stretchy="false">)</mo>
            </mrow>
          </msubsup>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msubsup>
            <mi>a</mi>
            <mn>3</mn>
            <mrow class="MJX-TeXAtom-ORD">
              <mo stretchy="false">(</mo>
              <mn>2</mn>
              <mo stretchy="false">)</mo>
            </mrow>
          </msubsup>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
  <mo stretchy="false"> → </mo>
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
</math>


对第二层每个activation节点的计算公式如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>1</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>10</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>0</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>11</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>1</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
			<mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>12</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>2</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>13</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>3</mn>
        </msub>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>2</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>20</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>0</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>21</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>1</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>22</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>2</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>23</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>3</mn>
        </msub>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>3</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>30</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>0</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>31</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>1</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>32</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>2</mn>
        </msub>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>33</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msub>
          <mi>x</mi>
          <mn>3</mn>
        </msub>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub>
          <mi>h</mi>
			<mi mathvariant="normal">Θ</mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <msubsup>
          <mi>a</mi>
          <mn>1</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>3</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>10</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msubsup>
          <mi>a</mi>
          <mn>0</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>11</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msubsup>
          <mi>a</mi>
          <mn>1</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>12</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msubsup>
          <mi>a</mi>
          <mn>2</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>+</mo>
        <msubsup>
          <mi mathvariant="normal">Θ</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>13</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msubsup>
          <mi>a</mi>
          <mn>3</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

上面可以看到<math><msup><mi>θ</mi><mi>(1)</mi></msup></math>矩阵是3x4的，X矩阵是4x1的，这样相乘得出的矩阵是3x1的，对应每个a节点的值。最终输出结果是另一个θ矩阵<math><msup><mi>θ</mi><mi>(2)</mi></msup></math>乘以上面的3x1矩阵，由此可知<math><msup><mi>θ</mi><mi>(2)</mi></msup></math>是3x1的矩阵，这个矩阵的意义是第二层节点的权重值。

因此可以知道，每一层神经网络节点都有各自的权重矩阵<math><msup><mi>θ</mi><mi>(j)</mi></msup></math>，矩阵维度的计算方法是：

> <math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><mtext>If network has&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mi>j</mi></msub></mrow><mtext>&#xA0;units in layer&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><mi>j</mi></mrow><mtext>&#xA0;and&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow><mtext>&#xA0;units in layer&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow><mtext>,then&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="normal">Θ</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></mrow><mtext>&#xA0;will be of dimension&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>×</mo><mo stretchy="false">(</mo><msub><mi>s</mi><mi>j</mi></msub><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow><mtext>.</mtext></math>


推算可知+1来自<math><msup><mi>θ</mi><mi>(j)</mi></msup></math>矩阵的"bias nodes"，即<math><msub><mi>x</mi><mn>0</mn></msub></math>和<math><msubsup><mi>θ</mi><mi>(j)</mi><mn>0</mn></msubsup></math>。如下图：

![](/assets/images/2017/09/ml-6-6.png)

举例来说：

如果layer 1有两个输入节点，layer 2 有4个activation节点，那么<math><msup><mi>θ</mi><mi>(1)</mi></msup></math>是4x3的，<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>×</mo><mo stretchy="false">(</mo><msub><mi>s</mi><mi>j</mi></msub><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><mo>×</mo><mn>3</mn></math>

- 向量化表示

上面计算中间节点的式子可以用向量化表示，我们定义一个新的变量<math><msubsup><mi>z</mi><mi>k</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup></math>来表示g函数的参数，则上述公式可表示如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>1</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi>z</mi>
          <mn>1</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>2</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi>z</mi>
          <mn>2</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msubsup>
          <mi>a</mi>
          <mn>3</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi>z</mi>
          <mn>3</mn>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

其中，上角标用来表示第几层layer，下角标表示该层的第几个节点。例如layer j=2，第k个节点，z 表示为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msubsup>
    <mi>z</mi>
    <mi>k</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>2</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <mo>=</mo>
  <msubsup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
      <mo>,</mo>
      <mn>0</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <msub>
    <mi>x</mi>
    <mn>0</mn>
  </msub>
  <mo>+</mo>
  <msubsup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
      <mo>,</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <mo>&#x22EF;<!-- ⋯ --></mo>
  <mo>+</mo>
  <msubsup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>k</mi>
      <mo>,</mo>
      <mi>n</mi>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
  <msub>
    <mi>x</mi>
    <mi>n</mi>
  </msub>
</math>

用向量表示x和<math><msup><mi>z</mi><mrow><mi>j</mi></mrow></msup></math>如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mi>x</mi>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>0</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>1</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <mo>&#x22EF;<!-- ⋯ --></mo>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mi>n</mi>
                </msub>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
      <mtd>
        <msup>
          <mi>z</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>j</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msubsup>
                  <mi>z</mi>
                  <mn>1</mn>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mi>j</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msubsup>
                  <mi>z</mi>
                  <mn>2</mn>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mi>j</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <mo>&#x22EF;<!-- ⋯ --></mo>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msubsup>
                  <mi>z</mi>
                  <mi>n</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mi>j</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
    </mtr>
  </mtable>
</math>

令x为第一层节点<math><mi>x</mi><mo>=</mo><msup><mi>a</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup></math>，则每层的向量化表示为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo>&#x2212;<!-- − --></mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo>&#x2212;<!-- − --></mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
</math>

其中，<math><msup><mi>Θ</mi><mi>(j-1)</mi></msup></math>是 jx(j+1) 的，<math><msup><mi>a</mi><mi>(j-1)</mi></msup></math>是(j+1)x1的，因此<math><msup><mi>z</mi><mi>j</mi></msup></math>是jx1的，即

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>g</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
</math>

当我们计算完<math><msup><mi>a</mi><mi>(j)</mi></msup></math>后，我们可以给<math><msup><mi>a</mi><mi>(j)</mi></msup></math>增加一个bias unit，即<math><msubsup><mi>a</mi><mn>0</mn><mi>(j)</mi></msubsup><mo>=</mo><mn>1</mn></math>，则a变成了(j+1)x1的。以此类推：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo>+</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
</math>

最终的预测函数h表示为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>h</mi>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo>+</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>g</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>j</mi>
      <mo>+</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
</math>

注意到在每层的计算上，我们的预测函数和逻辑回归基本相同。我们增加了这么多层，即神经网络是为了更好的得到非线性函数的预测结果，这个算法也叫做**Forward Propagation**，后面简称**FB**算法，Octave实现为：

```matlab

function g = sigmoid(z)
	g = 1.0 ./ (1.0 + exp(-z));
end

function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1 = [ones(m, 1), X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(m,1) a2];
h = sigmoid(a2*Theta2');

% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
[max,index] = max(h,[],2);
p = index;

end

```



### Neural Network Example

- 单层神经网络实现与或门

神经网络的一个简单应用是预测<math><msub><mi>x</mi><mn>1</mn></msub></math> AND <math><msub><mi>x</mi><mn>2</mn></msub></math>，当<math><msub><mi>x</mi><mn>1</mn></msub></math>和<math><msub><mi>x</mi><mn>2</mn></msub></math>都为1的时候，结果是true，预测函数如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>0</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>1</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>2</mn>
                </msub>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
        <mo stretchy="false">&#x2192;<!-- → --></mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mi>g</mi>
                <mo stretchy="false">(</mo>
                <msup>
                  <mi>z</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mn>2</mn>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo stretchy="false">)</mo>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
        <mo stretchy="false">&#x2192;<!-- → --></mo>
        <msub>
          <mi>h</mi>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

<math><msub><mi>x</mi><mn>0</mn></msub></math>为1，我们假设 <math><msup><mi>θ</mi><mi>(1)</mi></msup></math>的值如下：<math><msup><mi>Θ</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mo stretchy="false">[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mo>−</mo><mn>30</mn></mtd><mtd><mn>20</mn></mtd><mtd><mn>20</mn></mtd></mtr></mtable><mo stretchy="false">]</mo></math>


![](/assets/images/2017/09/ml-6-7.png)


如图所示，我们构建了一个一层的神经网络来处理计算机的"AND"请求，来代替原来的“与门”。神经网络可用来构建所有的逻辑门，比如"OR"运算如下图：

![](/assets/images/2017/09/ml-6-8.png)

- 二级神经网络构建同或门

上面我们实现了与或非(非的推导忽略)，对应的theta矩阵如下：

<br></br>

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mi>A</mi>
        <mi>N</mi>
        <mi>D</mi>
        <mo>:</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mo>&#x2212;<!-- − --></mo>
                <mn>30</mn>
              </mtd>
              <mtd>
                <mn>20</mn>
              </mtd>
              <mtd>
                <mn>20</mn>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mi>N</mi>
        <mi>O</mi>
        <mi>R</mi>
        <mo>:</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mn>10</mn>
              </mtd>
              <mtd>
                <mo>&#x2212;<!-- − --></mo>
                <mn>20</mn>
              </mtd>
              <mtd>
                <mo>&#x2212;<!-- − --></mo>
                <mn>20</mn>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mi>O</mi>
        <mi>R</mi>
        <mo>:</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mo>&#x2212;<!-- − --></mo>
                <mn>10</mn>
              </mtd>
              <mtd>
                <mn>20</mn>
              </mtd>
              <mtd>
                <mn>20</mn>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
      </mtd>
    </mtr>
  </mtable>
</math>

我们可以通过上面的矩阵来构建XNOR门：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>0</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>1</mn>
                </msub>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msub>
                  <mi>x</mi>
                  <mn>2</mn>
                </msub>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
        <mo stretchy="false">&#x2192;<!-- → --></mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msubsup>
                  <mi>a</mi>
                  <mn>1</mn>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mn>2</mn>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <msubsup>
                  <mi>a</mi>
                  <mn>2</mn>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mn>2</mn>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msubsup>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
        <mo stretchy="false">&#x2192;<!-- → --></mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msup>
                  <mi>a</mi>
                  <mrow class="MJX-TeXAtom-ORD">
                    <mo stretchy="false">(</mo>
                    <mn>3</mn>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
        <mo stretchy="false">&#x2192;<!-- → --></mo>
        <msub>
          <mi>h</mi>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
  </mtable>
</math>

第一层节点的θ矩阵为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>1</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mfenced open="[" close="]">
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>30</mn>
        </mtd>
        <mtd>
          <mn>20</mn>
        </mtd>
        <mtd>
          <mn>20</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>10</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>20</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>20</mn>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
</math>

第二层节点的θ矩阵为:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>2</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mfenced open="[" close="]">
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>10</mn>
        </mtd>
        <mtd>
          <mn>20</mn>
        </mtd>
        <mtd>
          <mn>20</mn>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
</math>

每层节点的计算用向量化表示为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd />
      <mtd>
        <msup>
          <mi>a</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo>&#x22C5;<!-- ⋅ --></mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msup>
          <mi>a</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>3</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo>=</mo>
        <mi>g</mi>
        <mo stretchy="false">(</mo>
        <msup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo>&#x22C5;<!-- ⋅ --></mo>
        <msup>
          <mi>a</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>2</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <msub>
          <mi>h</mi>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <msup>
          <mi>a</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mn>3</mn>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
    </mtr>
  </mtable>
</math>


![](/assets/images/2017/09/ml-6-9.png)


### Multiclass Classification

使用神经网络进行多种类型分类的问题，我们假设最后的输出是一个向量，如下图所示

![](/assets/images/2017/09/ml-6-10.png)

上面的例子中，对于输出结果y的可能情况有：

![](/assets/images/2017/09/ml-6-11.png)

每一个<math><msup><mi>y</mi><mi>(i)</mi></msup></math>向量代表一中分类结果，抽象来看，多级神经网络分类可如下表示：

![](/assets/images/2017/09/ml-6-12.png)


### Cost Function

- 先定义一些变量:
	- L = 神经网络的层数
	- <math><msub><mi>S</mi><mi>l</mi></msub></math> = 第l层的节点数
	- K = 输出层的节点数，即输出结果的种类。
		- 对0和1的场景，K=1， <math><msub><mi>S</mi><mi>l</mi></msub><mo>=</mo><mn>1</mn></math>
		- 对于多种分类的场景，K>=3， <math><msub><mi>S</mi><mi>l</mi></msub><mo>=</mo><mi>K</mi></math>
		- 用<math><msub><mi>h</mi><mi>Θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><msub><mo stretchy="false">)</mo><mi>k</mi></msub></math>表示第K个分类的计算结果

- Cost Function

参考之前的逻辑回归cost函数：

<math display="block">
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
  <mo stretchy="false">]</mo>
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
</math>

对神经网络来说，输出结果不再只有两种类型，而是有K种分类，cost函数也更加抽象和复杂：

<math display="block">
  <mtable rowspacing="3pt" columnspacing="1em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>k</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>K</mi>
        </munderover>
        <mfenced open="[" close="]">
          <mrow>
            <msubsup>
              <mi>y</mi>
              <mi>k</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mo stretchy="false">(</mo>
                <mi>i</mi>
                <mo stretchy="false">)</mo>
              </mrow>
            </msubsup>
            <mi>log</mi>
            <mo>&#x2061;<!-- ⁡ --></mo>
            <mo stretchy="false">(</mo>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>h</mi>
              <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
            <msub>
              <mo stretchy="false">)</mo>
              <mi>k</mi>
            </msub>
            <mo stretchy="false">)</mo>
            <mo>+</mo>
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo>&#x2212;<!-- − --></mo>
            <msubsup>
              <mi>y</mi>
              <mi>k</mi>
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
            <mo stretchy="false">(</mo>
            <msub>
              <mi>h</mi>
              <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
            <msub>
              <mo stretchy="false">)</mo>
              <mi>k</mi>
            </msub>
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
            <mi>l</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>L</mi>
            <mo>&#x2212;<!-- − --></mo>
            <mn>1</mn>
          </mrow>
        </munderover>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <msub>
              <mi>s</mi>
              <mi>l</mi>
            </msub>
          </mrow>
        </munderover>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <msub>
              <mi>s</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>l</mi>
                <mo>+</mo>
                <mn>1</mn>
              </mrow>
            </msub>
          </mrow>
        </munderover>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>,</mo>
            <mi>i</mi>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>l</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </mtd>
    </mtr>
  </mtable>
</math>

为了计算多个输出结果，括号前的求和表示对K层分别进行计算后再累加计算结果。中括号后面是regularization项，是每层θ矩阵元素的平方和累加，公式里各层θ矩阵的列数等同于对应层的节点数，行数等它对应层的节点数+1，其中<math><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>s</mi><mi>l</mi></msub></mrow></munderover><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>s</mi><mrow><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow></munderover><mo stretchy="false">(</mo><msubsup><mi>Θ</mi><mrow><mi>j</mi><mo>,</mo><mi>i</mi></mrow><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup><msup><mo stretchy="false">)</mo><mn>2</mn></msup></math>是每个θ矩阵项的平方和，<math><munderover><mo>∑</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>L</mi><mo>-</mo><mn>1</mn></mrow></munderover></math>代表各个层θ矩阵的平方和累加。理解了这两部分就不难理解regularization项了，它和之前逻辑回归的regularization项概念是一致的。

理解上述式子着重记住以下三点：

1. 前两个求和符号是对每层神经网络节点进行逻辑回归cost function运算后求和
2. 后面三个求和符号是是每层神经网络节点的θ矩阵平方和的累加求和
3. 特殊注意的是，后面三个求和符号中第一个求和符号中的i代表层数的index，不代表训练样本的index

- Octave demo

假设有三层神经网络，已知权重矩阵Theta1，Theta2，代价函数使用代数形式描述为：

```matlab

function [J grad] = nnCostFunction(num_labels,X, y, Theta1, Theta2, lambda)

% X:5000x400
% y:5000x1
% num_labels:10
% Theta1: 25x401
% Theta2: 10x26

% Setup some useful variables
m = size(X, 1);
J = 0;


% make Y: 5000x10
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% cost function
J = (1/m)*sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2));
% regularization item
r = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

% add r
J = J+r;

end
```


## Backpropagation algotrithm

"Backpropagation"是神经网络用来求解**Cost Function**最小值的算法，类似之前线性回归和逻辑回归中的梯度下降法。上一节我们已经了解了**Cost Function**的定义，我们的目标是求解：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <munder>
    <mo movablelimits="true">min</mo>
    <mi mathvariant="normal">Θ</mi>
  </munder>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi mathvariant="normal">Θ</mi>
  <mo stretchy="false">)</mo>
</math>

即找到合适的θ值使**Cost Function**的值最小，即通过一个合适的算法来求解对θ的偏导

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mstyle displaystyle="true">
    <mfrac>
      <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
      <mrow>
        <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
        <msubsup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>,</mo>
            <mi>j</mi>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>l</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
      </mrow>
    </mfrac>
  </mstyle>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  <mo stretchy="false">)</mo>
</math>

我们先假设神经网络只有一个训练样本(x,y)，我们使用**"Forward Propagation"**的向量化方式来逐层计算求解出最后的结果

![](/assets/images/2017/09/ml-6-13.png)

接下来我们要求θ矩阵的值，用到的算法叫做**"Backpropagation"**，我们定义：

<math display="block"><msubsup><mi>δ</mi><mi>j</mi><mi>(l)</mi></msubsup><mo>=</mo><mtext>"error" of node j in layer l </mtext></math>

假设 **Layer L=4** 那么有：

<math display="block"><msubsup><mi>δ</mi><mi>j</mi><mi>(4)</mi></msubsup><mo>=</mo><msubsup><mi>a</mi><mi>j</mi><mi>(4)</mi></msubsup><mo>-</mo><msub><mi>y</mi><mi>j</mi></msub></math>

向量化表示为：

<math display="block"><msup><mi>δ</mi><mi>(4)</mi></msup><mo>=</mo><msup><mi>a</mi><mi>(4)</mi></msup><mo>-</mo><mi>y</mi></math>

其中，δ，a，y向量行数等于最后一层节点的个数，这样我们首先得到了最后一层的δ值，接下来我们要根据最后一层的δ值来向前计算前面各层的δ值，第三层的公式如下：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>δ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>3</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mo stretchy="false">(</mo>
  <msup>
    <mi mathvariant="normal">Θ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>3</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mo stretchy="false">)</mo>
    <mi>T</mi>
  </msup>
  <msup>
    <mi>δ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>4</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mtext>&#xA0;</mtext>
  <mo>.</mo>
  <mo>∗</mo>
  <mtext>&#xA0;</mtext>
 <msup>
    <mi>g</mi>
    <mo>'</mo>
  </msup>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>3</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
 </math>
 
 第二层的计算公式如下：
 
 <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>δ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>2</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mo stretchy="false">(</mo>
  <msup>
    <mi mathvariant="normal">Θ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>2</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mo stretchy="false">)</mo>
    <mi>T</mi>
  </msup>
  <msup>
    <mi>δ</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>3</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mtext>&#xA0;</mtext>
  <mo>.</mo>
  <mo>∗</mo>
  <mtext>&#xA0;</mtext>
 <msup>
    <mi>g</mi>
    <mo>'</mo>
  </msup>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mn>2</mn>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
 </math>
 
 <br></br>
 第一层是输入层，是样本数据，没有错误，因此不存在<math><msup><mi>δ</mi><mi>(1)</mi></msup></math>
<br></br>

在上述的式子中，<math><msup><mi>g</mi><mo>'</mo></msup></math>对<math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi>z</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math>的求导等价于下面式子：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>g</mi>
    <mo>'</mo>
  </msup>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>z</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>l</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>l</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mtext>&#xA0;</mtext>
  <mo>.</mo>
  <mo> ∗ </mo>
  <mtext>&#xA0;</mtext>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>-</mo>
  <msup>
    <mi>a</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>l</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
</math>

因此我们可以看到所谓的**Backpropagation Algorithm**即是先计算最后一层的δ值，然后依次向前计算各层的δ值。

如果忽略regularization项，即<math><mi>λ</mi><mo>=</mo><mn>0</mn></math>，我们能够发现如下式子:

<math display="block"><mstyle displaystyle="true"><mfrac><mi>∂</mi><mrow><mi>∂</mi><msubsup><mi>Θ</mi><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>,</mo><mi>j</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac></mstyle><mi>J</mi><mo stretchy="false">(</mo><mi>Θ</mi><mo stretchy="false">)</mo><mo>=</mo><msubsup><mi>a</mi><mi>j</mi><mi>(l)</mi></msubsup><msubsup><mi>δ</mi><mi>i</mi><mi>(l+1)</mi></msubsup></math> 

上面是**Layer L=4**的例子，让我们对Backpropagation Algorithm先有了一个直观的感受，接下来从通用的角度给出Backpropagation Algorithm的计算步骤

假设有训练集 <math><mo fence="false" stretchy="false">{</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mrow  class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo>⋯</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo fence="false" stretchy="false">}</mo></math>，令

- 对所有<math><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>l</mi></math> ，令 <math xmlns="http://www.w3.org/1998/Math/MathML"><msubsup><mi mathvariant="normal">Δ</mi><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>,</mo><mi>j</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup></math> = 0，得到一个全零矩阵

- For i=1 to m 做循环，每个循环体执行下面操作

	1. <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>a</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>:=</mo><msup><mi>x</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></msup></math> 让神经网络第一层等于输入的训练数据
	2. 对 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>a</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup></math> 进行**"Forward Propagation"**计算，其中 `l=2,3,…,L` 计算过程如上文图示

	3. 使用 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>y</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></msup></math>来计算 <math><msup><mi>δ</mi><mi>(L)</mi></msup><mo>=</mo><msup><mi>a</mi><mi>(L)</mi></msup><mo>-</mo><msup><mi>y</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></msup></math>

	4. 根据 <math><msup><mi>δ</mi><mi>(L)</mi></msup></math> 向前计算 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>δ</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>L</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>δ</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>L</mi><mo>−</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>…</mo><mo>,</mo><msup><mi>δ</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup></math>，公式为：<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>&#x03B4;<!--δ--></mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mo stretchy="false">(</mo><mo stretchy="false">(</mo><msup><mi mathvariant="normal">&#x0398;<!--Θ--></mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><msup><mo stretchy="false">)</mo><mi>T</mi></msup><msup><mi>&#x03B4;<!--δ--></mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mtext>&#xA0;</mtext><mo>.</mo><mo>&#x2217;<!--∗--></mo><mtext>&#xA0;</mtext><msup><mi>a</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mtext>&#xA0;</mtext><mo>.</mo><mo>&#x2217;<!--∗--></mo><mtext>&#xA0;</mtext><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;<!--−--></mo><msup><mi>a</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></math> 这个过程涉及到了链式法则，在下一节会介绍

	5. <math xmlns="http://www.w3.org/1998/Math/MathML"> <msubsup> <mi mathvariant="normal">&#x0394;<!-- Δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> <mo>,</mo> <mi>j</mi> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo>:=</mo> <msubsup> <mi mathvariant="normal">&#x0394;<!-- Δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mi>i</mi> <mo>,</mo> <mi>j</mi> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo>+</mo> <msubsup> <mi>a</mi> <mi>j</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> <msubsup> <mi>&#x03B4;<!-- δ --></mi> <mi>i</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo>+</mo> <mn>1</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </math> 对每层的θ矩阵偏导不断叠加，进行梯度下降，前面的式子也可以用向量化表示 <math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi mathvariant="normal">&#x0394;<!-- Δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo>:=</mo> <msup> <mi mathvariant="normal">&#x0394;<!-- Δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo>+</mo> <msup> <mi>&#x03B4;<!-- δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo>+</mo> <mn>1</mn> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">(</mo> <msup> <mi>a</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msup> <msup> <mo stretchy="false">)</mo> <mi>T</mi> </msup> </math>


- 加上Regularization项得到最终的θ矩阵
  - 当 j≠0 时，$ D_{i,j}^{(l)} \thinspace := \thinspace \frac{1}{m}(\Delta_{i,j}^{(l)} + \lambda \Theta_{i,j}^{(l)}), \thinspace if \thinspace $
  - 当 j=0 时，$ D_{i,j}^{(l)} \thinspace := \thinspace \frac{1}{m}\Delta_{i,j}^{(l)} $

大写的D矩阵用来表示θ矩阵的计算是不断叠加的，我们最终得到的偏导式子为：

$$
\frac{\partial{J(\Theta)}}{\partial\Theta_{ij}^{(l)}} \thinspace = \thinspace D_{(ij)}^{(l)}
$$

### Backpropagation Intuition

这一小节对上面提到的Backpropagation(后面简称BP算法)做一个简单的数学推到，来搞清楚<math> <msubsup> <mi>δ </mi> <mi>j</mi> <mrow> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> </math>的计算过程。

还是先看Forward Propagation，我们还是拿前面的图距离，假设神经网络如下图

![](/assets/images/2017/09/ml-6-5.png)

他只有一个输出节点，由之前提到的Forward Propagation得到的预测函数为：

<math display="block">
<msub>
  <mi>h</mi>
	<mi mathvariant="normal">Θ</mi>
</msub>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo stretchy="false">)</mo>
<mo>=</mo>
<msubsup>
  <mi>a</mi>
  <mn>1</mn>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>3</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<mo>=</mo>
<mi>g</mi>
<mo stretchy="false">(</mo>
<msubsup>
  <mi mathvariant="normal">Θ</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mn>10</mn>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<msubsup>
  <mi>a</mi>
  <mn>0</mn>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<mo>+</mo>
<msubsup>
  <mi mathvariant="normal">Θ</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mn>11</mn>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<msubsup>
  <mi>a</mi>
  <mn>1</mn>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<mo>+</mo>
<msubsup>
  <mi mathvariant="normal">Θ</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mn>12</mn>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<msubsup>
  <mi>a</mi>
  <mn>2</mn>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<mo>+</mo>
<msubsup>
  <mi mathvariant="normal">Θ</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mn>13</mn>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<msubsup>
  <mi>a</mi>
  <mn>3</mn>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">(</mo>
    <mn>2</mn>
    <mo stretchy="false">)</mo>
  </mrow>
</msubsup>
<mo stretchy="false">)</mo>
</math>

这个函数中：

1. <math> <msub> <mi>h</mi> <mi mathvariant="normal">Θ</mi> </msub> <mo stretchy="false">(</mo> <mi>x</mi> <mo stretchy="false">)</mo> </math> 是以Θ为变量的函数
2. 它的计算过程是从第一层开始向最后一层逐层计算，每一层每个节点的值是由它后一层的节点乘以权重矩阵Θ

BP的计算和推导不如Forward容易理解，也不直观，它的特点类和FB类似

1. δ也是自变量为θ的函数
2. 它的计算过程是从最后一层开始向第一层逐层计算，每层的δ值是由它前面一层的δ值乘以权重矩阵θ
3. 它的计算包含两部分，第一部分是求梯度（对θ求偏导），第二部分是梯度下降

先说第一部分求梯度，由上节给出的代价函数为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable rowspacing="3pt" columnspacing="1em" displaystyle="true" minlabelspacing=".8em">
    <mtr>
      <mtd>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>k</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>K</mi>
        </munderover>
        <mfenced open="[" close="]">
          <mrow>
            <msubsup>
              <mi>y</mi>
              <mi>k</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mo stretchy="false">(</mo>
                <mi>i</mi>
                <mo stretchy="false">)</mo>
              </mrow>
            </msubsup>
            <mtext>&#xA0;</mtext>
            <mi>log</mi>
            <mo>&#x2061;<!-- ⁡ --></mo>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>h</mi>
              <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
            <msub>
              <mo stretchy="false">)</mo>
              <mi>k</mi>
            </msub>
            <mo>+</mo>
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo>&#x2212;<!-- − --></mo>
            <msubsup>
              <mi>y</mi>
              <mi>k</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mo stretchy="false">(</mo>
                <mi>i</mi>
                <mo stretchy="false">)</mo>
              </mrow>
            </msubsup>
            <mo stretchy="false">)</mo>
            <mtext>&#xA0;</mtext>
            <mi>log</mi>
            <mo>&#x2061;<!-- ⁡ --></mo>
            <mo stretchy="false">(</mo>
            <mn>1</mn>
            <mo>&#x2212;<!-- − --></mo>
            <msub>
              <mi>h</mi>
              <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
            <msub>
              <mo stretchy="false">)</mo>
              <mi>k</mi>
            </msub>
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
            <mi>l</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>L</mi>
            <mo>&#x2212;<!-- − --></mo>
            <mn>1</mn>
          </mrow>
        </munderover>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <msub>
              <mi>s</mi>
              <mi>l</mi>
            </msub>
          </mrow>
        </munderover>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <msub>
              <mi>s</mi>
              <mi>l</mi>
            </msub>
            <mo>+</mo>
            <mn>1</mn>
          </mrow>
        </munderover>
        <mo stretchy="false">(</mo>
        <msubsup>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>,</mo>
            <mi>i</mi>
          </mrow>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>l</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msubsup>
        <msup>
          <mo stretchy="false">)</mo>
          <mn>2</mn>
        </msup>
      </mtd>
    </mtr>
  </mtable>
</math>


如果将regularization项忽略，令K=1,对于单一节点<math><msup><mi>x</mi><mi>(i)</mi></msup></math>,<math><msup><mi>y</mi><mi>(i)</mi></msup></math>的代价函数简化为：

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
	<mi>J</mi>
    <mo stretchy="false">(</mo>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
    <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>-</mo>
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
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
  <mo>−</mo>
  <msub>
    <mi>h</mi>
    <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
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
</math>

上面函数近似约等于:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
	<mi>J</mi>
    <mo stretchy="false">(</mo>
    <mi mathvariant="normal">Θ</mi>
    <mo stretchy="false">)</mo>
  <mo>≈</mo>
  <mo>(</mo>
  <msub>
    <mi>h</mi>
    <mi mathvariant="normal">Θ</mi>
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
	  <mo>-</mo>
	<msup>
	<mi>y</mi>
	<mrow class="MJX-TeXAtom-ORD">
	  <mo stretchy="false">(</mo>
	  <mi>i</mi>
	  <mo stretchy="false">)</mo>
	</mrow>
	</msup>
	<msup>
	<mo>)</mo>
	<mn>2</mn>
	</msup>
</math>


其中`l`代表神经网络layer的index，回忆这个函数的含义是计算样本 <math xmlns="http://www.w3.org/1998/Math/MathML"> <mo stretchy="false">(</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo>,</mo> <msup> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> </math> 和预测结果的误差值，接下来的任务就是找到使这个函数的达到最小值的Θ，找到的办法是通过梯度下降的方式，使<math><mi>J(Θ)</mi></math>沿梯度下降最快的方向达到极值（注意：<math><mi>J(Θ)</mi></math>不是convex函数，不一定有最小值，很可能收敛到了极值点），梯度下降需要用到 <math xmlns="http://www.w3.org/1998/Math/MathML"> <mi mathvariant="normal">∂</mi> <mi>J(Θ)</mi><mrow class="MJX-TeXAtom-ORD"> <mo>/</mo> </mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>Θ</mi> <mrow class="MJX-TeXAtom-ORD"> <mi>j</mi> <mi>i</mi> </mrow> <mi>(l)</mi> </msubsup> </math> ，并将计算结果保存到<math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi mathvariant="normal">&#x0394;<!-- Δ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math> 中。

下面我们以3层神经网络为例，分解这个推导过程，神经网络如下图所示

![](/assets/images/2017/09/ml-6-14.png)

再来回顾一下各个变量的含义为：

- 另 <math><msub><mi>x</mi><mn>1</mn></msub></math>和<math><msub><mi>x</mi><mn>2</mn></msub></math> 表示神经网络的输入样本，两个特征
- 另 <math><msubsup><mi>z</mi><mi>j</mi><mi>(l)</mi></msubsup></math>表示第l层的第`j`个节点的输入值
- 另 <math><msubsup><mi>a</mi><mi>j</mi><mi>(l)</mi></msubsup></math>表示第l层的第`j`个节点的输出值
- 另 <math><msubsup><mi>Θ</mi><mrow><mi>j</mi><mi>i</mi></mrow><mi>(l)</mi></msubsup></math>表示第`l`层到第`l+1`层的权重矩阵
- 另 <math> <msubsup> <mi>δ </mi> <mi>j</mi> <mrow> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> </math>表示第`l`层第`j`个节点的预测偏差值，他的数学定义为<math><msubsup> <mi>δ</mi><mi>j</mi><mi>(l)</mi></msubsup> <mo>=</mo> <mfrac> <mi mathvariant="normal">∂</mi> <mrow> <mi mathvariant="normal"> ∂</mi> <msubsup> <mi>z</mi> <mi>j</mi><mi>(l)</mi> </msubsup> </mrow> </mfrac> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal">Θ</mi><mo stretchy="false">)</mo></math>

我们的目的是求解Θ矩阵的值，以得出最终的预测函数，这个例子中以求解<math><msubsup><mi>Θ</mi><mn>11</mn><mi>(3)</mi></msubsup></math>和<math><msubsup><mi>Θ</mi><mn>11</mn><mi>(2)</mi></msubsup></math>为例

1. 参考上面几节，我们令<math><msub> <mi>h</mi> <mi> Θ</mi> </msub> <mo stretchy="false">(</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>t</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> <mo>=</mo> <mi>g</mi> <mo stretchy="false">(</mo> <msup> <mi>z</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>t</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo><mo>=</mo><msup><mi>a</mi><mi>(t)</mi></msup></math>，其中<math><mi>g</mi></math>为sigmoid函数<math><mi>g</mi><mi>(z)</mi><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>

2. 先求 <math><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>1</mi></mrow><mi>(3)</mi></msubsup></math>，由链式规则，可以做如下运算：<math> <mfrac> <mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal">∂ </mi> <msubsup> <mi>Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo>=</mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal"> Θ </mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mi>(4)</mi> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi mathvariant="normal"> Θ </mi> <mrow class="MJX-TeXAtom-ORD"> <mn>11</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>

3. 参考上面δ的定义，可知上面等式后两项为：<math><msubsup><mi>δ</mi><mn>1</mn><mi>(4)</mi></msubsup><mo>=</mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal"> Θ </mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mi>(4)</mi> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>， 即输出层第一个节点的误差值，展开计算如下：

	<math display="block"><msubsup><mi>δ</mi><mn>1</mn><mi>(4)</mi></msubsup><mo>=</mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal"> Θ </mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mi>(4)</mi> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac><mo>=</mo><mo>&#x2212;<!-- − --></mo> <mo stretchy="false">[</mo> <mi>y</mi> <mo>&#x2217;<!-- ∗ --></mo> <mo stretchy="false">(</mo> <mn>1</mn> <mo>&#x2212;<!-- − --></mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo stretchy="false">)</mo> <mo>+</mo> <mo stretchy="false">(</mo> <mi>y</mi> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> <mo stretchy="false">)</mo> <mo>&#x2217;<!-- ∗ --></mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo stretchy="false">]</mo><mo>=</mo> <mo>&#x2212;<!-- − --></mo> <mo stretchy="false">[</mo> <mi>y</mi> <mo>&#x2217;<!-- ∗ --></mo> <mo stretchy="false">(</mo> <mn>1</mn> <mo>&#x2212;<!-- − --></mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo stretchy="false">)</mo> <mo>+</mo> <mo stretchy="false">(</mo> <mi>y</mi> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> <mo stretchy="false">)</mo> <mo>&#x2217;<!-- ∗ --></mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo stretchy="false">]</mo> <mo>=</mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo>&#x2212;<!-- − --></mo> <mi>y</mi> <mo>=</mo> <msup> <mi>a</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msup> <mo>&#x2212;<!-- − --></mo> <mi>y</mi></math>，其中用到了sigmoid函数一个特性：<math xmlns="http://www.w3.org/1998/Math/MathML"> <mi>g</mi> <mi class="MJX-variant" mathvariant="normal">&#x2032;<!-- ′ --></mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo>=</mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo>&#x2217;<!-- ∗ --></mo> <mo stretchy="false">(</mo> <mn>1</mn> <mo>&#x2212;<!-- − --></mo> <mi>g</mi> <mo stretchy="false">(</mo> <mi>z</mi> <mo stretchy="false">)</mo> <mo stretchy="false">)</mo></math>
	
4. 这样我们得到了<math><msubsup><mi>δ</mi><mn>1</mn><mi>(4)</mi></msubsup></math>（参考上一节BP算法的步骤(3)），接下来继续求解<math><mfrac><mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow><mrow> <mi>∂ </mi> <msubsup> <mi mathvariant="normal">Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow></mfrac></math>，前面第二步等号后的最有一项<math><mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi mathvariant="normal"> Θ </mi> <mrow class="MJX-TeXAtom-ORD"> <mn>11</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>，将<math><msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo></mrow></msubsup></math>展开有：<math><msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo></mrow></msubsup><mo>=</mo><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>0</mi></mrow><mi>(3)</mi></msubsup><mo> * </mo><msubsup><mi>a</mi><mn>0</mn><mi>(3)</mi></msubsup><mo>+</mo><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>1</mi></mrow><mi>(3)</mi></msubsup><mo> * </mo><msubsup><mi>a</mi><mn>1</mn><mi>(3)</mi></msubsup><mo>+</mo><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>2</mi></mrow><mi>(3)</mi></msubsup><mo> * </mo><msubsup><mi>a</mi><mn>2</mn><mi>(3)</mi></msubsup></math>，对<math><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>1</mi></mrow><mi>(3)</mi></msubsup></math>求偏导的结果为<math><msubsup><mi>a</mi><mn>1</mn><mi>(3)</mi></msubsup></math>

5. 将第4步与第三步的式子合并，即得出 <math><mfrac><mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow><mrow> <mi>∂ </mi> <msubsup> <mi mathvariant="normal">Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow></mfrac><mo>=</mo><msubsup><mi>δ</mi><mn>1</mn><mi>(4)</mi></msubsup><mo> * </mo><msubsup><mi>a</mi><mn>1</mn><mi>(3)</mi></msubsup></math> 与上一节BP算法步骤(5)一致

6. 接下来计算<math><msubsup><mi>Θ</mi><mrow><mi>1</mi><mi>1</mi></mrow><mi>(2)</mi></msubsup></math>，链式规则可做如下运算<math> <mfrac> <mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal">∂ </mi> <msubsup> <mi>Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo>=</mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal"> Θ </mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mi>(4)</mi> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi >∂</mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac><mo> * </mo><mfrac> <mrow> <mi> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi >∂</mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac><mo> * </mo><mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi mathvariant="normal"> Θ </mi> <mrow class="MJX-TeXAtom-ORD"> <mn>11</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>

7. 参考上面δ的定义，可知<math> <msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo>=</mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi mathvariant="normal"> Θ </mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mi>(4)</mi> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo> ∗ </mo> <mfrac> <mrow> <mi> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi >∂</mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac><mo> * </mo><mfrac> <mrow> <mi> ∂ </mi> <msubsup> <mi>a</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi >∂</mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>，由上面的步骤3可知，等式的前两项为<math><msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup></math>。这里可以看出对δ值的计算和之前的FB算法类似，如果将神经网络反向来看，当前层的<math><msup> <mi>δ </mi> <mi>1</mi></msup></math>值是根据后一层的<math><msup> <mi>δ </mi> <mi>(1-1)</mi></msup></math>计算得来。等式的第三项，将<math><msubsup><mi>z</mi><mn>1</mn><mi>(4)</mi></msubsup></math>展开后对<math><msubsup><mi>a</mi><mn>1</mn><mi>(3)</mi></msubsup></math>求导后得到<math><msubsup><mi>Θ</mi><mn>11</mn><mi>(3)</mi></msubsup></math>，等式最后一项为<math> <mi>g</mi> <mi class="MJX-variant" mathvariant="normal">&#x2032;<!-- ′ --></mi> <mo stretchy="false">(</mo> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo stretchy="false">)</mo></math>

8. 将上一步的结果进行整理得到: <math><msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup><mo>=</mo><msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>4</mn> <mo stretchy="false">)</mo> </mrow> </msubsup><mo> * </mo><msubsup><mi>Θ</mi><mn>11</mn><mi>(3)</mi></msubsup> <mo> * </mo> <mi>g</mi> <mi class="MJX-variant" mathvariant="normal">&#x2032;<!-- ′ --></mi> <mo stretchy="false">(</mo> <msubsup> <mi>z</mi> <mn>1</mn> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> <mo stretchy="false">)</mo></math> 和上一节BP算步骤(4)一致

9. 将8的结果带入第6步，可得出<math> <mfrac> <mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal">∂ </mi> <msubsup> <mi>Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo>=</mo> <msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup>  <mo> * </mo><mfrac> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi>z</mi> <mn>1</mn> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> <mrow> <mi mathvariant="normal"> ∂ </mi> <msubsup> <mi mathvariant="normal"> Θ </mi> <mrow class="MJX-TeXAtom-ORD"> <mn>11</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac></math>，将<math><msubsup><mi>z</mi><mn>1</mn><mi>(3)</mi></msubsup></math>展开后对<math><msubsup> <mi mathvariant="normal"> Θ </mi> <mrow class="MJX-TeXAtom-ORD"> <mn>11</mn> </mrow> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup></math>求导得到<math><msubsup><mi>a</mi><mn>1</mn><mi>(2)</mi></msubsup></math>

10. 整理第9步结果可知 <math> <mfrac> <mrow> <mi>∂</mi> <mi>J</mi> <mo stretchy="false">(</mo> <mi>Θ</mi> <mo stretchy="false">)</mo> </mrow> <mrow> <mi mathvariant="normal">∂ </mi> <msubsup> <mi>Θ</mi> <mrow> <mn>11</mn> </mrow> <mrow> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup> </mrow> </mfrac> <mo>=</mo> <msubsup> <mi>δ </mi> <mi>1</mi> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup>  <mo> * </mo> <msubsup><mi>a</mi><mn>1</mn><mi>(2)</mi></msubsup></math>，与上一节步骤（5）一致

通过上面的推导，大概可以印证上一节的结论：

<math display="block"><mstyle displaystyle="true"><mfrac><mi>∂</mi><mrow><mi>∂</mi><msubsup><mi>Θ</mi><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>,</mo><mi>j</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac></mstyle><mi>J</mi><mo stretchy="false">(</mo><mi>Θ</mi><mo stretchy="false">)</mo><mo>=</mo><msubsup><mi>a</mi><mi>j</mi><mi>(l)</mi></msubsup><msubsup><mi>δ</mi><mi>i</mi><mi>(l+1)</mi></msubsup></math>


而关于对<math> <msubsup> <mi>δ </mi> <mi>j</mi> <mrow> <mo stretchy="false">(</mo> <mi>l</mi> <mo stretchy="false">)</mo> </mrow> </msubsup> </math>的计算则是BP算法的核心，继续上面的例子，计算<math><msubsup> <mi>δ </mi> <mi>2</mi> <mrow> <mo stretchy="false">(</mo> <mn>3</mn> <mo stretchy="false">)</mo> </mrow> </msubsup></math>和<math><msubsup> <mi>δ </mi> <mi>2</mi> <mrow> <mo stretchy="false">(</mo> <mn>2</mn> <mo stretchy="false">)</mo> </mrow> </msubsup></math>：

![](/assets/images/2017/09/ml-6-15.png)


可以观察到 BP 算法两个突出特点：

1. 自输出层向输入层（即反向传播），逐层求偏导，在这个过程中逐渐得到各个层的参数梯度。

2. 在反向传播过程中，使用 δ(l)δ(l) 保存了部分结果，避免了大量的重复运算，因而该算法性能优异。


### Implementation Nodte: Unrolling parameters

这一小节介绍如何使用Advanced optimization来计算神经网络，对于优化函数，前面有讲过，的定义如下:

```matlab
function [jVal, gradient] = costFunction(theta)

...

optTheta = fminunc(@costFunction, initialTheta, options)

```

`fminunc`的第二个参数initialTheta需要传入一个vector，而我们之前推导的神经网络权重矩阵Θ显然不是一维的向量，对于一个四层的神经网络来说：

- Θ矩阵：<math><msup><mi>Θ</mi><mi>(1)</mi></msup></math>，<math><msup><mi>Θ</mi><mi>(2)</mi></msup></math>，<math><msup><mi>Θ</mi><mi>(3)</mi></msup></math> - matrices(Theta1, Theta2, Theta3)
- 梯度矩阵：<math><msup><mi>D</mi><mi>(1)</mi></msup></math>，<math><msup><mi>D</mi><mi>(2)</mi></msup></math>，<math><msup><mi>D</mi><mi>(3)</mi></msup></math> - matrices(D1, D2, D3)

因此我们需要将矩阵转换为向量，在Octave中，可用如下命令

```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]

```
这种写法会将3个Θ矩阵排成一维向量，假设<math><msup><mi>Θ</mi><mi>(1)</mi></msup></math>是10x11的，<math><msup><mi>Θ</mi><mi>(2)</mi></msup></math>是10x11的，<math><msup><mi>Θ</mi><mi>(3)</mi></msup></math>是1x11的，也可以从`thetaVector`取出原始矩阵

```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

总结一下：

- 前面得到的`thetaVector`代入到`fminunc`中，替换`initialTheta`
- 在`costFunction`中，输入的参数是`thetaVec`
	
	```matlab
	function[jVal,gradientVec] = costFunction(thetaVec)
	```
	在`costFunction`中，我们需要使用`reshape`命令从`theVec`取出<math><msup><mi>Θ</mi><mi>(1)</mi></msup></math>，<math><msup><mi>Θ</mi><mi>(2)</mi></msup></math>，<math><msup><mi>Θ</mi><mi>(3)</mi></msup></math> 用来计算FB和BP算法，得到<math><msup><mi>D</mi><mi>(1)</mi></msup></math>，<math><msup><mi>D</mi><mi>(2)</mi></msup></math>，<math><msup><mi>D</mi><mi>(3)</mi></msup></math> 梯度矩阵和<math><mi>J</mi><mi>(Θ)</mi></math>，然后再unroll <math><msup><mi>D</mi><mi>(1)</mi></msup></math>，<math><msup><mi>D</mi><mi>(2)</mi></msup></math>，<math><msup><mi>D</mi><mi>(3)</mi></msup></math>得到`gradientVec`


### Gradient Checking

在计算神经网络的梯度时，要确保梯度计算正确，最好在计算过程中进行Gradient Checking。对于代价函数在某个点导数可近似为:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mstyle displaystyle="true">
    <mfrac>
      <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
      <mrow>
        <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
        <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
      </mrow>
    </mfrac>
  </mstyle>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2248;<!-- ≈ --></mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mrow>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
        <mo>+</mo>
        <mi>&#x03F5;<!-- ϵ --></mi>
        <mo stretchy="false">)</mo>
        <mo>&#x2212;<!-- − --></mo>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03F5;<!-- ϵ --></mi>
        <mo stretchy="false">)</mo>
      </mrow>
      <mrow>
        <mn>2</mn>
        <mi>&#x03F5;<!-- ϵ --></mi>
      </mrow>
    </mfrac>
  </mstyle>
</math>


上面式子是单个Θ矩阵的梯度近似，对于过个Θ矩阵的梯度近似，计算方法相同:

<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mstyle displaystyle="true">
    <mfrac>
      <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
      <mrow>
        <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mi>j</mi>
        </msub>
      </mrow>
    </mfrac>
  </mstyle>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2248;<!-- ≈ --></mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mrow>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mn>1</mn>
        </msub>
        <mo>,</mo>
        <mo>&#x2026;<!-- … --></mo>
        <mo>,</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>+</mo>
        <mi>&#x03F5;<!-- ϵ --></mi>
        <mo>,</mo>
        <mo>&#x2026;<!-- … --></mo>
        <mo>,</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mi>n</mi>
        </msub>
        <mo stretchy="false">)</mo>
        <mo>&#x2212;<!-- − --></mo>
        <mi>J</mi>
        <mo stretchy="false">(</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mn>1</mn>
        </msub>
        <mo>,</mo>
        <mo>&#x2026;<!-- … --></mo>
        <mo>,</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mi>j</mi>
        </msub>
        <mo>&#x2212;<!-- − --></mo>
        <mi>&#x03F5;<!-- ϵ --></mi>
        <mo>,</mo>
        <mo>&#x2026;<!-- … --></mo>
        <mo>,</mo>
        <msub>
          <mi mathvariant="normal">&#x0398;<!-- Θ --></mi>
          <mi>n</mi>
        </msub>
        <mo stretchy="false">)</mo>
      </mrow>
      <mrow>
        <mn>2</mn>
        <mi>&#x03F5;<!-- ϵ --></mi>
      </mrow>
    </mfrac>
  </mstyle>
</math>

为了保证计算结果相近，其中<math><mi>ϵ</mi><mo>=</mo><msup><mn>10</mn><mi>(-4)</mi></msup></math>，注意过小的ϵ会导致计算问题，由于我们只能对一个Θ矩阵进行ϵ的加减，对于多个Θ矩阵，在Octave中需要使用循环计算

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

得到近似的梯度之后，我们可以将计算得到的`Appprox`和上一节的`deltaVector`进行比较，查看是否`gradApprox ≈ deltaVector`，由于计算`Approx`代价很大，速度很慢，一般在确认了BP算法正确后，就不在计算`Appox`了

结合前一节做一个简单的总结:

1. 通过实现BP算法得到δ矩阵`DVec`（Unrolled <math><msup><mi>D</mi><mi>(1)</mi></msup></math>，<math><msup><mi>D</mi><mi>(2)</mi></msup></math>，<math><msup><mi>D</mi><mi>(3)</mi></msup></math>）
2. 进行梯度检查，计算`gradApprox`
3. 确保计算结果足够相近
4. 停止梯度检查，使用BP得到的结果
5. 确保在用神经网络训练数据的时候梯度检查是关闭的，否则会非常耗时

###Random Initialization

计算神经网络，将theta初始值设为0不合适，这会导致在计算BP算法的过程中所有节点计算出的值相同。我们可以使用随机的方式产生Θ矩阵，比如将<math><msubsup><mi>Θ</mi><mi>ij</mi><mi>(l)</mi></msubsup></math>初始化范围控制在[-ϵ,ϵ]：

```matlab
Theta1=rand(10,11)*(2*INIT_EPSILON)-INIT_EPSILON #初始化10x11的矩阵
Theta1=rand(1,11)*(2*INIT_EPSILON)-INIT_EPSILON #初始化1x11的矩阵
```

rand(x,y)函数会为矩阵初始化一个0到1之间的实数，上面的INIT_EPSILON和上一节提到的ϵ不是一个ϵ。

### 小结

这一章先介绍了如何构建一个神经网络，包含如下几个步骤

- 第一层输入单元的个数 = 样本<math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math>的维度
- 最后一层输出单元的个数 = 预测结果分类的个数
- Hidden Layer的个数= 默认为1个，如果有多余1个的hidden layer，通常每层的unit个数相同，理论上层数越多越好

接下来介绍了如何训练一个神经网络，包含如下几步

1. 随机初始化Θ矩阵
2. 实现FP算法，对任意<math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math>得出预测函数<math xmlns="http://www.w3.org/1998/Math/MathML"> <msub> <mi>h</mi> <mi mathvariant="normal">&#x0398;<!-- Θ --></mi> </msub> <mo stretchy="false">(</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> </math>
3. 实现代价函数
4. 使用BP算法对代价函数求偏导，得到<math><mstyle displaystyle="true"><mfrac><mi>∂</mi><mrow><mi>∂</mi><msubsup><mi>Θ</mi><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>,</mo><mi>j</mi></mrow><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac></mstyle><mi>J</mi><mo stretchy="false">(</mo><mi>Θ</mi><mo stretchy="false">)</mo></math>的算式
5. 使用梯度检查，确保BP算出的Θ矩阵结果正确，然后停止梯度检查
6. 使用梯度下降或者其它高级优化算法求解权重矩阵Θ，使代价函数的值最小

不论是求解FP还是BP算法，都要loop每一个训练样本

```matlab
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

BP梯度下降的过程如下图所示：

![](/assets/images/2017/09/ml-6-16.png)

再回忆一下梯度下降，函数在极值点处的导数
