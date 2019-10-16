---
layout: post
list_title: 机器学习 | Machine Learning | 线性回归 | Linear Regression 
title: 线性回归   
meta: Coursera Stanford Machine Learning Cousre Note, Chapter2
categories: [Machine Learning,AI]
mathjax: true
---

> 文中所用到的图片部分截取自Andrew Ng在[Cousera上的课程](https://www.coursera.org/learn/machine-learning)

### Cost Function

* Hypothesis 函数：

$$
h_{\theta}(x)=\theta_0 + {\theta_1}x
$$

怎么计算参数 θ 呢？需要通过代价函数求解

* cost 函数：

$$
J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^m(\hat{y}_{i} - y_i)^2=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x_i)-y_i)^2
$$

这个式子的含义是找到<math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub></math>的值使 <math><mi>J</mi><mo stretchy="false">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo stretchy="false">)</mo></math>的值最小，为了求导方便系数乘了 1/2

### Cost Function - Intuition(1)

对于 Hypothesis 函数：

$$
h_{\theta}(x)=\theta_0 + \theta_1x
$$

当$\theta_0=0$时，简化为：

$$
h_{\theta}(x) = \theta_1x
$$

对于 cost 函数简化为：

$$
J(\theta_1) = \frac{1}{2m}\sum_{i=1}^m(\hat{y}_{i} - y_i)^2=\frac{1}{2m}\sum_{i=1}^m(\theta_1(x_i)-y_i)^2
$$

假设有一组训练集：`(1,1),(2,2),(3,3)`

- 当 $\theta_1=1$ 时，$h_\theta(x) = x$, 有 $J(1) = \frac{1}{2m}(0^2+0^2+0^2)=0$
- 当 $\theta_1=0.5$ 时，$h_\theta(x) = 0.5x$，有 $J(0.5) = \frac{1}{2m}((0.5-1^2+(1-2)^2+(1.5-3)^2)=0.58$
- 当 $\theta_1=0$ 时，$h_\theta(x) = 0$, 有 $J(0) = \frac{1}{2m}(1^2+2^2+3^2)=2.3$

以此类推，通过不同的`θ`值可以求出不同的`J(θ)`，如下图所示：

![](/assets/images/2017/09/ml-2.png)

我们的目标是找到一个`θ`值使`J(θ)`最小。显然上述案例中，当`θ=1`时，`J(θ)`最小，因此我们可以得到 Hypothesis 函数：

$$
h_{\theta}(x) = x
$$

### Cost Function - Intuition(2)

使用 contour plots 观察<math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mi>,</mi><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>在二维平面的投影，

> 关于 contour plot[参考](https://nb.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/visualizing-scalar-valued-functions/v/contour-plots）

![](/assets/images/2017/09/ml-3.png)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(θ0,θ1) and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when θ0 = 800 and θ1= -0.15

Taking another h(x) and plotting its contour plot, one gets the following graphs:

![](/assets/images/2017/09/ml-3-1.png)

When θ0 = 360 and θ1 = 0, the value of J(θ0,θ1) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![](/assets/images/2017/09/ml-3-2.png)

The graph above minimizes the cost function as much as possible and consequently, the result of θ1 and θ0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

* Ocatave Demo

```matlab
function J = computeCost(X, y, theta)

m = length(y); % number of training examples

predictions = X*theta;

sqrErrors = ( predictions - y ).^2;

J = 1/(2*m)*sum(sqrErrors);

end
```

### Gradient descent

对 Cost 函数：$J(\theta_0, \theta_1)$，找到$\theta_0, \theta_1$ 使 $J(\theta_0, \theta_1)$值最小

* 方法 1. 选择任意<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>，例如：<math><msub><mi>θ</mi><mi>0</mi></msub><mo>=</mo><mn>1</mn><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>=</mo><mn>1</mn></math> 

* 方法 2. 不断改变<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>使<math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>按梯度方向进行减少，直到找到最小值

* 图形理解

![Altext](/assets/images/2017/09/ml-4.png)

* 梯度下降法： 
  - `:=` 代表赋值，例如 a:=b 代表把 b 的值赋值给 a，类似的比如 a:=a+1。因此 := 表示的是计算机范畴中的赋值。而=号则代表 truth assertion，a = b 的含义是 a 的值为 b 
  - `α` 代表 learning rate 是梯度下降的步长 - <math> <mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo stretchy="false">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo stretchy="false">)</mo></math>代表对`θ`求偏导

<math display="block">
  <msub>
    <mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>:=</mo>
  <msub>
	<mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>-</mo>
  <mi>α</mi>
  <mfrac>
    <mi mathvariant="normal">∂</mi>
    <mrow>
      <mi mathvariant="normal">∂</mi>
      <msub>
        <mi>θ</mi>
        <mi>j</mi>
      </msub>
    </mrow>
  </mfrac>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>θ</mi>
    <mn>0</mn>
  </msub>
  <mo>,</mo>
  <msub>
    <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math>

* 理解梯度下降

梯度下降是求多维函数的极值方法，因此公式是对 <math><msub><mi>θ</mi><mi>j</mi></msub></math> 求导，每一个<math><msub><mi>θ</mi><mi>j</mi></msub></math>代表一元参数，也可以理解为一维向量，上述 case 中，只有<math><msub><mi>θ</mi><mn>0</mn></msub></math>和<math><msub><mi>θ</mi><mn>1</mn></msub></math>两个参数，可以理解在这两个方向上各自下降，他们的向量方向为<math><msup><mi>J</mi><mi>(θ)</mi></msup></math>下降的方向，下降过程是一个同步迭代的过程：

![](/assets/images/2017/09/ml-4.png)

理解二维梯度下降之前，可以先假设<math><msup><mi>J</mi><mi>(θ)</mi></msup></math>是一维的，即只有一个参数，那么上述梯度下降公式简化为：

<math display="block">
  <msub>
    <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <mo>:=</mo>
  <msub>
	 <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <mo>−</mo>
  <mi>α</mi>
  <mfrac>
    <mi>d</mi>
    <mrow>
      <mi>d</mi>
      <msub>
		 <mi>θ</mi>
        <mn>1</mn>
      </msub>
    </mrow>
  </mfrac>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math>

问题简化为对一元函数求导，假设<math><msup><mi>J</mi><mi>(θ)</mi></msup></math>如下图所示：
![](/assets/images/2017/09/ml-3-2.png)
`θ`会逐渐向极值点出收敛，当`θ`到达极值点时，该处导数为 0，则`θ`值不再变化。
<math display = "block">
<msub>
<mi>θ</mi>
<mn>1</mn>
</msub>
<mo>:=</mo>
<msub>
<mi>θ</mi>
<mn>1</mn>
</msub>
<mo>−</mo>
<mi>α</mi>
<mo>∗</mo>
<mn>0</mn>
</math>
理解了一维的梯度下降，接下来看怎么把它应用到<math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>上，对<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>分别求偏导，得到下面公式：

<math display="block">
  <mtable>
    <mtr>
      <mtd>
        <mtext>repeat until convergence:{</mtext>
        <mo fence="false" stretchy="false">{</mo>
      </mtd>
      <mtd />
    </mtr>
    <mtr>
      <mtd>
        <msub>
          <mi>θ</mi>
          <mn>0</mn>
        </msub>
        <mo>:=</mo>
      </mtd>
      <mtd>
        <msub>
			<mi>θ</mi>
          <mn>0</mn>
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
        <msub>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
          </mrow>
        </msub>
        <mo stretchy="false">)</mo>
        <mo>−</mo>
        <msub>
          <mi>y</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
          </mrow>
        </msub>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub>
			<mi>θ</mi>
          <mn>1</mn>
        </msub>
        <mo>:=</mo>
      </mtd>
      <mtd>
        <msub>
			<mi>θ</mi>
          <mn>1</mn>
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
        <mfenced open="(" close=")">
          <mrow>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>h</mi>
				<mi>θ</mi>
            </msub>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <mo>−</mo>
            <msub>
              <mi>y</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
          </mrow>
        </mfenced>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mo fence="false" stretchy="false">}</mo>
      </mtd>
      <mtd />
    </mtr>
  </mtable>
</math>
	
对于线性回归，<math><msup><mi>J</mi><mi>(θ)</mi></msup></math>是凸函数(convex function)，因此上述两个式子没有局部极值点，只有全局唯一的一个极值点。梯度下降法通常在离极值点远的地方下降很快，但在极值点附近时会收敛速度很慢。并且，在目标函数是凸函数时，梯度下降法的解是全局最优解。而在一般情况下，梯度下降法不保证求得全局最优解。

## Multiple features

上几节讨论的问题是：已知一个房子大小和价格样本数据集，来推导房价和房屋大小的关系函数：

| Size(x) | Price(y) |
| ------- | -------- |
| 2104    | 460      |
| 1035    | 224      |
| 868     | 230      |
| 642     | 126      |
| ...     | ...      |

`x`为房子的 size，`y`是房价，上述的一维线性回归函数：

$$
h_\theta(x) = \theta_0+ \theta_1x
$$

但是影响房价的因素很多，比如房屋数量，楼层数等等：

| Size(x1) | number of bed room (x2) | number of floors(x3) | Price(y) |
| -------- | ----------------------- | -------------------- | -------- |
| 2104     | 5                       | 2                    | 460      |
| 1035     | 4                       | 1                    | 224      |
| 868      | 3                       | 2                    | 230      |
| 642      | 2                       | 1                    | 126      |
| ...      | ...                     | ...                  | ...      |

对应到公式里，则表现为$x$是多维时，公式如下：

$$
h_\theta(x) = \sum_{j=0}^n\theta_jx_j = \theta_0 + \theta_1x_1 + + \theta_2x_2 + ... + + \theta_nx_n
$$

其中：

- $x_j^{(i)}$ 表示第j个feature的第i个样本
- $x^{(i)}$ 表示第i组训练样本
- $m$ 表示样本数
- $n$ 表示feature个数

举例来说，$x^{(2)}$表示第二组训练集：

<math display="block">
<msup><mi>x</mi><mi>(2)</mi></msup>
<mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr><mtd><mi>1035</mi></mtd></mtr>
	<mtr><mtd><mi>4</mi></mtd></mtr>
	<mtr><mtd><mi>1</mi></mtd></mtr>
	<mtr><mtd><mi>224</mi></mtd></mtr>
</mtable>
</mfenced>
</math>

$$
x^{(2)} = 
\begin{bmatrix}
1035  \\
4  \\
1  \\
224 \\
\end{bmatrix}
$$

而$x_3^{(2)}$表示上面向量中中第三个元素：

$$
x_3^{(2)} = 1
$$

还是举个买房子的例子，假如我们得到如下函数:

$$
h_\theta(x) = 80 + 0.1x_1 + 0.01x_2 + 3x_3 - 2x_4
$$

其中$h_\theta(x)$表示房子的总价，$\theta_0=80$ 代表房子的基础价格，<math><msub><mi>x</mi><mn>1</mn></msub></math>代表这栋房子的 size，<math><msub><mi>θ</mi><mn>1</mn></msub></math>是用 cost function 计算出来对<math><msub><mi>x</mi><mn>1</mn></msub></math>的系数，类似的<math><msub><mi>x</mi><mn>2</mn></msub></math>代表房子的房间数，<math><msub><mi>θ</mi><mn>2</mn></msub></math>是对<math><msub><mi>x</mi><mn>2</mn></msub></math>的系数，等等

在这个式子中$x_0$ 默认为1，即$x_0^{(i)}=1$，可以把每条样本和对应的参数看成两条vector:

<math display="block">
<mi>x</mi><mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr><mtd><msub><mi>x</mi><mn>0</mn></msub></mtd></mtr>
	<mtr><mtd><msub><mi>x</mi><mn>1</mn></msub></mtd></mtr>
	<mtr><mtd><mi>...</mi></mtd></mtr>
	<mtr><mtd><msub><mi>x</mi><mi>n</mi></msub></mtd></mtr>
</mtable>
</mfenced>
<mspace width="2em"></mspace>
<mi>θ</mi><mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr><mtd><msub><mi>θ</mi><mn>0</mn></msub></mtd></mtr>
	<mtr><mtd><msub><mi>θ</mi><mn>1</mn></msub></mtd></mtr>
	<mtr><mtd><mi>...</mi></mtd></mtr>
	<mtr><mtd><msub><mi>θ</mi><mi>n</mi></msub></mtd></mtr>
</mtable>
</mfenced>
</math>

对于`x`和`θ`来说，都是`(n+1)x1`的矩阵，而<math><msup><mi>θ</mi><mi>T</mi></msup></math>为`1x(n+1)`，则上述式子也可以表示为：

<math display="block">
<mtable>
    <mtr>
      <mtd>
        <msub>
          <mi>h</mi>
          <mi>θ</mi>
        </msub>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mo>=</mo>
        <mfenced open="[" close="]">
          <mtable rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <msub>
                  <mi>θ</mi>
                  <mn>0</mn>
                </msub>
                <mspace width="2em" />
                <msub>
          		  <mi>θ</mi>
                  <mn>1</mn>
                </msub>
                <mspace width="2em" />
                <mo>.</mo>
                <mo>.</mo>
                <mo>.</mo>
                <mspace width="2em" />
                <msub>
          		 <mi>θ</mi>
                  <mi>n</mi>
                </msub>
              </mtd>
            </mtr>
          </mtable>
        </mfenced>
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
                <mo>⋮</mo>
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
        <mo>=</mo>
        <msup>
	       <mi>θ</mi>
          <mi>T</mi>
        </msup>
        <mi>x</mi>
      </mtd>
    </mtr>
  </mtable>
</math>

> 上述式子也叫做`Multivariate linear regression`

### Gradient Descent for Multiple variables

参考一维线性回归的的 cost 函数，多维线性回归的 cost 函数为:

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
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <mo>−</mo>
            <msub>
              <mi>y</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
          </mrow>
        </mfenced>
        <mn>2</mn>
      </msup>
    </mstyle>
</math>

多维梯度下降公式和前面类似：

<math display="block">
  <msub>
    <mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>:=</mo>
  <msub>
	<mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>-</mo>
  <mi>α</mi>
  <mfrac>
    <mi mathvariant="normal">∂</mi>
    <mrow>
      <mi mathvariant="normal">∂</mi>
      <msub>
        <mi>θ</mi>
        <mi>j</mi>
      </msub>
    </mrow>
  </mfrac>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>θ</mi>
  <mo stretchy="false">)</mo>
</math>

对<math><msub><mi>θ</mi><mi>j</mi></msub></math>求偏导，得到：
<math display="block">
<mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em 2em 0.278em" displaystyle="true" minlabelspacing=".8em">
<mtr>
<mtd />
<mtd>
<mtext>repeat until convergence:</mtext>
<mspace width="thickmathspace" />
<mo fence="false" stretchy="false">{</mo>
</mtd>
</mtr>
<mtr>
<mtd>
<mspace width="thickmathspace" />
</mtd>
<mtd>
<msub>
<mi>θ</mi>
<mn>0</mn>
</msub>
<mo>:=</mo>
<msub>
<mi>θ</mi>
<mn>0</mn>
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
<mtd>
<mspace width="thickmathspace" />
</mtd>
<mtd>
<msub>
<mi>θ</mi>
<mn>1</mn>
</msub>
<mo>:=</mo>
<msub>
<mi>θ</mi>
<mn>1</mn>
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
<mn>1</mn>
<mrow class="MJX-TeXAtom-ORD">
<mo stretchy="false">(</mo>
<mi>i</mi>
<mo stretchy="false">)</mo>
</mrow>
</msubsup>
</mtd>
</mtr>
<mtr>
<mtd>
<mspace width="thickmathspace" />
</mtd>
<mtd>
<msub>
<mi>θ</mi>
<mn>2</mn>
</msub>
<mo>:=</mo>
<msub>
<mi>θ</mi>
<mn>2</mn>
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
<mn>2</mn>
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
<mo>⋯</mo>
</mtd>
</mtr>
<mtr>
<mtd>
<mspace width="thickmathspace" />
</mtd>
<mtd>
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
</mtd>
</mtr>
<mtr>
<mtd>
<mo fence="false" stretchy="false">}</mo>
</mtd>
</mtr>
</mtable>
</math>

* 线性回归梯度计算的 Ocatave Demo

```matlab
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    num_features = size(X,2);
    h = X*theta;

    for j = 1:num_features
        x = X(:,j);
        theta(j) = theta(j) - alpha*(1/m)*sum((h-y).* x);
    end

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
```

## Feature Scaling

Idea: Make sure features are on a similar scale.

E.g.:`x1 = size(0-200 feet)`,`x2=number of bedrooms(1-5)`

这种情况 contour 图是一个瘦长的椭圆，如图：

在不优化的情况下，这类梯度下降速度很慢。如果我们将`x1,x2`做如下调整：

`x1 = size(0-200 feet)/5`,`x2=(number of bedrooms)/5`,则 contour 图会变为接近圆形，梯度下降收敛的速度会加快。通常为了加速收敛，会将每个 feature 值(每个`xi`)统一到某个区间里，比如 <math><mn>0</mn><mo>≤</mo><msub><mi>x</mi><mi>1</mi></msub><mo>≤</mo><mn>3</mn></math>，<math><mn>-2</mn><mo>≤</mo><msub><mi>x</mi><mi>2</mi></msub><mo>≤</mo><mn>0.5</mn></math>等等

## Mean normalization

Replace <math><msub><mi>x</mi><mi>i</mi></msub></math>with <math><msub><mi>x</mi><mi>i</mi></msub><mo>-</mo><msub><mi>μ</mi><mi>i</mi></msub></math> to make features have approximately zero mean.实际上就是将 feature 归一化，

例如`x1=(size-1000)/2000 x2=(#bedrooms-2)/5`

则有：<math><mn>-0.5</mn><mo>≤</mo><msub><mi>x</mi><mi>1</mi></msub><mo>≤</mo><mn>0.5</mn></math>，<math><mn>-0.5</mn><mo>≤</mo><msub><mi>x</mi><mi>2</mi></msub><mo>≤</mo><mn>0.5</mn></math>

* <math><msub><mi>μ</mi><mi>i</mi></msub></math> 是所有 <math><msub><mi>x</mi><mi>i</mi></msub></math>

* <math><msub><mi>μ</mi><mi>i</mi></msub></math> 是`xi`的区间范围，(max-min)

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if xi represents housing prices with a range of 100 to 2000 and a mean value of 1000, then

<math display="block">
  <msub>
    <mi>x</mi>
    <mi>i</mi>
  </msub>
  <mo>:=</mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mrow>
        <mi>p</mi>
        <mi>r</mi>
        <mi>i</mi>
        <mi>c</mi>
        <mi>e</mi>
        <mo>−</mo>
        <mn>1000</mn>
      </mrow>
      <mn>1900</mn>
    </mfrac>
  </mstyle>
</math>

* `μ`表示所有 feature 的平均值
* `s = max - min`

## Learning Rate

<math display="block">
  <msub>
    <mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>:=</mo>
  <msub>
	<mi>θ</mi>
    <mi>j</mi>
  </msub>
  <mo>-</mo>
  <mi>α</mi>
  <mfrac>
    <mi mathvariant="normal">∂</mi>
    <mrow>
      <mi mathvariant="normal">∂</mi>
      <msub>
        <mi>θ</mi>
        <mi>j</mi>
      </msub>
    </mrow>
  </mfrac>
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <mi>θ</mi>
  <mo stretchy="false">)</mo>
</math>

这节讨论如何选取`α`。

为了求解`J(θ)`的最小值，梯度下降会不断的迭代找出最小值，理论上来说随着迭代次数的增加，`J(θ)`将逐渐减小，如图：

![Altext](/assets/images/2017/09/ml-4-3.png)

但是如果`α`选取过大，则可能会导致越过极值点的情况，导致随着迭代次数的增加，`J(θ)`的值增加或忽高忽低不稳定的情况:

![Altext](/assets/images/2017/09/ml-4-4.png)

解决办法都是选取较小的`α`值

* Summary: - if `α` is too small: slow convergence - if `α` is too large: `J(θ)`may not decrease on every iteration; may not converge - To choose `α` , try: ..., 0.001, 0.003, 0.01,0.03, 0.1,0.3, 1, ...

## Polynomial regression

对于线性回归函数:

<math display="block">
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
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
  <msub>
    <mi>x</mi>
    <mn>3</mn>
  </msub>
  <mo>+</mo>
  <mo>⋯</mo>
  <mo>+</mo>
  <msub>
    <mi>θ</mi>
    <mi>n</mi>
  </msub>
  <msub>
    <mi>x</mi>
    <mi>n</mi>
  </msub>
</math>

其中<math><msub><mi>x</mi><mi>i</mi></msub></math>代表 feature 种类，有些情况下使用这些 feature 制作目标函数不方便，因此可以考虑重新定义 feature 的值

We can improve our features and the form of our hypothesis function in a couple different ways.
We can **combine** multiple features into one. For example, we can combine x1 and x2 into a new feature x3 by taking x1⋅x2.

例如我们可以将两个 feature 合成一个:`x3 = x1*x2`，使用`x3`作为先行回归的 feature 值。

另外，如果只有一个 feature，而使用线性函数又不适合描述完整的数据集，可以考虑多项式函数，比如使用二次函数或者三次函数：

<math display="block">
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
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
<msubsup>
<mi>x</mi>
<mn>1</mn>
<mn>2</mn>
</msubsup>
<mspace width="1em"></mspace>
<mi>or</mi>
<mspace width="1em"></mspace>
<msub>
<mi>h</mi>
<mi>θ</mi>
</msub>
<mo stretchy="false">(</mo>
<mi>x</mi>
<mo stretchy="false">)</mo>
<mo>=</mo>
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
<msubsup>
<mi>x</mi>
<mn>1</mn>
<mn>2</mn>
</msubsup>
<mo>+</mo>
<msub>
<mi>θ</mi>
<mn>3</mn>
</msub>
<msubsup>
<mi>x</mi>
<mn>1</mn>
<mn>3</mn>
</msubsup>
</math>

可以令 <math><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>,</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup></math> 但是这么选择的一个问题在于 feature scaling 会比较重要，如果 x1 的 range 是[1,1000]，那么 x2 的 range 就会变成[1,1000000]等

## Normal Equation

对于 cost 函数：

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
            <msub>
              <mi>x</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
            <mo stretchy="false">)</mo>
            <mo>−</mo>
            <msub>
              <mi>y</mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi>i</mi>
              </mrow>
            </msub>
          </mrow>
        </mfenced>
        <mn>2</mn>
      </msup>
  </mstyle>
</math>

前面提到的求`J(θ)`最小值的思路是使用梯度下降法，对<math><msub><mi>θ</mi><mi>j</mi></msub></math>求偏导得到各个 θ 值:

<math display="block">
<mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
<mo>=</mo>
<mn>0</mn>
<mspace width="1em"></mspace>
<mi>(for every j)</mi>
</math>

出了梯度下降法之外，还有一种方法叫做**Normal Equation**，这种方式不需要迭代，可以直接计算出 θ 值 。

假设我们有 m 个样本。特征向量的维度为 n。因此，可知样本为 <math><mo>{</mo><mo>(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo>)</mo><mo>,</mo><mo>...</mo><mo>(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo>)</mo><mo>}</mo></math>，其中对于每一个样本中的<math><msup><mi>x</mi><mi>(i)</mi></msup></math>，都有 <math><msup><mi>x</mi><mi>(i)</mi></msup><mo>=</mo><mo>{</mo><msubsup><mi>x</mi><mi>1</mi><mi>(i)</mi></msubsup><msubsup><mi>x</mi><mi>2</mi><mi>(i)</mi></msubsup><mo>,</mo><mo>...</mo><msubsup><mi>x</mi><mi>n</mi><mi>(i)</mi></msubsup><mo>}</mo></math>，令线性回归函数 <math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>3</mn></msub><msub><mi>x</mi><mn>3</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>θ</mi><mi>n</mi></msub><msub><mi>x</mi><mi>n</mi></msub></math>，则有：

<math display="block">
<mi>X</mi>
<mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr>
		<mtd><mn>1</mn></mtd>
		<mtd><msubsup><mi>x</mi><mi>1</mi><mi>(1)</mi></msubsup></mtd>
		<mtd><msubsup><mi>x</mi><mi>2</mi><mi>(1)</mi></msubsup></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><msubsup><mi>x</mi><mi>n</mi><mi>(1)</mi></msubsup></mtd>
	</mtr>
	<mtr>
		<mtd><mn>1</mn></mtd>
		<mtd><msubsup><mi>x</mi><mi>1</mi><mi>(2)</mi></msubsup></mtd>
		<mtd><msubsup><mi>x</mi><mi>2</mi><mi>(2)</mi></msubsup></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><msubsup><mi>x</mi><mi>n</mi><mi>(2)</mi></msubsup></mtd>
	</mtr>
	<mtr>
		<mtd><mn>1</mn></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><mo>...</mo></mtd>
	</mtr>
	<mtr>
		<mtd><mn>1</mn></mtd>
		<mtd><msubsup><mi>x</mi><mi>1</mi><mi>(m)</mi></msubsup></mtd>
		<mtd><msubsup><mi>x</mi><mi>2</mi><mi>(m)</mi></msubsup></mtd>
		<mtd><mo>...</mo></mtd>
		<mtd><msubsup><mi>x</mi><mi>n</mi><mi>(m)</mi></msubsup></mtd>
	</mtr>
</mtable>
</mfenced>
<mo>,</mo>
<mspace width="1em"></mspace>
<mi>θ</mi>
<mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr>
		<msub><mi>θ</mi><mi>1</mi></msub>
	</mtr>
	<mtr>
		<msub><mi>θ</mi><mi>2</mi></msub>
	</mtr>
	<mtr>
		<mtd><mo>...</mo></mtd>
	</mtr>
	<mtr>
		<msub><mi>θ</mi><mi>n</mi></msub>
	</mtr>
</mtable>
</mfenced>
<mo>,</mo>
<mspace width="1em"></mspace>
<mi>Y</mi>
<mo>=</mo>
<mfenced open="[" close="]">
<mtable>
	<mtr>
		<msub><mi>y</mi><mi>1</mi></msub>
	</mtr>
	<mtr>
		<msub><mi>y</mi><mi>2</mi></msub>
	</mtr>
	<mtr>
		<mtd><mo>...</mo></mtd>
	</mtr>
	<mtr>
		<msub><mi>y</mi><mi>m</mi></msub>
	</mtr>
</mtable>
</mfenced>
</math>

其中：

* <math><mi>X</mi></math> 是 <math><mi>m</mi><mo>\*</mo><mi>(n+1)</mi></math>的矩阵
* <math><mi>θ</mi></math> 是 <math><mi>(n+1)</mi><mo>\*</mo><mn>1</mn></math>的矩阵
* <math><mi>Y</mi></math> 是 <math><mi>m</mi><mo>\*</mo><mn>1</mn></math>的矩阵

看个例子：

![](/assets/images/2017/09/ml-4-5.png)

若希望<math><msub><mi>h</mi><mi>(θ)</mi></msub><mo>=</mo><mi>y</mi></math>，则有<math><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mi>Y</mi></math>，回想**单位矩阵** 和 **矩阵的逆**的性质：

* 单位矩阵 E，<math><mi>AE</mi><mo>=</mo><mi>EA</mi><mo>=</mo><mi>A</mi></math>
* 矩阵的逆<math><msup><mi>A</mi><mi>-1</mi></msup></math>，A 必须为方阵，<math><mi>A</mi><msup><mi>A</mi><mi>-1</mi></msup><mo>=</mo><msup><mi>A</mi><mi>-1</mi></msup><mi>A</mi><mo>=</mo><mi>E</mi></math>

再来看看式子 <math><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mi>Y</mi></math> 若想求出 θ，那么我们需要做一些转换：

1. 先把 θ 左边的矩阵变成一个方阵。通过乘以<math><msup><mi>X</mi><mi>T</mi></msup></math>可以实现，则有 <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>

2. 把 θ 左边的部分变成一个单位矩阵，这样左边就只剩下 θ，<math><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>

3. 由于<math><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>=</mo><mi>E</mi></math>，因此式子变为<math><mi>θ</mi><mo>=</mo><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>，这就**Normal Equation**的表达式。

如果用 Octave 表示，命令为：`pinv(X'*X)*X'*Y`

什么 case 适合使用 Normal Equation，什么 case 适合使用 Gradient Descent？

| Gradient Descent                                                                       | Normal Equation                                                                                                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Need to choose alpha                                                                   | No need to choose alpha                                                                                                                                             |
| Needs many iterations                                                                  | No need to iterate                                                                                                                                                  |
| <math><mi>O</mi><mo>(</mo><mi>k</mi><msup><mi>n</mi><mn>2</mn></msup><mo>)</mo></math> | <math><mi>O</mi><mo> (</mo><msup><mi>n</mi><mn>3</mn></msup><mo>)</mo></math> need to calculate inverse of <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> |
| Works well when n is large                                                             | Slow if n is very large                                                                                                                                             |

当样本数量 n>=1000 时使用梯度下降，小于这个数量使用 normal equation 更方便，当 n 太大时，计算 <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> 会非常慢

When implementing the normal equation in octave we want to use the `pinv` function rather than `inv`. The `pinv` function will give you a value of θ even if<math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> is not invertible(不可逆).

If <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> is noninvertible, the common causes might be having :

* Redundant features, where two features are very closely related (i.e. they are linearly dependent)
* Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.
