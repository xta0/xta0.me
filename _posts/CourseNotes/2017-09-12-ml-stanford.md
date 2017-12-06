---
layout: post
title: Machine Learning Course
meta: Stanford 机器学习课程笔记
categories: course
---

> 所有文章均为作者原创，转载请注明出处

## Chapter1

### Macine Learning definition

机器学习的定义：

- Arthur Samuel(1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell(1998). Well-posed Learning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

### Machine Learning Algorithms:

- 监督学习：Supervised learning
- 非监督学习：Unsupervised learning
- 其它：
	- Reinforcement learning
	- recommender systems

	
### Supervised Learning

监督学习定义: "Define supervised learning as problems where the desired output is provided for examples in the training set." 监督学习我们对给定数据的预测结果有一定的预期，输入和输出之间有某种关系，监督学习包括"Regression"和"Classification"。其中"Regression"是指预测函数的预测结果是连续的，"Classification"指的是预测函数的结果是离散的


### Unspervised Learning

- definition

Define unsupervised learning as problems where we are not told what the desired output is.

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

- Example:

	- Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

	- Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).


## Chapter2

### Model Representation

To establish notation for future use, we’ll use `x(i)` to denote the “input” variables (living area in this example), also called input features, and y(i) to denote the “output” or target variable that we are trying to predict (price). A pair `(x(i),y(i))` is called a training example, and the dataset that we’ll be using to learn—a list of m training examples`(x(i),y(i));i=1,...,m`—is called a training set. Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use `X` to denote the space of input values, and `Y` to denote the space of output values. In this example, `X = Y = ℝ.`

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function `h : X → Y` so that` h(x) `is a “good” predictor for the corresponding value of `y`. For historical reasons, this function `h` is called a hypothesis. Seen pictorially, the process is therefore like this:

![Altext](/images/2017/09/ml-1.png)
 
When the target variable that we’re trying to predict is **continuous**, such as in our housing example, we call the learning problem a `regression problem`. When `y` can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

- regression

回归在数学上来说是给定一个点集，能够用一条曲线去拟合之，如果这个曲线是一条直线，那就被称为线性回归，如果曲线是一条二次曲线，就被称为二次回归，回归还有很多的变种，如locally weighted回归，logistic回归，等等


### Cost Function

- Hypothesis函数：

<math display="block">
	<msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi>
	<mo>=</mo>
	<msub><mi>θ</mi> <mi>0</mi></msub>
	<mo>+</mo>
	<msub><mi>θ</mi>
	<mi>1</mi></msub>
	<mi>x</mi>
</math>


怎么计算参数θ呢？

根据训练数据集，找到最合适的θ值

- cost函数：

<math display="block">
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
  <mo>=</mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <mi>m</mi>
      </mrow>
    </mfrac>
  </mstyle>
  <mstyle displaystyle="true">
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
            <mrow class="MJX-TeXAtom-ORD">
              <mover>
                <mi>y</mi>
                <mo stretchy="false"> ^</mo>
              </mover>
            </mrow>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>i</mi>
            </mrow>
          </msub>
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
  </mstyle>
</math>

这个式子的含义是找到<math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub></math>的值使 <math><mi>J</mi><mo stretchy="false">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo stretchy="false">)</mo></math>的值最小，为了求导方便系数乘了1/2


### Cost Function - Intuition(1)

对于Hypothesis函数：<math>
	<msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi>
	<mo>=</mo>
	<msub><mi>θ</mi> <mi>0</mi></msub>
	<mo>+</mo>
	<msub><mi>θ</mi>
	<mi>1</mi></msub>
	<mi>x</mi>
</math>


当<math><msub><mi>θ</mi><mi>0</mi></msub><mo>=</mo><mi>0</mi></math>时，简化为：<math>
	<msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi>
	<mo>=</mo>
	<msub><mi>θ</mi>
	<mi>1</mi></msub>
	<mi>x</mi>
</math>

对于cost函数简化为：

<math display="block">
  <mi>J</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>θ</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mstyle displaystyle="true">
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <mi>m</mi>
      </mrow>
    </mfrac>
  </mstyle>
  <mstyle displaystyle="true">
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
            <mrow class="MJX-TeXAtom-ORD">
              <mover>
                <mi>y</mi>
                <mo stretchy="false"> ^</mo>
              </mover>
            </mrow>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>i</mi>
            </mrow>
          </msub>
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
              <mi>θ</mi>
              <mi>1</mi>
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
  </mstyle>
</math>

假设有一组训练集：`(1,1),(2,2),(3,3)`

当<math><msub><mi>θ</mi><mi>1</mi></msub><mo>=</mo><mi>1</mi></math>时，<math><msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi><mo>=</mo><mi>x</mi></math>, 有<math><mi>J</mi><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="true"><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac></mstyle><mfenced open="(" close=")"><mrow><msup><mi>0</mi><mn>2</mn></msup><mo>+</mo><msup><mi>0</mi><mn>2</mn></msup><mo>+</mo><msup><mi>0</mi><mn>2</mn></msup></mrow></mfenced><mo>=</mo><mi>0</mi></math>

当<math><msub><mi>θ</mi><mi>1</mi></msub><mo>=</mo><mi>0.5</mi></math>时，<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi>	<mo>=</mo>	<mi>0.5</mi>	<mi>x</mi></math>，<math><mi>J</mi><mo stretchy="false">(</mo><mn>0.5</mn><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="true"><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac></mstyle><mo stretchy="false">[</mo>		<mo stretchy="false">(</mo>			<mi>0.5</mi>			<mo>-</mo>			<mi>1</mi>		<msup>			<mo stretchy="false">)</mo>			<mn>2</mn>		</msup>		<mo>+</mo>			<mo stretchy="false">(</mo>				<mi>1</mi>				<mo>-</mo>				<mi>2</mi>		<msup>			<mo stretchy="false">)</mo>			<mn>2</mn>		</msup>			<mo>+</mo>			<mo stretchy="false">(</mo>	<mi>1.5</mi>				<mo>-</mo>				<mi>3</mi>		<msup>			<mo stretchy="false">)</mo>			<mn>2</mn>		</msup><mo stretchy="false">]</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mo>x</mo><mi>3</mi></mrow></mfrac>	<mo stretchy="false">(</mo>	<mi>3.5</mi>	<mo stretchy="false">)</mo>	<mo>=</mo>	<mi>0.58</mi></math>

当<math><msub><mi>θ</mi><mi>1</mi></msub><mo>=</mo><mi>0</mi></math>时，<math><msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi><mo>=</mo><mi>0</mi></math>，有<math><mi>J</mi><mo stretchy="false">(</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="true"><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac></mstyle><mo stretchy="false">(</mo><mrow>		<msup>			<mi>1</mi>			<mn>2</mn>		</msup>		<mo>+</mo><msup><mi>2</mi><mn>2</mn>		</msup>		<mo>+</mo>		<msup>			<mi>3</mi>			<mn>2</mn>		</msup></mrow><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>14</mn>		<mn>6</mn></mfrac>	<mo>=</mo>	<mi>2.3</mi></math>

以此类推，通过不同的`θ`值可以求出不同的`J(θ)`，如下图所示：

![](/images/2017/09/ml-2.png)

我们的目标是找到一个`θ`值使`J(θ)`最小。显然上述案例中，当`θ=1`时，`J(θ)`最小，因此我们可以得到Hypothesis函数：

<math display="block"><msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi>
	<mo>=</mo>
	<mi>x</mi>
</math>

### Cost Function - Intuition(2)

使用contour plots观察<math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mi>,</mi><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>在二维平面的投影，

> 关于contour plot[参考](https://nb.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/visualizing-scalar-valued-functions/v/contour-plots）

![](/images/2017/09/ml-3.png)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(θ0,θ1) and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when θ0 = 800 and θ1= -0.15

Taking another h(x) and plotting its contour plot, one gets the following graphs:

![](/images/2017/09/ml-3-1.png)

When θ0 = 360 and θ1 = 0, the value of J(θ0,θ1) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.


![](/images/2017/09/ml-3-2.png)

The graph above minimizes the cost function as much as possible and consequently, the result of θ1 and θ0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

### Gradient descent

对Cost函数： <math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>，找到<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>使 <math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>值最小

- 方法
	1. 选择任意<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>，例如：<math><msub><mi>θ</mi><mi>0</mi></msub><mo>=</mo><mn>1</mn><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>=</mo><mn>1</mn></math>
	2. 不断改变<math><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub></math>使<math><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>0</mi></msub><mo>,</mo><msub><mi>θ</mi><mi>1</mi></msub><mo>)</mo></math>按梯度方向进行减少，直到找到最小值

- 图形理解

![Altext](/images/2017/09/ml-4.png)

- 梯度下降法：
	- `:=` 代表赋值，例如 a:=b 代表把b的值赋值给a，类似的比如 a:=a+1。因此 := 表示的是计算机范畴中的赋值。而=号则代表truth assertion，a = b的含义是a的值为b
	- `α` 代表learning rate是梯度下降的步长
	- <math> <mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo stretchy="false">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo stretchy="false">)</mo></math>代表对`θ`求偏导

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

- 理解梯度下降

梯度下降是求多维函数的极值方法，因此公式是对 <math><msub><mi>θ</mi><mi>j</mi></msub></math> 求导，每一个<math><msub><mi>θ</mi><mi>j</mi></msub></math>代表一元参数，也可以理解为一维向量，上述case中，只有<math><msub><mi>θ</mi><mn>0</mn></msub></math>和<math><msub><mi>θ</mi><mn>1</mn></msub></math>两个参数，可以理解在这两个方向上各自下降，他们的向量方向为<math><msup><mi>J</mi><mi>(θ)</mi></msup></math>下降的方向，下降过程是一个同步迭代的过程：	
	
![](/images/2017/09/ml-4.png)
	
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
	
![](/images/2017/09/ml-3-2.png)
	
`θ`会逐渐向极值点出收敛，当`θ`到达极值点时，该处导数为0，则`θ`值不再变化。
		
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

### Multiple features

上几节讨论的问题是：已知一个房子大小和价格样本数据集，来推导房价和房屋大小的关系函数：


Size(x)  | Price(y)
------| -------------
2104  | 460
1035  | 224
868  | 230
642  | 126
...	   | ...

`x`为房子的size，`y`是房价，上述的一维线性回归函数：<math>
	<msub><mi>h</mi> <mi>θ</mi></msub><mi>(x)</mi>
	<mo>=</mo>
	<msub><mi>θ</mi> <mi>0</mi></msub>
	<mo>+</mo>
	<msub><mi>θ</mi>
	<mi>1</mi></msub>
	<mi>x</mi>
</math>，但是影响房价的因素很多，比如房屋数量，楼层数等等：

Size(x1)| number of bed room (x2) | number of floors(x3) | Price(y)
------| ------------| ----------- | ----|
2104  | 5				| 	2			|		460 |
1035  | 4				|	1			|		224 |
868  |  3				|	2			|		230 | 
642  |  2				|	1			|		126 |
...	  | ...			| 	...			|		... | 

对应到公式里，则表现为`x`是多维时，公式如下：

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

其中：

<math display="block">
    <mtr>
      <mtd>
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
      <mtd>
        <mo>=</mo>
        <mtext>value of feature</mtext>
        <mi>j</mi>
        <mtext>in the</mtext>
        <msup>
          <mi>i</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>t</mi>
            <mi>h</mi>
          </mrow>
        </msup>
        <mtext>training example</mtext>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msup>
          <mi>x</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>i</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mtext>the input (features) of the</mtext>
        <msup>
          <mi>i</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>t</mi>
            <mi>h</mi>
          </mrow>
        </msup>
        <mtext>training example</mtext>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mi>m</mi>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mtext>the number of training examples</mtext>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mi>n</mi>
      </mtd>
      <mtd>
        <mo>=</mo>
        <mtext>the number of features</mtext>
      </mtd>
    </mtr>
  </mtable>
</math>


举例来说，<math><msup><mi>x</mi><mi>(2)</mi></msup></math>表示第二组训练集：

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

而<math><msubsup><mi>x</mi><mi>3</mi><mi>(2)</mi></msubsup></math>表示上面向量中中第三个元素：

<math display="block">
<msubsup>
<mi>x</mi>
<mi>3</mi>
<mi>(2)</mi>
</msubsup>
<mo>=</mo>
<mn>1</mn>
</math>

还是举个买房子的例子，假如我们得到如下函数:

<math display="block">
<msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo>
<mo>=</mo>
<mn>80</mn>
<mo>+</mo>
<mn>0.1</mn><msub><mi>x</mi><mi>1</mi></msub>
<mo>+</mo>
<mn>0.01</mn><msub><mi>x</mi><mi>2</mi></msub>
<mo>+</mo>
<mn>3</mn><msub><mi>x</mi><mi>3</mi></msub>
<mo>-</mo>
<mn>2</mn><msub><mi>x</mi><mi>4</mi></msub>
</math>

其中<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math>表示房子的总价，<math><msub><mi>θ</mi><mn>0</mn></msub></math> = 80代表房子的基础价格，<math><msub><mi>x</mi><mn>1</mn></msub></math>代表这栋房子的size，<math><msub><mi>θ</mi><mn>1</mn></msub></math>是用cost function计算出来对<math><msub><mi>x</mi><mn>1</mn></msub></math>的系数，类似的<math><msub><mi>x</mi><mn>2</mn></msub></math>代表房子的房间数，<math><msub><mi>θ</mi><mn>2</mn></msub></math>是对<math><msub><mi>x</mi><mn>2</mn></msub></math>的系数，等等

在这个式子中

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
<math><msub><mi>x</mi><mn>0</mn></msub></math> 默认为1，即<math><msubsup><mi>x</mi><mi>0</mi><mi>(i)</mi></msubsup><mo>=</mo><mn>1</mn></math>可以把每条样本和对应的参数看成两条vector:

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
<mo>,</mo>
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

参考一维线性回归的的cost函数，多维线性回归的cost函数为:

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


### Feature Scaling

Idea: Make sure features are on a similar scale.

E.g.:`x1 = size(0-200 feet) `,`x2=number of bedrooms(1-5)`

这种情况contour图是一个瘦长的椭圆，如图：


在不优化的情况下，这类梯度下降速度很慢。如果我们将`x1,x2`做如下调整：

`x1 = size(0-200 feet)/5`,`x2=(number of bedrooms)/5`,则contour图会变为接近圆形，梯度下降收敛的速度会加快。通常为了加速收敛，会将每个feature值(每个`xi`)统一到某个区间里，比如 <math><mn>0</mn><mo>≤</mo><msub><mi>x</mi><mi>1</mi></msub><mo>≤</mo><mn>3</mn></math>，<math><mn>-2</mn><mo>≤</mo><msub><mi>x</mi><mi>2</mi></msub><mo>≤</mo><mn>0.5</mn></math>等等

### Mean normalization

Replace <math><msub><mi>x</mi><mi>i</mi></msub></math>with <math><msub><mi>x</mi><mi>i</mi></msub><mo>-</mo><msub><mi>μ</mi><mi>i</mi></msub></math> to make features have approximately zero mean.实际上就是将feature归一化，

例如`x1=(size-1000)/2000	x2=(#bedrooms-2)/5`

则有：<math><mn>-0.5</mn><mo>≤</mo><msub><mi>x</mi><mi>1</mi></msub><mo>≤</mo><mn>0.5</mn></math>，<math><mn>-0.5</mn><mo>≤</mo><msub><mi>x</mi><mi>2</mi></msub><mo>≤</mo><mn>0.5</mn></math>

- <math><msub><mi>μ</mi><mi>i</mi></msub></math> 是所有 <math><msub><mi>x</mi><mi>i</mi></msub></math>

- <math><msub><mi>μ</mi><mi>i</mi></msub></math> 是`xi`的区间范围，(max-min)

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

- `μ`表示所有feature的平均值
- `s = max - min`

### Learning Rate

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

![Altext](/images/2017/09/ml-4-3.png)

但是如果`α`选取过大，则可能会导致越过极值点的情况，导致随着迭代次数的增加，`J(θ)`的值增加或忽高忽低不稳定的情况:

![Altext](/images/2017/09/ml-4-4.png)

解决办法都是选取较小的`α`值

- Summary:
	- if `α` is too small: slow convergence
	- if `α` is too large: `J(θ)`may not decrease on every iteration; may not converge
	- To choose `α` , try: ..., 0.001, 0.003, 0.01,0.03, 0.1,0.3, 1, ...


### Polynomial regression

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

其中<math><msub><mi>x</mi><mi>i</mi></msub></math>代表feature种类，有些情况下使用这些feature制作目标函数不方便，因此可以考虑重新定义feature的值

We can improve our features and the form of our hypothesis function in a couple different ways.
We can **combine** multiple features into one. For example, we can combine x1 and x2 into a new feature x3 by taking x1⋅x2.

例如我们可以将两个feature合成一个:`x3 = x1*x2`，使用`x3`作为先行回归的feature值。

另外，如果只有一个feature，而使用线性函数又不适合描述完整的数据集，可以考虑多项式函数，比如使用二次函数或者三次函数：

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

可以令 <math><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>,</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup></math> 但是这么选择的一个问题在于feature scaling 会比较重要，如果x1的range是[1,1000]，那么x2的range就会变成[1,1000000]等

### Normal Equation

对于cost函数：

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
  </mstyle>
</math>

前面提到的求`J(θ)`最小值的思路是使用梯度下降法，对<math><msub><mi>θ</mi><mi>j</mi></msub></math>求偏导得到各个θ值:

<math display="block">
<mfrac><mi mathvariant="normal">∂</mi><mrow><mi mathvariant="normal">∂</mi><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
<mo>=</mo>
<mn>0</mn>
<mspace width="1em"></mspace>
<mi>(for every j)</mi>
</math>

出了梯度下降法之外，还有一种方法叫做**Normal Equation**，这种方式不需要迭代，可以直接计算出θ值 。

假设我们有m个样本。特征向量的维度为n。因此，可知样本为 <math><mo>{</mo><mo>(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo>)</mo><mo>,</mo><mo>...</mo><mo>(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo>)</mo><mo>}</mo></math>，其中对于每一个样本中的<math><msup><mi>x</mi><mi>(i)</mi></msup></math>，都有 <math><msup><mi>x</mi><mi>(i)</mi></msup><mo>=</mo><mo>{</mo><msubsup><mi>x</mi><mi>1</mi><mi>(i)</mi></msubsup><msubsup><mi>x</mi><mi>2</mi><mi>(i)</mi></msubsup><mo>,</mo><mo>...</mo><msubsup><mi>x</mi><mi>n</mi><mi>(i)</mi></msubsup><mo>}</mo></math>，令线性回归函数 <math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>3</mn></msub><msub><mi>x</mi><mn>3</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>θ</mi><mi>n</mi></msub><msub><mi>x</mi><mi>n</mi></msub></math>，则有：

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

- <math><mi>X</mi></math> 是 <math><mi>m</mi><mo>*</mo><mi>(n+1)</mi></math>的矩阵
- <math><mi>θ</mi></math> 是 <math><mi>(n+1)</mi><mo>*</mo><mn>1</mn></math>的矩阵
- <math><mi>Y</mi></math> 是 <math><mi>m</mi><mo>*</mo><mn>1</mn></math>的矩阵

看个例子：

![](/images/2017/09/ml-4-5.png)

若希望<math><msub><mi>h</mi><mi>(θ)</mi></msub><mo>=</mo><mi>y</mi></math>，则有<math><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mi>Y</mi></math>，回想**单位矩阵** 和 **矩阵的逆**的性质：

- 单位矩阵E，<math><mi>AE</mi><mo>=</mo><mi>EA</mi><mo>=</mo><mi>A</mi></math>
- 矩阵的逆<math><msup><mi>A</mi><mi>-1</mi></msup></math>，A必须为方阵，<math><mi>A</mi><msup><mi>A</mi><mi>-1</mi></msup><mo>=</mo><msup><mi>A</mi><mi>-1</mi></msup><mi>A</mi><mo>=</mo><mi>E</mi></math>

再来看看式子 <math><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mi>Y</mi></math> 若想求出θ，那么我们需要做一些转换：

1. 先把θ左边的矩阵变成一个方阵。通过乘以<math><msup><mi>X</mi><mi>T</mi></msup></math>可以实现，则有 <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>

2. 把θ左边的部分变成一个单位矩阵，这样左边就只剩下θ，<math><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>·</mo><mi>θ</mi><mo>=</mo><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>

3. 由于<math><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>=</mo><mi>E</mi></math>，因此式子变为<math><mi>θ</mi><mo>=</mo><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><msup><mo>)</mo><mn>-1</mn></msup><msup><mi>X</mi><mi>T</mi></msup><mi>Y</mi></math>，这就**Normal Equation**的表达式。

如果用Octave表示，命令为：`pinv(X'*X)*X'*Y`

什么case适合使用Normal Equation，什么case适合使用Gradient Descent？

Gradient Descent | Normal Equation
------------- | -------------
Need to choose alpha  | No need to choose alpha
Needs many iterations  | No need to iterate
<math><mi>O</mi><mo>(</mo><mi>k</mi><msup><mi>n</mi><mn>2</mn></msup><mo>)</mo></math> | <math><mi>O</mi><mo> (</mo><msup><mi>n</mi><mn>3</mn></msup><mo>)</mo></math> need to calculate inverse of <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math>
Works well when n is large | Slow if n is very large
	
当样本数量n>=1000时使用梯度下降，小于这个数量使用normal equation更方便，当n太大时，计算 <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> 会非常慢

When implementing the normal equation in octave we want to use the `pinv` function rather than `inv`. The `pinv` function will give you a value of θ even if<math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> is not invertible(不可逆).

If <math><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi></math> is noninvertible, the common causes might be having :

- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

### 附录：Octave Cheet Sheet

#### Basic Operations

- 退出
	- `exit`,`quit` 
- 赋值
	- `a=3`
	- `b='hi'` 
	
- print
	- `disp(a)`
	- `sprintf('%0.6f',a)`

- 矩阵与向量
	- `A = [1 2; 3 4; 5 6]` 3x2的矩阵
	- `A(2,:)`第二行的向量，`A(:,2)`第二列的向量，`A(:)`将A所有的元素放到一个vector里
	- `size(A)`得到矩阵的维度, `size(A,1)` 得到矩阵的行数, `size(A,2)`得到矩阵的列数
	- `length(A)`得到max(row,col)
	- `A = [A, [100;101;102]];`在A后面增加一栏，`C=[A B]`将A和B（B在A右边）组合后赋值给C，`C=[A;B]`则是将B放到A的下边
	- `V = [1 2 3]` 1x3的向量； `V=[1;2;3]` 3X1向量
	- `v=1:6` 1x6 向量，步长为1；`v=[0:0.01;0.98]` 1x99的向量，从0到0.98，步长0.01
	- `ones(2,3)` 2x3单位阵，类似的 `2*ones(2,3)`
	- `w=zeros(1,3)` 1x3的0矩阵
	- `I=eye(3)` 3x3的单位矩阵
	- `rand(3,3)` 随机3x3矩阵，类似的`randn(3,3)`产生类似高斯分布的3x3矩阵
	- `A=magic(3)`产生magic矩阵，每行每列和相等

- 控制语句
	- for循环： `for i=1:10, v(i)=2^i; end;` 
	- while循环：`i=1; while i<=5, v(i)=100; i=i+1; end;`
	- if-elseif-else `if cond1, xxx; elseif cond2, xxx; else xxx; end;`

- 定义函数
	- `function y = squareThisNumber(x) y=x^2; end; z = squareThisNumber(10)`
	- 另一种方式是将行数定义在文件里，文件已.m结尾，在文件目录下使用会自动识别函数名。或者指定search path：`addPath('/Users/ang/Desktop')`
	- `function v = squareAndCubeThisNum(x) y1=x^2; y2=x^3; end; [a,b]= squareAndCubeThisNum(5)`返回两个值
	- 以线性回归为例，加入我们有三个数据集(1,1),(2,2),(3,3)，回归函数为`h(x)=θ*x`，求cost function的最小值。首先定义costFunction.m:

```matlab

	function J = costFunction(X,Y,theta)

	%X is "designed matrix" , containing the training example
	%y is the class labels

	m = size(X,1); %number of the training example

	predictions = X*theta; %predictions of hypothesis on all examples

	sqrErrors = ( predictions - Y ).^2; %squared errors

	J = 1/(2*m)*sum(sqrErrors)

```

然后输入参数：`X=[1 1; 1 2; 1 3]`(x0 = 1),`y = [1;2;3]`,`theta = [0;1]`(θ0=0，θ1=1)，最后调用函数:`j=costFunction(X,y,theta)`

#### Moving data arround

- 读取数据
	- 到指定目录下，`load('featureX.dat')` 
	- `who`展示当前已有的变量,`whos`展示细节
	- `clear`清空所有数据

- 存数据
	- `save hello.mat v`将v保存到hello.mat中，并以二进制的形式保存到磁盘 
	- `save hello.txt v -ascii` 将v以text的形式保存


#### Computing Data

- 加减乘除
	- 矩阵相乘`A*C`，矩阵每项对应相乘`A .* B` 
	- 矩阵每项平方运算`A .^ 2`, 矩阵每项倒数运算 `1 ./ A `
	- `log`运算，`exp`运算，`abs`
	- 加法 `v+1`，v的每项都加1
	- 转置`A'`
	- 求逆矩阵`pinv(A)`
	- 求A各列项的和`sum(A,1)`，求A各行项的和`sum(A,2)`，类似的,`ceil(a)`,`floor(a)`

- 逻辑运算
	- `a<3`，矩阵a中每项是否小于3来返回0或1
	- `max(a)`,返回每列最大值组成的向量，`max(max(a))`矩阵中最大项
	- `find(A>=7)`找到A中大于7的项的index

#### Ploting Data

- macos上设置环境`setenv("GNUTERM","qt")`
- `clf`清空当前图片
- 画图
	- `t=[0.0:0.01:0.98], y=sin(2*pi*t); plot(t,y,'r')`横坐标是t，纵坐标是y，r代表color
	- `hold on`保留当前图片的内容，在此基础上叠加
	- `xlabel`,`ylabel`横纵坐标标注，`legend('sin','cos')`做图解，`title`标题
	- `axis[0.5 1 -1 1]`设置y轴，x轴显示范围
	- `print -dpng 'plotpng'`在当前目录保存图片
	- `close`关掉图片窗口，`figure(1)`打开一个窗口，``figure(2)`同时打开两个窗口，`subplot(1,2,1)`将窗口切分成1x2个并占用第一个，`subplot(1,2,2)` 将窗口切分成1x2个并占用第2个
- 直方图
	- `hist(w,50)`绘制w的高斯直方图，50个柱，

- 矩阵图
	- `imagesc(A)` 
	- `iamgesc(magic(15)),colorbar, colormap gray`

#### Vectorization

将数学公式转为向量或矩阵运算求解，例如<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="true"><munderover><mo>∑</mo><mrow class="MJX-TeXAtom-ORD">	<mi>j</mi>	<mo>=</mo>	<mn>0</mn></mrow><mrow>	<mi>n</mi></mrow></munderover></mstyle><msub><mi>θ</mi><mi>j</mi></msub><msub><mi>x</mi><mi>j</mi></msub></math>如果是数学运算，计算方法为：

```matlab
prediction = 0.0;
for j=1:n+1
	prediction = prediction + theta(j)*X(j)
end;
```

而如果使用向量化，则可以将上述式子理解为<math><msup><mi>θ</mi><mi>T</mi></msup><mi>X</mi></math>，代码描述为：`predication = theta'*X`，更简单。 

## Chapter3

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

- 0："Negative Class"(e.g., benign tumor)
- 1: "Positive Class"(e.g., malignant tumor)

对于分类场景，使用线性回归模型不适合，原因是 <math><msub><mi>h</mi><mi>θ</mi></msub></math> 的值区间不能保证在[0,1]之间，因此需要一种新的模型，叫做logistic regression，逻辑回归。

### Logistic Regression Model

在给出模型前，先不考虑y的取值是离散的，我们希望能使：<math><mn>0</mn><mo>≤</mo><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≤</mo><mn>1</mn></math>，可以将式子做一些变换：

- 线性函数：<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>X</mi></math>
- 做如下变换：<math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>)</mo></math>, 另<math><mi>z</mi><mo>=</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo>,</mo><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>

- 得到函数：<math><mi>g</mi><mo stretchy="false">(</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mstyle></math>
函数曲线如下:

![](/images/2017/09/ml-5-1.png)

函数`g(z)`, 如上图将所有实数映射到了(0,1]空间内，这使他可以将任意一个h(x)的值空间转化为适合分类器取值的空间, `g(z)`也叫做**Sigmoid Function**`hθ(x)`的输出是结果是`1`的概率，比如`hθ(x)=0.7`表示70%的概率我们的输出结果为`1`，因此输出是`0`的概率则是30%：

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

- <math><mo>"</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>"</mo></math> if <math><math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>0.5</mn></math>

- <math><mo>"</mo><mi>y</mi><mo>=</mo><mn>0</mn><mo>"</mo></math> if <math><math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo><</mo><mn>0.5</mn></math>


通过观察函数<math><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></mstyle></math>上一节的曲线可以发现，当`z`大于0的时候`g(z)≥0.5`，因此只需要<math><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi><mo> > </mo><mn>0</mn></math>即可，即：

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

![](/images/2017/09/ml-5-2.png)

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

![](/images/2017/09/ml-5-3.png)

则落在圈外的样本点，可以预测`y=1`

### Cost Function

由上面可知，给定：

- 训练集<math><mo>{</mo><mo>(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo>)</mo><mo>,</mo><mo>...</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo>)</mo><mo>}</mo></math>，m个样本，其中:<math><mi>x</mi><mo>∈</mo><mo>[</mo><mtable>	<mtr>		<msub><mi>x</mi><mi>1</mi></msub>	</mtr>	<mtr>		<msub><mi>x</mi><mi>2</mi></msub>	</mtr>	<mtr>		<mtd><mo>...</mo></mtd>	</mtr>	<mtr>		<msub><mi>x</mi><mi>n</mi></msub>	</mtr></mtable><mo>]</mo><mspace width="1em"></mspace><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo><mi>y</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0,1</mn><mo stretchy="false">}</mo></math>

- 预测函数: <math><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mstyle></math>

问题还是怎么求解<math><mo>θ</mo></math>，如果使用之前先行回归的cost function，即 <math><mi>J</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow class="MJX-TeXAtom-ORD"><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover>		<mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi>		<mi>i</mi></msub><mo stretchy="false">)</mo><mo>−</mo><msub><mi>y</mi><mi>i</mi></msub>		<msup><mo>)</mo><mn>2</mn></msup></math>

这时会出现`J(θ)`不是convex function的情况，原因是`h(x)`变成了复杂的非线性函数，因此梯度下降无法得到最小值（得到极小值的概率更高）。

- 逻辑回归的Cost Function：

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

当`y=1`的时候，`J(θ) = 0` -> `h(x)=1`；`J(θ) = ∞ ` -> `h(x)=0`，如下图所示

![](/images/2017/09/ml-5-2.png)

当`y=0`的时候，`J(θ) =0 ` -> `h(x)=0`，`J(θ) = ∞ ` -> `h(x)=1`，如下图所示

![](/images/2017/09/ml-5-3.png)

图上可以看出`J(θ)`有极值点，接下来的问题就是分别求解`h(x)=0`和`h(x)=1`两种情况下的`θ`值

### Simplifed Cost Function

上述Cost Function 可以简化为一行：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
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

- Conjugate gradient
- BFGS
- L-BFGS

这三种算法的优点是：

- 不需要人工选择`α`
- 比低度下降更快

缺点是：比较复杂

开发者不需要自己实现这些算法，在一般的数值计算库里都有相应的实现，例如python，Octave等。我们只需要关心两个问题，如何给出:

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

我们可以使用Octave写这样一个函数:

```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end

```

然后使用Octave自带的`fminunc()`优化算法计算出`θ`的值：

```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

```
`fminunc()`函数接受三个参数：costFunction，初始的θ值（至少是2x1的向量），还有options

### Multiclass classification

前面我们解决的问题都是针对两个场景进行分类，即 y = {0,1} 实际生活中，常常有多个场景需要归类，例如

- Email foldering/tagging：Work，Friends，Family，Hobby
- Weather：Sunny，Cloudy，Rain，Snow

即分类结果不只是0或1，而是多个 y = {0,1...n}，解决多个分类我们使用one-vs-all的方式，即选取某一个场景进行归类时，将其他场景合并为起对立的场景。如图：

![](/images/2017/09/ml-5-4.png)


上图可知我们先取一个class进行计算，将其他的归类为另一个class，这样就可以使用前面提到的binary regression model进行计算，即


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


- One-vs-all(one-vs-rest):


Train a logistic regression classifier<math><msubsup><mi>h</mi><mi>θ</mi><mi>(i)</mi></msubsup><mi>(x)</mi></math> for each class <math><mi>i</mi></math> to predict the probability that <math><mi>y</mi><mo>=</mo><mi>i</mi></math>.

On a new input <math><mi>x</mi></math>, to make a prediction, pick the class <math><mi>i</mi></math> that maximizes

- 总结一下就是对每种分类先计算他的<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi></math>，当有一个新的`x`需要分类时，选一个让<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi></math>值最大的分类器。

### 附录：Regularization

### The problem of overfitting

以线性回归的预测房价为例，如下图所示：

![](/images/2017/09/ml-5-5.png) 

可以看到:

- 通过线性函数预测房价不够精确(underfit)，术语叫做"High Bias"，其原因主要是样本的feature太少了。
- 通过二次函数拟合出来的曲线刚好可以贯穿大部分数据样本，术语叫做"Just Right"
- 通过四阶多项式拟合出来的曲线虽然能贯穿所有数据样本，但是曲线本身不够规则，当有新样本出现时不能很好的预测。这种情况我们叫做**Over Fitting（过度拟合）**，术语叫做"High variance"。If we have too many features, the learned hypothesis may fit the training set very well（`J(θ)=0`）, but fail to generalize to new examples(predict prices on new examples) Over Fitting的问题在样本少，feature多的时候很明显

- Addressing overfitting:
	- Reduce number of features
		- Manually select which features to keep.
		- Model selection algorithm
	- Regularization
		- Keep all features, but reduce magnitude/values of parameters <math><msub><mi>θ</mi><mi>j</mi></msub></math>
		- Works well when we have a lot of features, each of which contributes a bit to predicting<math><mi>y</mi></math>

		
### Regularization Cost Function

如果要减少overfitting的情况，我们可以降低一些参数的权重，假设我们想要让下面的函数变成二次方程：

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

我们想要在不去掉θ3和θ4的前提下，降低<math><msub><mi>θ</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup></math>和<math><msub><mi>θ</mi><mn>4</mn></msub><msup><mi>x</mi><mn>4</mn></msup></math>的影响，我们可以修改cost function为：

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

我们在原来的cost function后面增加了两项，为了让cost function接近0，我们需要让<math><msub><mi>θ</mi><mn>3</mn></msub></math>,<math><msub><mi>θ</mi><mn>4</mn></msub></math>近似为0，这样即会极大的减少<math><msub><mi>θ</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup></math>和<math><msub><mi>θ</mi><mn>4</mn></msub><msup><mi>x</mi><mn>4</mn></msup></math>的值，减少后的曲线如下图粉色曲线，更接近二次函数：

![](/images/2017/09/ml-5-6.png)

Small values for parameters <math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>n</mn></msub></math>

- "Simpler" hypothesis，选取更小的`θ`值能得到更简单的预测函数，例如上面的例子，如果将 <math><msub><mi>θ</mi><mn>3</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>4</mn></msub></math>近似为0的话，那么上述函数将变为二次方程，更贴近合理的假设函数
- Housing example:
	- Feature: <math><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo><msub><mi>x</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>100</mn></msub></math>
	- Parameters: <math><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>,...,</mo><msub><mi>θ</mi><mn>100</mn></msub></math>

100个feature，如何有效的选取这些`θ`呢，改变cost function：

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

参数λ叫做**regularization parameter**，它的作用是既要保证曲线的拟合程度够高同时又要确保θ的值尽量小。如果λ的值选取过大，例如<math><mi>λ</mi><mo>=</mo><msup><mi>10</mi><mi>10</mi></msup></math>，会导致计算出的所有的θ值都接近0，从而使<math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi><mo>=</mo><msub><mi>θ</mi><mi>0</mi></msub></math> 即产生"Under fitting"



### Regularized linear regression

有了上面的式子，我们可以将它应用到线性回归：

- 修改梯度下降公式为：

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

上述式子中<math xmlns="http://www.w3.org/1998/Math/MathML"><mn>1</mn><mo>−</mo><mi>α</mi><mfrac><mi>λ</mi><mi>m</mi></mfrac></math>必须小于1，这样就减小了<math><msub><mi>θ</mi><mi>j</mi></msub></math>的值，后面的式子和之前梯度下降的式子相同

- 应用到Normal Equation

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

L是一个（n+1)x(n+1)的对单位阵，第一项是0。在引入λ.L之后<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>+</mo><mi>λ</mi><mi>L</mi></math>保证可逆

### Regularized logistic regression

逻辑回归也有overfitting的问题，如图所示

![](/images/2017/09/ml-5-7.png)

处理方式和线性回归相同，之前知道逻辑回归的cost function 如下：

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

我们可以在最后加一项来regularize这个函数：

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

![](/images/2017/09/ml-5-8.png)

## Neural Networks

### Non-linear hypotheses

神经网络是一个很老的概念，在机器学习领域目前还是主流，之前已经介绍了linear regression和logistics regression，为什么还要学习神经网络？以non-linear classification为例，如下图所示

![](/images/2017/09/ml-6-1.png)

可以通过构建对feature的多项式<math><msub><mi>g</mi><mi>θ</mi></msub></math>来确定预测函数，但这中方法适用于feature较少的情况，比如上图中，只有两个feature。
当feature多的时候，产生多项式就会变得很麻烦，还是预测房价的例子，可能的feature有很多，比如：x1=size,x2=#bedrooms,x3=#floors,x4=age,...x100=#schools等等，
假设n=100，有几种做法：

- 构建二阶多项式
	- 如`x1^2,x1x2,x1x3,x1x4,...,x1x100,x2^2,x2x3...`有约为5000项(n^2/2)，计算的代价非常高。
	- 取一个子集，比如`x1^2, x2^2,x3^2...x100^2`，这样就只有100个feature，但是100个feature会导致结果误差很高

- 构建三阶多项式
	- 如`x1x2x3, x1^2x2, x1^2x3...x10x11x17...`, 有约为n^3量级的组合情况，约为170,000个


另一个例子是图像识别与分类，例如识别一辆汽车，对于图像来说，是通过对像素值进行学习（车共有的像素特征vs非汽车特征），那么feature就是图片的像素值，如下图所示，加入有两个像素点是车图片都有的：

![](/images/2017/09/ml-6-2.png)

假设图片大小为50x50，总共2500个像素点，即2500个feature（灰度图像，RGB乘以三），如果使用二次方程，那么有接近300万个feature，显然图像这种场景使用non linear regression不合适，需要探索新的学习方式


### Neural Networks and Brain

- Origins: Algorithms that try to mimic the brain
	- Was vary widely used in 80s and early 90s; popularity diminished in late 90s
	- Recent resurgence: State of the art technique for many applications，近几年大数据推动

- 脑神经试验
	- 大脑负责识别听力的区域同样可以被训练来识别视觉信号

	

### Mode Representation 1

- 大脑中的神经元结构如下图：

![](/images/2017/09/ml-6-3.png)

所谓神经元就是一种计算单元，它的输入是**Dendrite**，输出是**Axon**。类比之前的机器学习模型，dendrite可以类比为<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>1</mn></msub><mo>⋯</mo><msub><mi>x</mi><mi>n</mi></msub></math>，输出Axon可以类比为预测函数的结果。

- Neuron model: Logistic unit

![](/images/2017/09/ml-6-4.png)

如上图，是一个单层的神经网络，其中：

- 输入端<math><msub><mi>X</mi><mn>0</mn></msub></math>默认为1，也叫做"bias unit."
- <math><msub><mi>h</mi><mi>θ</mi></msub><mi>(x)</mi></math> 和逻辑回归预测方程一样，用<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow class="MJX-TeXAtom-ORD"><mo>−</mo><msup><mi>θ</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></math>表示，也叫做asigmoid(logistic)**activation**函数
- θ矩阵在神经网络里也被叫做权重weight

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
    </mtable>
  </mfenced>
  <mo stretchy="false">→</mo>
  <mfenced open="[" close="]">
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mtext>&#xA0;</mtext>
          <mtext>&#xA0;</mtext>
          <mtext>&#xA0;</mtext>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
  <mo stretchy="false">→</mo>
  <msub>
    <mi>h</mi>
    <mi>θ</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
</math>


多层神经网络如下图

![](/images/2017/09/ml-6-5.png)

- 第一层是叫"Input Layer"，最后一层叫"Output Layer"，中间叫"Hidden Layer"，上面例子中，我们使用<math xmlns="http://www.w3.org/1998/Math/MathML"><msubsup><mi>a</mi><mn>0</mn><mn>2</mn></msubsup><mo>⋯</mo><msubsup><mi>a</mi><mi>n</mi><mn>2</mn></msubsup></math>表示，他们也叫做"activationunits."

- <math><msubsup><mi>a</mi><mi>i</mi><mi>(j)</mi></msubsup></math>="activation" of unit <math><mi>i</mi></math> in layer <math><mi>j</mi></math>

- <math><msup><mi>θ</mi><mi>(j)</mi></msup></math>=matrix of weights controlling function mapping from layer <math><mi>j</mi></math> to <math><mi>j+1</mi></math>

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

> <math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>If network has&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mi>j</mi></msub></mrow><mtext>&#xA0;units in layer&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><mi>j</mi></mrow><mtext>&#xA0;and&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow><mtext>&#xA0;units in layer&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow><mtext>,then&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msup><mi mathvariant="normal">Θ</mi><mrow class="MJX-TeXAtom-ORD"><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></mrow><mtext>&#xA0;will be of dimension&#xA0;</mtext><mrow class="MJX-TeXAtom-ORD"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>×</mo><mo stretchy="false">(</mo><msub><mi>s</mi><mi>j</mi></msub><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow><mtext>.</mtext></math>


推算可知+1来自<math><msup><mi>θ</mi><mi>(j)</mi></msup></math>矩阵的"bias nodes"，即<math><msub><mi>x</mi><mn>0</mn></msub></math>和<math><msubsup><mi>θ</mi><mi>(j)</mi><mn>0</mn></msubsup></math>。如下图：

![](/images/2017/09/ml-6-6.png)

举例来说：

如果layer 1有两个输入节点，layer 2 有4个activation节点，那么<math><msup><mi>θ</mi><mi>(1)</mi></msup></math>是4x3的，<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mrow class="MJX-TeXAtom-ORD"><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>×</mo><mo stretchy="false">(</mo><msub><mi>s</mi><mi>j</mi></msub><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><mo>×</mo><mn>3</mn></math>

- 向量化表示

上面计算中间节点的式子可以用向量化表示


