---
layout: post
title: Machine Learning - Chap7
meta: Coursera Stanford Machine Learning Cousre Note, Chapter7
categories: [ml-stanford,course]
---

## Chapter7 Unsupervised Learning

- 监督学习，它的特点是对已知样本产生的结果是有预期的，因此我们可以对样本进行标记，Training Set可以表示为<math><mo>{</mo><mo>(</mo><msup><mi>x</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(1)</mi></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(2)</mi></msup><mo>)</mo><mo>,</mo><mo>...</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>,</mo><msup><mi>y</mi><mi>(m)</mi></msup><mo>)</mo><mo>}</mo></math>
- 非监督性学习对样本产生的结果是无法预期的，因此无法对样本数据进行标，Training Set可表示为<math><mo>{</mo><msup><mi>x</mi><mi>(1)</mi></msup><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>}</mo></math>，非监督学习要做的是在这些无标注的数据中寻找某种规则或者模式。比如将数据归类，也叫做聚类( **Clustering** )就是一种非监督学习算法

### K-Means 算法

K-Means是解决分类问题的一种很常用的迭代算法，具体步骤如下(以K=2为例)

- 在数据集中随机初始化两个点（红蓝叉），作为聚类中心(Cluster Centrioid)

![](/images/2017/09/ml-9-1.png)	

- 对样本进行簇分配（Cluster Assignment）：对数据集中的所有样本计算到聚类中心的距离，按照样本据某个中心点距离远近进行归类

![](/images/2017/09/ml-9-2.png)

- 移动聚类中心 (Move Centroid): 对分类好的样本，求平均值，得到新的聚类中心

![](/images/2017/09/ml-9-3.png)

- 重复上述两个步骤，直到中心点不再变化

![](/images/2017/09/ml-9-4.png)


K-Means函数的输入有两个参数：聚类数量K和训练集（默认<math><msub><mi>x</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn></math> )

随机初始化K个聚类中心点:<math><msub><mi>μ</mi><mn>1</mn></msub><mo>,</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>...,</mo><msub><mi>μ</mi><mi>k</mi></msub></math>

Repeat{

<math><mtext>for</mtext><mspace width="0.5em"/><mi>i</mi><mo>=</mo><mn>1</mn><mspace width="0.5em"/><mtext>to</mtext><mspace width="0.5em"/><mi>m</mi></math><br>
<math><mspace width="2em"/><msup><mi>c</mi><mi>(i)</mi></msup> <mo>:=</mo> <munder><mrow><mtext>min</mtext></mrow><mrow><mi>k</mi></mrow></munder><mo>||</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><msub><mi>μ</mi><mi>k</mi></msub><msup><mo>||</mo><mn>2</mn></msup></math>//index of cluster(1,2,...K) to which example <math><msup><mi>x</mi><mi>(i)</mi></msup></math> is currently assigned <br><math><mtext>end</mtext></math>
<br>
<br>
<math><mtext>for</mtext><mspace width="0.5em"/><mi>k</mi><mo>=</mo><mn>1</mn><mspace width="0.5em"/><mtext>to</mtext><mspace width="0.5em"/><mi>K</mi></math>
<br>
<math><mspace width="2em"/><msub><mi>μ</mi><mi>k</mi></msub> <mo>:=</mo> <mtext>average(mean) of points assigned to cluster k</mtext></math>
</br>
<math><mtext>end</mtext></math>

}

- 第一步的Octave实现为：

```matlab

function idx = findClosestCentroids(X, centroids)

	% Set K
	K = size(centroids, 1);
	
	% You need to return the following variables correctly.
	idx = zeros(size(X,1), 1);
	
	
	for i=1:size(X,1)
	    d_min=Inf;
	    for k=1:K
	        d=sum((X(i,:)-centroids(k,:)).^2); %这一步也可以用向量化实现
	        %  
	        %	 diff = X(i, :)'-centroids(k, :)';
	    	 %	 d = diff'*diff;
	        %
	        if d<d_min
	            d_min=d;
	            idx(i)=k;
	        end  
	    end 
	end
end

```

- 第二步的Octave实现为：

```matlab

function centroids = computeCentroids(X, idx, K)

	% Useful variables
	[m n] = size(X);
	centroids = zeros(K, n);
	
	for k=1:K 
	    num=0;
	    sum=zeros(1,n);
	    for i=1:m
	        if (k==idx(i))
	            sum=sum + X(i,:);
	            num=num+1;
	        end 
	    end 
	    centroids(k,:)= sum/num;
	end 

end

```

- K-Means的优化目标函数为：找到<math><msup><mi>c</mi><mn>(1)</mn></msup><mo>,...,</mo><msup><mi>c</mi><mi>(m)</mi></msup><mo>,...,</mo><msub><mi>μ</mi><mi>1</mi></msub><mo>,...,</mo><msub><mi>μ</mi><mi>k</mi></msub></math>使<math><mi>J</mi></math>最小：

<math display="block"> 
<mtext>min</mtext>
<mspace width="0.5em"/>
<mi>J</mi>
<mo stretchy="false">(</mo>
<msup><mi>c</mi><mn>(1)</mn></msup><mo>,...,</mo><msup><mi>c</mi><mi>(m)</mi></msup><mo>,...,</mo><msub><mi>μ</mi><mi>1</mi></msub><mo>,...,</mo><msub><mi>μ</mi><mi>k</mi></msub>
<mo stretchy="false">)</mo>
<mo>=</mo>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
 <munderover>
  <mo>∑</mo>
  <mrow>
    <mi>i</mi>
    <mo>=</mo>
    <mn>1</mn>
  </mrow>
  <mrow>
    <mi>m</mi>
  </mrow>
</munderover>
<mo>||</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><msub><mi>μ</mi><msup><mi>c</mi><mi>(i)</mi></msup></msub><msup><mo>||</mo><mn>2</mn></msup>
</math>

- 其中
	- <math><msup><mi>c</mi><mi>(i)</mi></msup></math>:样本<math><msup><mi>x</mi><mi>(i)</mi></msup></math>被分配到某个聚类的index值，例如：<math><msup><mi>x</mi><mi>(i)</mi></msup><mo>-></mo><mn>5</mn></math>，则<math><msup><mi>c</mi><mi>(i)</mi></msup><mo>=</mo><mn>5</mn></math>
	- <math><msub><mi>μ</mi><mi>k</mi></msub></math>:聚类中心点
	- <math><msub><mi>μ</mi><msup><mi>c</mi><mi>(i)</mi></msup></msub></math>:样本<math><msup><mi>x</mi><mi>(i)</mi></msup></math>被分配到某个聚类的中心点，例如：<math><msup><mi>x</mi><mi>(i)</mi></msup><mo>-></mo><mn>5</mn></math>，<math><msup><mi>c</mi><mi>(i)</mi></msup><mo>=</mo><mn>5</mn></math>，<math><msub><mi>μ</mi><msup><mi>c</mi><mi>(i)</mi></msup></msub><mo>=</mo><msub><mi>μ</mi><mn>5</mn></msub></math>
  - J也叫做**Distortion Function**，在实际求解上，参照上一节提供的步骤运算更好理解。


前面提到了在进行K-Means运算之前要先随机初始化K个聚类中心点:<math><msub><mi>μ</mi><mn>1</mn></msub><mo>,</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>...,</mo><msub><mi>μ</mi><mi>k</mi></msub></math>（<math><mi>K</mi><mo><</mo><mi>m</mi></math>）,通常的做法是在训练样本中随机选取<math><mi>K</mi></math>个样本作为<math><msub><mi>μ</mi><mn>1</mn></msub><mo>,</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>...,</mo><msub><mi>μ</mi><mi>k</mi></msub></math>的值，即<math xmlns="http://www.w3.org/1998/Math/MathML"> <msub> <mi>&#x03BC;<!-- μ --></mi> <mn>1</mn> </msub> <mo>=</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <msub> <mi>i</mi> <mn>1</mn> </msub> <mo stretchy="false">)</mo> </mrow> </msup> <mo>,</mo> <msub> <mi>&#x03BC;<!-- μ --></mi> <mn>2</mn> </msub> <mo>=</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <msub> <mi>i</mi> <mn>2</mn> </msub> <mo stretchy="false">)</mo> </mrow> </msup> <mo>,</mo> <mo>&#x2026;<!-- … --></mo> <mo>,</mo> <msub> <mi>&#x03BC;<!-- μ --></mi> <mi>k</mi> </msub> <mo>=</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <msub> <mi>i</mi> <mi>k</mi> </msub> <mo stretchy="false">)</mo> </mrow> </msup> </math>。如果随机选取的样本点不理想，均值点很可能会落到local optima，如下图所示：

![](/images/2017/09/ml-9-5.png)

为了避免这种情况出现，我们可以尝试多次计算:

For i = 1 to 100 {

1. Randomly initialize K-means.

2. Run K-means. Get <math><msup><mi>c</mi><mn>(1)</mn></msup><mo>,...,</mo><msup><mi>c</mi><mi>(m)</mi></msup><mo>,...,</mo><msub><mi>μ</mi><mi>1</mi></msub><mo>,...,</mo><msub><mi>μ</mi><mi>k</mi></msub></math>

3. Compute cost function(distortion) <math><mi>J</mi> <mo stretchy="false">(</mo> <msup><mi>c</mi><mn>(1)</mn></msup><mo>,...,</mo><msup><mi>c</mi><mi>(m)</mi></msup><mo>,...,</mo><msub><mi>μ</mi><mi>1</mi></msub><mo>,...,</mo><msub><mi>μ</mi><mi>k</mi></msub> <mo stretchy="false">)</mo> </math>
 
}

4 . Pick clustering that gave lowest cost <math><mi>J</mi> <mo stretchy="false">(</mo> <msup><mi>c</mi><mn>(1)</mn></msup><mo>,...,</mo><msup><mi>c</mi><mi>(m)</mi></msup><mo>,...,</mo><msub><mi>μ</mi><mi>1</mi></msub><mo>,...,</mo><msub><mi>μ</mi><mi>k</mi></msub> <mo stretchy="false">)</mo> </math>


### 维数约减 Dimensionality Reduction

对于多维度的样本数据我们希望可以将其缩减到低维度，这样可以对数据进行可视化处理，便于观察和理解数据。例如将二维数据映射到一维，3维数据映射到二维，映射过程主要是通过投影完成

![](/images/2017/09/ml-9-6.png)

![](/images/2017/09/ml-9-7.png)

一种常用的降维算法叫做主成分分析**PCA**( [Principal Compoent Analysis](https://zh.wikipedia.org/wiki/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90))，它可以将<math><mi>n</mi></math>维数据映射成<math><mi>k</mi></math>维数据。具体来说，是找到<math><mi>k</mi></math>个向量<math><msup><mi>u</mi><mn>(1)</mn></msup><mo>,</mo><msup><mi>u</mi><mn>(2)</mn></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>u</mi><mi>(k)</mi></msup></math>，使各个数据点在该向量上的投影误差（距离）最小。通俗的说，PCA就是尝试寻找一个低维平面将高维度数据投影到这个平面，且各个点到该平面的垂直距离最短（误差最小）

![](/images/2017/09/ml-9-8.png)

在使用PCA之前，通常要对数据进行预处理，使样本保持在统一数量级上。假设有训练样本：<math><mo>{</mo><msup><mi>x</mi><mi>(1)</mi></msup><msup><mi>x</mi><mi>(2)</mi></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>x</mi><mi>(m)</mi></msup><mo>}</mo></math>，对其进行feature scaling(mean normalization):

<math display="block">
<msub><mi>μ</mi><mi>j</mi></msub>
<mo>=</mo>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> 
<msubsup><mi>x</mi><mi>j</mi><mi>(i)</mi></msubsup>
</math>
<br>
<math display="block"><mtext>Replace each </mtext><mspace width="0.5em"/><msubsup><mi>x</mi><mi>j</mi><mi>(i)</mi></msubsup><mtext>with</mtext><mspace width="0.5em"/><msub><mi>x</mi><mi>j</mi></msub><mo>-</mo><msub><mi>μ</mi><mi>j</mi></msub></math>

接下来要解决两个问题，一是怎么计算投影平面<math><mo>[</mo><msup><mi>u</mi><mn>(1)</mn></msup><mo>,</mo><msup><mi>u</mi><mn>(2)</mn></msup><mo>,</mo><mo>...</mo><mo>,</mo><msup><mi>u</mi><mi>(k)</mi></msup><mo>]</mo></math>，而是怎么计算平面中的点

- 我们定义投影平面为<math><msub><mi>U</mi><mtext>reduce</mtext></msub></math>，维度为：nxk
- 平面中的点构成的矩阵为<math><mi>Z</mi></math>维度为kx1

第一步先计算协方差矩阵得到sigma矩阵<math><mo>∑</mo></math>

<math display="block">
<mo>∑</mo>
<mo>=</mo>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover> 
<mo stretchy="false">(</mo>
<msup><mi>x</mi><mi>(i)</mi></msup>
<mo stretchy="false">)</mo>
<mo stretchy="false">(</mo>
<msup><mi>x</mi><mi>(i)</mi></msup>
<msup><mo stretchy="false">)</mo><mi>T</mi></msup>
</math> 

向量化实现为:

<math display="block">
<mtext>Sigma</mtext>
<mo>=</mo>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
<mo> * </mo>
<msup><mi>X</mi><mi>T</mi></msup>
<mo> * </mo>
<mi>X</mi>
</math>

其中<math><msup><mi>x</mi><mi>(i)</mi></msup></math>为<math><mi>n</mi><mo> * </mo><mn>1</mn></math>矩阵，因此<math><mo>∑</mo></math> 为 <math><mi>n</mi><mo>*</mo><mn>n</mn></math>。

第二步计算<math><mo>∑</mo></math>的特征值和特征向量，Octave可直接使用svd函数对矩阵进行奇异值分解:

```matlab

[U,S,V] = svd(Sigma)

```

在返回的三个结果中，<math><mi>U</mi></math>为nxn矩阵，我们从中取出前<math><mi>k</mi></math>列，得到<math><msub><mi>U</mi><mtext>reduce</mtext></msub></math>矩阵，既是我们想要的投影平面

第三步计算<math><mi>Z</mi></math>矩阵：

<math display="block">
	<mi>Z</mi>
	<mo>=</mo>
	<msubsup><mi>U</mi><mtext>reduce</mtext><mi>T</mi></msubsup>
	<mo>*</mo>
	<mi>X</mi>
</math>

其中，<math><msubsup><mi>U</mi><mtext>reduce</mtext><mi>T</mi></msubsup></math>为kxn的矩阵，x为nx1的矩阵，因此<math><mi>Z</mi></math>为kx1的矩阵 

总结一下上述过程的Octave实现为:

```matlab

Sigma = (1/m) * X'*X;
[U,S,V] = svd(Sigma);

Z = zeros(size(X, 1), K);

U_reduce = U(:, 1:K); 
for i=1:size(X,1)
    x = X(i, :)';
    Z(i,:) = x' * U_reduce;
end 

```

> 这部分涉及到一些的数学推导，需要理解矩阵的特征值，特征向量，奇异值分解以及协方差矩阵的物理意义等等，此处暂时忽略这些概念，把PCA当做黑盒来看待

当我们有了低维度的<math><mi>Z</mi></math>，如何还原出高维度的<math><mi>X</mi></math>，可通过如下式子：

<math display="block">
	<msub><mi>X</mi><mtext>approx</mtext></msub>
	<mo>=</mo>
	<msubsup><mi>U</mi><mtext>reduce</mtext><mi>T</mi></msubsup>
	<mo>*</mo>
	<mi>Z</mi>
</math>

这里，<math><msubsup><mi>U</mi><mtext>reduce</mtext><mi>T</mi></msubsup></math>是nxk的，<math><mi>Z</mi></math>是kx1的，<math><msub><mi>X</mi><mtext>approx</mtext></msub></math>是nx1的, matlab实现为:

```matlab

function X_rec = recoverData(Z, U, K)

	X_rec = zeros(size(Z, 1), size(U, 1));
	for i=1:size(Z,1)
	    v=Z(i,:)';
	    X_rec(i,:)=v' * U(:, 1:K)';
	end 
end 

```



![](/images/2017/09/ml-9-9.png)

接下来的问题是如何选择<math><mi>k</mi></math>值，使用下面公式：

<math display="block">
<mfrac>
<mrow>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover>
<mo>||</mo><msup><mi>x</mi><mi>(i)</mi></msup><mo>-</mo><msubsup><mi>x</mi><mtext>approx</mtext><mi>(i)</mi></msubsup><msup><mo>||</mo><mn>2</mn></msup>
</mrow>
<mrow>
<mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac>
<munderover> <mo>∑</mo> <mrow> <mi>i</mi> <mo>=</mo> <mn>1</mn> </mrow> <mrow> <mi>m</mi> </mrow> </munderover>
<mo>||</mo><msup><mi>x</mi><mi>(i)</mi></msup><msup><mo>||</mo><mn>2</mn></msup>
</mrow>
</mfrac>
<mo><=</mo>
<mn>0.01</mn>
<mspace width='1em' />
<mo stretchy="false">(</mo>
<mn>1%</mn>
<mo stretchy="false">)</mo>
</math>

其中，分子为样本平均投影误差的平方和，分母为样本数据的平均方差和，选取最小的<math><mi>k</mi></math>值，使比值小于等于0.01。这个式子的意思是“在保留了99%的样本差异性的前提下，选择了K个主成分”，具体的做法如下：

- 选取<math><mi>k</mi><mo>=</mo><mn>1</mn></math>
- 计算<math><msub><mi>U</mi><mtext>reduce</mtext></msub><mo>,</mo><msup><mi>z</mi><mi>(1)</mi></msup><mo>,</mo><msup><mi>z</mi><mi>(1)</mi></msup><mo>...</mo><mo>,</mo><msup><mi>z</mi><mi>(m)</mi></msup><mo>,</mo><msubsup><mi>x</mi><mtext>approx</mtext><mi>(1)</mi></msubsup><mo>,</mo><mo>...</mo><mo>,</mo><msubsup><mi>x</mi><mtext>approx</mtext><mi>(m)</mi></msubsup></math>
- 检查上述式子结果是否满足条件，如果不满足，重复第一步令<math><mi>k</mi><mo>=</mo><mi>k</mi><mo>+</mo><mn>1</mn></math>

如果使用Octave，在`[U,S,V] = svd(Sigma);` 的返回值中，`S`矩阵是一个对角阵，我们可以用这个矩阵简化对<math><mi>k</mi></math>值的计算：对于给定的任意<math><mi>k</mi></math>值，计算

<math display="block">
  <mstyle displaystyle="true">
    <mfrac>
      <mrow>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>k</mi>
        </munderover>
        <msub>
          <mi>S</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mi>i</mi>
          </mrow>
        </msub>
      </mrow>
      <mrow>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>n</mi>
        </munderover>
        <msub>
          <mi>S</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mi>i</mi>
          </mrow>
        </msub>
      </mrow>
    </mfrac>
  </mstyle>
  <mo>>=</mo>
<mn>0.99</mn>
</math>

通常情况下选取<math><mi>k</mi><mo>=</mo><mn>1</mn></math>，缓慢增大<math><mi>k</mi></math>值，观察上面式子计算结果。这种方式的好处是只需要计算一次`svd`得到<math><mi>S</mi></math>即可，比较高效