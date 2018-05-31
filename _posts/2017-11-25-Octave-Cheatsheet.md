---
layout: default
list_title: Octave Cheatsheet
categories: [pl,cheatsheet,octave]
meta: Octave 简明操作
---

## Octave Cheat Sheet

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
	- `close`关掉图片窗口，`figure(1)`打开一个窗口，`figure(2)`同时打开两个窗口
	- `subplot(1,2,1)`将窗口切分成1x2个并占用第一个，`subplot(1,2,2)` 将窗口切分成1x2个并占用第2个
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

### Resource

- [Octave Doc](https://octave.org/doc/)
- [Octave 例子参考](https://octave.sourceforge.io/octave/overview.html)