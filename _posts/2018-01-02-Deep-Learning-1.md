---
updated: '2017-11-30'
list_title: 深度学习 | Logistic Regression as a Neural Network
title: Logistic Regression as a Neural Network
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

> 部分图片截取自课程视频[Nerual Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)

## Logistic Regression Recap

前面机器学习的课程中，我们曾[介绍过Logistic Regression的概念](https://xta0.me/2017/09/27/Machine-Learning-3.html)，它主要用来解决分类的问题，尤其是True和False的分类。我们可以将逻辑回归模型理解为一种简单的分类神经网络，`F(inputs) = {0 or 1}`，我们假设预测函数为线性函数：

$$
\hat{y} = \sigma(w^Tx + b)
$$

其中$\hat{y}$, $x$, $w$ 和 $b$ 皆为矩阵，$\sigma$为sigmoid函数，定义为$\sigma(z) = \frac{1}{1+e^{-z}}$。我们假设第$i$组训练样本用向量$x^{(i)}$表示，则$x^{(i)}$是$nx1$的（注意是n行1列），那么假设一共有$m$组训练样本，则样本矩阵$X$表示为

$$
X= 
\begin{bmatrix}
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
x^{(1)} & x^{(2)} & x^{(3)} & . & . & . & x^{(m)} \\
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
\end{bmatrix}
$$

类似的$w$是一个`nx1`的向量，则$w^Tx$是`1xm`的, 对应的常数项$b$也是`1xm`的矩阵。对任意第$i$个训练样本，有

$$
z^{(i)} = w^Tx^{(i)} + b 
$$

对所有训练集则可以使用矩阵运算来表示

$$
\begin{bmatrix}
z^{(1)} & z^{(2)} & . & . & . &z^{(m)}  
\end{bmatrix}
= w^{T}X + 
\begin{bmatrix}
b_1 & b_2 & . & . & . &b_n
\end{bmatrix}
$$

对任意训练集$i$，另$a^{(i)} = \sigma(z^{(i)})$，则最后的预测结果$\hat{y}$可以表示为

$$
\hat{y} = 
\begin{bmatrix}
a^{(1)} & a^{(2)} & . & . & . &a^{(m)}  
\end{bmatrix}
$$

因此预测结果$\hat{y}$为`1xm`的向量

### Cost function

对于某一组训练集可知其Loss函数为

$$
L(\hat(y),y) = - (y\log{\hat{y}}) + (1-y)\log{(1-\hat{y})} 
$$

然后我们对所有$m$组训练集都计算Loss函数，之后再求平均，则可以得到Cost function

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum_{i=1}^{m}[(y^{(i)}\log{\hat{y}^{(i)}}) + (1-y^{(i)})\log{(1-\hat{y}^{(i)})} ]
$$

## Gradient Descent

有了Cost funciton之后，我们就可以使用梯度下降来求解$w$和$b$，使$J(w,b)$最小。梯度下降的计算方式如下

$$
w := w - \alpha\frac{dJ(w,b)}{dw} \\
b := b - \alpha\frac{dJ(w,b)}{db} 
$$

上述式子通过不断的对$w$和$b$进行求偏导，最终使其收敛为一个稳定的值，其中$\alpha$为Learning Rate,用来控制梯度下降的幅度。在后面的Python代码中，使用`dw`表示 $\frac{dJ(w,b)}{dw}$，`db`表示$\frac{dJ(w,b)}{db}$，以此类推。

### Computation Graph

虽然我们有了上面的算式，但如何有效的计算它是我们接下来要讨论的问题，这里我们介绍一种使用**Computation Graph**的思路，所谓的Computation Graph的概念，基本思想是将每一步运算都用一个节点表示，然后将这些节点串联起来得到一个Graph，举例来说，假设有一个函数为

$$
J(a,b,c) = 3(a+bc)
$$

我们另

- $u = bc$
- $v = a+u$
- $J = 3v$

则该算式的Computation Graph可以表示如下

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-1.png" class="md-img-center">

接下来我们要思考如何对Graph中的每一项进行求导，这将是后面计算神经网络backpropagation的基础。显然如果有微积分基础的话，这并不难

- $\frac{dJ}{dv} = 3$
- $\frac{dJ}{du} = \frac{dJ}{dv} \times \frac{dv}{du} = 3 \times 1 = 3$
- $\frac{dJ}{da} = \frac{dJ}{dv} \times \frac{dv}{da} = 3 \times 1 = 3$
- $\frac{dJ}{db} = \frac{dJ}{dv} \times \frac{dv}{du} \times \frac{du}{db} = 3 \times 1 \times c = 3c$
- $\frac{dJ}{dc} = \frac{dJ}{dv} \times \frac{dv}{du} \times \frac{du}{dc} = 3 \times 1 \times b = 3b$

在接下来的代码中，我们需要表示上面的每个导数值，其表示方式为

$$
\frac{dFinalOutputVar}{dvar}
$$

这种写法太过冗余，因此，如果想表示$\frac{dJ}{da}$，在代码中可直接写成`da`，其余同理，计算这些变量到数值的过程，可类比于神经网络的backpropagation

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-2.png" class="md-img-center">

### Loss Function

理解了Computation Graph，我们回到算梯度下降上来，我们先看一种简单的情况，假设我们只有一组训练样本，该样本中只有两个feature，$x_1$和$x_2$，我们用$w_1$和$w_2$表示两个feature对应的权重，则预测的model可以表示为

$$
\hat{y} = \sigma(z) = \sigma(w_1x_1 + w_2x_2 + b) \\
$$

为了求解$w_1$和$w_2$，结合上面给出的Loss函数，得到生成的Computation Graph如下

> 注意这里使用的是Loss函数，而不是Cost函数，因为当前case为单一的样本，不涉及到所有的样本集

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-3.png" class="md-img-center">

为了使Loss函数的值最小，我们需要使用梯度下降来计算$w_1$,$w_2$和$b$

$$
w_1 := w_1 - \alpha\frac{dL(a,y)}{dw_1} \\
w_2 := w_2 - \alpha\frac{dL(a,y)}{dw_2} \\
b := b - \alpha\frac{dL(a,y)}{db} 
$$

接下来利用前面提到的求偏导的方式，一步步反向计算得到 $w_1$ 和 $w_2$的最终值，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-4.png" class="md-img-center">

- $da = \frac {dL(a,y)} {da} = - \frac{y}{a} + \frac{1-y}{1-a}$
- $dz = \frac {dL(a,y)} {dz} = \frac {dL(a,y)}{da} \times \frac {da}{dz} = a-y$
- $dw_1 = \frac {dL(a,y)} {dw_1} = x_1 \times dz$ 
- $dw_2 = \frac {dL(a,y)} {dw_2} = x_2 \times dz$

### Cost Function

接着我们可以考虑使用上述方法来计算逻辑回归的cost function，如前所述，对于任意一组的训练集，我们用$x^{(i)}$表示第i个样本， 每个样本$x^{(i)}$包含$n$个feature，则$x^{(i)}$是`n x 1`的，每组样本的预测结果用$\hat{y}^{(i)}$或$a^{(i)}$表示，假设整个训练集有$m$组样本，则对于cost function可以表示为

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(a^{(i)}, y) \\
\hat{y}^{(i)} = a^{(i)} = \sigma(z^{(i)}) = \sigma(w^tx^{(i)} + b)
$$

可以看到，cost函数只是loss函数的平均值，现在我们假设$n=2$，则每组样本都有两个feature，对应的$w^{(i)}$是$2\times1$的，即`[w1,w2]`，因此对$dw_1^{(i)}$的计算只需要循环$m$次累加$\frac{d(a^{(i)}, y^{(i)})}{dw_1}$，然后求平均即可，$dw_2^{(i)}$同理

$$
\frac{dJ(w,b)}{dw_1} = \frac{1}{m}\sum_1^{m}\frac{d(a^{(i)}, y^{(i)})}{dw_1}
$$

伪代码如下

```python
J=0, dw1=0, dw2=0, db=0
for i=1 to m 
    z[i] = w.tx[i] + b
    a[i] = sigmoid(z[i])
    J += -y[i]*log(a[i]) + (1-y[i])log(1-a[i])
    dz[i] = a[i] - y[i]
    for j=1 to n
        dw[j] += x[i][j] * dz[i] #第i组样本的第j个feature
    db += dz[i]
        # if n = 2
            #dw1 += x[i][1] * dz[i]
            #dw2 += x[i][2] * dz[i]
dw1 = dw1 / m
dw2 = dw2 / m
db  = db / m

w1 = w1 - alpha*dw1
w2 = w2 - alpha*dw2
b  = b - alpha*db
```
上面代码展示了某一次梯度下降的计算过程

## Vectorization 

由于深度学习涉及大量的矩阵间的数值计算，而且数据量有很大，使用`for`计算时间成本太高。Vectorization是用来取代for循环的一种针对矩阵数值计算的计算方式，其底层可以通过GPU或者CPU(SIMD)的并行指令集来实现。不少数值计算库都有相应的实现，比如Python的Numpy，C++的Eigen等。

比如我们想要计算$z = w^Tx+b$，我们假设$x$和$w$都是`nx1`的向量，我们使用Python来对比下两种计算方式的差别

```python
#for loop
z = 0
for i in range(1,n):
    z += w[i] * x[i]
z+=b

# use numpy
# vectorized version of doing 
# matrix multiplications
z = np.dot(w.T,x)+b 
```
numpy数组的另一个特点是可以做element-wise的矩阵运算，这样让我们避开了for循环的使用

```python
a = np.ones([1,2])  #[1,1]
a = a*2 #[2,2]
```
接下来我们可以使用numpy重新实现以下上一节计算`dw`代码的for循环部分

```python
J=0,db=0
dw = np.zeros([n,1]) 
for i=1 to m 
    #z是1xm的
    #x是nxm的
    z[i] = w.tx[i] + b
    a[i] = sigmoid(z[i])
    J += -y[i]*log(a[i]) + (1-y[i])log(1-a[i])
    dz[i] = a[i] - y[i]
    # for j=1 to n
        # dw[j] += x[i][j] * dz[i] #第i组样本的第j个feature
    dw += x[i]*dz[i]
    db += dz[i]
dw = dw/m
```
### Compute Forward Propagation

回到前面第一节计算矩阵$Z$的式子上

$$
Z=
\begin{bmatrix}
z^{(1)} & z^{(2)} & . & . & . &z^{(m)}  
\end{bmatrix}
= w^{T}X + 
\begin{bmatrix}
b_1 & b_2 & . & . & . &b_n
\end{bmatrix}
$$

如果使用numpy表示，则只需要一行代码

```python
Z = np.dot(w.T,X) + b #b is a 1x1 number
```
### Compute Backwward Propagation

在通过Loss函数计算w值时，我们曾给出过下面式子

$$
dz^{(1)} = a^{(1)} - y^{(1)} \\
dz^{(2)} = a^{(2)} - y^{(2)} \\
... \\
dz^{(i)} = a^{(i)} - y^{(i)} \\
$$

训练集共有$m$，则$dz$的矩阵(`1xm`)表示为

$$
dZ = [dz^{(1)}, dz^{(2)}, ... , dz^{(m)}]
$$

另 $A = [a^{(i)}...a^{(m)}]$, $Y = [y^{(i)}...y^{(m)}]$，则

$$
dZ = A - Y = [a^{(1)} - y^{(1)}, a^{(2)} - y^{(2)}, ... , a^{(m)} - y^{(m)}]
$$

在前面求解$dw$的代码中，我们虽然将$dw$向量化后减少了一重循环，但最外层还有一个`[1,m]`的for循环，接下来我们的任务是将这个for循环也向量化。

我们的目的是求解$dw$和$d$`，其中$db$为

$$
db = \frac{1}{m}\sum_{i=1}^{m}dz^{(i)}
$$

上述式子可以用numpy一行表示 `db = 1/m * np.sum(dZ)`，对于$dw$，有

$$
dw = \frac{1}{m}XdZ^T \\
= \frac{1}{m}
\begin{bmatrix}
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
x^{(1)} & x^{(2)} & x^{(3)} & . & . & . & x^{(m)} \\
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
\end{bmatrix}
\begin{bmatrix}
dz^{(1)} \\
. \\
. \\
. \\
dz^{(m)}
\end{bmatrix}
= \frac{1}{m}[x^{(1)}dz^{(1)},..., x^{(m)}dz^{(m)}]
$$

综上，一个简单的逻辑回归神经网络Python实现如下

```python
# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- number of features in a given example
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros([dim,1]) # dimx1
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    return s

# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (features, #examples)
    Y -- true "label" vector of size (1, #examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X)+b) # compute activation
    cost = - 1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))  #compute cost
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (features, #examples)
    Y -- true "label" vector of shape (1, #examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    costs = []
    for i in range(num_iterations):
        # Run propagation
        grads, cost = propagate(w,b,X,Y)        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule 
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (features, #examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X)+b)    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0][i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    #1. initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    #2. Gradient descent
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations, learning_rate, print_cost)
    
    #3. Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    #4. Predict test/train set examples
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    #5. Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#Run the following cell to train your model.
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# result
# Cost after iteration 0: 0.693147
# Cost after iteration 100: 0.584508
# Cost after iteration 200: 0.466949
# Cost after iteration 300: 0.376007
# Cost after iteration 400: 0.331463
# Cost after iteration 500: 0.303273
# Cost after iteration 600: 0.279880
# Cost after iteration 700: 0.260042
# Cost after iteration 800: 0.242941
# Cost after iteration 900: 0.228004
# Cost after iteration 1000: 0.214820
# Cost after iteration 1100: 0.203078
# Cost after iteration 1200: 0.192544
# Cost after iteration 1300: 0.183033
# Cost after iteration 1400: 0.174399
# Cost after iteration 1500: 0.166521
# Cost after iteration 1600: 0.159305
# Cost after iteration 1700: 0.152667
# Cost after iteration 1800: 0.146542
# Cost after iteration 1900: 0.140872
# train accuracy: 99.04306220095694 %
# test accuracy: 70.0 %
```
我们可以观察一下cost值得变化情况

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-5.png">