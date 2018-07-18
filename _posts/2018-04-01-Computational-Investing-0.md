---
layout: post
list_title: 计算投资（一）| Computational Investment Part 0
title: Course Overview
mathjax: true
---

> Online course from Geogia Tech

## Overview

- Course Objectives
    - Understand electronic markets
    - Understand market data
    - Write software to visualize
    - Write software to discover
    - Create market simulator
- Prerequisites
    - Advanced Programming (Python)
    - Fundamental concepts of investing, financial markets
- Course Logistics
    - 8 weeks, 2 modules per week
    - Projects in Excel and Python


### Quick Hands on Demo

In this demo we are going to 

```python
import panda as pd

def 

```



## Course Resource

- [wiki](http://wiki.quantsoftware.org/index.php?title=QuantSoftware_ToolKit)

- [Active Portfolio Management](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826/ref=sr_1_1?ie=UTF8&s=books&qid=1263182044&sr=1-1)
- [All about Hedge Funds](https://www.amazon.com/All-About-Hedge-Funds-Started/dp/0071393935)
- [Applied Quantitative Methods for Trading and Investment](https://www.amazon.com/Applied-Quantitative-Methods-Trading-Investment/dp/0470848855/ref=sr_1_1?ie=UTF8&s=books&qid=1263181752&sr=8-1)

- [Other Resources](https://www.coursera.org/learn/computational-investing/supplement/TPxSD/course-resources))


## 附：Install QSTK on MacOS

> [官方指南](https://github.com/QuantSoftware/QuantSoftwareToolkit/wiki/Mac-Installation) 有些步骤已经过时，这里整理一份暂时可用的，测试通过了`\Examples\Basics\`下的所有`tutorial.py`，<mark>环境为Python2.7</mark>

- Install Pip

```
brew install python
pip install --upgrade setuptools
pip install --upgrade pip
```

- Install virtualenv

```
pip install nose
pip install virtualenv
```

- Install Numpy, Scipy and Matplotlib

这些库依赖`gfortran`，后者在brew库中已经被移到`gcc`，因此这里`brew install gfortran`会失败，可以用`gcc`替代:

```
brew install gcc
```
如果在执行`make`时卡住，尝试更新XCode插件`sudo xcode-select --install`。
安装GCC成功后，还需要先更新home brew的两个Science库:

```
brew tap brewsci/homebrew-science
brew tap brewsci/bio
```

然后安装 Numpy, Scipy and Matplotlib
```
brew install numpy
brew install scipy
brew install matplotlib
```

- 创建一个QSTK的测试目录

```
mkdir ~/QSTK
cd ~/QSTK
```

- 使用virtualenv进行环境隔离

```
virtualenv env --distribute --system-site-packages
source ~/QSTK/env/bin/activate
```
`activate`之后的安装都将与全局环境隔离，这里要先check一下`env/lib/`下的python版本， 如果误使用了python3，要还原回来，需要重新指定python版本

```
virtualenv --python=/usr/bin/python2.7 ~/QSTK/env
```

- 安装QSTK及其依赖

```
pip install pandas
pip install scikits.statsmodels
pip install scikit-learn
pip install cvxopt
pip install QSTK
```
创建matplotlib配置文件:`echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc`

- 测试QSTK demo

```
curl -O https://spark-public.s3.amazonaws.com/compinvesting1/QSTK-Setups/Examples.zip
unzip Examples.zip
```
使用`Examples`目录中的`Validation.py`进行测试，如果发现`TimeSeries`类找不到：

```
2.7.14_3/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/QSTK/qstkutil/qsdateutil.py", line 36, in _cache_dates
    return pd.TimeSeries(index=dates, data=dates)
AttributeError: 'module' object has no attribute 'TimeSeries'
```
原因是`panda`版本过新，还原到0.7.3版本

```
pip install pandas==0.7.3
```
- 测试`Basic`下的一系列`tutorial.py`

```
python tutorial1.py
```
如果发现

```
numpy TypeError: The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.
```
则表示numpy的版本不对，具体是哪个版本还不清楚，暂时的解法是按照提示去出错的地方，修改代码。上述步骤完成后，测试`tutorial1.py`，成功后可发现`Basics`目录下生成了多个`pdf`文件

- 退出`virtutalen`环境

```
/bin/deactive
```

- 卸载QSTK

```
rm -rf ~/QSTK
```