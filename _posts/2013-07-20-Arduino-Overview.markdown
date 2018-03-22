---
title: Arduino 101
layout: post
tag: Arduino
categories: 随笔

---

<em>所有文章均为作者原创，转载请注明出处</em>

想象一下，有一个小车，上面绑上摄像头，可以用iPhone控制，实时采集视频，发送到手机上。类似[这种](http://blog.miguelgrinberg.com/post/building-an-arduino-robot-part-i-hardware-components)。基本上采用Arduino和它上面一些扩展的shield就能实现。在淘宝上买了Arduino，摆弄了一阵，开始还以为和c51，DSP一样，需要一堆堆的汇编呢，上手发现arduino居然提供api，IDE也很友好。动手写了一些demo，复习了一些数电模电的知识，还想了半天面包板该怎么插，想通了之后，后面就很容易了，但烧坏了一支LED。


##Arduino 101

<a href="/assets/images/2013/07/arduino-robot-11.png"><img src="/assets/images/2013/07/arduino-robot-11.png" alt="arduino-robot-11" width="360" height="260" class="alignnone size-full wp-image-1026" /></a>

- 电源：左边灰色的部分是USB口，调试程序的时候可以顺便用来供电，下面黑色口是独立的电源input。

- 数字信号管脚：在最上面从又到左有14个输入/输出引脚，由于传输数字信号，只有高电平和低电平两个值。有些管脚是具有特殊功能的，0，1管脚被标识为RX,TX，代表这两个管脚是串口通信用的。13管脚一般会接一个LED。3，5，6，9，10，11被标有"pwm"或者"~"，意思是<a href="http://zh.wikipedia.org/zh-cn/%E8%84%88%E8%A1%9D%E5%AF%AC%E5%BA%A6%E8%AA%BF%E8%AE%8A">脉冲宽度调制</a>，他可以通过输入的数字新号来产生输出的模拟新号。剩下的2，4，7，8，12可以被自由使用。

- GND：地线，

- AREF：很少用，用来告诉Arduino模拟新号电压的范围

- 模拟信号管脚：底部从右到左有六个管脚0~5，这些是模拟信号输入管脚，不同的电压会产生不同的模拟信号值，电压范围可以通过AREF管脚控制。有一点需要注意，这些模拟信号的管脚同样可以作为数字信号的管脚使用，尤其当数字信号管脚不够用的时候，可以把它们当做14-19号管脚使用。

- VIN：Arduino板子的电压值。可以通过这个管脚读取当前的电压值，例如，如果Arduino接上9v的外部电源，这个管脚的值为9

- GND：地线

- 5V and 3,3V ：不论当前电压为多少，这两个管脚始终返回5v，3.3v 

- RESET：当这个管脚接地的时候，Arduino会被reset，可以用这个管脚做一个reset button



##Programming in Arduino

Arduino的开发很简单，推荐大家在Mac或者Linux上开发，因为不需要安装驱动，如果在windows上开发，需要参照这个连接来<a href="http://arduino.cc/en/Guide/windows#.UwmxNUKSwVI">安装驱动</a>。

假设你现在可以成功运行IDE并且能找到你的Arduino了，现在我们来讨论如何编写Arduino代码，这可能会涉及到一些电子的知识，首先Arduino芯片是可编程逻辑器件，你可以写好一段程序，把它烧到这个器件中，然后Arduino就会按照你程序的逻辑工作。如果你有过单片机的开发经验，相信这些不会太陌生，唯一不同的是，你不再需要使用汇编了，<a href="http://arduino.cc/en/Reference/HomePage#.Uwmzr0KSwVI">Arduino为你提供了一些简单的C/C++的API</a>，这也是我喜欢它的地方，实际上许多情况下，我们不需要很多复杂的API，太多的API会导致开发起来很疑惑，你总是在想我这个API用的对不对？干这个事情是不是该用其它的API？少量简单的API可以消除这些疑惑并能让你很快的上手开发Arduino程序。

此外，Arduino还为你准备了一个IDE:

<a href="/assets/images/2013/07/arduino-ide2.png"><img src="/assets/images/2013/07/arduino-ide2.png" alt="arduino-ide" width="454" height="440" class="alignnone size-full wp-image-1037" /></a>

Arduino程序包含两个函数——setup()和loop()。

上面是通过Arduino点亮LED的demo。

Setup()函数中所写的程序只会在程序的最开始执行一次，用来做一些初始化工作。

loop()函数中的代码会一遍遍重复执行。Arduino程序不允许同时执行多个函数，也没有退出和关闭的功能。

整个程序的开始和停止取决于Arduino芯片的电源开关。

API：

```

//////////////////////////////
//  setup()中调用
//////////////////////////////

指定引脚的输入输出模式：

pinMode(int number, int MODE)          // MODE : OUTPUT/INPUT

OUTPUT : 只可以向该管脚进行写操作

INPUT    :  只可以向该管脚进行读操作

```


```

//////////////////////////////
//  loop()中调用
//////////////////////////////

1，数字信号管脚：2，4，7，8，12，13

digitalRead（int number）                        ：读取某个管脚上的电平（0：low，1：high）

digitalWrite (int number, int LEVEL)           :  向某个管脚写电瓶（0：low，1：high）

2，模拟信号管脚：3，5，6，9，10，11

analogRead（int number）                         : 读取某个管脚的值（0-255）

analogWrite（int number, int LEVEL）         : 向某个管脚写入值（0-255）

3，串行通信：

让Arduino与PC通信 ： serial()函数指令集

Serial.begin( 9600 );      9600为波特率

Serial.println(val);

```

<a href="/assets/images/2013/11/arduino.png"><img src="/assets/images/2013/11/arduino.png" alt="arduino" width="353" height="260" class="alignnone size-full wp-image-239" /></a>