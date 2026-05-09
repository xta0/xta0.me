---
layout: post
list_title: 计算机图形学 | Computer Graphics | 数学基础
title: 计算机图形学
mathjax: true
categories: [Computer Graphics, OpenGL]
---

### Course Overview

- Systems: Write complex 3D graphics programs
	- Real-time scene viewer in OpenGL,GL Shading Language,  offline raytracer
- Theory: Mathematical aspects and algotrithms underlying modern 3D graphics systems

- Homework
	- HW1: **Transformations**. Place objects in world, view them simple viewer for a teapot
	- HW2: **Scene Viewer**. View scene, Lighting and Shading ( with GLSL programming shaders )
	- HW3: **Ray Tracer**. Realistic images with ray tracing. (two basic approaches: rasterize and raytrace image)

- Workload
	- 3 programming projects, time consuming
	- Course involve understanding of mathematical, geometrical concepts tought
	- Prerequisites: Solid C/C++/Java programming
	- Linear algebra and basic math skills

- GPU programming
	- Modern 3D Graphics Programming with GPUs
	- GLSL + Programmable Shaders in HW0,1,2
	- Should be very portable, but need to setup your envrionment, compliation framework



## Overview and History of Computer Graphics

- Computer Graphics
	- Coined by William Fetter of Boeing in 1960
	- First graphic system in mid 1950s USAF SAGE radar data (developed MIT)

- Text
	- Text itself is major development in computer graphics
	- In Alan Turing's biography, text is missed out.Manchester Mark I uses LED to display what happened,not text.
- GUI
	- Xerox Star
		- Invented at Palo Alto Research Center in 1970's , around 1975
		- Used in Apple Machintosh
	- Windows
- Drawing
	- Sketchpad (suthrland , MIT 1963)
- Paint Systems
	- SuperPaint System: Richard Shoup, Alvy Ray Smith (PARC, 1973 - 79)
	- Precursor to Photoshop: general image processing
- Image PRocessing
	- Filter, Crop, Scale, Composite
	- Add or remove objects
- Modeling
	- Spline curves, surfaces: 70s - 80s
	- Utah teapot: Famous 3D model
	- More recetly: Triangle meshes often acquired from real objects

- Rendering
	- 1960s (visibiliy)
		- Hidden Line Algorithms: Roberts(63),Appel(67)
		- Hidden Surface Algorithms: Roberts(63),Appel(67)
		- Visibllity = Sorting: Sutherland(74)
	- 1970s(lighting)
		- Diffuse Lighting: (Gouraud 1971)
		- Specular Lighting: (Phong 1974)
		- Curved Surfaces,Texture: (Blinn 1974)
		- Z-Buffer Hidden Surface (Catmull 1974)
	- 1980s,90s (Global illumination)
		- Ray Tracing - Whitted(1980)
		- Radiosity - Goral, Torrance et al(1984)
		- The Rendering equation - Kajiya(1986)


## Basic Math

- Vectors
- Matrix

### Vectors

#### 点积（Dot Product）

假设二维向量$a=[x_{a},y_{a}]$ 和 $b=[x_{b},y_{b}]$

- 点积的**代数表达**为：


$$
a·b=x_{a} x_{b}+y_{b} y_{b}
$$


由上式可知，点积的结果是标量（Scalar），无方向

- 假设向量$a$, $b$间的夹角为θ, 点积的**几何表达**为：


$$
a·b=|a||b|cos \theta
$$

<br>

$$
\theta=\text{arccos}(\frac{a·b}{|a||b|})
$$


- 点积的几何意义：
	- 计算向量$a$, $b$间的夹角，判断是否是同一方向以及是否正交
		- $a·b>0$，同向，夹角在0-90之间
		- $a·b=0$，正交，互相垂直
		- $a·b<0$，反向，夹角在90-180之间
	- 向量 $b$在向量$a$上的投影长度 再乘以向量$a$的长度。
		- 向量 $b$在向量$a$上的投影长度表示为:$\|b→a \|$
		- $\|b→a \|=\|b \|cos \theta=\frac{a·b}{\|a \|}$
		- $b→a=\|b→a \|\frac{a}{\|a \|}(\text{unit vector})=\frac{a·b}{\|a \|^{2}} a$

#### 叉积（Cross Product）

假设三维向量$a=[x_{a},y_{a},z_{a}]$ 和 $b=[x_{b},y_{b},z_{b}]$

- 叉积的**代数表达**为：


$$
a x b=|\begin{bmatrix}i & j & k \\ x_{a} & y_{a} & z_{a} \\ x_{b} & y_{b} & z_{b}\end{bmatrix}|=(y_{a} z_{b}-z_{a} y_{b}) i+(z_{a} x_{b}-x_{a} z_{b}) j+(x_{a} y_{b}-y_{a} x_{b}) k
$$


上式可知，叉积的结果是矩阵的行列式的值，是向量，另一种表达方式是使用向量 $a$的对偶矩阵（dual matrix）$A^{\cdot}$


$$
a x b=A^{\cdot} b=[\begin{bmatrix}0 & -z_{a} & y_{a} \\ z_{a} & 0 & -x_{a} \\ -y_{a} & x_{a} & 0\end{bmatrix}][\begin{bmatrix}x_{b} \\ y_{b} \\ z_{b}\end{bmatrix}]
$$


假设向量$a$, $b$间的夹角为θ, 叉积的**几何表达**为：


$$
a x b=|a||b|sin \theta
$$


- 几何意义
	- 向量 $a$,$b$叉乘的结果向量为为向量 $a$,$b$所构成的平行四边形平面的法向量，法向量方向遵守“右手”定律
	- 向量 $a$,$b$叉乘的模为向量 $a$,$b$所构成的平行四边形面积
		- $a x b=-b x a$


#### Orthonormal Basic Frames

如何使用向量的点积和叉积创建直角坐标系。

- 坐标系种类
	- Global， Local， World， Model， Parts of model
- 关键问题
	- 物体在不同坐标系中的位置和相互关系

- 坐标系
	- 3D 坐标系
		- 单位向量：$\|u \|=\|v \|=\|w \|=1$
		- 相互正交：$u·v=v·w=u·w=0$
		- 满足叉乘：$w=u x v$
	- p向量在三个方向上的投影
		- $p=(p·u) u+(p·v) v+(p·w) w$

	- 给定向量$a$, $b$（不正交），如何构建一组三维直角坐标系的基底
		- 在a方向上构建单位向量w：$w=\frac{a}{\|a \|}$
		- 找到一个和ab构成平面垂直的单位法向量u： $u=\frac{b x w}{\|b x w \|}$
		- 根据右手规则找到单位向量v：$v=w x u$


### Matrix

- 矩阵乘法
	- 向量的点积可用矩阵乘法表示：
		- $a·b=a^{T} b$
	- 向量的叉积可以用向量a的对偶矩阵乘以向量b得到，如上文所示
- 矩阵变换
	- 矩阵transform时图形学重点,通常是使用一个转换矩阵乘以一条向量（或一个点），得到转换后的结果
		- 例如对某坐标点进行y轴镜像：


$$
[\begin{bmatrix}-1 & 0 \\ 0 & -1\end{bmatrix}][\begin{bmatrix}x \\ y\end{bmatrix}]=[\begin{bmatrix}-x \\ y\end{bmatrix}]
$$


	- 转置矩阵
		- $(A B)^{T}=B^{T} A^{T}$

	- 逆矩阵
		-  $(A B)^{-1}=B^{-1} A^{-1}$


###  Recommended Textbook

- < Real-Time Rendering>
- < Computer Graphics: Principles and Practice>, 2nd Edition (3rd would be released around mid 2013) 	- < Computer Graphics, C Version>, 2nd Edition (not 3rd or 4th)
- < Fundamentals of Computer Graphics>, 3rd Edition
- < Computer Graphics using OpenGL>, 2nd or 3rd Edition*
- < Interactive Computer Graphics: A Top-Down Approach with Shader-Based OpenGL>, 6th Edition
- < 3D Computer Graphics: A Mathematical Introduction with OpenGL>
