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

假设二维向量<math><mi>a</mi><mo>=</mo><mo stretchy="false">[</mo><msub><mi>x</mi><mi>a</mi></msub><mo>,</mo><msub><mi>y</mi><mi>a</mi></msub><mo stretchy="false">]</mo></math> 和 <math><mi>b</mi><mo>=</mo><mo stretchy="false">[</mo><msub><mi>x</mi><mi>b</mi></msub><mo>,</mo><msub><mi>y</mi><mi>b</mi></msub><mo stretchy="false">]</mo></math>

- 点积的**代数表达**为：

<math display="block">
	<mi>a</mi><mo>·</mo><mi>b</mi>
	<mo>=</mo>
	<msub><mi>x</mi><mi>a</mi></msub><msub><mi>x</mi><mi>b</mi></msub>
	<mo>+</mo>
	<msub><mi>y</mi><mi>b</mi></msub><msub><mi>y</mi><mi>b</mi></msub>
</math>

由上式可知，点积的结果是标量（Scalar），无方向

- 假设向量<math><mi>a</mi></math>, <math><mi>b</mi></math>间的夹角为θ, 点积的**几何表达**为：

<math display="block">
	<mi>a</mi><mo>·</mo><mi>b</mi>
	<mo>=</mo>
	<mo stretchy="false">|</mo><mi>a</mi><mo stretchy="false">|</mo>
	<mo stretchy="false">|</mo><mi>b</mi><mo stretchy="false">|</mo>
	<mo>cos</mo>
	<mi>θ</mi>
</math>
<br>
<math display="block">
	<mi>θ</mi>
	<mo>=</mo>
	<mtext>arccos</mtext>
	<mo stretchy="false">(</mo>
	<mfrac>
	<mrow>
		<mi>a</mi><mo>·</mo><mi>b</mi>
	</mrow>
	<mrow>
		<mo stretchy="false">|</mo><mi>a</mi><mo stretchy="false">|</mo>
		<mo stretchy="false">|</mo><mi>b</mi><mo stretchy="false">|</mo>
	</mrow>
	</mfrac>
	<mo stretchy="false">)</mo>
</math>

- 点积的几何意义： 
	- 计算向量<math><mi>a</mi></math>, <math><mi>b</mi></math>间的夹角，判断是否是同一方向以及是否正交
		- <math><mi>a</mi><mo>·</mo><mi>b</mi><mo>></mo><mn>0</mn></math>，同向，夹角在0-90之间
		- <math><mi>a</mi><mo>·</mo><mi>b</mi><mo>=</mo><mn>0</mn></math>，正交，互相垂直
		- <math><mi>a</mi><mo>·</mo><mi>b</mi><mo><</mo><mn>0</mn></math>，反向，夹角在90-180之间
	- 向量 <math><mi>b</mi></math>在向量<math><mi>a</mi></math>上的投影长度 再乘以向量<math><mi>a</mi></math>的长度。
		- 向量 <math><mi>b</mi></math>在向量<math><mi>a</mi></math>上的投影长度表示为:<math><mo>||</mo><mi>b</mi><mo>→</mo><mi>a</mi><mo>||</mo></math>
		- <math><mo>||</mo><mi>b</mi><mo>→</mo><mi>a</mi><mo>||</mo> <mo>=</mo> <mo>||</mo><mi>b</mi><mo>||</mo><mo>cos</mo><mi>θ</mi><mo>=</mo><mfrac><mrow><mi>a</mi><mo>·</mo><mi>b</mi></mrow><mrow><mo>||</mo><mi>a</mi><mo>||</mo></mrow></mfrac></math>
		- <math><mi>b</mi><mo>→</mo><mi>a</mi><mo>=</mo><mo>||</mo><mi>b</mi><mo>→</mo><mi>a</mi><mo>||</mo><mfrac><mrow><mi>a</mi></mrow><mrow><mo>||</mo><mi>a</mi><mo>||</mo></mrow></mfrac><mo stretchy="false">(</mo><mtext>unit vector</mtext><mo stretcchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>a</mi><mo>·</mo><mi>b</mi></mrow><mrow><mo>||</mo><mi>a</mi><msup><mo>||</mo><mn>2</mn></msup></mrow></mfrac><mi>a</mi></math>

#### 叉积（Cross Product）

假设三维向量<math><mi>a</mi><mo>=</mo><mo stretchy="false">[</mo><msub><mi>x</mi><mi>a</mi></msub><mo>,</mo><msub><mi>y</mi><mi>a</mi></msub><mo>,</mo><msub><mi>z</mi><mi>a</mi></msub><mo stretchy="false">]</mo></math> 和 <math><mi>b</mi><mo>=</mo><mo stretchy="false">[</mo><msub><mi>x</mi><mi>b</mi></msub><mo>,</mo><msub><mi>y</mi><mi>b</mi></msub><mo>,</mo><msub><mi>z</mi><mi>b</mi></msub><mo stretchy="false">]</mo></math>

- 叉积的**代数表达**为：

<math display="block">
<mi>a</mi><mo>x</mo><mi>b</mi>
<mo>=</mo>
<mo>|</mo>
<mtable>
<mtr>
	<mtd><mi>i</mi></mtd>
	<mtd><mi>j</mi></mtd>
	<mtd><mi>k</mi></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>x</mi><mi>a</mi></msub></mtd>
	<mtd><msub><mi>y</mi><mi>a</mi></msub></mtd>
	<mtd><msub><mi>z</mi><mi>a</mi></msub></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>x</mi><mi>b</mi></msub></mtd>
	<mtd><msub><mi>y</mi><mi>b</mi></msub></mtd>
	<mtd><msub><mi>z</mi><mi>b</mi></msub></mtd>
</mtr>
</mtable>
<mo>|</mo>
<mo>=</mo>
<mo stretchy="false">(</mo>
<msub><mi>y</mi><mi>a</mi></msub><msub><mi>z</mi><mi>b</mi></msub>
<mo>-</mo>
<msub><mi>z</mi><mi>a</mi></msub>
<msub><mi>y</mi><mi>b</mi></msub>
<mo stretchy="false">)</mo>
<mi>i</mi>
<mo>+</mo>
<mo stretchy="false">(</mo>
<msub><mi>z</mi><mi>a</mi></msub><msub><mi>x</mi><mi>b</mi></msub>
<mo>-</mo>
<msub><mi>x</mi><mi>a</mi></msub><msub><mi>z</mi><mi>b</mi></msub>
<mo stretchy="false">)</mo>
<mi>j</mi>
<mo>+</mo>
<mo stretchy="false">(</mo>
<msub><mi>x</mi><mi>a</mi></msub><msub><mi>y</mi><mi>b</mi></msub>
<mo>-</mo>
<msub><mi>y</mi><mi>a</mi></msub><msub><mi>x</mi><mi>b</mi></msub>
<mo stretchy="false">)</mo>
<mi>k</mi>
</math>

上式可知，叉积的结果是矩阵的行列式的值，是向量，另一种表达方式是使用向量 <math><mi>a</mi></math>的对偶矩阵（dual matrix）<math><msup><mi>A</mi><mo>*</mo></msup></math>

<math display="block">
<mi>a</mi><mo>x</mo><mi>b</mi>
<mo>=</mo>
<msup><mi>A</mi><mo>*</mo></msup><mi>b</mi>
<mo>=</mo>
<mo>[</mo>
<mtable>
<mtr>
	<mtd><mn>0</mn></mtd>
	<mtd><mo>-</mo><msub><mi>z</mi><mi>a</mi></msub></mtd>
	<mtd><msub><mi>y</mi><mi>a</mi></msub></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>z</mi><mi>a</mi></msub></mtd>
	<mtd><mn>0</mn></mtd>
	<mtd><mo>-</mo><msub><mi>x</mi><mi>a</mi></msub></mtd>
</mtr>
<mtr>
	<mtd><mo>-</mo><msub><mi>y</mi><mi>a</mi></msub></mtd>
	<mtd><msub><mi>x</mi><mi>a</mi></msub></mtd>
	<mtd><mn>0</mn></mtd>
</mtr>
</mtable>
<mo>]</mo>
<mo>[</mo>
<mtable>
<mtr>
	<mtd><msub><mi>x</mi><mi>b</mi></msub></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>y</mi><mi>b</mi></msub></mtd>
</mtr>
<mtr>
	<mtd><msub><mi>z</mi><mi>b</mi></msub></mtd>
</mtr>
</mtable>
<mo>]</mo>
</math>

假设向量<math><mi>a</mi></math>, <math><mi>b</mi></math>间的夹角为θ, 叉积的**几何表达**为：

<math display="block">
<mi>a</mi><mo>x</mo><mi>b</mi>
<mo>=</mo>
<mo stretchy="false">|</mo><mi>a</mi><mo stretchy="false">|</mo>
<mo stretchy="false">|</mo><mi>b</mi><mo stretchy="false">|</mo>
<mo>sin</mo>
<mi>θ</mi>
</math>

- 几何意义  
	- 向量 <math><mi>a</mi></math>,<math><mi>b</mi></math>叉乘的结果向量为为向量 <math><mi>a</mi></math>,<math><mi>b</mi></math>所构成的平行四边形平面的法向量，法向量方向遵守“右手”定律
	- 向量 <math><mi>a</mi></math>,<math><mi>b</mi></math>叉乘的模为向量 <math><mi>a</mi></math>,<math><mi>b</mi></math>所构成的平行四边形面积
		- <math><mi>a</mi><mo>x</mo><mi>b</mi><mo>=</mo><mo>-</mo><mi>b</mi><mo>x</mo><mi>a</mi></math> 


#### Orthonormal Basic Frames

如何使用向量的点积和叉积创建直角坐标系。

- 坐标系种类
	- Global， Local， World， Model， Parts of model
- 关键问题
	- 物体在不同坐标系中的位置和相互关系

- 坐标系
	- 3D 坐标系
		- 单位向量：<math><mo stretchy="false">||</mo><mi>u</mi><mo>||</mo><mo>=</mo><mo>||</mo><mi>v</mi><mo>||</mo><mo>=</mo><mo>||</mo><mi>w</mi><mo>||</mo><mo>=</mo><mn>1</mn></math>
		- 相互正交：<math><mi>u</mi><mo>·</mo><mi>v</mi><mo>=</mo><mi>v</mi><mo>·</mo><mi>w</mi><mo>=</mo><mi>u</mi><mo>·</mo><mi>w</mi><mo>=</mo><mn>0</mn></math>
		- 满足叉乘：<math><mi>w</mi><mo>=</mo><mi>u</mi><mo>x</mo><mi>v</mi></math>
	- p向量在三个方向上的投影
		- <math><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mi>p</mi><mo>·</mo><mi>u</mi><mo stretchy="false">)</mo><mi>u</mi><mo>+</mo><mo stretchy="false">(</mo><mi>p</mi><mo>·</mo><mi>v</mi><mo stretchy="false">)</mo><mi>v</mi><mo>+</mo><mo stretchy="false">(</mo><mi>p</mi><mo>·</mo><mi>w</mi><mo stretchy="false">)</mo><mi>w</mi></math>

	- 给定向量<math><mi>a</mi></math>, <math><mi>b</mi></math>（不正交），如何构建一组三维直角坐标系的基底
		- 在a方向上构建单位向量w：<math><mi>w</mi><mo>=</mo><mfrac><mrow><mi>a</mi></mrow><mrow><mo>||</mo><mi>a</mi><mo>||</mo></mrow></mfrac></math>
		- 找到一个和ab构成平面垂直的单位法向量u： <math><mi>u</mi><mo>=</mo><mfrac><mrow><mi>b</mi><mo>x</mo><mi>w</mi></mrow><mrow><mo>||</mo><mi>b</mi><mo>x</mo><mi>w</mi><mo>||</mo></mrow></mfrac></math>
		- 根据右手规则找到单位向量v：<math><mi>v</mi><mo>=</mo><mi>w</mi><mo>x</mo><mi>u</mi></math>
 
 
### Matrix

- 矩阵乘法
	- 向量的点积可用矩阵乘法表示：
		- <math><mi>a</mi><mo>·</mo><mi>b</mi><mo>=</mo><msup><mi>a</mi><mi>T</mi></msup><mi>b</mi></math>  
	- 向量的叉积可以用向量a的对偶矩阵乘以向量b得到，如上文所示
- 矩阵变换
	- 矩阵transform时图形学重点,通常是使用一个转换矩阵乘以一条向量（或一个点），得到转换后的结果 
		- 例如对某坐标点进行y轴镜像：
		
		<math display="block">
			<mo>[</mo>
			<mtable>
				<mtr>
					<mtd><mn>-1</mn></mtd>
					<mtd><mn>0</mn></mtd>
				</mtr>
				<mtr>
					<mtd><mn>0</mn></mtd>
					<mtd><mn>-1</mn></mtd>
				</mtr>
			</mtable>
			<mo>]</mo>
			<mo>[</mo>
			<mtable>
				<mtr>
					<mtd><mi>x</mi></mtd>
				</mtr>
				<mtr>
					<mtd><mi>y</mi></mtd>
				</mtr>
			</mtable>
			<mo>]</mo>
			<mo>=</mo>
			<mo>[</mo>
			<mtable>
				<mtr>
					<mtd><mo>-</mo><mi>x</mi></mtd>
				</mtr>
				<mtr>
					<mtd><mi>y</mi></mtd>
				</mtr>
			</mtable>
			<mo>]</mo>
		</math>
	
	- 转置矩阵
		- <math><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><msup><mo>)</mo><mi>T</mi></msup><mo>=</mo><msup><mi>B</mi><mi>T</mi></msup><msup><mi>A</mi><mi>T</mi></msup></math> 

	- 逆矩阵
		-  <math><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><msup><mo>)</mo><mi>-1</mi></msup><mo>=</mo><msup><mi>B</mi><mi>-1</mi></msup><msup><mi>A</mi><mi>-1</mi></msup></math> 


###  Recommended Textbook

- < Real-Time Rendering> 	
- < Computer Graphics: Principles and Practice>, 2nd Edition (3rd would be released around mid 2013) 	- < Computer Graphics, C Version>, 2nd Edition (not 3rd or 4th)
- < Fundamentals of Computer Graphics>, 3rd Edition
- < Computer Graphics using OpenGL>, 2nd or 3rd Edition*
- < Interactive Computer Graphics: A Top-Down Approach with Shader-Based OpenGL>, 6th Edition
- < 3D Computer Graphics: A Mathematical Introduction with OpenGL>
