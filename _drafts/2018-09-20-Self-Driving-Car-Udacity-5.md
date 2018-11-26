---
updated: "2018-09-30"
layout: post
list_title: Autonomous Driving | Sensors
title: Sensors
categories: [AI,Autonomous-Driving]
mathjax: true
---

## Introduction

- Odometers, Speedometers, and Derivatives
Understanding motion means understanding quantities like position, velocity, and acceleration and how they relate to each other. And it turns out that calculus gives us two incredible tools for understanding these relationships: derivatives and integrals.

In this lesson you will learn about the derivative and what it can tell us about motion. By the end of this lesson you will be able to take a car's odometery data (distance traveled) and use it to infer new knowledge about velocity and acceleration.
使用导数求速度，通过里程表推到汽车速度和

- Accelerometers, Rate Gyros, and Integrals

Every self driving car has at least one inertial measurement unit in it. These small sensors are able to measure acceleration in three directions as well as rotation rates around all three axes (pitch, roll, and yaw).

But what can we do with this data? In this lesson you'll learn how the integral can be used to accumulate changes in data (and motion）

- Two Dimensional Robot Motion and Trigonometry

In this lesson you'll use knowledge about a vehicle's heading and displacement(朝向和位移) to calculate horizontal and vertical changes in its motion.

- LAB - Reconstructing Trajectories

In the (optional) final project for this course you will use data like this.

<table class="index--table--3PMj4 index--table-striped--1V45V">
<thead>
<tr>
<th style="text-align:center">timestamp</th>
<th style="text-align:center">displacement</th>
<th style="text-align:center">yaw_rate</th>
<th style="text-align:center">acceleration</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">0.0</td>
<td style="text-align:center">0</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">0.25</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">19.6</td>
</tr>
<tr>
<td style="text-align:center">0.5</td>
<td style="text-align:center">1.225</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">19.6</td>
</tr>
<tr>
<td style="text-align:center">0.75</td>
<td style="text-align:center">3.675</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">19.6</td>
</tr>
<tr>
<td style="text-align:center">1.0</td>
<td style="text-align:center">7.35</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">19.6</td>
</tr>
<tr>
<td style="text-align:center">1.25</td>
<td style="text-align:center">12.25</td>
<td style="text-align:center">0.0</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">1.5</td>
<td style="text-align:center">17.15</td>
<td style="text-align:center">-2.829</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">1.75</td>
<td style="text-align:center">22.05</td>
<td style="text-align:center">-2.829</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">2.0</td>
<td style="text-align:center">26.95</td>
<td style="text-align:center">-2.829</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">2.25</td>
<td style="text-align:center">31.85</td>
<td style="text-align:center">-2.829</td>
<td style="text-align:center">0.0</td>
</tr>
<tr>
<td style="text-align:center">2.5</td>
<td style="text-align:center">36.75</td>
<td style="text-align:center">-2.829</td>
<td style="text-align:center">0.0</td>
</tr>
</tbody>
</table>

to reconstruct plots of the vehicle's trajectory like this:

![](/assets/images/2018/09/example-trajectory.png)

## Navigation Sensors

We will be discussing the following sensors in this course:

- **Odometers** , An odometer measures how far a vehicle has traveled by counting wheel rotations. These are useful for measuring distance traveled (or displacement), but they are susceptible to bias (often caused by changing tire diameter). A "trip odometer" is an odometer that can be manually reset by a vehicle's operator.

- **Inertial Measurement Unit**, 惯性测量单元， An Inertial Measurement Unit (or IMU) is used to measure a vehicle's heading, rotation rate, and linear acceleration using magnetometers, rate gyros, and accelerometers. We will discuss these sensors more in the next lesson.