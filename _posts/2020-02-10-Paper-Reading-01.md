---
list_title: Paper Reading Notes | [2018] Machine Learning at Facebook - Understanding Inference at the Edge
title: Machine Learning at Facebook - Understanding Inference at the Edge
layout: post
mathjax: true
---

## Introduction

In our dataset,an overwhelming majority of mobile CPUs use in-order ARM Cortex-A53 and Cortex-A7 cores

Considering theoretical peak FLOP performance, less than 20% of mobile SoCs have a GPU 3× more powerful than CPUs and, on a
median mobile device, GPUs are only as powerful as CPUs.

This paper makes the following key observations:

- Nearly all mobile inference run on CPUs and most deployed mobile CPU cores are **old and low-end**. In 2018, only a fourth of smartphones implemented CPU cores designed in 2013 or later. In a median Android device, GPU provides only as much performance as its CPU. Only **11%** of the Android smartphones have a GPU that is 3 times more performant than its CPU.

- System diversity makes porting code to co-processors, such as DSPs, challenging. We find it more effective to provide general, algorithmic level optimizations that can target all processing environments. When we have control over the system environment (e.g., Portal [4] or Oculus [5] virtual reality platforms) or when there is little diversity and amature SW stack (e.g., iPhones), performance acceleration with co-processors becomes more viable

