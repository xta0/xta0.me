---
list_title: Paper Notes | Machine Learning at Facebook - Understanding Inference at the Edge
title: Machine Learning at Facebook - Understanding Inference at the Edge
layout: post
mathjax: true
---

## Introduction

In our dataset,an overwhelming majority of mobile CPUs use in-order ARM Cortex-A53 and Cortex-A7 cores

Considering theoretical peak FLOP performance, <mark>less than 20% of mobile SoCs have a GPU 3× more powerful than CPUs</mark> and, on a
median mobile device, GPUs are only as powerful as CPUs.

This paper makes the following key observations:

- Nearly all mobile inference run on CPUs and most deployed mobile CPU cores are <mark>old and low-end</mark>. In 2018, only a fourth of smartphones implemented CPU cores designed in 2013 or later. In a median Android device, GPU provides only as much performance as its CPU. Only <mark>11%</mark> of the Android smartphones have a GPU that is 3 times more performant than its CPU.

- System diversity makes porting code to co-processors, such as DSPs, challenging. <mark>We find it more effective to provide general, algorithmic level optimizations that can target all processing environments</mark>. When we have control over the system environment (e.g., Portal [4] or Oculus [5] virtual reality platforms) or when there is little diversity and amature SW stack (e.g., iPhones), performance acceleration with co-processors becomes more viable

- <mark>The main reason to switch to an accelerator/coprocessor is power-efficiency and stability in execution time. Speedup is largely a secondary effect</mark>.

- Inference performance variability in the field is much worse than standalone benchmarking results. Variability poses a problem for user-facing applications with real-time constraints. To study these effects,there is a need for system-level performance modeling.

## THE LAY OF THE LAND: A LOOK AT SMARTPHONES FACEBOOK RUNS ON

### 2.2 Mobile CPUs show little diversity

Figure 3 shows a breakdown of the year smartphone CPU cores were designed or released. 72% of primary CPU cores being used in mobile devices today were designed over 6 years ago. Cortex A53 represents more than 48% of the entire mobile processors whereas Cortex A7 represents more than 15% of the mobile processors

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/02/1.png">

iOS devices tend to use fewer, more powerful cores while Android devices tend to have more cores, which are often less powerful. A similar observation was made in 2015 [6]. To optimize a production application for this degree of hardware diversity, <mark>we optimize for the common denominator: the cluster of most performant CPU cores.</mark>

About half of the SoCs have two CPU clusters: a cluster of high-performance cores and another cluster of energy-efficient cores. Only a small fraction include three clusters of cores. Cores in the different clusters may differ in microarchitectures, frequency settings, or cache sizes. A few SoCs even have two clusters consisting of identical cores. <mark>In nearly all SoCs, cores within the same cluster have a shared cache, but no cache level is shared between cores in the different clusters.</mark> The lack of a shared cache imposes a high synchronization cost between clusters. For this reason, <mark>Facebook apps target the high-performing cluster by, for example, matching thread and core count for neural network inference</mark>.

