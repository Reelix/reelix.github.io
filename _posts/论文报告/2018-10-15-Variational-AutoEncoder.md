---
layout: post
title: An introduction to Variational Autoencoders-Background,Loss function and Application
date: 2018-10-15 12:32:00
categories: 深度学习
tags: Unsupervised-Learning Feature-Extractor
mathjax: true
figure: /images/spectral-cluster/mainfold_hypothesis.png
---

* content
{:toc}

## 前言

Variational autoencoders(VAE)是一类生成模型，它将深度学习与统计推断相结合，可用于学习高维数据$X$的低维表示$Z$。与传统自编码器不同，Variational autoencoders 假设$X$与$Z$都是满足某种分布假设的随机变量(向量)，因此Variational autoencoder 本质是对随机向量分布参数的估计(如均值，方差等矩估计)。在这个假设下，我们可以利用分布函数假设与预测参数进对$p(x\vert z)$与$p(z\vert x)$进行估计，用最大似然设计损失函数，并利用概率分布$p($x$\vert z)$来对$X$进行采样与生成。

本文旨在对VAE进行基于背景，损失函数以及应用方面的介绍。本文将先对VAE所需要的数学知识与基本假设进行简要描述，并在主体部分对文献[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)进行翻译，同时对该文章中省略或笔者认为叙述不清数学证明与显然性描述进行补全与解释。

本文写作过程中主要参考资料为:

* [Introduction to variational autoencoders](https://tensorchiefs.github.io/bbs/files/vae.pdf)
* [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
* [Pattern Recognition and Machine Learning(Chap 1.6)](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-16/issue-4/049901/Pattern-Recognition-and-Machine-Learning/10.1117/1.2819119.short?SSO=1)
* [Wikipedia for some terms](https://www.wikipedia.org/)





## 数学知识与基本假设

### 流形假设,定义与性质

#### 流形的定义与性质

##### 直观理解

流形是局部具有欧几里得性质的空间，是曲线与曲面的推广。欧式空间就是一个最简单的流形实例。几何形体的拓扑结构是柔软的，因为所有的isomorphic会保持拓扑结构不变，而解析几何则是“硬”的，因为整体结构是固定的。光滑流形可以认为是两者之间的模型，它的无穷小结构是硬的，但是整体结构是软的。硬度是容纳微分结构的原因，而软度则可以成为很多需要独立的局部扰动的数学与物理模型。一般流形可以通过把许多平直的片折弯粘连而成，一个直观的理解就是生活中用的麻将席，麻将席整体是柔软的，但是其局部(一小片一小片竹板)则是硬的。

##### 严格定义

**流形定义为：**

假设$M$是豪斯多夫空间(即空间中任意两个点都可以通过邻域来分离)，假设对于任意一点 $x\in M$，都存在$x$的一个邻域，这个邻域同胚(即存在两个拓扑空间中的双连续函数)于$m$维欧式空间$R^m$的一个开集，就称$M$是一个$m$维流形。

#### 流形假设
流形假设就是真实世界中的高维数据(比如说图像)位于一个镶嵌在高维空间中的低维流形上。比如我们获得了二维随机向量$(x,y)$的一组采样:

$$
(1,1),(2,2),(3,3),...,(n,n)
$$

那么这组数据就是在二维空间中的一条线上(镶嵌在二维空间中的一维流形),即该组二维数据可以由$x=t,y=t,t\in R$生成。

流形假设即数据$x=(x_1,....,x_N)$由k维空间中的$y=(y_1,...,y_k)$经连续函数(不一定光滑，不一定线性，也不一定是双射)所构成:

$$
x_i=f_i(y_1,...,y_k),\forall i \in \{1,..,N\}
$$

一个直观理解是手写数字的生成模型，如图所示。手写数字由$(v_1,v_2)$所生成，在$v_2$上的线性变换即为手写数字的旋转，在$v_1$上的线性变换则对应了手写数字的放缩。

![fig1](/images/VAE/mainfold_hypothesis.png)

流形假设是非常重要的假设，该假设可以部分解释为什么深度学习work。同时当考虑到符合流形假设的一些问题的时候（如图像处理等），这个假设可以给我们一些直观，并能够在工程中进行应用(如VAE)。

### 潜变量空间模型假设
该部分是对文献[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) Section1.1的翻译，提前放在此处用于与流形假设对比从而辅助理解。

当训练一个生成模型的时候，生成数据不同维度之间的依赖越强，那么模型就越难进行训练。以生成手写数字0-9为例，如果数字的左半部分是数字*5*的左半部分，那么右半部分就不可能是数字*0*的右半部分，不然生成的图像显然不是一个真实的数字。直观上，一个好的生成模型应该在生成每一个像素所对应值之前先决定到底要生成哪个数字(这样可以避免生成四不像的目标)，而所谓的**决定**就是潜变量。
### 信息论