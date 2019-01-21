---
layout: post
title: Some notes on hierarchical vae
date: 2019-1-13 20:10:00
categories: 机器学习
tags: Variational-Inference
mathjax: true
figure: /images/hierarchical-vae/basic_structure.png
---

* content
{:toc}

参考资料:


**FengHZ's Blog 首发原创**

## 前言

我们在[An introduction to Variational Autoencoders-Background,Loss function and Application](https://fenghz.github.io/Variational-AutoEncoder/)中对**Kingma**提出的**VAE**进行了详细的介绍. VAE采用了潜变量假设, 将输入空间$X$与潜变量空间$Z$都看作是概率空间, 利用贝叶斯公式

$$
p(x)=\int_{z}p(x\vert z)p(z)dz
$$

建立从潜变量空间$Z$到输入空间$X$分布间的映射, 并利用神经网络来拟合该映射. 在拟合过程中，我们对$p(x\vert z),q(z\vert x),p(z)$的分布进行正态性假设, 并用$q(z\vert x)$来拟合$p(z\vert x)$, 要求拟合结果要令样本点$x\in X$的出现概率尽可能大，并利用**Jensen**不等式构造具有解析形式的变分下界

$$
\log(p(x))=log(E_{q(z\vert x)}\frac{p(x,z)}{q(z\vert x)})\geq E_{q(z\vert x)}\log(\frac{p(x,z)}{q(z\vert x)})=ELBO \tag{1}
$$








通过最大化$(1)$中的**ELBO(Evidence Lower Bound)**, 我们可以间接最大化$log(p(x))$, 这就是**Kingma**所提出**VAE**的基本思想. 但是, **VAE**将潜变量空间看作是一个随机层(**Stochastic Layer**), 这就会带来一些局限性. 假如流形假设为以下形式

$$
x=f_1(z_1,f_2(z_2),f_2\circ f_3(z_3),\ldots,f_1\circ f_{2}\circ \ldots \circ f_{n}(z_n))\\
z_i =g_{i}\circ g_{i-1}\circ \ldots \circ g_{1}(x),i=1,2,\ldots,n\\
z_i\in Z,x\in X

$$
那么潜变量之间具有较为明显的层次关系,而**Kingma**所提出**VAE**仅构建了一层随机层,此时$z_1,\ldots,z_n$的关系是并列关系, 因此没有办法揭示潜变量之间的层次关系. 基于此问题, 文献[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)与**Kingma**同时独立提出了变分自编码器. 该文献主要有两个贡献, 一是提出了具有层次结构的变分自编码器, 二是对于重参数化技巧给出了严谨的数学证明, 是**VAE**领域的基石文章之一. 在该篇文章后, 也有诸多论文对**Hierarchical VAE**的结构与损失函数进行了优化, 如[Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)对不同层次的因子进行了加权处理, [Ladder Variational Autoencoders](https://arxiv.org/abs/1602.02282)采用**ladder**结构, 将对潜变量分布的预测分为编码预测与解码预测两部分, 并用两部分预测结果的平均来作为最终预测以保证编码解码中的信息对称. [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)将**Hierarchical VAE**的结构扩展到半监督任务上, 将数据标签看作是服从多项式分布的离散潜变量. 

本文将先给出**Hiearachical VAE**的基本形式与损失函数, 然后证明与重参数化技巧息息相关的几个结论，并给出从该结论出发的对于其他分布假设的计算方法, 最后对上文列出的**Hierarchical VAE**的扩展结构进行讨论总结.

## Hierarchical VAE基本结构与损失函数

![basic structure](/images/hierarchical-vae/basic_structure.png)
