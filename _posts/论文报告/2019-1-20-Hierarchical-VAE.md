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

建立从潜变量空间$\mathcal{Z}$到输入空间$\mathcal{X}$分布间的映射, 并利用神经网络来拟合该映射. 在拟合过程中，我们对$p(x\vert z),q(z\vert x),p(z)$的分布进行正态性假设, 并用$q(z\vert x)$来拟合$p(z\vert x)$, 要求拟合结果要令样本点$x\in \mathcal{X}$的出现概率尽可能大，并利用**Jensen**不等式构造具有解析形式的变分下界

$$
\log(p(x))=log(E_{q(z\vert x)}\frac{p(x,z)}{q(z\vert x)})\geq E_{q(z\vert x)}\log(\frac{p(x,z)}{q(z\vert x)})=ELBO \tag{1}
$$








通过最大化$(1)$中的**ELBO(Evidence Lower Bound)**, 我们可以间接最大化$log(p(x))$, 这就是**Kingma**所提出**VAE**的基本思想. 但是, **VAE**将潜变量空间看作是一个随机层(**Stochastic Layer**), 这就会带来一些局限性. 假如流形假设为以下形式

$$
x=f_1(z_1,f_2(z_2),f_2\circ f_3(z_3),\ldots,f_1\circ f_{2}\circ \ldots \circ f_{n}(z_n))\\
z_i =g_{i}\circ g_{i-1}\circ \ldots \circ g_{1}(x),i=1,2,\ldots,n\\
z_i\in \mathcal{Z},x\in \mathcal{X}

$$
那么潜变量之间具有较为明显的层次关系,而**Kingma**所提出**VAE**仅构建了一层随机层,此时$z_1,\ldots,z_n$的关系是并列关系, 因此没有办法揭示潜变量之间的层次关系. 基于此问题, 文献[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)与**Kingma**同时独立提出了变分自编码器. 该文献主要有两个贡献, 一是提出了具有层次结构的变分自编码器, 二是对于重参数化技巧给出了严谨的数学证明, 是**VAE**领域的基石文章之一. 在该篇文章后, 也有诸多论文对**Hierarchical VAE**的结构与损失函数进行了优化, 如[Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)对不同层次的因子进行了加权处理, [Ladder Variational Autoencoders](https://arxiv.org/abs/1602.02282)采用**ladder**结构, 将对潜变量分布的预测分为编码预测与解码预测两部分, 并用两部分预测结果的平均来作为最终预测以保证编码解码中的信息对称. [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)将**Hierarchical VAE**的结构扩展到半监督任务上, 将数据标签看作是服从多项式分布的离散潜变量. 

本文将先给出**Hiearachical VAE**的基本形式与损失函数, 然后证明与重参数化技巧息息相关的几个结论，并给出从该结论出发的对于其他分布假设的计算方法, 最后对上文列出的**Hierarchical VAE**的扩展结构进行讨论总结.

## Hierarchical VAE基本结构与损失函数

![basic structure](/images/hierarchical-vae/basic_structure.png)

如图所示为**Hierarchical VAE**的基本结构，该结构与**U-Net**的结构非常类似, 不过**U-Net**是直接进行特征融合, 而**Hiearachical VAE**则是进行因子融合. 

我们对该结构图做如下解释。首先输入生成空间中的样本$X\in \mathcal{X}$, 我们用$L$个层次排列的**Feature Encoder Block**来进行特征提取, 得到层次特征$F_1,F_2,\ldots,F_L$(一般认为用于图像处理的卷积神经网络低层特征提取形状信息, 高层特征提取语义信息), 然后再用相应的层次特征$F_i$对当前层次的潜变量$z_i$的分布$p(z_i\vert X) \sim \mathcal{N}(\mu_{i},C_{i})$进行推断, 得到层次潜变量$z_1,z_2,\ldots,z_L$, 然后通过这些层次潜变量构建解码器来对$p(X\vert z_1,z_2,\ldots,z_L)\sim \mathcal{N}(X,\sigma^2)$进行推断. 编码过程中我们需要对$L$组潜变量的分布进行推断, 而解码过程中我们需要对$L$组潜变量分别采样并进行分层解码.

与**Kingma**的**VAE**有所不同, 基于层次结构的**VAE**在原有假设基础上基于**Bayes**链式法则进行了扩展, 记左边的推断网络参数为$\theta^r$, 右边的生成网络参数为$\theta ^g$, $z=(z_1,\ldots,z_L),h=(h_1,\ldots,h_L)$, 其基本假设如下

$$
p(\theta^g)\sim \mathcal{N}(0,\kappa I);\\
p(z_i)\sim \mathcal{N}(0,I),i\in {1,\ldots,L};\\
p(z)=\Pi_{i=1}^L p(z_i);\\
q(z_i \vert X,\theta^r)\sim \mathcal{N}(\mu_i,C_i),i\in {1,\ldots,L};\\
q(z \vert X,\theta^r)=\Pi_{i=1}^L q(z_i \vert X,\theta^r);\\
p(X,z)=p(X\vert h_1(z_{1,\ldots,L},\theta^g))p(\theta^g)\Pi_{i=1}^L p(z_i);\\
p(X,h)=p(X\vert h_1,\theta^g)p(h_{L}\vert \theta^g)p(\theta^g)\Pi_{l=1}^{L-1}p_{l}(h_l\vert h_{l+1},\theta^g);\\
p(X\vert h_1,\theta^g)\sim \mathcal{N}(g(h_1),\sigma^2)
$$

注意

$$
p(X,h)=p(X\vert h_1,\theta^g)p(h_{L}\vert \theta^g)p(\theta^g)\Pi_{l=1}^{L-1}p_{l}(h_l\vert h_{l+1},\theta^g)
$$

$$
p(X,z)=p(X\vert h_1(z_{1,\ldots,L},\theta^g))p(\theta^g)\Pi_{i=1}^L p(z_i)\tag{1}
$$

考虑到

$$
z_i \sim \mathcal{N}(\mu_i,C_i),i\in {1,\ldots,L}\\
 h_{L}=G_Lz_L\\
h_i=Decoder_{i+1}(h_{i+1})+G_iz_i,i\in {1,\ldots,L-1}\\
$$

其中$G_L$为一个线性矩阵, 这两个式子在我们的模型中是等价的,其中

$$
p_l(h_l\vert h_{l+1},\theta^g) = N(Decoder_{i+1}(h_{i+1})+G_i\mu_i,G_iC_iG_i')
$$

我们一般使用$p(X,z)$公式以获得显式的结果, 利用如上假设, 我们对**Hierarchical VAE**的损失函数进行推导如下

$$
\log(p(X))=\int_{z}q(z\vert X,\theta^r)\log(p(X))dz=\int_{z}q(z\vert X,\theta^r)\log(\frac{p(X,z)}{q(z\vert X,\theta^r)}\frac{q(z\vert X,\theta^r)}{p(z\vert X)})dz\\
=\int_{z}q(z\vert X,\theta^r)\log(\frac{p(X,z)}{q(z\vert X,\theta^r)})dz+\mathcal{D}[q(z\vert X,\theta^r)\Vert p(z\vert x)]
$$

其中

$$
\int_{z}q(z\vert X,\theta^r)\log(\frac{p(X,z)}{q(z\vert X,\theta^r)})dz\tag{2}
$$

即为我们的目标损失函数, 也是层次潜变量下的**ELBO**, 我们给出在假设下代入$(1)$

$$
\int_{z}q(z\vert X,\theta^r)\log(\frac{p(X,z)}{q(z\vert X,\theta^r)})dz=\int_{z}q(z\vert X,\theta^r)\log(\frac{p(X\vert h_1(z_{1,\ldots,L},\theta^g))p(\theta^g)\Pi_{i=1}^L p(z_i)}{q(z\vert X,\theta^r)})dz \\
=E_{z\sim q(z\vert X,\theta^r)}\log(p(X\vert h_1(z_{1,\ldots,L},\theta^g))+\log(p(\theta^g))-\mathcal{D}[q(z\vert X,\theta^r),p(z)]
$$

其中, 因为$z_1,z_2,\ldots,z_L$彼此不相关, 因此$p(z_1,\ldots,z_L) \sim  \mathcal{N}(0,I),dim(I)=\sum_{l=1}^L dim(z_l)$, 同时

$$
q(z\vert X,\theta^r)\sim N((\mu_{1,1},\ldots,\mu_{1,dim(z_1)},\ldots,\mu_{L,1},\ldots,\mu_{L,dim(z_L)}),
\begin{pmatrix}
C_1 & \\
& C_2 & \\
&&\ddots \\
&&& C_L
\end{pmatrix}
)
$$

因此

$$
\mathcal{D}[q(z\vert X,\theta^r),p(z)]=\sum_{l=1}^L[\Vert \mu_l\Vert_2^2+Tr(C_l)-log\vert C_l\vert]+b
$$

其中$b$是一个固定常数, 我们用*K*个采样$(\hat{z}_1,\hat{z}_2,\ldots,\hat{z}_K)\sim q(z\vert X,\theta^r)$来估计期望$E_{z\sim q(z\vert X,\theta^r)}\log(p(X\vert h_1(z_{1,\ldots,L},\theta^g))$, 此时可以给出$(2)$的解析形式, 即目标函数为

$$
arg\max_{\theta^g,\theta^r}\frac{1}{K}\sum_{k=1}^K-\frac{\Vert X- h_1(\hat{z_k},\theta^g \Vert_2^2)}{\sigma ^2} -\frac{\Vert \theta^g \Vert _2^2}{2*\kappa}  + \sum_{l=1}^L[\Vert \mu_l\Vert_2^2+Tr(C_l)-\log\vert C_l\vert]
$$

它等价于

$$
arg\min_{\theta^g,\theta^r}\frac{1}{K}\sum_{k=1}^K\frac{\Vert X- h_1(\hat{z_k},\theta^g \Vert_2^2)}{\sigma ^2} +\frac{\Vert \theta^g \Vert _2^2}{2*\kappa}  - \sum_{l=1}^L[\Vert \mu_l\Vert_2^2+Tr(C_l)-\log\vert C_l\vert] \tag{3}
$$

## 重参数化技巧

如果我们用**Deep Learning**模型进行建模,而$\theta^r$与$\theta^g$ 分别是识别网络(或者说编码网络)以及生成网络(或者说解码网络)的如何优化$(3)