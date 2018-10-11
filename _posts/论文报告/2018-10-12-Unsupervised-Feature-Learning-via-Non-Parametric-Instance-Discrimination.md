---
layout: post
title: A paper report for paper - Unsupervised Feature Learning via Non-Parametric Instance Discrimination
date: 2018-10-11 12:32:00
categories: 深度学习
tags: Network-Architecture
mathjax: true
---

* content
{:toc}

## 简介

[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/abs/1805.01978)是CVPR2018的一篇无监督特征提取方法，且是一篇Oral文章。它采用实例区分(Instance Discrimination)构造实例分类器对图像进行无监督特征提取，所提取的特征可以很好地用于图像的相似度度量任务中。

我为此文撰写了一个report进行深度分析，report以slice形式进行呈现。论文中为了解决实例分类器类别不均衡的问题采用了Noise Contrastive Estimation方法，该方法的数学推导我放于Applendix中。





## Report

![report](/images/unsupervised-feature-extractor/report.png)

## Applendix

**Article:Noise-contrastive estimation: A new estimation principle for unnormalized statistical models**

$p(:,v)$ is a density function with parameters $v$, s.t. $\sum_{i=1}^np(i,v)=1$

Our target is to find the suitable parameter $\theta$ for the network which can make :
$$
p(i,v_i)=1,\forall i\\
p(i,v_i)=\frac{exp(<v_i^{t-1},v_i>/\gamma)}{\sum_{j=1}^nexp(<v_j^{t-1},v_i>/\gamma)}=\frac{exp(<v_i^{t-1},v_i>/\gamma)}{Z_i}
$$

We transfer the problem into an binary estimation problem, and the noise vector $(v_1',...,v_m')$ is  uniformly chosen from $(v_1,...,v_n)$ with $p(v_i')=\frac{1}{n}$, then we have a vector $v_i$, from the data distribution，and $(v_1',...,v_m')$ from the noise distribution, we use $P(C=1\vert i,v)$ to  represent the probability for  $(i,v)$ belong to the data distribution, and we have:

$$
P(i\vert C=1;v)=p(i,v)\\
P(i\vert C=0;v)=\frac{1}{n}
$$
Then we can calculate $P(C=1|i,v)$ use Bayes Formulation:
$$
h(i,v)=P(C=1\vert i,v)=\\
\frac{P(i\vert C=1;v)P(C=1;v)}{P(i\vert C=1)P(C=1;v)+P(i\vert C=0;v)P(C=0;v)}\\
=\frac{p(i,v)}{p(i,v)+\frac{P(C=0;v)}{P(C=1;v)}P(i\vert C=0;v)}\\
$$
And
$$
\frac{P(C=0;v)}{P(C=1;v)}P(i\vert C=0;v)=\frac{m}{n}\\
P(C=0\vert i,v)=1-P(C=1\vert i,v)
$$
Then the binary cross entropy loss function is:
$$
l(\theta)=-[ln(P(C=1\vert i,v_i))+\sum_{j=1}^m ln(P(C=0\vert i,v_j'))]
$$
