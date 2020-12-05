---
layout: post
title: A Literature Survey on Domain Generalization
date: 2020-11-28 10:49:00
categories: 机器学习
tags: Deep-Learning, Transfer Learning, Domain Adaptation
mathjax: true
---

* content
{:toc}

在分布式的深度学习训练场景下，训练数据与测试数据可能来自不同的分布，同时训练数据来自多个数据源，数据源之间也存在分布偏移。由于深度学习模型本身具有[对训练集的过拟合特性](https://openreview.net/forum?id=Sy8gdB9xx)，对于不同分布数据的微调也会[导致深度学习模型的"灾难性遗忘"](https://www.zhihu.com/question/360374828/answer/1225597056)，因此这种域偏移会严重影响模型的泛化能力。







在之前的博客中，我们用两篇长文介绍了解决这一问题的著名技术：域适应(Domain Adaptation)，分别为[域适应的基本原理](https://www.fenghz.xyz/Domain-Adaptation-A-Survey/)，以及[域适应算法中的核函数](https://www.fenghz.xyz/RHKS-DA/)。此外，经过四个月的工作，我们撰写的一篇名为["KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation"](https://arxiv.org/abs/2011.09757)的文章也已公开并投稿到AAAI2021，在这篇工作中，我们提出了一个新的解决联邦无监督域适应问题的算法，并在当前最大规模的验证集DomainNet上取得了51.1%的准确率。但是，域适应问题要求取得目标域的有标注或无标注数据，这与真实场景具有较大的区别。首先，真实场景中，我们往往希望模型能够适配于多个目标域，并可以进行快速的，小样本的微调。其次，真实场景的训练与测试往往是分离的，训练用于调试模型的数据，往往测试并不可用。域泛化(Domain Generalization)是研究这一问题的有效方法，它假设模型的输入为来自多个源域的数据集，而希望模型能够学到域无关的特征，这种特征可以容易地泛化到新的测试域上。

在本文中，我们对现有的域泛化算法进行综述，并按如下顺序展开：首先，我们介绍域泛化算法的基本理论模型；然后，我们将现有域泛化方法分为四类，基于元学习的域泛化、基于域无关特征的域泛化、基于生成模型的域泛化、以及基于自监督任务的域泛化，并逐一介绍这些方法的优缺点。

## 域泛化问题的基本定义与误差界

我们用如下形式刻画在二分类问题上的域泛化过程。首先，我们将输入空间记作$\mathcal{X}$，将预测的标签空间记作$\mathcal{Y}$，其中$\mathcal{Y}=\{0,1\}$。为了突破“源域未知”这一限制，我们引入在输入空间与标签空间的直积上的概率分布的集合，记作

$$
\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}
$$

$$\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$$中的每一个元素$P_{\mathbf{X}\mathbf{Y}}\in \mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$都代表着输入空间与标签空间的一种可能的联合分布，而不同的联合分布对应着训练过程中的不同域。为方便起见，我们记$\mathcal{B}_{\mathcal{X}}$为输入空间$\mathcal{X}$上所有可能的概率分布集合，而记$\mathcal{B}_{\mathcal{Y}\vert \mathcal{X}}$为给定观测$\mathbf{X}$后标签空间的条件后验分布。同样的，我们有$P_{\mathbf{X}\mathbf{Y}}=P_{\mathbf{X}}\cdot P_{\mathbf{Y\vert X}}$，其中$P_{\mathbf{X}}\in \mathcal{B}_{\mathcal{X}}$，$P_{\mathbf{Y\vert X}}\in \mathcal{B}_{\mathcal{Y\vert X}}$。

假设在$\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$上存在一类分布，记作$\mu$，而我们观察到的$N$个域（也就是$N$个概率分布）都是对$\mu$的独立同分布采样，即

$$
P_{\mathbf{X}\mathbf{Y}}^{(1)},\ldots,P_{\mathbf{X}\mathbf{Y}}^{(N)}\sim^{\text{i.i.d}}\mu
$$


我们考虑域泛化问题的目标函数如下。考虑在概率分布集合与输入空间上的联合映射
$$
f:\mathcal{B}_{\mathcal{X}}\times \mathcal{X}\rightarrow\mathbb{R}; \hat{\mathbf{Y}}=f(P_{\mathbf{X}},\mathbf{X})
$$

我们希望对于任意测试分布$P^{T}_{\mathbf{X}\mathbf{Y}}$，模型预测与真实标签的损失尽量小，即最小化

$$
\epsilon(f) :=\mathbb{E}_{P^{T}_{\mathbf{X}\mathbf{Y}}\sim \mu}\mathbb{E}_{(\mathbf{X}^T,\mathbf{Y}^T)\sim P^{T}_{\mathbf{X}\mathbf{Y}}}[l(f(P^T_{\mathbf{X}},\mathbf{X}^T),\mathbf{Y}^T)] \tag{1}
$$

然而，在实际的采样过程中，我们往往只能采样若干个测试域，而每个测试域也一般只能采样若干个样本。在这种情况下，我们在测试域上的泛化误差与公式(1)所述的理想泛化误差之间有一定的偏移。假设我们采样了$N$个测试域，记作

$$
P_{\mathbf{XY}}^{(1)},\ldots,P_{\mathbf{XY}}^{(N)}\sim \mu
$$

对于对采样的每一个测试域$P_{\mathbf{XY}}^{(i)}$，我们选取尺寸为$n$的样本集合$S_i$，其中

$$
S_i=(\mathbf{X}_{i,j},\mathbf{Y}_{i,j})_{}
$$

## 基于元学习的域泛化

## 基于域无关特征的域泛化

## 基于生成模型的域泛化

## 基于自监督任务的域泛化

## 可用代码与模型验证