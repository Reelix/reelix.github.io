---
layout: post
title: From Representation Learning to VAE - A brief history
date: 2018-11-27 12:32:00
categories: 机器学习
tags: Variational-Inference, Representation-Learning
mathjax: true
---

* content
{:toc}

## 简介

从主成分分析(PCA)，线性因子模型到自动编码器，基于无监督学习的特征提取方法不断发展。同时自从引入深度学习技术后，我们对潜变量与因子空间的推断效率与推断方法都有极大的突破，此时原始数据可以在因子空间(通常是潜变量空间)得到更简洁的表示。一个很自然的问题是，如何来比较两种表示的优越程度呢？到底是什么因素决定了一种表示比另外一种表示更好呢?

*No Free Lunch Theorem*告诉我们，所有的机器学习算法的期望效果都是一样的，但是有一些算法在特定的任务上表现会更好。因此，表示的优劣与否往往取决于我们采用该表示与所进行的学习任务是否契合。但是表示学习方法往往是无监督的，如何在无监督过程中注入我们将要进行的学习任务的偏好，以使得某一种表示方法在某种任务上优越于其他表示方法，这就是表示学习领域所关注的任务。





表示学习指出，我们可以以潜先验(*meta-prior*)的形式为模型注入关于潜变量的一些先验信息，如潜变量的独立性信息，层次信息，因果信息，稀疏性信息等，以使得模型能够令表示体现相应的特征。这些潜先验可以用模型约束的形式注入(如稀疏性信息可以用一阶正则化项表示)，可以作用于模型结构上(如层次信息可以用*feature pyramid*表示)，也可以用统计推断方法的后验概率独立性进行约束(如VAE的潜变量后验概率)，同时，我们更可以用半监督学习方法(*semi-supervised learning*)来迫使模型习得我们所偏好任务的因果关系。表示学习认为，如果表示向量$\mathbf{h}$表示观察值$\mathbf{X}$的很多潜在因素，而我们的任务所需要推断的因素$\mathbf{y}$恰好包含在$\mathbf{h}$中，那么通过$\mathbf{h}$来预测$\mathbf{y}$将非常容易。

表示学习与深度学习的结合力量是非常强大的，深度学习是一般函数近似任务的几个世纪进步的结晶，它作为万能近似工具可以轻松地学习到任何变量间的映射关系，而表示学习则为深度学习提供了正确的学习方向。

本文旨在对表示学习领域进行初学者的阶段性总结式介绍，并梳理从表示学习到变分自动编码器之间的关系。本文主要参考资料为[Deep Learning Book](https://www.deeplearningbook.org/)中的13，14，15，19章。本文将以抽象层次由低到高的顺序展开，在*Section.1*中，本文将介绍PCA与线性因子模型的主要假设与结构, 在*Section.2*中，我们将*PCA*扩展到非线性*PCA*，同时引入自动编码器这个概念，并给出自动编码器的一些约束模式，在*Section.3*中，我们将这些约束模式放在表示学习(*representation learning*)的框架下进行讨论，并引入无监督预训练(*unsupervised pre-training*)，迁移学习与领域自适应(*transfer learning and domain adaptation*)，半监督学习(*semi-supervised learning*)，特征分布式表示(*distributed representation*)以及潜先验(*meta-prior*)这些表示学习的热门领域，从而对表示学习进行概览。最后，我们将研究表示学习领域下的一类强力的非线性因子模型变分自动编码器(**VAE**)，在此前的Blog[An introduction to Variational Autoencoders](https://fenghz.github.io/Variational-AutoEncoder/)，以及[Some notes on hierarchical vae](https://fenghz.github.io/Hierarchical-VAE/)中，我们已经对(**VAE**)有了一个比较全面的了解，本文主要将重点放在变分推断的发展历程与假设分析方面。

## Section.1 线性因子模型

### PCA与因子模型
PCA与因子模型是两大主要的降维模型，虽然在求解技术方面，两大模型在正态假设下都对协方差矩阵施加特征值分解变换以在有限维度下求得包含最大方差的变换，但是在模型原理与模型解释方面两大模型有较大的差距。其中，PCA可以看作是自动编码器的一种简单情况，而因子模型则可以看作变分自动编码器的雏形，同时PCA可以施加概率变换变成因子模型。

#### 模型结构
假设$\mathbf{X}=(X_1,\ldots,X_p)'$为$p$维随机向量，其均值为$E(\mathbf{X})=\mathbf{\mu}$，协方差矩阵$D(\mathbf{X})=\mathbf{\Sigma}$，PCA模型要求找到$\mathbf{h}=(h_1,\ldots,h_q)'$为$\mathbf{X}$的前$q$个主成分，满足

$$
\begin{aligned}
    h_i &= \mathbf{a_i'}\mathbf{X}\\
    \mathbf{a_i'a_i }&= 1, i = 1,\ldots,p\\
    当i>1,\mathbf{a_i'\Sigma a_j}&=0,j=1,\ldots,i-1\\
    Var(h_i) &= \max_{a'a=1,a'\Sigma a_j=0,j=1,\ldots,i-1}Var(\mathbf{a'X})
\end{aligned}
$$

求解过程就是很简单的对对称半正定协方差矩阵$\mathbf{\Sigma}$进行正交分解

$$
\mathbf{\Sigma} (\mathbf{a_1,a_2,\ldots,a_n})=\mathbf{\Lambda (a_1,a_2,\ldots,a_n)}\tag{1}
$$

然后对$\mathbf{\Lambda}=diag(\lambda_1,...,\lambda_p)$进行排序，满足$\lambda_1\geq \lambda_2\geq\lambda_p$，然后挑选出前$q$个$\mathbf{a}$，得到$\mathbf{h}$，此时$Var(h_i)=\lambda_i$，同时$h_i,h_j$线性无关。

注意到PCA所分解的对象$\mathbf{X}$仍然是随机变量，但是PCA所用的手法却属于数值代数的范畴。PCA等价于一个坐标旋转变换，它将数据协方差矩阵进行了旋转，并在新的坐标系上进行了投影，然后只选择那些投影度量比较高的分量作为最后的结果，它的几何解释可以参考下图。

![PCA](/images/representation-learning/PCA.gif)

因子模型可以看作是一个生成模型，它描述了一个以潜变量$\mathbf{h}$为因，以可观测变量$\mathbf{X}$为果的生成关系。假定$\mathbf{h}\sim p(\mathbf{h})$, 其中$\mathbf{h}$满足独立性假设，即$p(\mathbf{h})$$=\Pi_{i=1}^{n}p(z_i)$。我们对$\mathbf{h}$进行一次观测，得到一个观测结果$\mathbf{\tilde{h}}$，然后用如下生成模式生成其对应的观测变量$\mathbf{\tilde{X}}$:

$$
\mathbf{\tilde{X}} = \mathbf{W\tilde{h}+b+z},z\sim \mathcal{N}(\mathbf{0,\Phi *I}),\mathbf{\Phi}= diag(\sigma_1^2,\ldots,\sigma_p^2)\tag{2}
$$

它诱导出了$\mathbf{X}$关于给定$\mathbf{h}$后的后验概率$p(\mathbf{X}\vert \mathbf{h})$，其中$\mathbf{W}$是可求解的参数。一般我们拥有大量可观测变量$\mathbf{\tilde{X}}$，给定因子模型后，我们需要求得最优的$\mathbf{W,b}$。注意仍然可以用主成分估计来求解$\mathbf{W,b}$，注意到当$\mathbf{\Phi} \rightarrow 0$时，因子模型可以看成是PCA模型。

### 因子模型中的*meta-prior*
如果我们将已知大量可观测变量$\mathbf{\tilde{X}}$后的因子模型$(2)$看作是一个统计推断问题, 即已知模型$(2)$，给定一个新的样本$\mathbf{X}_{test}$后，我们来推断到底是什么样的$\mathbf{h}_{test}$生成了$\mathbf{X}_{test}$，一个很自然的想法是最大似然估计，即找到单个最可能的编码值

$$
\mathbf{h}^{*}_{test} = \arg\max_{\mathbf{h}} p(\mathbf{h}\vert \mathbf{x}) \tag{3}
$$

考虑贝叶斯公式:

$$
p(\mathbf{h}\vert \mathbf{x}) = \frac{p(\mathbf{x}\vert \mathbf{h})p(\mathbf{h})}{p(\mathbf{x})}
$$

我们可以把$(3)$改写为:

$$
\mathbf{h}^{*}_{test} = \arg\max_{\mathbf{h}} \log(p(\mathbf{x}\vert \mathbf{h})) + \log(p(\mathbf{h}))\tag{4}
$$

$(4)$将$\mathbf{h}$的先验与因子模型中$\mathbf{X}$的后验结合起来了，此时我们可以通过对$\mathbf{h}$的先验施加一些约束来体现*meta prior*，一个典型的例子是稀疏编码约束，即我们希望$\mathbf{h}$中是稀疏的，仅仅在几个分量上具有权重，这样可以将样本依据在各因子上的权重进行分类或聚类。

#### 稀疏编码约束

一个典型的稀疏编码约束是将$\mathbf{h}$的先验看成*Laplacian*分布，即

$$
p(h_i)=\frac{\lambda}{4}e^{-\frac{1}{2}\lambda \vert h_i \vert}
$$

此时$(4)$在因子模型与*Laplacian*分布下的解析形式为

$$
\arg\min_{\mathbf{h}} \lambda \Vert \mathbf{h} \Vert_1+ \mathbf{(x-Wh)'\Phi^{-1}(x-Wh)}
$$

该形式对因子$\mathbf{h}$增加了一阶范数正则化，体现了稀疏编码约束。

### 不变特征(慢特征)分析
如果我们将$(4)$中计算$\mathbf{h}$的过程看作是一个关于$\mathbf{X}$的函数$\mathbf{h=f(X)}$，从这个视角下我们可以得到更多关于*meta prior*的形式，其中一个与时间序列数据有关的形式是不变特征(慢特征)分析。

慢特征分析是针对时间序列数据学习时序变化下不变特征的因子模型，它的想法来源于慢性准则(*slowness principle*)，即虽然对于单一场景而言有很多变化迅速的变量，但是对于整个时序场景而言，构成整个时序场景主要语义的特征变化是缓慢的。比如说，对于计算机视觉而言，视频上的像素改变是非常快的，如一匹斑马泵跑的视频，每一个像素点将在黑白之间不断变换，可是表达"图中是否有一匹斑马"的特征在整个视频中并不会变化，同时表达"图中斑马的位置"的特征在整个视频中变化的也很慢。如果我们希望将"习得变化缓慢的特征"作为因子模型的*meta prior*的话，我们需要在损失函数中加入慢性正则化

$$
\eta \sum_{t}L(f(x^{(t+1)}),f(x^{(t)})) \tag{5}
$$

SFA算法将函数$\mathbf{f(X;\theta)}$看作是优化参数$\theta$的问题(这个视角对于深度学习而言是非常自然的)，并将慢性正则化条件改为求解优化问题

$$
\begin{aligned}
    \min_{\theta} & E_{t}\Vert \mathbf{f(x^{(t+1)})-f(x^{(t)})} \Vert_2,\\
    s.t. & \\
    E_t \mathbf{f(x^{(t)})}_i&=0\\
    E_t [\mathbf{f(x^{(t)})}_i]^2&=1 \\
    E_t[\mathbf{f(x^{(t)})}_i\mathbf{f(x^{(t)})}_j]&=0,\forall i < j
\end{aligned}
$$

这些慢性条件可能可以在序列预测中得到一些好的结果，如果我们对物体运动环境已经进行了详细了解(如3D渲染环境的随机运动中，我们需要对相机位置，速度概率分布进行了解)，那么SFA算法能够比较准确地对深度模型预测SFA将学到什么样的特征。但是，慢性条件仍然没有大规模进行使用，这可能是因为慢性先验过于强大，使得模型对特征的预测坍缩到一个常数上，而无法捕捉时间上的连续微小变化。在物体检测方面，SFA能够捕捉到物体的位置特征，但是会忽视那些高速运动的物体。

## Section.2 自动编码器

### 从PCA到自动编码器
对于$\mathbf{X}=(X_1,\ldots,X_p)'$为$p$维随机向量，其均值为$E(\mathbf{X})=\mathbf{\mu}$，协方差矩阵$D(\mathbf{X})=\mathbf{\Sigma}$，一个简单的自动编码器可以表示为


$$
\begin{aligned}
    \mathbf{h=W^T(X-\mu)}\\
    \mathbf{\hat{X} = g(h) = b+Vh}
\end{aligned}
$$

我们希望它能够最小化重构误差

$$
E[\Vert\mathbf{ X-\hat{X} }\Vert^2] \tag{6}
$$

我们可以证明，能使得$(6)$最小的解为$V=W,b=\mu$，其中$W$为$(1)$中PCA所对应的解。因此一个简单的线性编码器的编码部分即为PCA所对应的映射，而解码部分则为PCA的逆映射。如果$\mathbf{dim(h)=dim(X)}$，则在最优解下$(6)$为0，否则解码部分将会丢失$\sum_{i=\mathbf{dim(h)}}^{\mathbf{dim(X)}}\lambda_i$ 的信息。在这个例子中我们可以看到，PCA可以视作自动编码器的一种特殊情况。


## Reference
