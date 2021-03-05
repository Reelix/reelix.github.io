---
layout: post
title: A Literature Survey on Mixup-based Methods
date: 2021-3-5 10:49:00
categories: 机器学习
tags: Regularization, Mixup, Domain Adaptation
mathjax: true
---

**Mixup** [1]是ICLR2017年提出的针对计算机视觉的一项简单的数据增广策略。通过对输入数据进行简单的线性变换（即$$\tilde{X}=\lambda*X_0+(1-\lambda)*X_1$$）,可以增加模型的泛化能力，并且能够提高模型对于对抗攻击(Adversarial Attack)的鲁棒性。笔者同时发现，采用**Mixup** 技术在各种半监督与类半监督任务上（如半监督学习，无监督域迁移等）都能极大提高模型的泛化能力，在某些任务上甚至达到了*State-of-the-Art*的结果。然而，对于**Mixup**为什么能够提高泛化能力，以及为什么能够有对对抗攻击的鲁棒性，现有研究仍然没有给出好的解释。在本篇Blog中，我将搜集现有对**Mixup**进行理论解释的文章，并对这些方法进行综述式叙述。此外，我也将对现有的**Mixup**策略进行综述。









但是，笔者对现有的理论研究并不抱有赞成态度。几乎所有的理论研究套路都是令$$1-\lambda\rightarrow 0$$，此时**Mixup**策略中的第二项，即$$(1-\lambda)*X_1$$这一项几乎是微小扰动。现有理论研究都是在这种近似基础上采用泰勒分解，并得出一些结论。但是，在笔者自己所做的实验中，在大部分半监督场景下，当$$\lambda$$的取值在$$0.5$$附近时，如$$\lambda\sim\beta(2,2)$$，模型泛化能力会提高一个台阶（如在DomainNet上相对于$$\lambda\sim\beta(0.2,0.2)$$的分布会提高4-5个点）。对于这种现象，现有所有的理论似乎都失效了，这也是值得研究的一个点。

## Mixup的介绍与理论分析

**Mixup**是一种数据增广策略，通过对模型输入与标签构建具有“凸”性质的运算，构造新的训练样本与对应的标签，提高模型的泛化能力。对于具有层次特征的深度学习模型$$\mathbf{h}=f_{m}\circ \cdots \circ f_{1}\circ f_{0}(\mathbf{X})$$，广义的**Mixup**[1,2]策略将任意层的特征以及对应的标签进行混合，并构造损失函数如下：

**Definition 1.** 对于任意两组输入-标签对$$(\mathbf{X}_0,\mathbf{y}_0)$$以及$$(\mathbf{X}_1,\mathbf{y}_1)$$，令$$\mathbf{h}^{k}_0,\mathbf{h}^{k}_1$$为$$\mathbf{X}_0,\mathbf{X}_1$$所对应的第$k$层输入特征（注意到第0层输入特征是原始输入，即$$\mathbf{h}_0=\mathbf{X}$$），则**Mixup**方法可以描述为 

$$
\tilde{\mathbf{h}}^{k} = (1-\lambda)*\mathbf{h}^{k}_0+\lambda*\mathbf{h}^{k}_1
\\
\tilde{\mathbf{y}}= (1-\lambda)*\mathbf{y}_0+\lambda*\mathbf{y}_1
\\
\text{L}_{mixup}(f_m\circ\cdots\circ f_{k},\mathbf{X}_0,\mathbf{y}_0,\mathbf{X}_1,\mathbf{y}_1)=\text{dist}(f_m\circ\cdots\circ f_{k}(\tilde{\mathbf{h}}^{k}),\tilde{\mathbf{y}})
$$

其中，$$\text{dist}$$函数可以用多种距离计算，如**norm-2**距离$$\Vert\cdot\Vert_2^2$$以及KL散度，而模型$$f_m\circ\cdots\circ f_{k}$$通过损失函数$$\text{L}_{mixup}$$计算梯度，通过梯度下降法进行更新。

大部分论文 [3]将**Mixup**方法视作一种*data-dependent*的正则化操作，即要求模型在特征层面对于运算满足线性约束，利用这种约束对模型施加正则化。但是，相比于其他的数据增广策略，**Mixup**还对标签空间进行了软化，其目标函数往往并不是**one-hot**的对象，而是一组概率。此外，**Mixup**并不需要刻意挑选融合的对象，就算是对输入的猫狗进行加权相加，得到看似毫无意义的输入数据以及对应的标签，都能对模型起到良好的引导。近来的工作还发现**Mixup**对于对抗攻击具有良好的鲁棒性，这些特性使得**Mixup**成为一个需要独立研究的对象。

在笔者自己的实验中，**Mixup**，**Label-smoothing**，**Knowledge** **distillation**这些概念往往在实践中都有千丝万缕的联系，它们的共同特点都是对标签空间进行了合理的“软化”。鉴于**Mixup**已经是我的常用涨点trick之一，对它进行全面的理论分析（然后看看能不能水文章）自然义不容辞。在本节中，我将以近年来Mixup上理论分析做的最深入的文献**On Mixup Regularization**[4]作为基础，对**Mixup**背后的理论进行介绍。


## 现有Work的Mixup方法

### 模型正则化Mixup

### 半监督学习Mixup

### 域迁移学习Mixup

## 参考文献

[1] Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond empirical risk minimization[J]. arXiv preprint arXiv:1710.09412, 2017.

[2] Verma V, Lamb A, Beckham C, et al. Manifold mixup: Better representations by interpolating hidden states[C]//International Conference on Machine Learning. PMLR, 2019: 6438-6447.

[3] Guo H, Mao Y, Zhang R. Mixup as locally linear out-of-manifold regularization[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 3714-3722.

[4] Carratino L, Cissé M, Jenatton R, et al. On mixup regularization[J]. arXiv preprint arXiv:2006.06049, 2020.