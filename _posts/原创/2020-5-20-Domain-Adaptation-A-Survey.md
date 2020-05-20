---
layout: post
title: A Brief Introduction of Domain Adaptation
date: 2020-05-20 10:49:00
categories: 机器学习
tags: Deep-Learning
mathjax: true
---

* content
{:toc}

在所有机器学习与深度学习的模型落地中，域偏移(Domain Shift)，即训练数据与真实数据来自于不同的分布，是一个很常见的问题，而这个问题在很多落地场景中都是很致命的。如在医学深度学习模型中，用A医院的数据(Source Domain)训练的模型往往在B医院(Target Domain)预测不准。在摄像头行人重识别(Re-ID)问题中，多个摄像头捕捉的场景分布完全不一致，导致单个行人在多个摄像头中的"重识别"变得较为困难。在联邦学习问题中，我们会单独获得多个数据源的数据，而只有部分数据有标签，这使得我们利用某些数据源的标签对其他数据进行建模的过程中也会出现偏移。有很多学者对这个问题做了不同的诠释，提供了不同的解决办法。有些流派将该问题解释为 "covariate shift", 认为解决该方法需要对每一个样本进行赋权操作。还有的流派将解决该问题的办法称作迁移学习(Transfer Learning), 通过先后训练与特征微调来*bridge the gap*. 以上流派着眼于方法论，其思想与文章写作都偏向实用主义。我比较喜欢的流派是以域适应(Domain Adaptation)作为解决办法的流派，该流派的基础思想是，在保持源域任务精度的前提下，缩小模型所习得的表示空间上源域与目标域的特征距离。本文主要介绍该流派自洽的数学表达，同时介绍现在较为实用的算法，包括对抗域迁移学习，以及联邦对抗域迁移学习。








本文主要参考文献为：
1. A Theory of learning from different domains
2. Analysis of Representations for Domain Adaptation
3. Domain-Adversarial Training of Neural Networks
4. Federated Adversarial Domain Adaptation
5. CYCADA: Cycle-Consistent Adversarial Domain Adaptation
   

## Domain Adaptation: 基本定义
从上文中的几个例子，我们大概隐隐约约知道了Domain Adaptation是解决两个数据集分布不一致的一种办法。但是，要给出严谨的数学定义，我们仍然要思考一些问题，比如，我们应该如何用符号来严格刻画Domain与Domain Shift呢？任意给两个不一样的数据集，我们都可以进行Domain Adaptation吗？对于两个域之间距离应该如何计算呢？给定的度量是否可以用样本进行经验估计呢？这些问题是定义上的问题。此外，在现实场景中，我们所需要处理的数据源是复杂的，有可能Source Domain和Target Domain都有充足的数据和标签，有可能只有Source Domain有标签，还有可能有充足的Source和Target Domain的源数据，但是两个数据集都只有部分标签。在这些场景下，如何去分配Source Domain和Target Domain的权重，这又是一个重要的问题。良好地描述一个问题，是解决一个问题的前提。在本节中，我们在概率论的角度，给出Domain以及Domain Adaptation的基本定义。

我们用如下的数学形式刻画在二分类任务上的Domain Adaptation问题。首先，我们将输入集记为$\mathcal{X}$，标注空间是一个概率空间，因为是二分类问题，我们的标注集就是$[0,1]$区间。在$\mathcal{X}$上，我们定义一个domain是由$<\mathcal{D},f>$所构成的一个对。其中，$\mathcal{D}$是一个分布函数，刻画输入集$\mathcal{X}$上每个样本的出现概率，而$f$是一个标签函数，$f:\mathcal{X}\rightarrow [0,1]$，表示从输入集到标注集的一个映射，其中，$f(\mathbf{x})\in [0,1],x\in \mathcal{X}$可以是$[0,1]$区间内的任意实数值，表示样本$\mathbf{x}$的标签为1的概率。一般而言，我们用字母$S,T$区分源域(Source Domain)和目标域(Target Domain)，而两个域的数学表示分别记为$<\mathcal{D}_S,f_S>$和$<\mathcal{D}_T,f_T>$。

给定两个域$<\mathcal{D}_S,f_S>$与$<\mathcal{D}_T,f_T>$后，我们希望学习一个函数$h:\mathcal{X}\rightarrow\{0,1\}$，这个目标函数直接预测输入所对应的标签，我们把这个模型习得的函数$h$称作是一个假设(hypothesis)，将$h$与$f$的差距记作

$$
\epsilon(h,f;\mathcal{D})= \mathbf{E}_{\mathbf{x}\sim \mathcal{D}}\vert h(\mathbf{x})-f(\mathbf{x})\vert \tag{1}
$$

为简化标记，在源域上，我们记$\epsilon_S(h)=\epsilon_S(h,f_S;\mathcal{D}_S)$，如果我们只有有限的样本，则我们采用这些样本对$(1)$的期望进行经验估计，记经验估计的结果为$\hat{\epsilon}_S(h)$。同样地，我们可以得到$\epsilon_T(h),\hat{\epsilon}_T(h)$.

给定两个domain$<\mathcal{D}_S,f_S>$和$<\mathcal{D}_T,f_T>$，我们采用基于$L_1$范数的变分散度(Variational Distance)来刻画它们的距离，即

$$
d_1(\mathcal{D}_S,\mathcal{D}_T) = 2 \sup_{B\subset \mathcal{X}}\vert \Pr_{\mathcal{D}_S}(B)-\Pr_{\mathcal{D}_T}(B)\vert \tag{2}
$$

其中，$\Pr(B)$表示集合$B$的概率。这个距离可以这么理解，我们找两个不同数据分布上分布差距最大的子集，将它们的概率的差值绝对值作为距离。但是，实际使用过程中，该距离又有诸多不便。首先，如果分布函数是离散的，那么找到$(2)$的上确界就是一个NP-Hard问题。其次，如果分布函数是连续的，那么$(2)$中取得上确界的$B$往往是由多个小区间联立拼凑而成，这实际上是无法计算的，因此该距离的计算意义不大。但是该距离却具有很高的实际用处。我们做Domain Adaptation，是希望在源域$\mathcal{D}_S$上训练的模型$h$能泛化到目标域$\mathcal{D}_T$上，即希望$\epsilon_S(h)$很小的时候，$\epsilon_T(h)$也比较小，这个时候，$(2)$中刻画的距离就给出了一个很漂亮的不等式，我们先不加证明地将其写出：

$$
\epsilon_T(h)\leq \epsilon_S(h)+d_1(\mathcal{D}_S,\mathcal{D}_T) +\min \{\mathbf{E}_{\mathcal{D}_S}\vert f_S(\mathbf{x})-f_T(\mathbf{x})\vert,\\ \mathbf{E}_{\mathcal{D}_T}\vert f_S(\mathbf{x})-f_T(\mathbf{x})\vert \} \tag{3}
$$

该不等式表示，在$\mathcal{D}_S$上训练的$h$，当$\epsilon_S(h)$很小的时候，$\epsilon_T(h)$是否也比较小，这取决于两个数据集的距离以及两个数据集的标签函数$f_S,f_T$是否一致。但是,$(3)$所展现出对于$\epsilon_T(h)$的上确界确实过于宽松了。一个很典型的想法是，我的模型只需要关注于对学到的假设$h$而言，两个数据集的距离不会很远就行了。在现实任务中，如果我有Domain Adaptation的需求，那么在目标任务上，这两个数据集应该是有一定相似度的，那么我们为什么要考虑整个数据分布的距离呢？是否只需要考察在目标假设$h$上的距离就够了呢？在这个思想下，$\mathcal{H}$距离被提出了

$$
d_{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)=2 \sup_{h\in \mathcal{H}}\vert \Pr_{\mathcal{D}_S}(I(h))-\Pr_{\mathcal{D}_T}(I(h))\vert \tag{4}
$$

其中，$\mathcal{H}$是我们模型训练出的所有可能假设$h$的集合，我们可以把它看成是模型容量。$I(h)=\{\mathbf{x}:h(\mathbf{x})=1,\mathbf{x}\in \mathcal{X}\}$是$\mathcal{X}$的一个子集，但是与$h$有关。通过$(4)$的定义，我们就把数据集的距离与目标假设联系在一起了。

## Domain Adaptation：几个重要的Bound
从Domain Adaptation的一系列定义中，我们大概已经有了一种感觉，就是






