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

## 域泛化问题的基本定义与误差界[1]

我们用如下形式刻画在二分类问题上的域泛化过程。首先，我们将输入空间记作$\mathcal{X}$，将预测的标签空间记作$\mathcal{Y}$，其中$\mathcal{Y}=\{0,1\}$。为了突破“源域未知”这一限制，我们引入在输入空间与标签空间的直积上的概率分布的集合，记作

$$
\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}
$$

$$\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$$中的每一个元素$$P_{\mathbf{X}\mathbf{Y}}\in \mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$$都代表着输入空间与标签空间的一种可能的联合分布，而不同的联合分布对应着训练过程中的不同域。为方便起见，我们记$$\mathcal{B}_{\mathcal{X}}$$为输入空间$$\mathcal{X}$$上所有可能的概率分布集合，而记$$\mathcal{B}_{\mathcal{Y}\vert \mathcal{X}}$$为给定观测$$\mathbf{X}$$后标签空间的条件后验分布。同样的，我们有$$P_{\mathbf{X}\mathbf{Y}}=P_{\mathbf{X}}\cdot P_{\mathbf{Y\vert X}}$$，其中$$P_{\mathbf{X}}\in \mathcal{B}_{\mathcal{X}}$$，$$P_{\mathbf{Y\vert X}}\in \mathcal{B}_{\mathcal{Y\vert X}}$$。

假设在$$\mathcal{B}_{\mathcal{X}\times \mathcal{Y}}$$上存在一类分布，记作$$\mu$$，而我们观察到的$$N$$个域（也就是$$N$$个概率分布）都是对$$\mu$$的独立同分布采样，即

$$
P_{\mathbf{X}\mathbf{Y}}^{(1)},\ldots,P_{\mathbf{X}\mathbf{Y}}^{(N)}\sim^{\text{i.i.d}}\mu
$$


我们考虑域泛化问题的目标函数如下。考虑在概率分布集合与输入空间上的联合映射

$$
f:\mathcal{B}_{\mathcal{X}}\times \mathcal{X}\rightarrow\mathbb{R}; \hat{\mathbf{Y}}=f(P_{\mathbf{X}},\mathbf{X})
$$

我们希望对于任意测试分布$$P^{T}_{\mathbf{X}\mathbf{Y}}$$，模型预测与真实标签的损失尽量小，即最小化

$$
\epsilon(f) :=\mathbb{E}_{P^{T}_{\mathbf{X}\mathbf{Y}}\sim \mu}\mathbb{E}_{(\mathbf{X}^T,\mathbf{Y}^T)\sim P^{T}_{\mathbf{X}\mathbf{Y}}}\ [l(f(P^T_{\mathbf{X}},\mathbf{X}^T),\mathbf{Y}^T)] \tag{1}
$$

然而，在实际的采样过程中，我们往往只能采样若干个测试域，而每个测试域也一般只能采样若干个样本。在这种情况下，我们在测试域上的泛化误差与公式(1)所述的理想泛化误差之间有一定的偏移。假设我们采样了$$N$$个测试域，记作

$$
P_{\mathbf{XY}}^{(1)},\ldots,P_{\mathbf{XY}}^{(N)}\sim \mu
$$

对于对采样的每一个测试域$$P_{\mathbf{XY}}^{(i)}$$，我们选取尺寸为$$n$$的样本集合$$S_i$$，其中

$$
S_i=(\mathbf{X}_{i,j},\mathbf{Y}_{i,j})_{1\leq j\leq n_i}
$$

我们记通过采样集$$\{S_i\}_{i=1}^{N}$$所得的域泛化损失为

$$
\epsilon(f,\sum_{i=1}^{N}n_i):=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{n_i}\sum_{j=1}^{n_i}\ [l(f(\hat{P}^{(i)}_{\mathbf{X}},\mathbf{X}_{i,j}),\mathbf{Y}_{i,j})]\tag{2}
$$

易得，当$$N\rightarrow \infty,n\rightarrow \infty$$时，通过$$(2)$$所计算出的泛化损失逼近于真实泛化损失$$(1)$$，因此我们也可以记$$(1)$$为$$\epsilon(f,\infty)$$。那么，一个很自然的问题是，我们通过$$(2)$$所估算的误差是否能收敛到真实误差$$\epsilon(f,\infty)$$，他们之间的距离是否可以用[PAC可学习理论](https://tangshusen.me/2018/12/09/vc-dimension/)进行表示。通过研究Reproducing Kernel Hilbert Space (RKHS)上的目标函数$$f$$，我们可以得到一个漂亮的理论上界。

### 在RKHS空间上进行泛化误差界分析

考虑损失函数$$l:\mathbb{R}\times \mathcal{Y}\rightarrow \mathbb{R}_{+}$$。令$$\bar{k}$$表示空间$$\mathcal{B}_{\mathcal{X}}\times \mathcal{X}$$上的核函数，令$$\mathcal{H}_{\bar{k}}$$表示对应的Reproducing Kernel Hilbert Space (RKHS)。给定采样集$$\{S_i\}_{i=1}^{N}$$，我们令$$\hat{P}_{\mathbf{X}}^{(i)}$$表示通过采样样本对第$$i$$个域的分布$P_{\mathbf{X}}^{(i)}$的经验估计，并通过采样集得对目标函数进行优化如下

$$
\hat{f}_{\lambda}=\arg_{f\in \mathcal{H}_{\bar{k}}}\min\frac{1}{N}\sum_{i=1}^{N}\frac{1}{n_i}\sum_{j=1}^{n_i}\ [l(f(\hat{P}_{\mathbf{X}}^{(i)},\mathbf{X}_{i,j}),\mathbf{Y}_{i,j})]+\lambda\Vert f\Vert.\tag{3}
$$

根据RKHS空间的基本性质，我们可以通过核函数计算的距离给出预测，即

$$
\hat{f}_{\lambda}(\hat{P}_{\mathbf{X}},\mathbf{X})=\sum_{i=1}^{N}\sum_{j=1}^{n_i}\alpha_{i,j}\mathbf{Y}_{i,j}\bar{k}((\hat{P}_{\mathbf{X}},\mathbf{X}),(\hat{P}_{\mathbf{X}}^{(i)},\mathbf{X}_{i,j}))
$$

而其中$$\alpha_{i,j}$$是优化器根据$$(3)$$所计算而来的参数。那么，对于这种特殊的输入$$(\hat{P}_{\mathbf{X}},\mathbf{X})$$，如何构造核函数，使得它们之间的距离能够被正确度量呢？我们已经在之前的博客中系统性介绍了[核函数的基本性质](http://www.fenghz.xyz/RHKS-DA/#reproducing-hilbert-kernel-space)，基于两个核函数的乘积也是核函数这一原理，一个很自然的想法是对$$\bar{k}$$进行如下所示的分解

$$
\bar{k}((P_{\mathbf{X}}^{(1)},\mathbf{X}_1),(P_{\mathbf{X}}^{(2)},\mathbf{X}_2))=k_{P}(P_{\mathbf{X}}^{(1)},P_{\mathbf{X}}^{(2)})k_{\mathbf{X}}(\mathbf{X}_1,\mathbf{X}_2)
$$

对于$$k_{\mathbf{X}}(\mathbf{X}_1,\mathbf{X}_2)$$，先前工作已经提出了很多有用的核函数，如线性核，二次核，高斯核等，因此我们这里重点讨论对分布构造核函数。在[Kernel Mean Embedding of Distributions: A Review and Beyond](https://arxiv.org/abs/1605.09522)这篇论文的第三章中，给出了一个通过**Kernel Mean Embedding**来计算分布之间距离的方法，即构造从所有可能的概率分布的集合$$\mathcal{B}_{\mathcal{X}}$$到通过任意一个在输入空间上的核函数$$k'_{\mathbf{X}}$$所对应的RKHS空间的映射$$\Phi:\mathcal{B}_{\mathcal{X}}\rightarrow \mathcal{H}_{k’_{\mathbf{X}}}$$如下：

$$
P_{\mathbf{X}}\rightarrow \Phi(P_{\mathbf{X}}):=\int_{\mathcal{X}}k’_{\mathbf{X}}(\mathbf{X},\cdot)dP_{\mathbf{X}}
$$

此时，再利用$$\mathcal{H}_{k_{\mathbf{X}}}$$空间的基本性质，即

$$
k'_{\mathbf{X}}(\mathbf{X}_1,\mathbf{X}_2)=\langle  k'_{\mathbf{X}}(\mathbf{X}_1,\cdot),k'_{\mathbf{X}}(\cdot,\mathbf{X}_2)\rangle_{\mathcal{H}}
$$

我们就可以考虑对于分布的如下kernel：

$$
k_{P}(P_{\mathbf{X}}^{(1)},P_{\mathbf{X}}^{(2)})=\langle\Phi(P_{\mathbf{X}}^{(1)}),\Phi(P_{\mathbf{X}}^{(2)})\rangle=\int_{\mathcal{X_1}}\int_{\mathcal{X_2}}k'_{\mathbf{X}}(\mathbf{X}_1,\mathbf{X}_2)dP_{\mathbf{X}}^{(1)}dP_{\mathbf{X}}^{(2)}\tag{4}
$$

考虑通过采样估计的经验分布$$(\hat{P}_{\mathbf{X}}^{(1)},\hat{P}_{\mathbf{X}}^{(2)})$$，代入$$(4)$$之后有如下等式：

$$
k_{P}(\hat{P}_{\mathbf{X}}^{(1)},\hat{P}_{\mathbf{X}}^{(2)}):=\frac{1}{n_1 * n_2}\sum_{i=1}^{n_1}\sum_{j=1}^{n_2}k'_{\mathbf{X}}(\mathbf{X}_{1,i},\mathbf{X}_{2,j})\tag{5}
$$

式$(5)$定义了最基本的分布距离计算的内积形式。如果我们扩展到二阶矩核函数，即

$$
k_{P}^2(P_{\mathbf{X}}^{(1)},P_{\mathbf{X}}^{(2)})=\Vert\Phi(P_{\mathbf{X}}^{(1)})-\Phi(P_{\mathbf{X}}^{(2)})\Vert_2^2\\
=\langle\Phi(P_{\mathbf{X}}^{(1)}),\Phi(P_{\mathbf{X}}^{(1)})\rangle+\langle\Phi(P_{\mathbf{X}}^{(2)}),\Phi(P_{\mathbf{X}}^{(2)})\rangle
-2\langle\Phi(P_{\mathbf{X}}^{(1)}),\Phi(P_{\mathbf{X}}^{(2)})\rangle
$$

此时，我们就导出了著名的MMD距离。我们用一类函数$$\mathcal{T}$$来表示所有通过二阶矩函数的扩张所得到的核函数，记作

$$
\mathcal{T}(P_{\mathbf{X}}^{(1)},P_{\mathbf{X}}^{(2)})=F(\Vert\Phi(P_{\mathbf{X}}^{(1)})-\Phi(P_{\mathbf{X}}^{(2)})\Vert)
$$

显然，高斯核

$$
\mathcal{T}(P_{\mathbf{X}}^{(1)},P_{\mathbf{X}}^{(2)})=\exp(-\frac{\Vert\Phi(P_{\mathbf{X}}^{(1)})-\Phi(P_{\mathbf{X}}^{(2)})\Vert_2^2}{2\sigma^2})\tag{6}
$$

是$$F$$的一种特例。引入了核函数以后，我们就可以通过核函数的有界性以及**Lipschitz**连续性来搞事了。我们的泛化误差分析基于以下两条基本假设：

**损失函数假设：**损失函数$$l(\cdot,y)$$关于第一个变量满足**$$L_{l}$$-Lipschitz**连续，且具有上界$$B_{l}$$。

**核函数假设:**  核函数$$k_{\mathbf{X}},k'_{\mathbf{X}},\mathcal{T}$$ 都是有界函数，且上界分别为$$B^2_{k},B^2_{k'}\geq 1$$,以及$$B_{\mathcal{T}}^2$$。此外，注意到

$$
\Phi:\mathcal{B}_{\mathcal{X}}\rightarrow \mathcal{H}_{k’_{\mathbf{X}}}
$$

是一个将分布映射到核函数对应的**RKHS：**$$\mathcal{H}_{k’_{\mathbf{X}}}$$的映射，而在$$\mathcal{T}$$的作用下，我们在$$\Phi$$的基础上又得到了一个新的核，这个核也可以构建对应的**RKHS：**$$\mathcal{H}_{\mathcal{T}}$$，因此我们可以构造一个典范映射$$\psi_{\mathcal{T}}:\mathcal{H}_{k’_{\mathbf{X}}}\rightarrow\mathcal{H}_{\mathcal{T}}$$，我们对这个映射也施加一个约束如下：

$$
\forall v,w\in\mathcal{H}_{k’_{\mathbf{X}}},\exists \alpha\in(0,1],L_{\mathcal{T}}>0,\\
s.t.\Vert\psi_{\mathcal{T}}(v)-\psi_{\mathcal{T}}(w)\Vert\leq L_{\mathcal{T}}\Vert v-w\Vert^{\alpha} \tag{7}
$$

一个常用的结论是，对于如$$(6)$$式所述的高斯核函数，$$(7)$$式在$$\alpha=1$$的条件下是成立的。

利用这两个假设，我们可以得到如下定理。

**一致泛化误差定理：**基于损失函数假设与核函数假设，通过对采样集$$\{S_i\}_{i=1}^{N}$$进行训练所得的目标函数$$f$$遵循一致泛化误差定理。对任意$$R>0,s.t. \Vert f\Vert_2^2\leq R$$，以下不等式至少以$$1-\delta$$的概率成立：

$$
\sup_{\Vert f\Vert_2^2\leq R,f\in \mathcal{H}_{\bar{k}}}\vert \epsilon(f,\sum_{i=1}^{N}n_i)-\epsilon(f,\infty)\vert\leq \\
c(RB_{k}L_{l}(B_{k'}L_{\mathcal{T}}(\frac{\log N+\log\delta^{-1}}{n})^{\frac{\alpha}{2}}+B_{\mathcal{T}}\frac{1}{\sqrt{N}})+B_{l}\sqrt\frac{\log\delta^{-1}}{N}) \tag{8}
$$

### 对域泛化的采样方案与因果分析

公式$$(1)$$给出了域泛化问题测试误差界的计算，那么如何对测试集进行采样呢？我们可以通过如下条件分解进行因果探索：

$$
\epsilon(f,\infty) :=\mathbb{E}_{P_{\mathbf{X}\mathbf{Y}}\sim \mu}\mathbb{E}_{(\mathbf{X},\mathbf{Y})\sim P_{\mathbf{X}\mathbf{Y}}}\ [l(f(P_{\mathbf{X}},\mathbf{X}),\mathbf{Y})]\\
=\mathbb{E}_{P_{\mathbf{X}}\sim \mu_{\mathbf{X}}}\mathbb{E}_{P_{\mathbf{Y\vert X}}\sim \mu_{\mathbf{Y\vert X}}}\mathbb{E}_{\mathbf{X}\sim P_{\mathbf{X}}}\mathbb{E}_{\mathbf{Y\vert X}\sim P_{\mathbf{Y\vert X}}}\ [l(f(P_{\mathbf{X}},\mathbf{X}),\mathbf{Y})]\\
=\mathbb{E}_{P_{\mathbf{X}}\sim \mu_{\mathbf{X}}}\mathbb{E}_{\mathbf{X}\sim P_{\mathbf{X}}}\mathbb{E}_{P_{\mathbf{Y\vert X}}\sim \mu_{\mathbf{Y\vert X}}}\mathbb{E}_{\mathbf{Y\vert X}\sim P_{\mathbf{Y\vert X}}}\ [l(f(P_{\mathbf{X}},\mathbf{X}),\mathbf{Y})]
$$

因此，我们可以不必同时生成输入$$\mathbf{X}$$以及对应的标注$$\mathbf{Y}$$，我们可以先从$$\mu_{\mathbf{X}}$$中采样对应的输入域$$P_{\mathbf{X}}$$，然后从$$P_{\mathbf{X}}$$中采样$$\mathbf{X}$$，最后再根据$$\mathbf{X}$$得到对应的条件标注$$\mathbf{Y}$$。

## 基于元学习的域泛化

[元学习](https://drive.google.com/file/d/1DuHyotdwEAEhmuHQWwRosdiVBVGm8uYx/view)是一个**"learning to learn"**的机器学习领域，它的目的是期望我们所学到的特征能够容易地泛化到新的任务，新的数据集上去。基于域泛化问题中目标域的不可见，元学习可以应用于域泛化任务中。现有基于元学习的域泛化工作主要有三类：基于参数元学习的域泛化[3]，基于正则化元学习的域泛化[4]，以及基于**episode-training**的域泛化[5]。

**基于参数元学习的域泛化。**如何让模型学到**具有泛化能力**的特征呢？一个很自然的想法是在训练的过程中"模拟"泛化这一步骤，即对训练集进行重采样，将其分为若干训练-测试输入对，然后我们先模拟训练过程，用每一个输入对的训练部分更新模型参数，然后再用更新后的参数在测试部分进行模拟测试。最后，为了让模型参数学到"泛化"的能力，我们将测试的损失对原参数进行求导，再对原参数进行更新。其基本过程如下所示：

![Meta1](../../images/domain-generalization/meta1.png)

首先，我们通过`Line 5-7`计算在构建的输入对的训练部分的损失，对原参数$$\Theta$$进行求导并更新。注意，这一步更新并不会终止计算图，即更新后的参数$$\Theta'$$可以认为是原参数的函数$$f(\Theta)$$，然后我们代入更新后的参数，计算在测试部分的损失$$\mathcal{G}(f(\Theta))$$，再用这一部分损失去对$$\Theta$$求梯度，然后进行更新即可。这种方法有一个缺点，即在计算模拟测试损失$$\mathcal{G}$$时，计算图会被扩大一倍，因此基于参数元学习的域泛化方法限于模型本身的大小而无法进行有效的推广。

**基于正则化元学习的域泛化。**正如上文所述，直接采用元学习方法会使得计算图扩大，从而令方法受限于深度神经网络本身的参数量。一种解决思路是用模型正则化损失来代替测试部分的损失，减少参数量。此外，为了减少在元学习中所需要学习的参数，我们还可以把神经网络分为两个部分，即**Backbone**与**Classifier**，参数分别为$$\psi,\theta$$。此时模型可以表示为：

$$
M_{\Theta}(\mathbf{X})=(T_{\theta}\circ F_{\psi})(\mathbf{X})
$$

对参数量较少的$$\theta$$引入一个自适应的正则化损失如下，其参数为$$\phi$$：

$$
R_{\phi}(\theta)=\sum_{i}\phi_{i}\vert \theta_{i} \vert
$$

我们采样$$p$$个训练域与$$q$$个测试域，我们希望在训练域中得到的模型能够不经调整地泛化到测试域上。根据上文所述，训练算法可以描述为以下三部分：

![meta2](../../images/domain-generalization/meta2.png)

首先，我们对每一个训练域固定**Backbone**参数，而对后面的分类器参数进行区分，即参数空间为$$[\psi,\theta_1,\ldots,\theta_{p}]$$。在每一个训练的epoch中，我们都用$$p$$个训练域的数据以及对应的标签对整个参数空间进行训练，得到参数空间的良好初始化，如该算法中`Line 2-7`。然后，我们开始对训练域进行随机划分，模拟元学习的过程，如算法中的`Line 8-9`。接着，我们引入自适应的正则化工具，并采用联合损失，即

$$
L(\mathbf{X},\mathbf{y},\theta,\phi)=CE(\mathbf{X},\mathbf{y},\theta)+R_{\phi}(\theta)
$$

在训练集上更新后，得到更新参数$$\beta_l = f(\theta,\phi)$$，如`Line 9-14`。最后，我们在模拟的测试集上进行参数更新，此时我们并不直接更新$$\theta$$，而是更新正则化参数$$\phi$$，如`Line 15-16`。也就是说，整个元学习的过程由正则化部分完成。注意到对每一个训练域，我们都分配了一个独特的分类器参数$$\theta_{i}$$，因此，在最后测试的过程中，我们用每一个分类器参数进行预测，而对预测结果进行**ensemble**。

**基于episode-training的域泛化。**相对于更广泛的元学习问题，域泛化有其特殊性，即源域与目标域的特征接近，且预测任务基本一致。基于这一特性，文献[5]提出了**episode-training**，它的基本思路分为两块，首先，对每一个训练域，模型构造独立的分类参数$$[\theta_i,\psi_i]$$。然后，分别采用每个训练域的数据对分类参数进行训练，得到*domain-specific*的$$[\theta_i,\psi_i]$$。最后，用这些*domain-specific*的$$[\theta_i,\psi_i]$$训练一个*domain-agnostic*的全局模型$$[\theta,\psi]$$。训练的方法分为三阶段：

* Step 1. 全局训练

  $$
  \arg_{\theta,\psi}\min \text{CE}(\psi (\theta (\mathbf{X}),\mathbf{y})
  $$

* Step 2. Episodic training for $\theta$

  $$
  \arg_{\theta}\min \mathbb{E}_{i,j\sim[1,n],i\neq j}[\mathbb{E}_{(\mathbf{X}_i,\mathbf{y}_i)\sim S_i}[\text{CE}(\psi_j (\theta (\mathbf{X}_i),\mathbf{y}_i)]]
  $$


* Step 3. Episodic training for $\psi$

  $$
  \arg_{\psi}\min \mathbb{E}_{i,j\sim[1,n],i\neq j}[\mathbb{E}_{(\mathbf{X}_i,\mathbf{y}_i)\sim S_i}[\text{CE}(\psi (\theta_{j} (\mathbf{X}_i),\mathbf{y}_i)]]
  $$

其算法如下所述：

![meta3](../../images/domain-generalization/meta3.png)

此外，本文还探索了一种对异构域泛化问题的训练方案，即泛化任务与训练任务也不一致，基本思路是引入随机分类器来训练特征提取器：

$$
\arg_{\theta}\min \mathbb{E}_{i\sim[1,n]}[\mathbb{E}_{(\mathbf{X}_i,\mathbf{y}_i)\sim S_i}[\text{CE}(\psi_{\text{Random}} (\theta (\mathbf{X}_i),\mathbf{y}_i)]]
$$

## 基于域无关特征的域泛化

根据公式$$(8)$$所述的一致泛化误差定理，如果对于所有可能的域，它们在核空间上的分布距离都尽量接近，那么就意味着分布距离的上界$$B_{k'},B_{\mathcal{T}}$$以及$$L_{\mathcal{T}}$$都变得尽量小，同时泛化误差也会变小。将神经网络视作一个可以自主学习的核函数，一个从域迁移的工作中自然衍生到域泛化的思路为令不同训练域的输入在神经网络的输出特征层面不可分辨。那么，采用什么样的损失函数才能让所学的特征达到**不可分辨**的效果呢？我们首先回顾一下统计学中常用的方差分析(ANOVA)。

模型所学的特征**不可分辨**，等价于检验模型对于不同域的数据所预测特征的分布一致。假如存在$$p$$个训练域，在每个训练域上，对$$\mathbf{X}_{i,j}$$所预测特征为$$\mu_{i,j}$$，域$$S_i$$特征的均值为$$\mu_i$$，所有域的平均特征为$$\mu$$。那么要令模型特征不可分辨，即验证以下假设成立：

$$
H_0:\mu_1=\mu_2=\cdots=\mu_p
$$

基于正态性假设，我们可以用平方和来计算$$H_0$$成立所需的统计量：

$$
\text{SS}_{T}=\sum_{i=1}^{N}\sum_{j=1}^{n_i}\Vert\mu_{i,j}-\mu\Vert_2^2\\
\text{SS}_{E}=\sum_{i=1}^{N}\sum_{j=1}^{n_i}\Vert \mu_{i,j}-\mu_i\Vert_2^2\\
\text{SS}_{A}=\sum_{i=1}^{N}\sum_{j=1}^{n_i}\Vert \mu_{i}-\mu\Vert_2^2
$$

根据方差分析表，当统计量$$F=\frac{\text{SS}_{A}}{\text{SS}_{E}}$$比较小时，模型在不同域上特征的均值统计量基本一致。基于域无关特征的域泛化方案可以根据这种思路进行设计，现有方法基本分为x类：

## 基于生成模型的域泛化

## 基于自监督任务的域泛化

## 可用代码与模型验证

## 参考文献

[1] Blanchard G, Lee G, Scott C. Generalizing from several related classification tasks to a new unlabeled sample[C]//Advances in neural information processing systems. 2011: 2178-2186.

[2] Muandet K, Fukumizu K, Sriperumbudur B, et al. Kernel mean embedding of distributions: A review and beyond[J]. arXiv preprint arXiv:1605.09522, 2016.

[3] Li D, Yang Y, Song Y Z, et al. Learning to generalize: Meta-learning for domain generalization[J]. arXiv preprint arXiv:1710.03463, 2017.

[4] Balaji Y, Sankaranarayanan S, Chellappa R. Metareg: Towards domain generalization using meta-regularization[C]//Advances in Neural Information Processing Systems. 2018: 998-1008.

[5] Li D, Zhang J, Yang Y, et al. Episodic training for domain generalization[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 1446-1455.

