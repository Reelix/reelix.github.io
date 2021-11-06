---
layout: post
title: 隐私保护的深度学习系统——基于高斯机制的差分隐私DL
date: 2021-11-4 10:49:00
categories: 深度学习
tags: Differential Privacy, SGD, Privacy-preserving, Gaussian Mechanism
mathjax: true
---

在之前[关于差分隐私的Tutorial](https://www.fenghz.xyz/Differential-Privacy/)中，我们简单介绍了欧盟隐私保护条例，即个人对数据具有知情权，拒绝权，修正与遗忘权，以及对于自动决策过程的选择权。差分隐私对于数据分析过程提出了严格的隐私保护定义，即对于数据库数据的任何分析，以及根据分析结果与其他信息进行的进一步的合并加工，都不会泄露个人隐私。通俗而言，就是对于任何数据分析员，要求分析员在对数据库进行分析后，对数据库中每一个个体的了解不会超过其在分析开始之前的了解。差分隐私的基本原理是**控制单个数据对于整个分析结果的影响**，对于简单的数据处理过程（如计算平均工资，统计性别比例），通过在数据分析结果中增加高斯噪声，可以令数据分析的机制满足差分隐私的约束。但是，对于需要多轮训练的复杂深度学习系统，构建差分隐私保护则更为困难。本文主要介绍基于高斯机制的差分隐私深度学习系统：通过在训练过程中施加高斯噪声，构建满足**差分隐私要求**的深度学习训练系统，并对所得深度模型**计算隐私开销**。此外，我们也将以[**Opacus**](https://github.com/pytorch/opacus)这一基于pytorch的差分隐私训练库为例进行代码讲解。





本文主要参考的文献为

1. [A Tutorial to Differentially Private Machine Learning, Neurips17](https://www.ece.rutgers.edu/~asarwate/nips2017/NIPS17_DPML_Tutorial.pdf)
2. [Deep learning with differential privacy](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)
3. [Rényi differential privacy](https://ieeexplore.ieee.org/abstract/document/8049725/)
4. [Subsampled Rényi Differential Privacy and Analytical Moments Accountant](https://arxiv.org/abs/1808.00087)
5. [Rényi Differential Privacy of the Sampled Gaussian Mechanism](https://arxiv.org/abs/1908.10530)
6. [The Composition Theorem for Differential Privacy](http://proceedings.mlr.press/v37/kairouz15.html)
   


## 差分隐私深度学习系统的基本框架

深度学习的训练与推断流程通常由四步组成：构建深度模型并初始化模型参数；输入训练数据并计算梯度；用梯度下降法更新模型直到收敛；将收敛的模型用于数据预测。最直接的隐私保护方法是对模型预测直接添加噪声，由于深度学习的模型参数非常容易获取，因此仅仅在预测阶段添加噪声往往无法达到隐私保护的目的。利用差分隐私的传递性特征，即对于满足$$(\epsilon,\delta)-DP$$的随机算法$$\mathcal{M}$$，对其结果进行任何形式的处理所构成的新算法$$f\circ \mathcal{M}$$同样满足$$(\epsilon,\delta)-DP$$，可得对深度学习系统添加差分隐私的环节应当尽量靠前。然而，在数据集上直接增加噪声往往会使得数据集本身变得脏而不可用（想象一下需要令两张照片的统计信息不可分辨所需要的噪声尺度）。另一个想法是在训练好的模型参数上直接添加噪声，即$$\tilde{\theta}=\theta+\text{Noise}$$，利用高斯机制，添加噪声后的参数也具备差分隐私的特性，该特性可以传递到所有的后续推断过程。但是，深度神经网络是一个对参数非常敏感的黑箱，直接对训练好的参数添加高斯噪声会使得整个算法完全失效，因此差分隐私机制需要在训练过程中加入。通过在训练过程中对梯度进行裁剪并添加高斯噪声，可以保证训练过程的差分隐私性质。

首先，我们定义深度学习系统中的差分隐私如下：

**(Definition 1. Differential Privacy in Deep Training System)** 记数据集为$$D$$，深度学习的模型参数为$$\theta$$，由所有$$D$$的子集组成的训练数据库记为$$\mathcal{D}$$，参数空间为$$\mathcal{T}$$。一个具有随机性质的深度学习训练机制将训练数据集作为输入，采用梯度下降法进行训练，输出训练后的参数，记作$$\mathcal{M}:\mathcal{D}\rightarrow\mathcal{T}$$。我们称这一训练机制满足$$(\epsilon,\delta)$$-DP，如果对于任意两个毗邻的训练集$$d,d'\in \mathcal{D}$$, 以及任何参数范围$$\mathcal{S}\subset \mathcal{T}$$，其输出的参数分布满足：

$$
Pr[\mathcal{M}(d)\in S]\leq e^{\epsilon}Pr[\mathcal{M}(d')\in S]+\delta
$$

在该定义下，数据集中的单个数据对于模型的影响被控制在一定的范围内，实现了在模型层面对于差分攻击的“不可分辨性”。此外，为了叙述方便，我们提前引入差分隐私的一个变种——**Rényi Differential Privacy (RDP)**。我们将在计算隐私开销时体会到**RDP**的威力。

**(Definition 2. Rényi Differential Privacy in DL-system)** 对于**Definition 1**中所述的训练机制$$\mathcal{M}$$，用$$\mathcal{M}(d)(\theta)$$表示在训练集$$d$$上训练所得结果为$$\theta$$的概率密度，即$$\mathcal{M}(d)(\theta)=\Pr[\mathcal{M}(d)=\theta]$$。我们称训练机制$$\mathcal{M}$$对于所有的$$\alpha\in (1,\infty)$$满足$$(\alpha,\epsilon)-$$RDP，如果对于所有可能的相邻训练集$$d,d'\in\mathcal{D}$$满足

$$
D_{\alpha}(\mathcal{M}(d)\Vert \mathcal{M}(d'))=\frac{1}{\alpha-1}\log\mathbb{E}_{\theta\sim \mathcal{M}(d')}[\frac{\mathcal{M}(d)(\theta)}{\mathcal{M}(d')(\theta)}]^{\alpha}\leq \epsilon
$$

**(Proposition 1. From RDP to DP)** RDP和DP可以进行直接转换，如果训练机制$$\mathcal{M}$$服从$$(\alpha,\epsilon)-$$RDP，那么对于所有的$$\delta,0<\delta<1$$，$$\mathcal{M}$$服从$$(\epsilon+\log(1/\delta)/\alpha-1,\delta)-$$DP。


在深度学习的训练中，模型参数采用梯度下降法进行更新，即$$\theta_{t+1}\leftarrow \theta_{t}-\frac{\eta_t}{N}\sum_{i=1}^{N}\nabla_{\theta_t}\text{loss}(x_i,\theta_t)$$，并且$$\theta_0$$是随机初始化参数。梯度是利用输入数据直接进行计算的结果，也是模型参数进行更新的主要运算，因此在梯度上施加差分噪声是自然的事情。为了控制个体数据的影响，文献[1]利用高斯机制对梯度施加差分隐私。高斯机制的定义如下：

**(Definition 3. Gaussian Mechanism)** 假设存在一个确定函数$$f:\mathcal{D}\rightarrow\mathcal{T}$$，敏感度为$$\Delta_2(f)=\max_{d,d'\in \mathcal{D}}\Vert f(d)-f(d')\Vert_2$$，那么对于任意的$$\delta\in(0,1)$$，给定随机噪声服从正态分布$$\mathcal{N}(0,\sigma^2)$$，那么随机算法$$\mathcal{M}(d)=f(d)+\mathcal{N}(0,\sigma^2)$$服从$$(\epsilon,\delta)-$$DP，其中

$$
\epsilon\geq\frac{\sqrt{2 \ln (1.25 / \delta)}}{\frac{\sigma}{\Delta_2 f}}
$$

利用高斯机制，我们对梯度分三个步骤增加差分噪声：首先，对每一个样本对应的梯度裁剪到一个固定范围$$[-C,C]$$，以控制个体数据的影响，此时梯度的敏感度$$\Delta_2(f)=\max_{x_i\in D}\Vert \nabla_{\theta}\text{loss}(x_i,\theta)\Vert_2\leq C$$。然后，对裁剪后的梯度增加高斯噪声$$\mathcal{N}(0,\sigma^2)$$，以得到满足差分隐私的梯度数据。最后，用这些梯度更新模型，并计算模型的隐私损失。记噪声乘子(*noise multiplier*)为$$z=\frac{\sigma}{C}$$，那么该训练系统服从$$(\epsilon,\delta)-$$DP的条件为

$$
z=\frac{\sqrt{2 \ln (1.25 / \delta)}}{\epsilon} \tag{1}
$$

完整算法如下图所述：

![1](/images/differential_privacy_dl/1.PNG)

模型一共进行了$$T$$轮训练，在每轮训练中，对训练集的每一个样本计算梯度，进行裁剪加噪，最后用满足差分隐私的梯度$$\tilde{\mathbf{g}}_t$$进行参数更新。当$$z$$满足$$(1)$$中所述的条件时，对于第$$t$$轮所选择的训练集，所得的参数$$\theta_{t+1}$$满足$$(\epsilon,\delta)-$$DP。注意到该算法引入了一个新的过程，子采样(subsample)，即在每一轮的训练集是整个训练集的一个子集，通过概率为$$q=L/N$$的不放回采样进行选取。一个普遍的结论是，随着采样率的增加，训练时长变为$$1/q$$倍，但是隐私界与$$q^2$$成正比，因此subsample操作可以用采样率对差分隐私进行amplify，从而降低隐私损失。此外，由于模型要进行$$T$$轮训练，如何计算$$T$$轮训练的总和隐私损失也是一个关键问题。我们就Subsample与Composition两个问题进行描述。

### Subsample：用采样率对隐私进行Amplify，降低差分隐私损失
在深度学习的训练过程中，由于数据量比较大，我们往往采用基于随机采样的梯度下降法，即每次选取一个Batch，在该Batch上计算平均梯度，采用它们的平均梯度进行梯度下降。那么，这种基于采样的方法对隐私损失会有什么样的影响呢？我们先不加证明地给出一个结论，即采样会增强隐私保护的力度，降低隐私损失。首先对Subsample给出如下的定义：

**(Definition 4. Subsample)** 给定一个含有$$N$$个样本的数据集$$X=\{x_1,\cdots,x_N\}$$，**Subsample**操作对于$$X$$的所有大小为$$L$$的子集以等概率进行一次无放回采样，记$$q = \frac{N}{L}$$为采样率。

考虑先对数据集$$X$$进行Subsample，然后在子集上用具有差分隐私性质的训练机制$$\mathcal{M}$$进行训练，文献[4]指出了如下结论：如果训练机制$$\mathcal{M}$$在训练集上满足$$(\epsilon,\delta)-$$DP，那么$$\mathcal{M'}=\mathcal{M}\circ\text{Subsample}$$对于整个数据集$$X$$满足$$(\epsilon',\delta')-$$DP，其中$$\epsilon'=\log(1+q(e^{\epsilon}-1)),\delta'=q\delta$$。

当$$\epsilon$$很小的时候，我们有$$\epsilon'=\mathcal{O}(q\epsilon)$$，是$$q\epsilon$$的小阶，此时虽然模型需要的训练轮次变多了，但是从整体而言，模型的隐私损失变小了。

对于**Rényi Differential Privacy**，文献[3,4,5]都对高斯机制的Subsample进行了研究，提出了更紧的基于Subsample的隐私损失，如下所述：

![2](/images/differential_privacy_dl/2.PNG)

![3](/images/differential_privacy_dl/3.PNG)

在这些Bound中，某些情况下$$\epsilon'$$会成为$$q^2\epsilon$$的小阶，这大大减少了隐私损失。

### Composition Theorem：计算整个训练系统的差分隐私损失

深度学习系统在经过一轮由Subsample，梯度计算，梯度裁剪，高斯加噪组成的训练后，得到了一组满足$$(\epsilon,\delta)-$$DP的参数$$\theta_t$$，然后将$$\theta_t$$作为新的初始参数进行下一轮训练。经过$$T$$轮训练后，模型收敛。假设整个训练过程是公开的，即从第一到第$$T$$轮的所有模型参数都是可以获取的，那么如何去判断整个训练机制的隐私损失呢？这就是Composition Theorem所解决的问题，它用于计算整个训练系统的差分隐私损失。一个直觉是，一个由$$T$$个满足$$(\epsilon,\delta)-$$DP的机制$$\mathcal{M}_t$$所组成的队列系统$$M$$的隐私损失至多是$$(T\epsilon,T\delta)$$。这个损失界是否可以进一步减少呢？Strong Composition Theorem提出，T个机制进行Composition后，隐私损失变为$$(\tilde{\epsilon},\tilde{\delta})$$，其中

$$
\tilde{\epsilon}=\epsilon\sqrt{2T\ln(1/\delta')}+T\epsilon\frac{e^{\epsilon}-1}{e^{\epsilon}+1};\tilde{\delta}=T\delta+\delta'
$$

一般而言我们取$$\delta'=\delta$$。结合Subsample定理，当$$\epsilon\rightarrow 0$$时，Strong Composition给出了$$(\mathcal{O}(q\epsilon\sqrt{T\ln(1/\delta)},q(T+1)\delta)$$的隐私损失。但是这个隐私损失与$$\delta$$相关，当$$\delta$$很小时，该损失变得非常大。因此，文献[2]提出了一个在深度学习的训练过程中计算总隐私损失的有效方法，即矩会计方法（Moments Accountant），它将Composition的隐私损失界降低到了$$(q\epsilon\sqrt{T},\delta)$$，是广泛采用的隐私损失界。该隐私损失界的基本思想是将每一轮训练的隐私损失看成随机变量，而将总隐私损失看成是各轮随机变量的加和分布，通过计算随机变量的矩生成函数（moment generating function），得到更精准的隐私界。该方法最终可以归结为RDP的计算，并且在高斯机制下具有解析解，我们对其进行详细的介绍。

### The Moments Accountant and RDP



## Opacus库：基于Pytorch框架的隐私保护库

## PySyft + Opacus：结合差分隐私与联邦学习

## 我们能做的开放性问题


现在的隐私保护都与数据，模型独立。如何做modle specific，data specific的隐私保护呢？

## Appendix：Rényi 散度的基本性质与RDP的比较优势
**(Rényi Divergence).** 给定两个满足离散分布的随机变量$$\mathbf{X}$$和$$\mathbf{Y}$$，它们具有 $$n$$ 个可能的值，每个值分别具有正概率 $$p_i$$ 和$$ q_i$$，随机变量$$\mathbf{X}$$和$$\mathbf{Y}$$的 Rényi 散度定义为

$$
D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})=\frac{1}{\alpha-1} \log \left(\sum_{i=1}^{n} \frac{p_{i}^{\alpha}}{q_{i}^{\alpha-1}}\right)
$$

其中$$\alpha > 0$$并且$$\alpha \neq 1$$。

对于Rényi散度已经有很多研究，推荐阅读[Rényi散度与KL散度的关系](https://arxiv.org/pdf/1206.2459.pdf)。Rényi散度有如下几个性质：

1. 对于两个正态分布$$\mathbf{X}\sim N(\mu_0,\sigma_0^2),\mathbf{Y}\sim N(\mu_1,\sigma_1^2)$$：
   $$
   D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})=\frac{\alpha\Vert\mu_0-\mu_1\Vert_2^2}{2[(1-\alpha)\sigma_0^2+\alpha\sigma_1^2]}+\frac{1}{1-\alpha}\ln\frac{(1-\alpha)\sigma_0^2+\alpha\sigma_1^2}{\sigma_0^{1-\alpha}\sigma_1^\alpha}
   $$

2. $$D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})$$关于$$\alpha$$是单调不减函数。

3. 定义$$D_{0}(\mathbf{X}\Vert \mathbf{Y})=\lim_{\alpha\rightarrow 0,\alpha>0}D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})$$，则$$D_{0}(\mathbf{X}\Vert \mathbf{Y})=-\ln(\sum_{i:p_i>0}q_i)$$。

4. 定义$$D_{1}(\mathbf{X}\Vert \mathbf{Y})=\lim_{\alpha\rightarrow 1}D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})$$，则$$D_{1}(\mathbf{X}\Vert \mathbf{Y})=D_{KL}(\mathbf{X}\Vert \mathbf{Y})$$。

   注意这条性质的证明实际上有点复杂，需要先用$$\ln_{x\rightarrow 1} x=x-1$$，注意到$$ \sum_{i=1}^{n} \frac{p_{i}^{\alpha}}{q_{i}^{\alpha-1}}\rightarrow^{\alpha\rightarrow1}1$$，因此$$\log \left(\sum_{i=1}^{n} \frac{p_{i}^{\alpha}}{q_{i}^{\alpha-1}}\right)\rightarrow \sum_{i=1}^{n} \frac{p_{i}^{\alpha}}{q_{i}^{\alpha-1}}-1$$，然后将原式写成

   $$
   \lim_{\alpha\rightarrow 1}D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})=\lim_{\alpha\rightarrow 1}\frac{1}{\alpha-1} \left(\sum_{i=1}^{n} \frac{p_{i}^{\alpha}}{q_{i}^{\alpha-1}}-1\right)=\sum_{p_i,q_i>0}\lim_{\alpha\rightarrow 1}\frac{p_i-p_{i}^{\alpha}q_{i}^{1-\alpha}}{1-\alpha}
   $$

   再利用积分中值定理

   $$
   \forall p,q>0,\frac{p-p^\alpha q^{1-\alpha}}{1-\alpha}=\frac{1}{1-\alpha}\int_{\alpha}^1p^zq^{1-z}\ln \frac{p}{q}dz=p^{\sigma}q^{1-\sigma}\ln \frac{p}{q},\sigma \in (\alpha,1)
   $$

   代入后就是
   
   $$\lim_{\alpha\rightarrow 1}D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})=\sum p_i \ln\frac{p_i}{q_i}=D_{KL}(\mathbf{X}\Vert \mathbf{Y})
   $$

5. 定义$$D_{\infty}(\mathbf{X}\Vert \mathbf{Y})=\lim_{\alpha\rightarrow \infty}D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})$$，则$$D_{\infty}(\mathbf{X}\Vert \mathbf{Y})=\ln \sup_{\mathcal{S}\subset \mathcal{O}}\frac{\operatorname{Pr}[\mathbf{X} \in S]}{\operatorname{Pr}\left[\mathbf{Y} \in S\right]}$$。

### RDP的总结与比较优势

1. RDP是DP的一种自然的推广，当$$\alpha\rightarrow \infty$$，RDP等价于DP。而$$D_{\alpha}(\mathbf{X}\Vert \mathbf{Y})$$关于$$\alpha$$是单调不减函数，所以DP是RDP的一个上界，RDP的Bound更紧。
2. RDP与DP共享了很多特性，而RDP的计算更加简单，将复杂的遍历计算归结于一个散度计算，这使得RDP可以在实际中使用。
3. RDP在高斯机制下的计算很简单，结论更加有用。
4. 在相同的隐私损失下，RDP能够添加更小的噪声，使得查询更加准确。






