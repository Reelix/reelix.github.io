---
layout: post
title: Denoising Criterion for Variational Autoencoding Framework
date: 2018-12-18 12:32:00
categories: 深度学习
tags: Variational Inference
mathjax: true
---

* content
{:toc}

## 大纲

对输入图像加噪声并输入编码器(AutoEncoder)，并让编码器进行去噪处理(即重构出原图像)是提高自编码器泛化性的一个非常好的方法，基于这种方法的自编码器又称为[去噪自编码器(Denoising AutoEncoder)](https://arxiv.org/abs/1511.06406)。去噪规范(Denoising Criterion)能够增强在潜变量空间中那些比较近的数据点投射到高维观测空间后对于噪声的鲁棒性，提高模型的泛化能力。因此将去噪规范与变分自编码器(Variational AutoEncoder)相结合，对于提高变分自编码器泛化能力具有很大帮助。

变分自编码器在训练中用到的重参数化技巧$z=\mu +\Sigma \times \epsilon,\epsilon \sim N(0,I)$其实可以看成是对隐藏层的一种噪声注入，$\epsilon$即为注入的噪声，而去噪编码器可以看作是对于输入$x$的一种噪声注入$\tilde{x}=x+\epsilon$。由于变分自编码器本身具有与编码器类似的结构，因此在其中加入去噪规范的一个自然的训练流程可以如下所述：

1. 给定一个对原数据进行corrupt的噪声分布$p(\tilde{x}\vert x)$，并从中采样$\tilde{x}$作为网络的输入
   
2. 将$\tilde{x}$映射到潜变量空间，并对潜变量空间分布参数$\tilde{q}(z\vert \tilde{x})$进行预测，同时令给定加了噪声的$\tilde{x}$后对潜变量$z$分布的条件预测$\tilde{q}(z\vert \tilde{x})$尽量接近真实分布$p(z\vert x)$,也就是最小化$\mathcal{D}[\tilde{q}(z\vert \tilde{x})\Vert p(z\vert x)]$

3. 从$\tilde{q}(z\vert \tilde{x})$中对$z$进行采样，并对$p(x\vert z)$进行预测，要求预测$p(x\vert z)$接近于$x$的真实分布，并最大化$x$的出现概率(最大似然思想)





注意到，这个自然的训练流程包括在潜变量空间的去噪过程$\mathcal{D}[\tilde{q}(z\vert \tilde{x})\Vert p(z\vert x)]$以及在生成空间(高维空间,观测空间)的重构过程。

那么一个自然的问题是，$\mathcal{D}[\tilde{q}(z\vert \tilde{x})\Vert p(z\vert x)]$这个公式与$log(p(x))$之间可以进行怎样的联系呢？传统变分自编码器有一个很漂亮的核心公式是

$$
log(P(X))-\mathcal{D}[Q(z\vert X)\Vert P(z\vert X)]=E_{z\sim Q}[log(P(X\vert z))]-\mathcal{D}[Q(z\vert X)\Vert P(z)]
$$

因此，本文最大的贡献是提出了$\mathcal{D}[\tilde{q}(z\vert \tilde{x})\Vert p(z\vert x)]$与$log(p(x))$的关系:

$$
log(p(x))=\mathcal{L}_{dvae}+E_{p(\tilde{x}\vert x)}[\mathcal{D}[q(z\vert \tilde{x})\Vert p(z\vert x)]] \tag{1}
$$

其中

$$
\mathcal{L}_{dvae}=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]\\
\tilde{q}(z\vert x)=\int_{\tilde{x}} q(z\vert \tilde{x})p(\tilde{x}\vert x)d\tilde{x}
$$

## 证明

为了证明$(1)$式中的结论，文中提出了以下几条引理与技巧：

1. 给图像加噪的过程等价于在推断网络(编码部分)的输入层加入一层或几层随机输出层，我们把推断网络的参数记作$\phi$,把随机输出层记作$\psi$，整个网络参数记录为$\Phi=\{\phi,\psi\}$，同时我们把从潜变量还原输入(对输入分布进行推断)的神经网络层的参数记作$\theta$.在这种记号下，我们可以对$(2),(3)$的表达进行如下改进:
   
   $$
   q(z\vert x)=q_{\Phi}(z\vert x)\\
   p(\tilde{x}\vert x)=p_{\psi}(\tilde{x}\vert x)\\
   q(z\vert \tilde{x})=q_{\phi}(z\vert \tilde{x})\\
   p(x\vert z)=p_{\theta}(x\vert z)\\
   q_{\Phi}(z\vert x)=\tilde{q}(z\vert x)=\int_{\tilde{x}} q_{\phi}(z\vert \tilde{x})p_{\psi}(\tilde{x}\vert x)d\tilde{x}
   $$

2. 引理$1$:
   
   $$
   q_{\Phi}(z\vert x)=\int_{\tilde{x}}q_{\phi}(z\vert \tilde{x})q_{\psi}(\tilde{x}\vert x)d\tilde{x}
   $$

   同时

   $$
   log(p_{\theta}(x))\geq E_{q_{\Phi}(z\vert x)}[log(\frac{p_{\theta}(x,z)}{q_\phi(z\vert \tilde{x})})]\geq E_{q_{\Phi}(z\vert x)}[log(\frac{p_{\theta}(x,z)}{q_{\Phi}(z\vert x)})] \tag{2}
   $$

3. 命题$1$:

   假设$q_{\phi}(z\vert \tilde{x})$是高斯分布，满足$q_{\phi}(z\vert \tilde{x})\sim N(z\vert \mu_{\phi}(\tilde{x}),\sigma_{\phi}(\tilde{x}))$。令$p(\tilde{x}\vert x)$是一个已知的加噪分布，我们有：

   $$
   E_{p(\tilde{x}\vert x)}[q_{\phi}(z\vert \tilde{x})]=\int_{\tilde{x}}q_{\phi}(z\vert\tilde{x})p(\tilde{x}\vert x)d\tilde{x}
   $$

   是多个高斯分布的和(也是高斯分布)

### 引理$1$的证明

在直觉上，引理$1$通过将网络分为推断部分$\phi$以及加噪部分$\psi$，将网络分为了两个部分，并用贝叶斯公式给出了这两个部分的关系，最后通过它们之间的关系给出了去噪部分的变分下界$\mathcal{L}_{dvae}$。

首先我们来证明一下

$$
E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]
$$

这是因为

$$
E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=\int_{z}\tilde{q}(z\vert x)[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]dz\\
=\int_{z}\int_{\tilde{x}} q(z\vert \tilde{x})p(\tilde{x}\vert x)d\tilde{x}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]dz\\
=\int_{z}q(z\vert \tilde{x})\int_{\tilde{x}} p(\tilde{x}\vert x)[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]dzd\tilde{x}\\
=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]
$$

然后我们证明一下引理$1$.

在引理$1$中，

$$
q_{\Phi}(z\vert x)=\int_{\tilde{x}}q_{\phi}(z\vert \tilde{x})q_{\psi}(\tilde{x}\vert x)d\tilde{x}
$$

这是利用贝叶斯公式所推导出的一个非常自然的结论。

式$(2)$中不等式的证明需要利用一个结论:

$$
E_{f(x)}[log\frac{g(x)}{f(x)}]\leq log[E_{f(x)}\frac{g(x)}{f(x)}]=0
$$

这是显然的，因为$log[E_{f(x)}\frac{g(x)}{f(x)}]=0$, 同时$E_{f(x)}[log\frac{g(x)}{f(x)}]=-\mathcal{D}[f(x)\Vert g(x)]\leq 0$

因此我们有

$$
E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]\leq log[E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}\frac{p(x,z)}{q(z\vert \tilde{x})}]=log(p(x))
$$

然后我们需要证明

$$
E_{q_{\Phi}(z\vert x)}[log(\frac{p_{\theta}(x,z)}{q_\phi(z\vert \tilde{x})})]\geq E_{q_{\Phi}(z\vert x)}[log(\frac{p_{\theta}(x,z)}{q_{\Phi}(z\vert x)})]\tag{3}
$$

应该怎么证明呢？这其实是显然的事情，因为等式就差一个期望项分母，$(3)$等价于证明

$$
E_{q_{\Phi}(z\vert x)}log(q_{\Phi}(z\vert x)) -E_{q_{\Phi}(z\vert x)}log(q_{\phi}(z\vert \tilde{x})) \geq 0
$$

因为$E_{q_{\Phi}(z\vert x)}log(q_{\Phi}(z\vert x)) -E_{q_{\Phi}(z\vert x)}log(q_{\phi}(z\vert \tilde{x})) =\mathcal{D}[q_{\Phi}(z\vert x) \Vert q_{\phi}(z\vert \tilde{x})]$

因此显然成立。

注意到引理$1$中不等式的最右项

$$
E_{q_{\Phi}(z\vert x)}[log(\frac{p_{\theta}(x,z)}{q_{\Phi}(z\vert x)})]
$$

正是如果我们把推断网络合并为$\Phi$并整体看待的情况下**VAE**的传统下界，因此这个等式的威力在于它通过仅使用推断网络$\phi$的参数以及用推断网络对直接输入加了噪声的图像$\tilde{x}$进行潜变量分布推断$q_{\phi}(z\vert \tilde{x})$就可以生成更好的变分下界，而将到底是如何加噪声这一部分与推断网络彻底独立开。同时，利用

$$
E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{p(\tilde{x}\vert x)}E_{q(z\vert \tilde{x})}log(p_{\theta}(x\vert z))-E_{p(\tilde{x}\vert x)}\mathcal{D}[q_{\phi}(z\vert \tilde{x})\Vert p(z)]
$$

这一等式，我们可以对方程进行估计

### 式$(1)$的证明

$$
log(p(x))=\mathcal{L}_{dvae}+E_{p(\tilde{x}\vert x)}[\mathcal{D}[q(z\vert \tilde{x})\Vert p(z\vert x)]]\tag{1}
$$



其实这就是一个非常简单的微积分证明过程，采用

$$
\mathcal{L}_{dvae}=E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]\\
log(p(x))=E_{\tilde{q}(z\vert x)}[log(p(x))]
$$

我们只需证明:

$$
log(p(x))-\mathcal{L}_{dvae}=E_{p(\tilde{x}\vert x)}[\mathcal{D}[q(z\vert \tilde{x})\Vert p(z\vert x)]] \tag{4}
$$

用到一个我们上文中反复使用的技巧

$$
E_{\tilde{q}(z\vert x)}=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}
$$

首先，我们得凑出$p(z\vert x)$这么一项，那么我们有$p(x)$了有$p(x,z)$了，显然$p(x,z)=p(x)p(z\vert x)$，因此$(4)$左边就是

$$
E_{\tilde{q}(z\vert x)}[\frac{q(z\vert \tilde{x})}{p(z\vert x)}] =E_{p(\tilde{x}\vert x)}E_{q(z\vert \tilde{x})}[\frac{q(z\vert \tilde{x})}{p(z\vert x)}]=E_{p(\tilde{x}\vert x)}[\mathcal{D}[q(z\vert \tilde{x})\Vert p(z\vert x)]]
$$

因此，式$(1)$告诉我们，当我们最大化$L_{dvae}$的时候，我们事实上在最大化$log(p(x))$的同时最小化$E_{p(\tilde{x}\vert x)}[\mathcal{D}[q(z\vert \tilde{x})\Vert p(z\vert x)]]$，也就是最小化真正的后验分布$p(z\vert x)$与我们用加了噪声的数据估计的后验分布分布$q(z\vert \tilde{x})$之间的差距。

## 训练过程细节与训练技巧

### 训练过程细节

我们训练的目标函数是

$$
\mathcal{L}_{dvae}=E_{\tilde{q}(z\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]=E_{q(z\vert \tilde{x})}E_{p(\tilde{x}\vert x)}[log(\frac{p(x,z)}{q(z\vert \tilde{x})})]\\ =E_{p(\tilde{x}\vert x)}E_{q(z\vert \tilde{x})}log(p_{\theta}(x\vert z))-E_{p(\tilde{x}\vert x)}\mathcal{D}[q_{\phi}(z\vert \tilde{x})\Vert p(z)]
$$

在训练过程中，我们要比较好地去估计期望，因为这里有两个期望(交换期望的过程就是交换变量的过程，没事，都可以换)，因此我们需要逐个对期望进行估计，估计次序如下:

1. 首先，从分布$p(\tilde{x}\vert x)$中采样$M$个$\tilde{x}^{m}$,对于每一个$\tilde{x}^m$依次计算$\mathcal{D}[q_{\phi}(z\vert \tilde{x}^m)\Vert p(z)]$，然后取$\sum _{m=1}^M\frac{\mathcal{D}[q_{\phi}(z\vert \tilde{x}^m)\Vert p(z)]}{M}$作为$E_{p(\tilde{x}\vert x)}\mathcal{D}[q_{\phi}(z\vert \tilde{x})\Vert p(z)]$的估计
   
2. 对于每一个$\tilde{x}^m$,从$q(z\vert \tilde{x}^m)$中采样$K$个$z^k$，然后计算$\sum_{m=1}^M\sum_{k=1}^K\frac{log(p_{\theta}(x\vert z^k))}{MK}$作为$E_{p(\tilde{x}\vert x)}E_{q(z\vert \tilde{x})}log(p_{\theta}(x\vert z))$的估计
   
3. 进行反向传播

### 训练技巧

1. 噪声对抗真的起作用了吗？

   当然起作用啦！$\mathcal{L}_{dvae}$在用了噪声之后比之前的更低了。

2. 噪声强度和噪声类型各有什么影响呢？

   噪声如果太强的话，就完全丢失了原来输入$x$的信息了，此时模型无法对潜变量空间进行推断

   作者实验了椒盐噪声以及胡椒噪声，发现什么样的噪声并没有关系

3. 怎样选$K,M$呢？

   你可以用$MCMC$方法来选择，但是其实你多选几次$K,M$不如多训练几次呢，所以最好也是最粗暴的选择:$K=1,M=1$.

4. Data Augmentation和 Data corruption哪个更有用呢？

   在MNIST上的VAE训练中，Data Augmentation比Data corruption效果更好。但是我们可能可以同时使用？

5. 除了椒盐噪声和其他噪声，有没有更sensible一点的噪声呢？

   这是一个很好的问题，也是下一步的探究方向。

   对于我这个做医学影像的人而言，我觉得这个噪声可以是设备之间的采样误差。我们知道医学影像本质上就是一台设备对于人体的采样，而不同医院的不同设备之间采样是有误差的。如果我们把这个采样误差分布用噪声刻画出来，然后用它来作为噪声批评，并让网络进行训练的话，是不是可以解决来自不同医院数据集不能混用的问题呢？

   这种噪声应该来说是非常有价值的。