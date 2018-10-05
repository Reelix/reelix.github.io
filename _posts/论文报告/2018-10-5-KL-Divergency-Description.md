---
layout: post
title: A tutorial of Kullback-Leibler divergence
date: 2018-10-5 20:10:00
categories: 机器学习
tags: "Probability Theory"
mathjax: true
---

* content
{:toc}

参考资料:[Jon Shlens' tutorial on Kullback–Leibler divergence and likelihood theory](https://arxiv.org/abs/1404.2000)

**FengHZ's Blog 首发原创**

## 笔记大纲

KL散度(Kullback-Leibler divergence)是用来度量两个概率分布相似度的指标,它作为经典损失函数被广泛地用于聚类分析与参数估计等机器学习任务中.

本文将从以下几个角度对KL散度进行介绍:

* KL散度的定义与基本性质
* 从采样角度出发对KL散度进行直观解释
* KL散度的应用以及常见分布KL散度的计算




## KL散度的定义与基本性质

### KL散度的定义

假设对随机变量$\xi$,存在两个概率分布$P,Q$.

如果$\xi$为离散随机变量,定义从$P$到$Q$的KL散度为:

$$
D_{KL}(P\vert \vert Q)=\sum_{i}P(i)ln(\frac{P(i)}{Q(i)})
$$

如果$\xi$为连续随机变量,则定义从$P$到$Q$的KL散度为:

$$
D_{KL}(P\vert \vert Q)=\int_{-\infty}^{\infty}p(x)ln(\frac{p(x)}{q(x)})dx
$$

注意连续随机变量的KL散度可以看作离散随机变量KL散度求和的极限,为了叙述简洁我们只讨论离散KL散度.

### KL散度的基本性质

#### 非负性

$D_{KL}(P\vert \vert Q)\geq 0$,且$D_{KL}=0$当且仅当$P=Q$.

证明(仅对离散情况进行证明,连续随机变量情况将积分化为求和的极限则同理可证)：

证明$\sum_{i}P(i)ln(\frac{P(i)}{Q(i)})\geq 0$等价于证明$\sum_{i}P(i)ln(\frac{Q(i)}{P(i)})\leq 0$.

采用$ln(x)\leq x-1,\forall x>0$,则:

$$
\sum_{i}P(i)ln(\frac{Q(i)}{P(i)})\leq \sum_{i}P(i)(\frac{Q(i)}{P(i)}-1)=0
$$

等号当且仅当 $\frac{Q(i)}{P(i)}=1,\forall i$ 时取得,此时有$P=Q$.

#### 非对易性

$D_{KL}(P\vert \vert Q)\neq D_{KL}(Q\vert \vert P)$ 

Trivial!

#### 值域

$D_{KL}(P\vert \vert Q)$在一定条件下可以趋向于无穷.  

Trivial！

## 从采样角度出发对KL散度进行直观解释

**KL散度描述了我们用分布$Q$来估计数据的真实分布$P$的编码损失.**

假设我们对于离散随机变量$\xi$进行了$n$次采样,并得到了$\xi$取值的一组观测$c=\{c_i\}$($c_i$描述了随机变量$\xi$取值为$i$的次数),观测$c$由分布$Q$生成的概率可以进行以下描述:

$$
L^n(c\vert Q)=C_n^{c_1}C_{n-c_1}^{c_2}...C_{n-c_1-...-c_{k-1}}^{c_k}q_1^{c_1}...q_k^{c_k}=\frac{n!}{\Pi_i c_i !}\Pi_i q_i^{c_i}\\(Note:q_i=Q(i))
$$

如果我们只观测一次,那么显然$\exists !j,s.t.c_j=1,L^1(c\vert Q)=q_j$.如果我们进行更多的观测, 那么$L$可能会收敛,也可能会发散.这是因为假设进行$n+1$次观测, 则$\exists !j$使得

$$
L^{n+1}(c\vert Q)=L^{n}(c\vert Q)\times \frac{n+1}{c_j +1}q_j
$$

注意到$\frac{c_j+1}{n+1}$服从伯努利大数定律并收敛到随机变量$\xi$取值为$j$的真实概率,如果$\frac{c_j+1}{n+1}\rightarrow q_j$,那么L会收敛,其它情况可能会发散.

为了更好地描述一次观测的平均概率,我们采用几何平均数,令

$$
\bar{L}=(L^n(c\vert Q))^{\frac{1}{n}}=(\frac{n!}{\Pi_i c_i !})^{\frac{1}{n}}\Pi_i q_i^{\frac{c_i}{n}}
$$

我们用$P$来描述$\xi$的真实分布.如果$\frac{c_i}{n}\rightarrow p_i=q_i$,那么$n\rightarrow \infty$,$\bar{L}\rightarrow 1$(证明留作习题).

如果$p_i\neq q_i$,那么$\bar{L}$则可能不会收敛.令$n\rightarrow \infty$,此时我们有:

$$
log_2(\bar{L})=(\frac{1}{n})(log (n!)-\sum _ilog(c_i!))+\sum_ip_ilog(q_i)\ \ \ (1)
$$

利用$n\rightarrow \infty$,$log(n!)\rightarrow nlogn-n$,以及当 $n\rightarrow \infty$,$\frac{c_i}{n}\rightarrow p_i>0$,此时有$c_i \rightarrow \infty$这两个结论,我们将$(1)$重写为:

$$
log_2(\bar{L})=\frac{(nlog(n)-n-\sum_i(c_ilog(c_i)-c_i)+\sum_ip_ilog(q_i)}{n}\\=log(n)-1-\frac{1}{n}\sum_i(c_ilog(c_i)-c_i)+\sum_ip_ilog(q_i)\ \ (2)
$$

注意到：

$$
log(n)=\sum_i\frac{c_i}{n}log(n)\\
\sum_i\frac{c_i}{n}=1\\
\frac{c_i}{n}\rightarrow p_i
$$

我们可以将$(2)$简化为:

$$
(2)=\sum_i\frac{c_i}{n}log(n)-\sum_{i}\frac{c_i}{n}log(c_i)+\sum_ip_ilog(q_i)=\\\sum_i[p_ilog(n)-p_ilog(c_i)+p_ilog(q_i)]=\\\sum_i[-p_ilog(p_i)+p_ilog(q_i)]=\\-D_{KL}(P\vert \vert Q)
$$

因此我们有:

$$
-log_2(\bar{L})(n\rightarrow \infty)=D_{KL}(P\vert \vert Q)
$$

由上所述，当$P,Q$两个分布的概率密度函数几乎处处相等的时候，此时有$\bar{L}=1$，也就是说$D_{KL}=0$，当两个分布相差太大的时候,$\bar{L}\rightarrow 0$,$D_{KL}=\infty$.它带给我们的一个直观是,KL散度度量了在对随机变量$\xi$的采样过程中,$\xi$的真实分布$P$与我们的假设分布$Q$的符合程度.

## KL散度的应用以及常见分布KL散度的计算

### 独立性度量

我们可以用KL散度来度量两个随机变量$x,y$的独立性：

$$
I(x;y)=D_{KL}(P=P(x,y)\vert \vert Q=P(x)P(y))=\\ \sum_{x,y}p(x,y)ln(\frac{p(x,y)}{p(x)p(y)})
$$

如果$x,y$统计独立,那么$I(x;y)=0$.

### 计算两个多元正态分布的KL散度

假设$x=(x_1,x_2,..,x_n)$为多元正态分布随机向量,且

$$
P_1(x)=\frac{1}{(2\pi)^{\frac{n}{2}}det(\Sigma_1)^{\frac{1}{2}}}exp(-\frac{1}{2}(x-u_1)^T\Sigma_1^{-1}(x-u_1))\\
P_2(x)=\frac{1}{(2\pi)^{\frac{n}{2}}det(\Sigma_2)^{\frac{1}{2}}}exp(-\frac{1}{2}(x-u_2)^T\Sigma_2^{-1}(x-u_2))
$$

那么

$$
D(P_1\vert \vert P_2)=\frac{1}{2}[log(\frac{det\Sigma_2}{det\Sigma_1})+tr(\Sigma_2^{-1}\Sigma_1)+(u_1-u_2)^T\Sigma_2^{-1}(u_1-u_2)-n]
$$

证明:

$$
D(P_1\vert \vert P_2)=E_{P_1}[log(P_1)-log(P_2)]\\=\frac{1}{2}E_{P_1}[-log(det(\Sigma_1))-(x-u_1)^T\Sigma_1^{-1}(x-u_1)+log(det(\Sigma_2))+\\(x-u_2)^T\Sigma_2^-1(x-u_2)]\\
\text{(drop the common part out)}\\
=\frac{1}{2}[log(\frac{det\Sigma_2}{det\Sigma_1})+E_{P_1}[(x-u_2)^T\Sigma_2^{-1}(x-u_2)-(x-u_1)^T\Sigma_1^{-1}(x-u_1)]\ \ (3)
$$

计算$E_{P_1}[(x-u_2)^T\Sigma_2^{-1}(x-u_2)-(x-u_1)^T\Sigma_1^{-1}(x-u_1)]$时有一个trick为:

$$
(x-u_2)^T\Sigma_2^{-1}(x-u_2)=tr(\Sigma_2^{-1}(x-u_2)(x-u_2)^T)
$$

其实这里利用了$a^Ta=tr(aa^T)$这个矩阵计算技巧.同时注意这里还有第二个trick为：在$P_1$的分布下$(x-u_1)(x-u_1)^T=\Sigma_1$,但是因为$x$服从$P_1$分布,$(x-u_2)(x-u_2)^T$则不等于$\Sigma_2$.用这些trick进行进一步计算为:

$$
E_{P_1}[(x-u_2)^T\Sigma_2^{-1}(x-u_2)-(x-u_1)^T\Sigma_1^{-1}(x-u_1)]\\=E_{P_1}[tr(\Sigma_2^{-1}(x-u_1+(u_1-u_2))(x-u_1+(u_1-u_2))^T)-1]\\=E_{P_1}[tr(\Sigma_2^{-1}(\Sigma_1+2(x-u_1)(u_1-u_2)^T+(u_1-u_2)(u_1-u_2)^T))]\\=tr(\Sigma_2^{-1}\Sigma_1)+(u_1-u_2)^T\Sigma_2^{-1}(u_1-u_2)-n
$$

带入$(3)$我们有:

$$
D(P_1\vert \vert P_2)=\frac{1}{2}[log(\frac{det\Sigma_2}{det\Sigma_1})+tr(\Sigma_2^{-1}\Sigma_1)+(u_1-u_2)^T\Sigma_2^{-1}(u_1-u_2)-n]
$$