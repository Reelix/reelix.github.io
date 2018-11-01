---
layout: post
title: Some notes on variational inference
date: 2018-11-1 12:32:00
categories: 机器学习
tags: Variational-Inference
mathjax: true
---

* content
{:toc}


## [变分是什么](https://en.wikipedia.org/wiki/Calculus_of_variations)


微积分是研究函数的手法，它研究在某一个特定函数上的微小变化。而变分则是研究泛函的手段，它通过研究函数的变化，来找到一类将函数映射到实数的泛函的极值点。这里泛函一般被表达为函数以及它们的导数的定积分(将函数映射到实数值)。最大化或者最小化一类泛函的函数往往可以用欧拉-拉格朗日方程来找到。

变分法用于计算泛函的极值。泛函是将函数映射到标量的映射，因此泛函在直观上可以被认为是函数的函数。泛函在函数空间的元素$y$上可以取得极值点，而该元素$y$定义于一个给定的域上。对于泛函$J[y]$，如果存在函数$f$，满足$\forall y \in f$的一个任意小的邻域中，$J[y]-J[f]$符号不变，那么我们说泛函$J[y]$在函数$f$处取得了极值点。

一类找极值点的手段是$Euler-Lagrange$方程，我们给出它的推导过程以及一个简单的例子(在这个例子中我们可以看到变分法的核心就是积分和求导互换哈哈哈哈哈哈)：

考虑泛函:

$$
    J[y]=\int_{x_1}^{x_2}L(x,y(x),y'(x))dx \tag{1}
$$

满足:

$$
x_1,x_2为常数\\
y(x) 有二阶连续微分\\
y'(x)=dy/dx\\
L(x,y,y')关于x,y,y'有二阶连续微分 \tag{2}
$$

如果泛函$J[y]$在$f$处达到了局部最小值，同时取$\eta(x)$是有一阶微分，同时在$x_1,x_2$处取0的任意函数，我们说此时局部极小值意味着:

$\forall \epsilon$足够接近0，我们有：

$$
J[f]\leq J[f+\epsilon \eta] \tag{3}
$$

我们定义$\phi(\epsilon)=J[f+\epsilon\eta]$，由极值点的定义，$\phi'(0)=0$，即(交换积分求导次序)

$$
\int_{x_1}^{x_2}\frac{dL}{d\epsilon}\vert_{\epsilon=0}dx=0 \tag{4}
$$

注意$y=f+\epsilon\eta$，因此我们有：

$$
\frac{dL}{d\epsilon}=\frac{\partial L}{\partial y}\frac{dy}{d\epsilon}+\frac{\partial L}{\partial y'}\frac{dy'}{d\epsilon}\tag{5}
$$

$y'=f'+\epsilon\eta',\frac{dy'}{d\epsilon}=\eta'$,代入$(4)$有：

$$
\int_{x_1}^{x_2}(\frac{\partial L}{\partial f}\eta+\frac{\partial L}{\partial f'}\eta')dx\tag{6}
$$

利用

$$
d(\frac{\partial L}{\partial f'}\eta)/dx=\frac{\partial L}{\partial f'}\eta'+\eta*\frac{d\frac{\partial L}{\partial f'}}{dx}\tag{7}
$$

$(6)$转化为：

$$
\int_{x_1}^{x_2}\eta(\frac{\partial L}{\partial f}-\frac{d\frac{\partial L}{\partial f'}}{dx})dx=0\tag{8}
$$

因为此时$\eta$是首尾为0的任意函数，因此该式成立当且仅当

$$
\frac{\partial L}{\partial f}-\frac{d\frac{\partial L}{\partial f'}}{dx}=0 \tag{9}
$$

这就是$Euler-Lagrange$等式，我们用一个简单的例子说明这个等式的威力：

假设有两个不同点$(x_1,y_1),(x_2,y_2)$，我们的目标是找到连接这两个点的最短曲线，我们假设曲线为$y(x),y(x_1)=y_1,y(x_2)=y_2$，那么我们的泛函为

$$
A[y]=\int_{x_1}^{x_2}\sqrt{1+[y'(x)]^2}dx\tag{10}
$$

我们利用$Euler-Lagrange$等式有：
$$
\frac{\partial L}{\partial y}=0,\frac{\partial L}{\partial y'}=\frac{y'}{\sqrt{1+(y')^2}}\tag{11}
$$
代入为
$$
\frac{y'}{\sqrt{1+(y')^2}}=c\tag{12}
$$
我们知道此时$y'=m$，因此$y$是一条直线，这也就证明了两点之间，直线最短。

## 变分推断的目的，推导公式，优化方法

### 目的

[回忆流型假设与潜变量模型](https://fenghz.github.io/2018/10/15/Variational-AutoEncoder/#12-%E6%BD%9C%E5%8F%98%E9%87%8F%E7%A9%BA%E9%97%B4%E6%A8%A1%E5%9E%8B%E5%81%87%E8%AE%BE)：

所谓变分推断就是用观测值来推断难以观测值的概率分布，一族概率分布构成了一个泛函空间，而我们的目的是找到在某种评价下的最优概率分布，而寻找的手段就是变分法。

我们假设$x=x_{1:n}$是一组观测变量，而$z=z_{1:m}$是一组潜变量，$\alpha$是分布的超参数。我们现在想通过$x$的观测来对$z$的后验分布进行估计:

$$
p(z\vert x,\alpha)=\frac{p(z,x\vert \alpha)}{\int_{z}p(z,x\vert \alpha)} \tag{13}
$$

这其实就是一个简单的Bayes推断问题，即如果我们知道了$(z,x)$的联合分布，或者知道了$p(x\vert z)$这样的条件分布，那么我们就只需要代公式就好了。然而这个过程是非常复杂的：

假设我们有一组关于$z$分布的超参数$u_{1:K}$，以及潜变量$z_{1:n}$，我们的观测变量为$x_{1:n}$，(注意这里n是观测的次数，一次观测对应于一个潜变量)我们用观测变量来对超参数和潜变量进行预测：

$$
p(u_{1:K},z_{1:n}\vert x_{1:n})=\Pi_{i=1}^n p(u_{1:K},z_i\vert x_i)\tag{14}
$$

同时我们可以进行bayes公式：
$$
p(u_{1:K},z_i\vert x_i)=\frac{p(x_i\vert u_{1:K},z_i)p(u_{1:K})p(z_i)}{p(x_i)}\tag{15}
$$

同时，分母部分可以进行推导：

$$
p(x_i)=\int_{u_1}\int...\int_{u_{K}}\sum_{z_i}\Pi_{k=1}^Kp(u_k)p(z_i)p(x_i\vert z_i,u_{1:K})\tag{16}
$$

代入$(14)$我们有：

$$
p(u_{1:K},z_{1:n}\vert x_{1:n})=\frac{\Pi_{i=1}^n\Pi_{k=1}^Kp(u_k)p(z_i)p(x_i\vert z_i,u_{1:K})}{\Pi_{i=1}^n\int_{u_{1:K}}\sum_{z_i}\Pi_{k=1}^Kp(u_k)p(z_i)p(x_i\vert z_i,u_{1:K})}\tag{17}
$$

注意z_i之间互相独立，x_i之间互相独立，u_k互相独立，u,z互相独立，同时有(14)做支撑，我们有

$$
p(x_{1:n})=\int_{u_{1:K}}[\Pi_{k=1}^Kp(u_k)]\Pi_{i=1}^n\sum_{z_i}p(z_i)p(x_i\vert z_i,u_{1:K})\tag{18}
$$

这就意味着当$n$很大的时候，我们需要对$K^n$个项进行计算，因此直接用贝叶斯方法进行后验估计不可行，而此时如何估计就成为贝叶斯统计学派中的核心问题，变分推断就是为了解决这个问题。

### 变分推断

为了进行变分推断，我们首先选中一个泛函空间，它的元素为潜变量的概率密度函数$q$，这组概率密度函数有一些参数，而这些参数就是我们要进行变分推断的参数:
$$
q(z_{1:m}\vert v)\tag{19}
$$
我们要用变分法来选择一组参数，这组参数能够确定泛函空间中对后验$p(z\vert x)$的最佳逼近，然后我们用$q$作为对于后验的替代，并用这个模型对接下来的数据进行推断，或者用$q$作为先验来进一步研究潜变量的分布。值得注意的是，真正的后验分布可能并不在变分族中，但是没有关系，我们可以进行一个最佳逼近，如我们可以用正态分布进行逼近，并证明在$\sigma \rightarrow0$的时候[我们的预测可以逼近后验分布](https://fenghz.github.io/2018/10/15/Variational-AutoEncoder/#%E4%B8%80%E7%BB%B4%E6%83%85%E5%86%B5%E7%9A%84%E8%AF%81%E6%98%8E).

那么现在的问题就转化为如何设置损失函数(或者说泛函空间到实数空间的映射了)，我们可以[用$KL$散度进行设置](https://fenghz.github.io/2018/10/05/KL-Divergency-Description/)：
$$
KL(q(z)\Vert p(z\vert x))=log\ p(x)-(E_q[log\ p(z,x)]-E_q[log\ q(z)])\tag{20}
$$

此时注意到我们需要在$q$的泛函空间中最小化$KL$散度，$log\ p(x)$与该泛函空间无关，因此其本质是最大化：
$$
\mathcal{L}=E_q[log\ p(z,x)]-E_q[log\ q(z)] \tag{21}
$$
注意
$$
p(z_{1:m},x_{1:n})=p(x_{1:n})\Pi_{j=1}^m p(z_{j}\vert z_{1:(j-1)},x_{1:n})\\
E_q[log\ q(z)]=\sum_{j=1}^m E_q[log\ q(z_j)] \tag{22}
$$
因此原式可以重写为：
$$
\mathcal{L}=log\ p(x_{1:n})+\sum_{j=1}^mE_q[log\ p(z_j\vert z_{1:j-1},x_{1:n})]-E_q[log\ q(z_j)] \tag{23}
$$

### 优化方法

$(23)$式是一个关于$q(z_1),...,q(z_k)$的优化函数，我们可以采用迭代优化方法(回忆Gauss-Seiders迭代方法)，比如先优化一个，再用它作为真实后验的替代去优化另外一个，逐个优化直到收敛，那么此时需要计算$\mathcal{L}_k$.

注意$(22)$中关于$m$的顺序是随机给定的，因此我们在计算关于$q(z_k)$的损失函数的时候，我们总可以将$z_k$作为$m$的最后一项，因此损失函数可以写成:

$$
\mathcal{L}_k=\int q(z_k)E_{-k}[log\ p(z_k\vert z_{-k},x)]d z_k -\int q(z_k)log\ q(z_k)dz_k\tag{24}
$$

注意这里下标$-k$意思是在$(z_1,..,z_n)$这些变量中除去$z_k$这个随机变量，而$E_{-k}$是指在$E_q$这个分布中除去$q(z_k)$这个分布的期望。

此时$\mathcal{L}_k$中的积分部分:

$$
J=q(z_k)E_{-k}[log\ p(z_k\vert z_{-k},x)]-\int q(z_k)log\ q(z_k)
$$

就是关于$z_k ,   q(z_k)$的一个泛函,我们用变分法求$\mathcal{L}_k$的极值：(积分求导换位置+等于0)

$$
E_{-k}[log\ p(z_k\vert z_{-k},x)]-log\ q(z_k)-1=0\tag{25}
$$

因此得到了
$$
q^*(z_k)\propto exp(E_{-k}[log\ p(z_k\vert z_{-k},x)]) \tag{26}
$$
通过这个关系我们可以进行更新。

### 梯度法

如果我们给定了$q(z\vert x)$的分布，那么我们可以只对这些分布参数进行推断，而不用计算具体的函数形式了(相当于给定了函数形式的先验)，如[变分自编码器的推导](https://fenghz.github.io/2018/10/15/Variational-AutoEncoder/#221-%E6%9E%84%E9%80%A0%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0)中给定了$x,q(z\vert x)$的分布假设，因此其最后的损失函数可以不用变分法，而只用对参数损失进行梯度下降法即可。





