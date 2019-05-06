---
layout: post
title: Some notes on Generative Adversarial Network
date: 2019-5-6 12:32:00
categories: 机器学习
tags: VAE,GAN
mathjax: true
---
* content
{:toc}

**FengHZ‘s Blog首发原创**


### 

Based on the latent variable assumption, the essential purpose of GAN is to map the distribution of latent variable $z\sim p(z)$ into the data distribution $x\sim p_{data}$. GAN utilizes the equilibrium theory 2-players game to find the optimal map. There have already been many articles introducing the structure of GAN with comparing GAN to the confrontation between counterfeiters and discriminators. However, understanding the basic mathemathical form of GAN is very important which can tell us why the equilibrium point can achieve the complex real data distribution $x\sim p_{data}$ without making any assumptions on the analytical form of $p_{data}$.

Given the discriminator $D$ and the generator $G$, the nash equilibrium point of the 2-players game 

$$
\min_{G}\max_{D} E_{x\sim p_{data}(x)} \log D(x) + E_{z\sim p(z)} \log (1-D(G(z))) \tag{1}
$$

* **proposition** The global optimal $(D,G)$of $(1)$ satisfy
  $$
  D(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} \tag{2}
  $$

  $$
  z\sim p(z), G(z) \sim p_{data}(x)\tag{3}
  $$
  Here we use $p_{g}$ to indicate the distribution of $G(z),z\sim p(z)$

  *proof for $(2)$*

  $$
  \max_{D} E_{x\sim p_{data}(x)} \log D(x) + E_{x\sim p_{g}(x)} \log (1-D(x)) \\
  =\max_{D} \int_{x} p_{data}(x) \log D(x)+p_{g}(x)\log (1-D(x)) dx\\
  =^{discretization} \sum_{i=1}^{N} p_{data}(x_i) \log D(x_i)+p_{g}(x_i)\log (1-D(x_i)) \Delta_{x} \\
  $$

  In the discretization form, we make $D$ achieve the maximum for each point $x_i$ and then get the global optimal, then we can get

  $$
  D(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} 
  $$

  We substitude $(2)$ into the equation $(1)$ and can get

  $$
    \min_{p_{g}} E_{x\sim p_{data}(x)} \log D(x) + E_{x\sim p_{g}(x)} \log (1-D(x))\\
    =\min_{p_{g}}\int_{x} p_{data}(x)\log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} + p_{g}(x)\log \frac{p_{g}(x)}{p_{data}(x)+p_{g}(x)}dx\\
    =\min _{p_{g}} J[p_{g}] \tag{4}
  $$
  
  We use the variational method to solve $(4)$ with the constraint $\int_{x} p_{g}(x) dx =1$, and following are the *Lagrange* form

  $$
  L[p_{g}] = J[p_{g}] + \lambda_{1}[\int_{x} p_{g}(x) dx -1] \tag{5}
  $$

  Using the *basic formulation of variational method*, we have

  $$
  \log \frac{p_{g}}{p_{data}+p_{g}} + \lambda_{1}=0
  $$

  which indicates

  $$
  p_{g}=\frac{p}{e^{\lambda_1}-1}
  $$

  combined with $\int_{x} p_{g}(x) dx =1$, we have

  $$
  p_g =p_{data}
  $$

  which is the essential part of GAN.