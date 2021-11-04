---
layout: post
title: 隐私保护的深度学习系统——基于高斯机制的差分隐私DL
date: 2021-11-4 10:49:00
categories: 深度学习
tags: Differential Privacy, SGD, Privacy-preserving, Gaussian Mechanism
mathjax: true
---

在之前[关于差分隐私的Tutorial](https://www.fenghz.xyz/Differential-Privacy/)中，我们简单介绍了欧盟隐私保护条例，即个人对数据具有知情权，拒绝权，修正与遗忘权，以及对于自动决策过程的选择权。差分隐私对于数据分析过程提出了严格的隐私保护定义，即对于数据库数据的任何分析，以及根据分析结果与其他信息进行的进一步的合并加工，都不会泄露个人隐私。通俗而言，就是对于任何数据分析员，要求分析员在对数据库进行分析后，对数据库中每一个个体的了解不会超过其在分析开始之前的了解。对于简单的数据处理过程（如计算平均工资，统计性别比例），通过在数据分析结果中增加高斯噪声，可以令数据分析的机制满足差分隐私的约束。但是，对于需要多轮训练的复杂深度学习系统，构建差分隐私保护则更为困难。本文主要介绍基于高斯机制的差分隐私深度学习系统：通过在训练过程中施加高斯噪声，构建满足**差分隐私要求**的深度学习训练系统，并对所得深度模型**计算隐私开销**。此外，我们也将以[**Opacus**](https://github.com/pytorch/opacus)这一基于pytorch的差分隐私训练库为例进行代码讲解。





本文主要参考的文献为

1. [A Tutorial to Differentially Private Machine Learning, Neurips17](https://www.ece.rutgers.edu/~asarwate/nips2017/NIPS17_DPML_Tutorial.pdf)
2. [Deep learning with differential privacy](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)
3. [Rényi differential privacy](https://ieeexplore.ieee.org/abstract/document/8049725/)
4. [Subsampled Rényi Differential Privacy and Analytical Moments Accountant](https://arxiv.org/abs/1808.00087)
5. [Rényi Differential Privacy of the Sampled Gaussian Mechanism](https://arxiv.org/abs/1908.10530)
6. [The Composition Theorem for Differential Privacy](http://proceedings.mlr.press/v37/kairouz15.html)
   


## 差分隐私深度学习系统的基本框架

## Subsample：用采样率amplify差分隐私，降低隐私损失

## Composition Theorem：计算整个训练系统的差分隐私损失

## Opacus库：基于Pytorch框架的隐私保护库

## PySyft + Opacus：结合差分隐私与联邦学习






