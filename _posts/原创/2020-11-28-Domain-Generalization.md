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







在之前的博客中，我们用两篇长文介绍了解决这一问题的著名技术：域适应(Domain Adaptation)，分别为[域适应的基本原理](https://www.fenghz.xyz/Domain-Adaptation-A-Survey/)，以及[域适应算法中的核函数](https://www.fenghz.xyz/RHKS-DA/)。此外，经过四个月的工作，我们撰写的一篇名为["KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation"](https://arxiv.org/abs/2011.09757)的文章也已公开并投稿到AAAI2021，在这篇工作中，我们提出了一个新的解决联邦无监督域适应问题的算法，并在当前最大规模的验证集DomainNet上取得了51.1%的准确率。但是，域适应问题要求取得目标域的有标注或无标注数据，这与真实场景具有较大的区别。首先，真实场景中，我们往往希望模型能够适配于多个目标域，并可以进行快速的，小样本的微调。其次，真实场景的训练与测试往往是分离的，训练用于调试模型的数据，往往测试并不可用。为了解决这些问题，域泛化(Domain Generalization)被提出了。
