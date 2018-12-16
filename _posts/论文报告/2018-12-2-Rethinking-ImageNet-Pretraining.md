---
layout: post
title: Rethinking ImageNet Pretraining
date: 2018-10-15 12:32:00
categories: 深度学习
tags: Transfer Learning
mathjax: true
---

* content
{:toc}

## 前言
本文是何凯明在`FAIR`的新作，主要针对使用ImageNet进行预训练的问题。

一直以来，在计算机视觉任务上，采用ImageNet数据集对设计好的神经网络进行预训练，然后再用目标数据集对神经网络进行训练是一套标准的流程，而很多在标准数据集上达到`State of Art`的结果也沿用了这一套训练模式。但是，预训练到底起到了多大的结果呢？它是否有助于最后训练结果的提升呢？本文主要给出了这个问题的量化。





## 主要结论

1. ImageNet预训练确实能够加快收敛速度，但是这是建立在不计算训练ImageNet的时间上的。实际上如果把预训练的时间也算上，比如用网络一共看过了多少张图像/多少个实例/多少个像素这些指标计算的话，ImageNet预训练与用整个数据集从随机初始化进行训练的收敛速度一致。同时，如果我们不进行预训练，直接在**COCO**数据集上对随机初始化的网络进行训练，只要时间足够长仍然能够收敛到*state of art*的结果。
2. ImageNet预训练并不能提供更好的正则化效果，也不能避免过拟合。作者用10\%的**COCO**数据集进行了实验，实验发现，如果用ImageNet进行预训练的话，在*fine-tune*阶段我们必须选取新的超参数以避免*over fitting*，同时，我们发现，使用10\%的**COCO**数据集并选取相同的超参数，我们用ImageNet进行预训练的结果与随机初始化并直接用10\%的数据训练的最终结果一致。
3. ImageNet预训练并不能提升最后定位的精确度，如果我们的任务是想要精确定位的话，ImageNet预训练并不能改进结果，这可能是因为分类任务与检测任务之间的GAP所导致的。
4. ImageNet预训练对于正则化方法具有一定的帮助。在训练网络的过程中，我们往往采用*batch normalization*技巧，但是*batch normalization*对于*batch size*超参数的选取非常敏感，如果*batch size <8*，*batch normalization*的正则化效果并不好。我们可以用ImageNet进行预训练，然后再冻结*batch normalization*的参数以达到*state of art*的结果，但是在不进行ImageNet预训练的时候，采用*group normalization*或者基于多块`GPU`的*Synchronized Batch Normalization*也可以从随机初始化的网络训练出*state of art*的结果。


## 如何使用这篇文章
1. 首先，文章只是在**COCO**数据集上进行了测试，而**COCO**数据集的图像真的很多，因此ImageNet预训练没用这是一个非常自然的结论。但是如果我们的数据很少，比如医学影像只有一点数据的话，那么预训练应该还是有用的。
2. 这篇文章总结了当前很多任务的*state of art*，可以当作业内标杆进行对比。
3. 训练时间要长一点


