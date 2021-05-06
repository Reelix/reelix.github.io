---
layout: post
title: An Introduction to the Differential Privacy on Deep Learning
date: 2021-5-5 10:49:00
categories: 机器学习
tags: Differential Privacy, Deep Learning, Privacy
mathjax: true
---

数据是人工智能的燃料，优秀的深度学习模型需要依靠大量高质量数据集进行训练。然而，随着模型精度的不断提升，对于个人隐私的泄露现象也变得越发严重。此外，随着互联网企业的扩展，用户数据开始担任重要生产资料的角色，成为各大垄断企业的护城河。欧盟，作为反对互联网垄断的桥头堡，同时也作为隐私保护的急先锋，在2018年正式施行法案《通用数据保护条例》(General Data protection Regulation, GDPR)。GDPR主张个人对数据的四项权利，请求权，拒绝权，修正权和删除、遗忘权。请求权，即个人有权了解其个人数据是否被处理，哪些个人数据以怎样的方式被处理以及进行了哪些数据处理操作；拒绝权，即个人有令人信服的合法理由，可禁止进行某些数据处理操作，比如个人可拒绝以营销为目的的个人数据处理。遗忘权，即个人有权寻求删除其个人数据的影响，比如用个人的微博，抖音数据训练的推荐算法，能够把个人的影响给忘掉。此外，GDPR还对数据的传输有明确的要求，比如欧盟境内的数据不得在境外被使用。







那么，一个自然的问题是，如何让人工智能模型的训练过程能够符合数据保护条例，保护个人隐私呢？依据GDPR的要求，首先数据的存储必须满足去中心化，而模型则需要在去中心化的数据库上进行分布式训练。联邦学习[1]是应对这种要求的解决方案，它允许模型在本地数据库上训练，并构建一个全局的调度器，通过对不同本地模型的更新进行汇总（即FedAvg算法），获得全局模型。得到的全局模型能够利用所有本地数据的信息，并得到更好的模型精度。其次，GDPR要求个人能够控制其数据对于模型的影响。人工智能模型是通过归纳所有数据信息构建的，它的输出结果是否会泄露个人隐私呢？差分隐私（Differential Privacy)[2]系统地探讨并解决了这一问题。在本篇Blog中，我们对应用于深度学习中的差分隐私技术进行入门式的介绍，包括差分隐私的动机与定义，满足差分隐私要求的梯度下降方法，以及如何对差分隐私的隐私保护程度进行计算。因为我们的主要目的是将差分隐私应用于隐私保护场景，而非对差分隐私进行研究，在本篇Blog中我们会略过一些数学证明的过程，将重心聚焦到差分隐私的应用上。



## References
[1] Konečný J, McMahan B, Ramage D. Federated optimization: Distributed optimization beyond the datacenter[J]. arXiv preprint arXiv:1511.03575, 2015.

[2] Dwork C, Roth A. The algorithmic foundations of differential privacy[J]. Foundations and Trends in Theoretical Computer Science, 2014, 9(3-4): 211-407.