---
layout: post
title: A paper report for paper - dynamic routing between capsules
date: 2018-10-11 11:32:00
categories: 深度学习
tags: Network-Architecture
mathjax: true
---

* content
{:toc}

## 简介

[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)是Hinton于2017年11月所撰写的一篇关于神经网络设计架构的文章。文章的主要观点是Vectorization，即用向量代替标量来表示特征，同时用向量的范数来代替分类可能性，向量之间的连结用动态路径进行连结。

我为此文撰写了一个report进行深度分析，report以slice形式进行呈现。





## Report

![Capsulenet](/images/capsule/Capsulenet.png)