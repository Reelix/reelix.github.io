---
layout: post
title: Data Augmentation 
date: 2019-3-17 12:32:00
categories: 机器学习
tags: Data-Augmentation
mathjax: true
---

* content
{:toc}

**FengHZ‘s Blog首发原创**

数据增广一直是模型训练的一个重要话题。如何确定*Data Augmentation*策略对于最后的精度具有重要的影响。在[AutoAugment:Learning Augmentation Strategies from Data](http://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html)一文中，采用了强化学习策略，对固定数据集给出了最佳数据增广方法。同时文章给出了我们现在常用的数据增广方法，这里主要对这些增广方法做一个叙述与实现.

```python
import math
import random
```

```python
import os
from os import path
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
image = Image.open(img_path)
image
```



![png](/images/data_augmentation/output_2_0.png)



## ShearX(Y)

ShearX(Y)本质是沿着x/y轴，固定另外一轴不变进行仿射变换的过程，如图所示

![image.png](/images/data_augmentation/1.png)

这里的$\theta$就是用于仿射变换的$\theta$，也是输入参数。

ShearX的输入参数是$\theta \in (-\frac{\pi}{2},\frac{\pi}{2})$,即对于坐标点$(x,y)$而言，相对于中心点$(x_c,y_c)$,其在变换下的最终位置是


$$
x'-x_c = (x-x_c)-tan(\theta)(y-y_c)\\
y'-y_c=y-y_c
$$


ShearY的输入参数也是$\theta \in (-\frac{\pi}{2},\frac{\pi}{2})$,但是它对$y$进行变换，即对于坐标点$(x,y)$而言，其在变换下的最终位置是


$$
x'-x_c = x-x_c\\
y'-y_c=(y-y_c)-tan(\theta)(x-x_c)
$$


一个$tan(\theta)=0.1$的例子如下，在实际操作中，我们一般令$\theta \in [-16^°,16^°]$

```python
theta = 10
theta_r = math.radians(theta)
tant = math.tan(theta_r)
w,h = image.size
shearx = tuple([1,-tant,tant*h/2,0,1,0])
plt.imshow(image.transform(image.size,Image.AFFINE,shearx,Image.BICUBIC,fillcolor=(128, 128, 128)))
plt.show()
sheary = tuple([1,0,0,-tant,1,tant*w/2])
plt.imshow(image.transform(image.size,Image.AFFINE,sheary,Image.BICUBIC,fillcolor=(128, 128, 128)))
plt.show()
```

![png](/images/data_augmentation/output_4_0.png)



![png](/images/data_augmentation/output_4_1.png)

## TranslateX(Y)

TranslateX(Y)就是在水平/数值两个方向平移一下图像，我们般选择平移范围在`[-150,150]`像素内。例子如下:

```python
transformx = tuple([1,0,10*random.choice([-1,1]),0,1,0])
plt.imshow(image.transform(image.size,Image.AFFINE,transformx,Image.BICUBIC,fillcolor=(128, 128, 128)))
plt.show()
transformy = tuple([1,0,0,0,1,10*random.choice([-1,1])])
plt.imshow(image.transform(image.size,Image.AFFINE,transformy,Image.BICUBIC,fillcolor=(128, 128, 128)))
plt.show()
```

![png](/images/data_augmentation/output_6_0.png)



![png](/images/data_augmentation/output_6_1.png)

## Rotate

旋转其实也是一种仿射变换，给定旋转角度$\theta$，旋转中心$(x_c,y_c)$我们有:


$$
x'-x_c = \cos(\theta) (x-x_c) - \sin(\theta)(y-y_c)\\
y'-y_c = \sin(\theta)(x-x_c) + \cos(\theta) (y-y_c)
$$


我们一 般取 $\theta \in [-\frac{\pi}{6},\frac{\pi}{6}]$

```python
theta = 10
thete_r = math.radians(theta)
cost = math.cos(thete_r)
sint = math.sin(thete_r)
w,h = image.size
xc=w/2
yc=h/2
rotate = tuple([cost,-sint,(1-cost)*xc+sint*yc,sint,cost,(1-cost)*yc-sint*xc])
plt.imshow(image.transform(image.size,Image.AFFINE,rotate,Image.BICUBIC,fillcolor=(128, 128, 128)))
plt.show()
```

![png](/images/data_augmentation/output_8_0.png)

## AutoContrast,Contrast

图像的对比度是指图像明亮的地方与灰暗地方的像素的差别。我们可以人为扩大这个差别从而增大对比度，也可以缩小这个差别减少对比度。自动增加对比度是指让图像中最大的灰度变为255，最小的灰度变为0，然后依次成比率改变图像的像素。

对比度增广的改变范围是$[0.1,1.9]$，其中1代表不变。例子如下

```python
plt.imshow(ImageEnhance.Contrast(image).enhance(1.2))
plt.show()
plt.imshow(ImageOps.autocontrast(image))
plt.show()
```

![png](/images/data_augmentation/output_10_0.png)



![png](/images/data_augmentation/output_10_1.png)

## Invert

Invert指对图像的像素值全部变成$255-value$

```python
plt.imshow(ImageOps.invert(image))
plt.show()
```

![png](/images/data_augmentation/output_12_0.png)

## Equalize

Equalize操作指把图像的密度直方图给规则化一下。图像本质上是一组采样数据，我们可以以像素值为划分，看在每一个像素值上有多少像素，这就是所谓的图像直方图。我们可以根据直方图做出$F(v<a)$的分布函数，`Equalize`操作本质上是希望这个分布函数是线性均匀上升到1的，这就是均衡化操作，如下所示

```python
ImageOps.equalize(image)
```



![png](/images/data_augmentation/output_14_0.png)



## Solarize

`Solarize`操作就是给定一个阈值$t$，对像素值大于$t$的所有像素点做invert操作。

```python
ImageOps.solarize(image,200)

```



![png](/images/data_augmentation/output_16_0.png)



## Posterize

把原来每个像素用8比特表示的图像压缩到更少的比特(一个更好的压缩方法是Vector Quantization)。

```python
ImageOps.posterize(image, 1)

```



![png](/images/data_augmentation/output_18_0.png)



## Color

[我们可以从RGB空间向HSV空间进行转移]，一个转移结果是

![image.png](/images/data_augmentation/2.png)

其中S代表颜色的饱和度，S为0时图像变为灰度，S增大后图像的色彩更加饱和

## Brightness

Brightness就是图像的亮度，它是HSV空间的V部分，为0的时候代表非常暗，而越高则代表视觉上的越亮。

```python
ImageEnhance.Color(image).enhance(0)

```



![png](/images/data_augmentation/output_20_0.png)



```python
ImageEnhance.Color(image).enhance(1.9)

```



![png](/images/data_augmentation/output_21_0.png)



```python
ImageEnhance.Brightness(image).enhance(0.2)

```



![png](/images/data_augmentation/output_22_0.png)



```python
ImageEnhance.Brightness(image).enhance(1.9)

```



![png](/images/data_augmentation/output_23_0.png)



## Sharpness

Sharpness代表图像的锐度。锐度计算是通过图像的梯度进行计算的，即对图像的像素空间进行差分，一般差分值大的部分代表图像变化剧烈，这 也就是所谓的边缘部分。增大这些边缘部分就会显得锐度变大

范围仍然取$[0.1,1.9]$

```python
ImageEnhance.Sharpness(image).enhance(0.1)

```



![png](/images/data_augmentation/output_25_0.png)



```python
ImageEnhance.Sharpness(image).enhance(1.9)

```



![png](/images/data_augmentation/output_26_0.png)



# AutoAugment

AutoAugment方法主要将它提到的14种数据增广方法进行了离散化：
![image.png](/images/data_augmentation/3.png)
对于每一个取值范围，文中都将其均匀划分为9份，取10个点作为10种策略并分别固定。用这些策略组成若干个Policy，每一个Policy由两个Action构成，每一个Action是一种数据增广方法，增广参数，以及做这种增广的概率。

文中采用强化学习技术给出了`Cifar10`,`SVHN`以及`ImageNet`上的最佳数据增广策略：

![image.png](/images/data_augmentation/4.png)

![image.png](/images/data_augmentation/5.png)