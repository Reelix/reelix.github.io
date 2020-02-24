---
layout: post
title: 深度学习服务器配置——解决内网穿透问题
date: 2020-02-16 10:49:00
categories: 实用工具
tags: Deep Learning
mathjax: true
---

* content
{:toc}


笔者长期负责实验室深度学习服务器的管理与维护，本文是对服务器管理维护过程中遇到的一些典型问题的一揽子解决方案。深度学习服务器主要由若干CPU(通常是2块CPU，共48-64进程)与若干GPU(2-8块GPU，通常为1080Ti，2080Ti，TITAN xp，Tesla V100等)组成，可以用于深度学习模型的训练。这些服务器往往放置于内网中，在外网无法通过ssh进行直接访问，需要通过麻烦的VPN验证进入内网方能进行访问。笔者还遇到的特殊问题是，管理的4台服务器位于不同的内网，换机登陆极其麻烦。此外，深度学习任务中还需要用到很多基于web服务的工具，如`jupyter notebook`，`tensorboard`等工具，这些web服务的配置也需要一些技巧。在本文中，我采用开源内网穿透工具[frp](https://github.com/fatedier/frp/blob/master/README_zh.md)对上述问题进行解决，从而实现内网服务器的公网ssh操作与公网访问。





### 用Frp实现内网穿透

运行服务器的典型场景是这样的：在内网A中有一台服务器，它只有内网ip而没有公网ip，因此在内网外无法访问。但是该服务器又可以访问公网资源，能够进行上传与下载。这时，我们想在公网访问它，一个很简单的思路是将该服务器与公网中我们能够访问的，具有公网ip的服务器进行捆绑，我们向公网服务器发送的请求全部变成数据包传给内网服务器，内网服务器再返回给公网服务器，并交由公网服务器展示给我们。基于这种思路，我们可以购买阿里云服务器作为中转，采用`frp`作为转发工具，从而实现内网穿透。

在云服务器方面，我购买的是阿里云学生优惠下的轻量应用服务器，每个月限1000G流量，一年113，峰值带宽5Mbps，方便下载与上传。[frp](https://github.com/fatedier/frp/blob/master/README_zh.md)的使用很简单，我们首先下载它最重要的两个部分，分别是frps(frp-server)与frpc(frp-client)。其中，frps控制公网服务器上接收内网服务器流量的端口，以及公网服务器打开内网web服务的端口，而frpc则控制内网服务器发送流量的ip地址，发送流量的端口，以及自己需要转发的服务在远程服务器上的对应端口. 一个例子如下：

我们拥有公网服务器`S`，其ip为`is`，系统为`ubuntu`。我们拥有内网服务器`C`，能够藉由`is`访问公网服务器。我们用`frp`建立内网穿透，希望通过ssh到公网服务器的某个端口就能进入内网服务器。在公网服务器中，我们放入程序`frps`，并创建一个名字为`frps.ini`的空配置文件。在内网服务器中，我们同样放入程序`frpc`，并创建名字为`frpc.ini`的空配置文件。

在公网服务器中，我们在配置文件`frps.ini`中写入
```
[common]
bind_port = 7000
```
这里`7000`是一个让内网服务器与公网服务器进行流量交换的端口，服从tcp文件传输协议，理论上可以取任意服务器允许的端口。然后我们在公网服务器上运行`frp`服务
```
frps -c frps.ini
```
这表示公网服务器开了一个端口7000，允许内网服务器进行流量上传与下载。接下来，我们让内网服务器通过这个端口，转发ssh服务。在内网服务器中，我们在配置文件`frpc.ini`中写入
```
[common]
server_addr = is
server_port = 7000

[ssh-C]
type = tcp
local_ip =127.0.0.1
local_port = 22
remote_port = xyz
```
这里`server_addr`是公网服务器的公网ip，`server_port`就是公网服务器开启的捆绑端口`bind_port`。与公网服务器用该端口进行捆绑后，我们设置`ssh`的服务`ssh-C`，从而转发`ssh`服务到公网服务器的`xyz`端口，因为`ssh`服务本质上是一个文本传输服务`tcp`，因此我们给`ssh`的传输类型也是`tcp`。然后我们在内网服务器上运行`frp`服务
```
frpc -c frpc.ini
```
然后，我们通过访问公网服务器xyz端口，就能访问到内网服务器，示例如下
```
ssh -p xyz username@is
```
注意，以上方法可以设置一台公网服务器对多台内网服务器的绑定，只需要设置不同的xyz端口即可。

### 服务器在线编程——Jupyter Lab
在自己电脑上写程序的时候，`jupyter notebook`是很好的辅助工具。我们希望在服务器上开一个类似的web服务，使其可以在自己的浏览器中访问，从而实现远程调试，在线编程。我们采用`jupyter lab`来实现。`jupyter lab`是一种可以调用服务器的python解释器后端处理网页上输入的工具，我们要让它可以随处访问，需要一些额外配置，包括配置访问ip，访问密码，文件存储目录，本机端口，配置好以后，我们就可以开启在线编程的web服务了。

jupyter lab可以采用Anaconda进行安装
```
conda install jupyterlab
```

安装好以后，我们在命令行输入
```
jupyter lab --generate-config
```
则会生成一个名为`.jupyter`的文件夹，里面有配置文件`jupyter_notebook_config.py`。我们通过修改它即可进行配置。首先我们对`ip`与`port`进行配置，即打开jupyter lab服务后，可以通过哪个ip的哪个端口访问。我们的建议是将`ip`设置为本机的内网ip，`port`随意设置为支持`http`服务的端口如下
```
c.NotebookApp.ip = "内网ip"
c.NotebookApp.port = "端口"
```
然后，我们可以设置登陆该jupyter lab所需要的密码。配置文件通过256位SHA密钥生成，我们把密码转换为密钥方能填写。在服务器上打开python，运行
```
from notebook.auth import passwd
passwd()
```
则会要求你输入密码，输入密码后将返回密码对应的密钥。如输入`123456`则会返回
```
'sha1:778d23c18ddb:d9afd76551b9aebc33ee628b792460aa7763b870'
```
我们将其如下粘贴到配置文件中
```
c.NotebookApp.password_required = True
c.NotebookApp.password = 'sha1:778d23c18ddb:d9afd76551b9aebc33ee628b792460aa7763b870'
```
最后，我们设置文件存储目录
```
c.NotebookApp.notebook_dir = "/home/user/mydir"
```
这样所有在jupyter lab中的文件与notebook都会存储于`/home/user/mydir`这个文件夹下，文件夹需要提前创建，不然会报错。注意到服务器是没有浏览器的，因此我们设置服务器的浏览器默认选项为false
```
c.NotebookApp.open_browser = False
```
然后，我们在命令行运行
```
jupyter lab
```
就可以在内网中通过`http:内网ip:端口`访问到服务器开启的jupyter lab服务了。

### 用公网中介访问内网web服务
如上文所述，我们可以用服务器在内网开启好用的在线编程服务。那么，内网的web服务如何通过公网访问呢？如果我们已经如第一章所述设置了对应的`ssh`, 那么一个很简单的方式是反向端口映射。以jupyter lab为例，如果我们已经设置了公网的ssh穿透，那么我们需要访问内网的web服务，只需要在自己机器的命令行中输入
```
ssh -p xyz -NfL localhost:端口:内网ip:端口 username@is
```
就可以把内网的web服务端口转发到本机。但是这种操作非常繁琐，每次开启都要重新输入一遍，更优雅的解决方案是通过frp再加二级域名来进行。我们用frp将内网的web端口转发到公网服务器的端口，通过访问公网服务器的某端口即可访问到内网。但是，往往公网服务器暴露的http服务端口只有1个，因此，我们可以通过二级域名的方式转发内网服务器的多个web服务，这里需要用到域名。
#### 域名注册
域名注册可以用腾讯云或阿里云服务器注册，在上面购买一个便宜的域名即可，国内购买的域名需要做实名认证，并等待10-12小时，域名就能被Dns服务器所记录，这样我们就可以用这个域名来进行定向了。腾讯云与阿里云都提供域名解析服务，我们点击解析，将域名与我们的服务器进行绑定，一个示例如下
```
主机记录:* 记录类型:A 记录值:公网ip
```
添加这个解析条目后，我们ping任何`xxyyzz.域名`都会直接ping到我们的公网ip上，这样就把域名和公网ip进行了绑定。我们利用该域名，对不同的web服务分配不同的前缀，然后在公网服务器上开一个支持`http`(超文本传输)协议的端口，就能实现内网web服务的公网访问，以jupyter lab为例。
#### Frp配置
首先，我们在服务器端的`frps.ini`文件中新增`http`端口与响应的域名为
```
[common]
bind_port = 7000
vhost_http_port = 8080
subdomain_host = web.域名
```
这样我们可以在服务器端开启`8080`端口来进行web访问，而内网的所有web服务将转发到公网服务器的8080端口。以转发上文开启的jupyter lab服务为例，我们对内网服务器进行配置，在`frpc.ini`文件中新增基于`http`协议的条目如下
```
[common]
server_addr = is
server_port = 7000

[ssh-C]
type = tcp
local_ip =127.0.0.1
local_port = 22
remote_port = xyz

[jupyterlab-C]
type = http
local_ip = 内网ip
local_port = 内网port
subdomain = jupyter
```
重新开启
```
frps -c frps.ini
frpc -c frpc.ini
```
我们就可以通过在浏览器访问`jupyter.web.域名:8080`转到内网服务器的web服务中，一切与在内网中访问`内网ip:内网port`完全一致。

**Notes：解决麻烦的备案问题**

注意我们的服务器和域名都是在大陆购买的，因此在公网开web服务其实是需要备案的。实名很简单，但是备案需要填写很多网站资料，还要贴网站的审核号，是很烦的事情。一个比较简单的方法是，把域名都改成`域名.`，即修改
```
subdomain_host = web.域名.
```
访问的时候也加`域名.:8080`即可，不然会被审查。
