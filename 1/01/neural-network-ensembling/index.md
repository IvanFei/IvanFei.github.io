# 

# 深度集成

所谓模型集成，顾名思义是将多个模型的整合成一个。它要做的关键点在于 “如何获得不同的模型” 以及 “如何有效的对多个模型输出的结果进行整合”。传统的集成方法有: bagging, boosting, stacking等方法，通过获得输出具有差异性的多个同质模型(bagging, boosting)或者异质模型(stacking)，对多个输出进行整合(取平均值或者投票)以便能够获得更好的得分。其中同质模型是指相同算法或者相同结构的神经网络，异质是指不同算法或者不同结构的神经网络。

然而如果将上述传统集成方法用在深度神经网络中的话，则需要训练多个不同的模型。这样的做法虽然能够行得通，但是却消耗了大量的计算资源和时间。**所以本文主要介绍如何集成深度网络，而不需要浪费太多训练时间和计算资源 （Motivation）**。



本文主要对三个方法进行介绍，其中包括 **「Snapshot Ensembles, SSE」**, **「Fast Geometric Ensembling, FGE」** 以及 **「Stochastic Weight Averaging, SWA」**。对应以下3篇paper如下：



[Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)

[Averaging Weights leads to Wider Optima and better generalization](https://arxiv.org/abs/1803.05407)



#####& 三种方法的**亮点**

**「Snapshot Ensemblies, SSE」**: 训练**一个**深度网络模型，使得训练过程中收敛到**M**个局部极小值进行保存权重，从而实现集成**M**个模型;

**「Fast Geometric Ensembling, FGE」**:  Snapshot Ensembles 收敛到每一个局部极小值需要太多时间，我要优化一下，使得两个局部极小值之间的训练时间不要这么长。

**「Stochastic Weight Averaging, SWA」**:  诶，我为什么要保存这么多模型进行集成，对局部极小值的权重的取平均 从而获得新的单模型可以媲美多个模型的集成，美哉美哉，都不用保存模型了，inference 阶段还节省了1/M的时间。



###1. 引子：Snapshot Ensembles

Snapshot Ensembles 的**动机**和**做法**很简单。

首先是**动机**： 为了解决集成需要训练多个模型，从而导致消耗过多时间和计算资源的问题。

在介绍**做法**前，需要先介绍几个基础的**概念**：

1. 在现有流行的网络框架中存在着百万数量级的局部极小值；

2. 局部极小值的test error其实已经足够低，影响网络test error主要是saddle point（鞍点），即saddle point 拥有较大的test error；

3. 不同的局部极小值虽然有着相似的loss 或者 test error。但是其实他们的预测结果是可以互补的，即他们错误预测的样本是不一样的；

4. 大的学习率能够帮助模型脱离鞍点和局部极小值。



**做法**其实很简单：

1. 采用重启的学习率曲线（重启的学习率足够大），如下图所示。模型在每个周期末端都收敛到局部极小值，然后由于学习率的重启，导致模型脱离当前的局部极小值，下一个周期收敛到新的局部极小值（直观的理解可以见图1）。从图1中可以看到，标准学习率的loss最后会收敛到一个局部极小值（左图）；而通过循环学习率，loss会收敛到不同的局部极小值。

2. 对每个周期末端都进行模型权重的保存，用于最后的集成，即测试的时候input 经过保存权重的几个模型，并对输出进行取平均或者投票。 



![图1](/Users/huangfeifei/fei/hugo-ivanfei/static/images/content/Optimization Method/Neural Network Ensembling/SSE_fig2.png)

<center>图1: Left -> 标准学习率下 loss 的收敛路径；Right -> 循环学习率下 loss的收敛路径。 </center>



**& 循环学习率**的设置：

 $$\alpha(t) = \frac{\alpha_0}{2}(cos(\frac{\pi \cdot mod(t-1, \lceil T/M \rceil)}{\lceil T/M \rceil}) + 1)$$

其中：

$t$: 当前的迭代步数

$T$: 总的迭代步数

$M$: 学习率循环次数

$\alpha_0$: 初始学习率

**学习率曲线** 如图2所示：

![图2](/Users/huangfeifei/fei/hugo-ivanfei/static/images/content/Optimization Method/Neural Network Ensembling/SSE_fig4.png)

<center> 图2: 学习率曲线 </center>



**& SSE 结果**

![图3](/Users/huangfeifei/fei/hugo-ivanfei/static/images/content/Optimization Method/Neural Network Ensembling/SSE_fig3.png)

<center> 图3: 在Cifar 数据上 两种不同学习率的下降趋势</center>

从图3中可以看到Cosine annealing with restart 的方法可以是的loss多次收敛。每一次的收敛，都由于学习率突跳到较大值而使得网络跳出局部极小值，从而达到获取不同模型。



### 2. 进阶：**Fast Geometric Ensembling**

