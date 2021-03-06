# 第一章 深度学习简介 #
通俗来说，**机器学习**是一门讨论各式各样的适用于不同问题的函数形式，以及如何使用数据来有效地获取函数参数具体值的学科。

**深度学习**是指机器学习中的一类函数，它们的形式通常为多层神经网络。

绝大多数神经网络都包含以下的**核心原则**：

1.交替使用线性处理单元与非线性处理单元，它们经常被称为“层”。
 
2.使用链式法则（即反向传播）来更新网络的参数（权值和偏置）。 

**深度学习的优点：**

1.优秀的容量控制方法，如丢弃法（即dropout），可避免大型网络的训练进入过拟合（过拟合即模型在训练集上的预测表现很好，而在测试集上的预测表现不好）。这是靠在整个网络中注入噪声而达到的，如训练时随机将权重替换为随机的数字。 

2.注意力机制使用了一个可学习的指针结构来构建出一个精妙的解决方法。也就是说，与其在像机器翻译这样的任务中记忆整个句子，不如记忆指向翻译的中间状态的指针。

3.模型允许重复修改深度网络的内部状态，这样就能模拟出推理链条上的各个步骤，就好像处理器在计算过程中修改内存一样。 

4.生成对抗网络的关键将采样部分替换成了任意的含有可微分参数的算法。这些参数将被训练到使辨别器不能再分辨真实的和生成的样本。生成对抗网络可使用任意算法来生成输出。

5.并行计算的能力

设计可扩展算法的最大瓶颈在于深度学习优化算法的**核心**：随机梯度下降需要相对更小的批量。

**框架：**

第一代框架：Caffe、Torch和Theano

第二代框架：TensorFlow（经常是以高层API Keras的形式被使用）、CNTK、Caffe2和ApacheMXNet。

第三代框架，即命令式深度学习框架，是由用类似NumPy的语法来定义模型的Chainer所开创的。这样的思想后来被PyTorch和MXNet的Gluon  API采用。

**深度学习的应用：**
1.智能语音助手
2.物体识别
3.博弈
4.自动驾驶汽车

**机器学习和深度学习的关系：**

**机器学习**研究如何使计算机系统利用经验改善性能。

**深度学习**是具有多级表示的表征学习方法（作为机器学习的一类，表征学习关注如何自动找出表示数据的合适方式，以便更好地将输入变换为正确的输出）。在每一级（从原始数据开始），深度学习通过简单的函数将该级的表示变换为更高级的表示。

深度学习基于神经网络模型可以逐级表示越来越抽象的概念或模式。以图像为例，它的输入是一堆原始像素值。深度学习模型中，图像可以逐级表示为特定位置和角度的边缘、由边缘组合得出的花纹、由多种花纹进一步汇合得到的特定部位的模式等。最终，模型能够较容易根据更高级的表示完成给定的任务，如识别图像中的物体。

**深度学习的特点：**

1.端到端的训练。即并不是将单独调试的部分拼凑起来组成一个系统，而是将整个系统组建好之后一起训练。

2.从含参数统计模型转向完全无参数的模型。当数据非常稀缺时，我们需要通过简化对现实的假设来得到实用的模型；当数据充足时，我们就可以用能更好地拟合现实的无参数模型来替代这些含参数模型。

3.相对其它经典的机器学习方法而言，深度学习的不同在于：对非最优解的包容、对非凸非线性优化的使用