# 深度学习基础

\[TOC\]

## 深度学习基础

机器学习的两个基本问题：分类与回归。

### 单层神经网络

#### 神经网络实现过程

1. 读取数据
2. 定义初始模型
3. 初始化模型参数
4. 定义损失函数
5. 训练模型

#### 神经网络回归模型——线性回归

线性回归是指在解决实际问题时，设定一个模型，各要素之间的关系是线性的，输出的值是连续值，主要用于解决回归问题。比如气温的变化，房价的涨跌等。基本数学公式表示为：

$$
y' = x1 * w1 + x2 * w2 + b
$$

y'是预测结果，x1,x2是输入的特征，w1,w2是权重，b是偏差

**线性回归模型训练**

线性回归模型训练涉及三个要素。训练数据（样本集）、损失函数、优化算法。

**训练数据**

机器学习的训练分为很多种，常见的分为监督学习和非监督学习。监督学习需要对数据集中的每个样本进行标注，标注样本的特征以及输出的结果。

**损失函数**

损失函数是用于衡量预测值与实际值之间的误差的函数，常见包括MSZ,Sigmoid等。模型训练的目的是为了是损失函数最小。

**优化算法**

迭代训练修改参数使得模型损失函数最小。常见方法mini-batch的梯度下降。线性回归模型参数迭代如下图：

![image-20200619130026013](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200619130026013.png)

注：模型训练的结果并不⼀定是最小化损失函数的最优解，而是对最优解的⼀个近似。

**线性回归模型的实现**

手工实现

d2l-zh/chapter\_deep-learning-basics/linear-regression-scratch.ipynb

gluon实现

d2l-zh/chapter\_deep-learning-basics/linear-regression-gluon.ipynb

#### 神经网络分类模型——softmax回归

softmax回归模型通过将输入特征与权重做线性叠加，输出分类结果，输出个数与数据集中的标签个数有关，有几个标签就有几个输出值。

softmax将输出值变换成值为正且和为1的概率分布：

![image-20200621165803618](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200621165803618.png)

**softmax回归模型的实现**

手工实现：d2l-zh/chapter\_deep-learning-basics/softmax-regression-scratch.ipynb

```text
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition
```

gluon实现: d2l-zh/chapter\_deep-learning-basics/softmax-regression-gluon.ipynb

#### 图像分类数据集（Fashion-MNIST）

代码：d2l-zh/chapter\_deep-learning-basics/fashion-mnist.ipynb

### 多层神经网络——MLP

多层感知机（multilayer perceptron，MLP），网络由一个输入层，一个输出层和多个隐藏层组成。

#### MLP模型

通过多层神经网络寻找特征信息。

![image-20200621232545342](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200621232545342.png)

#### 激活函数

隐藏层中，为了使特征更加可靠有效，可以引入非线性变换,即加入激活函数。常用激活函数包括ReLu,sigmoid,tanh.

![image-20200621233530430](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200621233530430.png)

#### MLP模型的实现

手工实现：d2l-zh/chapter\_deep-learning-basics/mlp-scratch.ipynb

gluon实现: d2l-zh/chapter\_deep-learning-basics/mlp-gluon.ipynb

### 模型误差

深度学习在训练过程中的误差主要是训练误差和泛化误差两个。由于模型是在训练集中进行训练调参，因此在训练过程中，往往模型的训练误差会不断的下降的，但**训练误差下降不代表泛化误差也会下降**，存在**欠拟合**，**过拟合**等问题。模型训练的目的是为了在随机生成的样本中，通过训练使泛化误差降低。

* 训练数据集：用于模型训练
* 测试数据集：用于计算模型的泛化误差，判断有无过拟合欠拟合
* 验证数据集：用于模型选择。

#### 欠拟合，过拟合

欠拟合：模型⽆法得到较低的训练误差

过拟合：模型的训练误差远小于它在测试数据集上的误差

![image-20200622122238978](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200622122238978.png)

**reason**

欠拟合：训练数据集过大，模型比较简单

过拟合：训练数据集较小，模型比较复杂

**solution**

过拟合：

权重衰减

> 权重衰减等价于_L_2范数正则化（regularization）。
>
> _L_2范数惩罚项指的是模型权重参数每个元素的平⽅和与⼀个正的常数的乘积。

丢弃法（dropout）

> 设定一定的概率，对隐藏单元判断是否丢弃掉。
>
> 设丢弃概率为p，那么有p的概率hi会被清零，有1/p的概率hi会除以1/p做拉伸。丢弃概率是丢弃法的超参数。
>
> ![image-20200622130402034](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200622130402034.png)
>
> ```text
> def dropout(X, drop_prob):
>     assert 0 <= drop_prob <= 1
>     keep_prob = 1 - drop_prob
>         # 这种情况下把全部元素都丢弃
>         if keep_prob == 0:
>             return X.zeros_like()
>     mask = nd.random.uniform(0, 1, X.shape) < keep_prob
>     return mask * X / keep_prob
> ```

### 模型正向传播、反向传播、计算图

⼀⽅⾯，正向传播的计算可能依赖于模型参数的当前值，而这些模型参数是在反向传播的梯度计算后通过优化算法迭代的。

另⼀⽅⾯，反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传播计算得到的。

#### 正向传播

对神经网络沿着从输入层到输出层的顺序，计算中间变量。

![image-20200622153816905](https://github.com/FZU-AI/AI_note/tree/f0e59590f527a02f21d7c32130bc58e19501defe/ch/动手学笔记/C:/Users/67592/AppData/Roaming/Typora/typora-user-images/image-20200622153816905.png)

#### 反向传播

指的是计算神经⽹络参数梯度的⽅法。

### 数值稳定性和模型初始化

深度模型有关数值稳定性的典型问题是衰减（vanishing）和爆炸（explosion）。指出现**梯度消失**或者**梯度爆炸**的情况。

