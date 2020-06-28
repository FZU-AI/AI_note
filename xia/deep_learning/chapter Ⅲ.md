# 第三章
## 线性回归
线性回归的输出是一个连续值因此适合回归问题。
### 1.1基础知识
#### 模型
以房屋模型为例，通过房屋面积和使用时间来预测房屋价格。
线性回归需要一个线性表达式，如：Y=(X1)(W1)+(X2)(W2)+b;  
其中X1，X2为变量；W1，W2分别为其对应的权重；b是偏差；Y是对真实值的预测。
#### 模型训练
##### 训练数据
例子中一个房屋被称为一个**样本**；多栋房屋的面积，使用时间和价格的一系列真实数据被称为**训练集**；Y的真实值被称为**标签**，自变量X被称为**特征**。
##### 损失函数
机器学习中，衡量误差的函数成为**损失函数（loss function)。**       
在模型训练中，常使用平方损失作为损失函数：ℓ=(Y-y)²/2  
![损失函数](https://www.cnblogs.com/xia-z/gallery/image/323901.html)
通常使用训练集中所有样本误差的算数平均数来横梁模型预测的质量。
在训练中，希望能够找到一组权重和偏差，使样本的损失最小。
##### 优化算法
当模型和损失函数较为简单时，误差最小化的解可以直接通过公式表达出来，这类解被称为**解析解**  
大多数深度学习模型没有解析解，只能通过优化算法有限迭代模型参数来尽可能降低损失函数的值，这类解被称为**数值解**
**小批量随机梯度下降**在深度学习中被广泛使用：  
*先选取一组模型的初始值，再对参数进行多次迭代，每次迭代都尽可能的降低损失函数的值。*  
*在每次迭代中，由固定数目的训练数据样本组成小批量B，然后求B的平均损失关于参数的导数，最后⽤此结果与预先设定的⼀个正数的乘积作为模型参数在本次迭代的减小量*
#### 模型预测
在模型完成后，优化算法停止时的值，为最优解或最优解的近似，然后使用此参数的线性回归模型来估算结果。  
## 1.2线性回归的表示方法
### 神经网络图
线性回归是一个单层神经网络图
神经网络图中，输入个数又称**特征数**或**特征向量维度**
### 矢量计算表达式
```python
#定义两个1000维的向量
from mxnet import nd 
from time import time
a = nd.ones(shape=1000) 
b = nd.ones(shape=1000) 
```
```python
#直接将向量做矢量相加
d = a + b 
```
矢量相加比按元素逐个标量相加的速度更快。
## 线性回归的从零开始实现
首先只利⽤NDArray和autograd来实现⼀个线性回归的训练
```python
%matplotlib inline//作图模块
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
```
### 2.1生成数据集
首先构造一个简单的数据集，定训练集样本数为1000，特征数为2，随机给出样本特征；使用给定的权重w = [2,−3.4]⊤和偏差b = 4.2，随机一个噪声ϵ来⽣成标签  
                                  y = Xw + b + ϵ,    
噪声项服从均值为0，标准差为0.01的正态分布
```python
num_inputs = 2//特征数为2
num_examples = 1000//训练集样本个数为1000
true_w = [2, -3.4]//给定的权重
true_b = 4.2//给定的偏差
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))//随机生成样本特征
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b//模型公式
labels += nd.random.normal(scale=0.01, shape=labels.shape)//给公式加上噪声
```
### 2.2读取数据
在训练模型时，需要遍历数据集并不断读取小批量数据样本。  
定义一个函数，每次返回批量大小个随机样本的标签和特征。
```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素
```
### 2.3初始化模型参数
我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
```python
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```
并且分别创建其梯度
```python
w.attach_grad()
b.attach_grad()
```
### 2.4定义模型
按照公式定义模型
```python
def linreg(X, w, b): 
    return nd.dot(X, w) + b
```
### 2.5定义损失函数
使用平方损失定义损失函数
```python
def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```
### 2.6定义优化算法
使用sgd函数实现小批量随机梯度下降算法
```python
def sgd(params, lr, batch_size): 
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```
### 2.7训练模型
在训练中，使用多次迭代模型参数；每次迭代中，根据当前读取的特征和标签，调用反向函数backward计算小批量随机梯度，并调用优化函数sgd迭代模型参数。
```python
lr = 0.03//学习率，超参数
num_epochs = 3//迭代周期个数，超参数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  //训练模型一共需要num_epochs个迭代周期
    // 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    // 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  // l是有关小批量X和y的损失
        l.backward()  //小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  //使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
```
训练得到的参数和真实参数十分接近。
## 线性回归的简洁实现
使用gulon可以简洁的实现模型。
