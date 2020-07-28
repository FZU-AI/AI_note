#               **第3章**

## **3.2 线性回归的从零开始实现**

先导入本例实验所需的模块和包，其中matplotlib包用于作图，且设置成嵌入显示

```
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
```

#### 3.2.1 生成数据集

构造⼀个简单的⼈⼯训练数据集，它可以使我们能够直观⽐较学到的参数和真实的模型参数的区别。设训练数据集样本数为1000，输⼊个数（特征数）为2。给定随机⽣成的批量样本特征**X** *∈* R 1000*×*2，我们使⽤线性回归模型真实权重**w** = [2*, ;*3*.*4]*⊤*和偏差*b* = 4*.*2，以及⼀个随机噪声项***ϵ***来⽣成标签

​                                                        **y** = **X w** + ***b*** + ***ϵ**,*

其中噪声项*ϵ*服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中⽆意义的⼲扰。

开始生成数据集

```
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
										//每一行均为长度为2的向量
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
										//每一行均为长度为1的向量
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

```
features[0], labels[0]
```

```
(
 [1.1630785 0.4838046]
 <NDArray 2 @cpu(0)>,
 
 [4.879625]
 <NDArray 1 @cpu(0)>)
```

通过⽣成第⼆个特征features[:, 1]和标签 labels 的散点图，可以更直观地观察两者间的线性关系

```
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);  # 加分号只显示图
```



#### 3.2.2  读取数据

在训练模型的时候，需要遍历数据集并不断读取小批量数据样本。这⾥定义⼀个函数：它每次返回batch_size（批量⼤小）个随机样本的特征和标签。

# 本函数已保存在d2lzh包中方便以后使用
```
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素
```

读取第⼀个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量⼤小和输⼊个数；标签形状为批量⼤小。

```
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

```
[[-0.55645716  1.7580789 ]
 [-0.6977626   0.23392938]
 [-1.270571    0.67174363]
 [ 0.5712555  -1.571084  ]
 [-0.44089758 -1.0183922 ]
 [ 0.9014263   1.0593758 ]
 [ 1.5194443   1.9040879 ]
 [-0.4658121  -0.28205368]
 [ 0.04718421  0.54256505]
 [ 1.0161712   0.5092231 ]]
<NDArray 10x2 @cpu(0)> 
[-2.9010134  2.006902  -0.6294592 10.672475   6.77526    2.390382
  0.7545429  4.217416   2.4509826  4.498421 ]
<NDArray 10 @cpu(0)>
```

#### 3.2.3 初始化模型参数

将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

```
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```

之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此需要创建它们的梯度。

```
w.attach_grad()
b.attach_grad()
```

#### 3.2.4 定义模型

线性回归的⽮量计算表达式的实现

```
def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    return nd.dot(X, w) + b
```

#### 3.2.5 定义损失函数

使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，需要把真实值`y`变形成预测值`y_hat`的形状。以下函数返回的结果也将和`y_hat`的形状相同。

```
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

#### 3.2.6 定义优化算法

以下的`sgd`函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。将它除以批量大小来得到平均值。

```
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh包中方便以后使用
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

#### 3.2.7. 训练模型

在训练中，将多次迭代模型参数。在每次迭代中，根据当前读取的小批量数据样本（特征`X`和标签`y`），通过调用反向函数`backward`计算小批量随机梯度，并调用优化算法`sgd`迭代模型参数。由于之前设批量大小`batch_size`为10，每个小批量的损失`l`的形状为(10, 1)。回忆一下自动求梯度那节，由于变量`l`并不是一个标量，运行`l.backward()`将对`l`中元素求和得到新的变量，再求该变量有关模型参数的梯度。

在一个迭代周期（epoch）中，将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。

```
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
```

```
epoch 1, loss 0.035054
epoch 2, loss 0.000124
epoch 3, loss 0.000049
```

训练完成后，可以比较学到的参数和用来生成训练集的真实参数，它们应该很接近。

```
true_w, w
```

```
([2, -3.4],
 
 [[ 1.9996009]
  [-3.4001083]]
 <NDArray 2x1 @cpu(0)>)
```

```
true_b, b
```

```
(4.2,
 
 [4.199646]
 <NDArray 1 @cpu(0)>)
```