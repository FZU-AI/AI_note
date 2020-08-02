#               **第3章**

## **3.2.3线性回归的简洁实现**

#### 使⽤MXNet提供的Gluon接口更⽅便地实现线性回归的训练

#### 3.3.1 ⽣成数据集

```
from mxnet import autograd, nd
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

#### 3.3.2. 读取数据集

Gluon提供了`data`包来读取数据。由于`data`常用作变量名，将导入的`data`模块用添加了Gluon首字母的假名`gdata`代替；在每一次迭代中，随机读取包含10个数据样本的小批量。

```
from mxnet.gluon import data as gdata
batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
#这里data_iter的使用与上一节中的一样。读取并打印第一个小批量数据样本。
```

```
for X, y in data_iter:
    print(X, y)
    break
```

```
[[ 1.60204     0.10594607]
 [ 0.99923676 -0.86133176]
 [ 0.00236897  1.8075399 ]
 [-0.52928144 -0.8213481 ]
 [ 0.11150218 -0.22487842]
 [ 0.882522    0.23611583]
 [ 1.733211   -0.08600121]
 [-1.4979132   0.35245472]
 [ 0.06527644 -0.9506362 ]
 [ 0.88619477 -0.73974735]]
<NDArray 10x2 @cpu(0)> 
[ 7.0403533  9.121296  -1.9449078  5.9434566  5.1916413  5.16984
  7.9535317  0.0114568  7.5680094  8.485436 ]
<NDArray 10 @cpu(0)>
```

#### 3.3.3. 定义模型

在上一节从零开始的实现中，需要定义模型参数，并使用它们一步步描述模型是怎样计算的。当模型结构变得更复杂时，这些步骤将变得更烦琐。其实，Gluon提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。

使用Gluon更简洁地定义线性回归：

首先，导入`nn`模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。我们先定义一个模型变量`net`，它是一个`Sequential`实例。在Gluon中，`Sequential`实例可以看作是一个串联各个层的容器。在构造模型时，在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。

```
from mxnet.gluon import nn

net = nn.Sequential()
```

回顾图3.1中线性回归在神经网络图中的表示。作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个`Dense`实例。

定义该层输出个数为1。

```
net.add(nn.Dense(1))
#在Gluon中无须指定每一层输入的形状，例如线性回归的输入个数。当模型得到数据时，例如后面执行net(X)时，模型将自动推断出每一层的输入个数
```

#### 3.3.4. 初始化模型参数

在使用`net`前，需要初始化模型参数，如线性回归模型中的权重和偏差。从MXNet导入`init`模块。该模块提供了模型参数初始化的各种方法。这里的`init`是`initializer`的缩写形式。通过`init.Normal(sigma=0.01)`指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数默认会初始化为零。

```
from mxnet import init

net.initialize(init.Normal(sigma=0.01))
```

#### 3.3.5. 定义损失函数

在Gluon中，`loss`模块定义了各种损失函数。用假名`gloss`代替导入的`loss`模块，并直接使用它提供的平方损失作为模型的损失函数。

```
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()  # 平方损失又称L2范数损失
```

#### 3.3.6. 定义优化算法

无须实现小批量随机梯度下降。在导入Gluon后，可以创建一个`Trainer`实例，并指定学习率为0.03的小批量随机梯度下降（`sgd`）为优化算法。该优化算法将用来迭代`net`实例所有通过`add`函数嵌套的层所包含的全部参数。这些参数可以通过`collect_params`函数获取。

```
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

#### 3.3.7. 训练模型

在使用Gluon训练模型时，通过调用`Trainer`实例的`step`函数来迭代模型参数。上一节中提到，由于变量`l`是长度为`batch_size`的一维`NDArray`，执行`l.backward()`等价于执行`l.sum().backward()`。按照小批量随机梯度下降的定义，在`step`函数中指明批量大小，从而对批量中样本梯度求平均。

```
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

```
epoch 1, loss: 0.035016
epoch 2, loss: 0.000123
epoch 3, loss: 0.000049
```

下面分别比较学到的模型参数和真实的模型参数。从`net`获得需要的层，并访问其权重（`weight`）和偏差（`bias`）。学到的参数和真实的参数很接近。

```
dense = net[0]
true_w, dense.weight.data()
```

```
([2, -3.4],
 
 [[ 2.0001278 -3.4000394]]
 <NDArray 1x2 @cpu(0)>)
```

```
true_b, dense.bias.data()
```

```
(4.2,
 
 [4.199906]
 <NDArray 1 @cpu(0)>)
```