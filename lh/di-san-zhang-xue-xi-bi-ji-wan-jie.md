# 第三章

从单层神经网络延伸到多层神经网络，并通过多层感知机引入深度学习模型

> * 线性回归
> * 线性回归的从零开始实现
> * 线性回归的简洁实现
> * softmax回归
> * softmax回归的简洁实现
> * 多层感知机
> * 模型选择、欠拟合和过拟合 
> * 权重衰减 
> * 丢弃法 
> * 正向传播、反向传播和计算图
> * 数值稳定性和模型初始化 
> * 实战Kaggle比赛：房价预测

## 线性回归

### 1.模型与模型训练

线性回归假设输出与各个输入之间是线性关系 如： y = x1w1 + x2w2 + b; 基于输入x1和x2来计算输出y的表达式，其中w1和w2是权重（weight），b是偏差（bias），且均为标量。 **它们是线性回归模型的参数（pa-rameter）** 。模型输出^ y是线性回归对真实价格y的预测或估计。 接下来我们需要通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。这个过程 叫作模型训练（model training）

### 2.损失函数

选取一个非负数作为误差，且数值越小表示误差越小。一个常用的选择是平方函数。它在评估索引为i的样本误差的表达式为： ℓ \(i\) \(w1; w2; b\) = 1/2 \* \( Y \(i\)- y \(i\) \)^2 误差越小表 示预测价格与真实价格越相近，且当二者相等时误差为0

### 3.矢量计算表达式

```python
from mxnet import nd 
from time import time 
#对两个向量相加的两种方法
a = nd.ones(shape=1000) 
b = nd.ones(shape=1000) 
#向量相加的一种方法是，将这两个向量按元素逐一做标量加法
start = time() 
c = nd.zeros(shape=1000) 
for i in range(1000): 
    c[i] = a[i] + b[i] 
time() - start #计算运算时间 为0.15223002433776855 
#向量相加的另一种方法是，将这两个向量直接做矢量加法
start = time() 
d = a + b 
time() - start #0.00029015541076660156 应该尽可能采用矢量计算，以提升计算效率
```

## 线性回归的从零开始实现

```python
#首先导入包
%matplotlib inline #用于绘图 
from IPython import display 
from matplotlib import pyplot as plt 
from mxnet import autograd, nd 
import random
```

### 1.生成数据集

设训练数据集样本数为1000，输入个数（特征数）为2，我们使用线性回归模型真实权重w = \[2,-3.4\]^⊤ 和偏差b = 4.2，以及一个随机噪声项ε来生成标签 y = X1w1+X2w2+b+ε

```python
num_inputs = 2 
num_examples = 1000 
true_w = [2, -3.4] 
true_b = 4.2 
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs)) 
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b 
labels += nd.random.normal(scale=0.01, shape=labels.shape) 
#features的每一行是一个⻓度为2的向量，而labels的每一行是一个⻓度为1的向量（标量）
def use_svg_display(): 
# 用矢量图显示 
    display.set_matplotlib_formats('svg') 
def set_figsize(figsize=(3.5, 2.5)): #figsize大小为宽、长
    use_svg_display() 
    # 设置图的尺寸 
    plt.rcParams['figure.figsize'] = figsize 
set_figsize() 
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1); # 加分号只显示图 
#将上面的plt作图函数以及use_svg_display函数和set_figsize函数定义在d2lzh包 里。以后在作图时，我们将直接调用d2lzh.plt。由于plt在d2lzh包中是一个全局变量，我们 在作图前只需要调用d2lzh.set_figsize()即可打印矢量图并设置图的尺寸
```

### 2.读取数据

读取数据 在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数： 它每次返回batch\_size（批量大小）个随机样本的特征和标签

```python
def data_iter(batch_size, features, labels): 
    num_examples = len(features) 
    indices = list(range(num_examples)) 
    random.shuffle(indices) # 样本的读取顺序是随机的 
    for i in range(0, num_examples, batch_size): 
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j) # take函数根据索引返回对应元素 
#每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小
batch_size = 10 
for X, y in data_iter(batch_size, features, labels): 
    print(X, y) 
    break
```

### 3.初始化以及定义模型

我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

```python
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1)) 
b = nd.zeros(shape=(1,))
#之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们需要创建它们的梯度。 
w.attach_grad() 
b.attach_grad() 

#线性回归的矢量计算表达式的实现。我们使用dot函数做矩阵乘法
def linreg(X, w, b): 
    # 本函数已保存在d2lzh包中方便以后使用 
    return nd.dot(X, w) + b
```

### 4.定义损失函数

方损失来定义线性回归的损失函数。在实现中，我们需要把真实值y变形成预测值y\_hat的形状。 以下函数返回的结果也将和y\_hat的形状相同

```python
def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 5.优化损失函数

以下的sgd函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优 化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量 大小来得到平均值

```python
def sgd(params, lr, batch_size): 
    for param in params: 
        param[:] = param - lr * param.grad / batch_size
```

### 6.训练模型

在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y），通过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数.在一个迭代周期（epoch）中，我们将完整遍历一遍data\_iter函数，并对训练数据集中所有 样本都使用一次

```python
lr = 0.03 #学习率
num_epochs = 3  #迭代周期个数 迭代周期数设得越大模型可能越有效，但是训练时间可能过⻓
net = linreg # 线性回归 
loss = squared_loss 
for epoch in range(num_epochs): 
    # 训练模型一共需要num_epochs个迭代周期 
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # 和y分别是小批量样本的特征和标签 
    for X, y in data_iter(batch_size, features, labels): 
        with autograd.record(): 
            l = loss(net(X, w, b), y) # l是有关小批量X和y的损失 
        l.backward() # 小批量的损失对模型参数求梯度 
        sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数 
        train_l = loss(net(features, w, b), labels) 
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
```

## 线性回归的简洁实现

介绍如何使用MXNet提供的Gluon接口更方便地实现线性回归的训练

### 1.生成数据集

```python
#features是训练数据特征，labels是标签
from mxnet import autograd, nd 
num_inputs = 2 
num_examples = 1000 
true_w = [2, -3.4] 
true_b = 4.2 
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs)) 
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b 
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

### 2.读取数据

Gluon提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用添加了Gluon首字母的假名gdata代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量

```python
from mxnet.gluon import data as gdata 
batch_size = 10 
# 将训练数据的特征和标签组合 
dataset = gdata.ArrayDataset(features, labels) 
# 随机读取小批量 
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

### 3.定义模型且初始化模型参数

```python
# 在Gluon中， Sequential实例可以看作是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加 层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。
from mxnet.gluon import nn net = nn.Sequential() 
# 作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1
net.add(nn.Dense(1)) 
#值得一提的是，在Gluon中我们无须指定每一层输入的形状，例如线性回归的输入个数。当模型 得到数据时，例如后面执行net(X)时，模型将自动推断出每一层的输入个数

#在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。我们从MXNet导入init模块。该模块提供了模型参数初始化的各种方法。这里的init是initializer的缩写形式。我们通过init.Normal(sigma=0.01)指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数默认会初始化为零
from mxnet import init 
net.initialize(init.Normal(sigma=0.01))
```

### 4.定义损失函数

在Gluon中，loss模块定义了各种损失函数。我们用假名gloss代替导入的loss模块，并直接 使用它提供的平方损失作为模型的损失函数。

```python
from mxnet.gluon import loss as gloss 
loss = gloss.L2Loss() # 平方损失又称L2范数损失,L2范数是指向量各元素的平方和然后求平方根
```

### 5.定义优化算法

我们也无须实现小批量随机梯度下降。在导入Gluon后，我们创建一个Trainer实例，并 指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。该优化算法将用来迭代net实例所 有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect\_params函数获取。

```python
from mxnet import gluon 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

### 6.训练模型

在使用Gluon训练模型时， 我们通过调用Trainer实例的step函数来迭代模型参数。上一节中我们提到，由于变量l是⻓度为batch\_size的一维NDArray，执行l.backward\(\)等价于执行l.sum\(\).backward\(\)。按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从 而对批量中样本梯度求平均

```python
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

## softmax回归

和线性回归不同，softmax回归的输出单元从一个变成了多个，且引入了softmax运算使输出更适合离散值的预测和训练

### 1.softmax回归模型

softmax回归跟线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，softmax回归的输出值个数等于标签里的类别数. 为一共有4种特征和3种输出动物类别，所以 权重包含12个标量\(带下标的w\) 、偏差包含3个标量\(带下标的b\)， 且对每个输入计算o1; o2; o3这3个输出 o1 = x1w11 + x2w21 + x3w31 + x4w41 + b1; o2 = x1w12 + x2w22 + x3w32 + x4w42 + b2; o3 = x1w13 + x2w23 + x3w33 + x4w43 + b3: softmax回归同线性回归一样，也是一个单层神经网络。由于每个输出o1; o2; o3的计算都要依赖于所有的输入x1; x2; x3; x4，softmax回归的输出层也是一个全连接层 分类问题需要得到离散的预测输出，一个简单的办法是将输出值oi当作预测类别是i的置信度，并将值最大的输出所对应的类作为预测输出，即输出argmax。例如，如果o1; o2; o3分别 为0:1; 10; 0:1，由于o2最大，那么预测类别为2，其代表猫 softmax运算符（softmax operator）解决了以上两个问题。它通过下式将输出值变换成值为正且 和为1的概率分布：

## 图像分类数据集（Fashion-MNIST）

Fashion-MNIST是一个10类服饰分类数据集,用于体现算法的性能

## softmax回归的简洁实现

使用Gluon来实现一个softmax回归模型

```python
#首先导入所需的包或模块。
%matplotlib inline 
import d2lzh as d2l 
from mxnet import gluon, init 
from mxnet.gluon import loss as gloss, nn 
batch_size = 256 
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
#softmax回归的输出层是一个全连接层。因此，我们添加一个输出个数为10的全连接层。我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。
net = nn.Sequential() 
net.add(nn.Dense(10)) 
net.initialize(init.Normal(sigma=0.01)) 
#Gluon提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳 定性更好
loss = gloss.SoftmaxCrossEntropyLoss() 
#使用学习率为0.1的小批量随机梯度下降作为优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1}) 
#训练模型
```

## 多层感知机

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）

### 1.隐藏层

位于输入层和输出层之间 H = XWh + bh; O = HWo + bo; 虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中 输出层权重参数为Wh,Wo，偏差参数为bhWo + bo。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价

### 2.激活函数

解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function） ReLU函数 ReLU（rectifiedlinear unit）函数提供了一个很简单的非线性变换。给定元素x，该函数定义为 ReLU\(x\) = max\(x;0\) 可以看出，ReLU函数只保留正数元素，并将负数元素清零

### 3.多层感知机

多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络， 且每个隐藏层的输出通过激 活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并 沿用本节之前定义的符号，多层感知机按以下方式计算输出 H = φ\(XW h + bh\); O = HW o + bo; 其中φ表示激活函数。在分类问题中，我们可以对输出O做softmax运算，并使用softmax回归中的交叉熵损失函数。 在回归问题中，我们将输出层的输出个数设为1，并将输出O直接提供给线性回归中使用的平方损失函数

### 4.多层感知机的简洁实现

```python
import d2lzh as d2l 
from mxnet import gluon, init 
from mxnet.gluon import loss as gloss, nn
#和softmax回归唯一的不同在于，我们多加了一个全连接层作为隐藏层。它的隐藏单元个数为256，并使用ReLU函数作为激活函数
net = nn.Sequential() 
net.add(nn.Dense(256, activation='relu'), nn.Dense(10)) 
net.initialize(init.Normal(sigma=0.01))
batch_size = 256 
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
loss = gloss.SoftmaxCrossEntropyLoss() 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5}) 
num_epochs = 5 
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
```

## 模型选择、欠拟合和过拟合

### 1.训练误差和泛化误差

训练误差和泛化误差。通俗来讲，前者指模型在训练数据集上表现出的误差，后者指模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差来近似. 当训练数据不够用时，预留大量的验证数据显得太奢侈。一种改善的方法是K折交叉验证（K-fold cross-validation）。在K折交叉验证中，我们把原始训练数据集分割成K个不重合的子数据集，然后我们做K次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他K-1个子数据集来训练模型。在这K次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这K次训练误差和验证误差分别求平均.

### 2.欠拟合和过拟合

模型无法得到较低的训练误差， 我们将这一现象称作欠拟合（underfitting） 模型的训练误差远小于它在测试数据集上 的误差，我们称该现象为过拟合（overfitting） 模型的复杂度过低， 很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。应对欠拟合和过拟合的一个办法是针对数据集选择合适复杂度的模型

## 权重衰减

权重衰减等价于L2范数正则化（regularization） 。正则化通过为模型损失函数添加惩罚项使学出 的模型参数值较小，是应对过拟合的常用手段 L2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积,L2范数正则化令权重w1和w2先自乘小于1的数，再减去不含惩罚项的梯度

### 1.简洁实现

```python
def fit_and_plot_gluon(wd): 
    net = nn.Sequential() 
    net.add(nn.Dense(1)) 
    net.initialize(init.Normal(sigma=1)) # 对权重参数衰减。权重名称一般是以weight结尾 
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd}) # 不对偏差参数衰减。偏差名称一般是以bias结尾 
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr}) 
    train_ls, test_ls = [], [] 
    for _ in range(num_epochs): 
        for X, y in train_iter: 
            with autograd.record(): 
                l = loss(net(X), y) 
            l.backward()            
            #对两个Trainer实例分别调用step函数，从而分别更新权重和偏差 
            trainer_w.step(batch_size)     
            trainer_b.step(batch_size) 
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar()) 
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
        d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test']) #画图
        print('L2 norm of w:', net[0].weight.data().norm().asscalar()) 
#fit_and_plot_gluon(0) L2 norm of w: 13.311798 
#fit_and_plot_gluon(3) L2 norm of w: 0.03225094 
使用权重衰减可以在一定程度上缓解过拟合问题
```

## 丢弃法

深度学习模型常常使用丢弃法（dropout）来应对过拟合问题 当对多层感知机的隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为p，那么有p的概率hi会被清零，有1-p的概率hi会除以1-p做拉伸

### 1.简洁实现

在Gluon中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时，Dropout层并不发挥作用,丢弃法只在训练模型时使用。

```python
drop_prob1, drop_prob2 = 0.2, 0.5 
net = nn.Sequential() 
net.add(nn.Dense(256, activation="relu"), 
nn.Dropout(drop_prob1), # 在第一个全连接层后添加丢弃层 nn.Dense(256, activation="relu"), nn.Dropout(drop_prob2), # 在第二个全连接层后添加丢弃层 nn.Dense(10)) net.initialize(init.Normal(sigma=0.01)) 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr}) 
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
```

## 正向传播、反向传播和计算图

使用数学和计算图两个方式来描述正向传播和反向传播

### 正向传播

正向传播是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量（包括输出） 假设输入是一个特征为x的样本，且不考虑偏差项，那么中间变量 z = W^\(1\)  _x 其中W^\(1\)是隐藏层的权重参数。把中间变量z输入按元素运算的激活函数φ后，将得到向量⻓度为h的隐藏层变量 h = φ\(z\) 隐藏层变量h也是一个中间变量。假设输出层参数只有权重W^\(2\),可以得到向量⻓度为q的输出层变量 o = W^\(2\)_  h 假设损失函数为ℓ，且样本标签为y，可以计算出单个数据样本的损失项 L = ℓ\(o; y\) 根据L2范数正则化的定义，给定超参数，正则化项即 s = λ/2 \( ∥W ^\(1\) ∥ ^2F+ ∥W \(2\) ∥ ^ 2\) 模型在给定的数据样 本上带正则化的损失为 J = L + s 将J称为有关给定数据样本的目标函数，并在以下的讨论中简称目标函数

### 反向传播和计算图

反向传播指的是计算神经网络参数梯度的方法。总的来说，反向传播依据微积分中的链式法则， 沿着从输出层到输入层的顺序， 依次计算并存储目标函数有关神经网络各层的中间变量以及参数的梯度 在训练深度学习模型时，正向传播和反向传播相互依赖

## 数值稳定性和模型初始化

深度模型有关数值稳定性的典型问题是衰减（vanishing）和爆炸（explosion）

### 衰减和爆炸

当神经网络的层数较多时，模型的数值稳定性容易变差。 假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入X分别与 0.2^30 =1  _10^21 （衰减）和5^30=9_10^20（爆炸）的乘积。类似地，当层数较多时，梯度的计算也更容易出现衰减或爆炸

### 随机初始化模型参数

在神经网络中，通常需要随机初始化模型参数. 使用net. initialize\(init.Normal\(sigma=0.01\)\)使模型net的权重参数采用正态分布的随机初始化方式。如果不指定初始化方法，如net.initialize\(\)，MXNet将使用默认的随机初始化方法：权重参数每个元素随机采样于-0.07到0.07之间的均匀分布，偏差参数全部清零

> **有人说随机初始化模型参数是为了“打破对称性” 。这里的“对称”应如何理解** 当我们把所有的参数都设成0的话，那么上面的每一条边上的权重就都是0，那么神经网络就还是对称的，对于同一层的每个神经元，它们就一模一样了 这样的后果是什么呢？我们知道，不管是哪个神经元，它的前向传播和反向传播的算法都是一样的，如果初始值也一样的话，不管训练多久，它们最终都一样，都无法打破对称（fail to break the symmetry）,那每一层就相当于只有一个神经元，最终L层神经网络就相当于一个线性的网络，如Logistic regression，线性分类器对我们上面的非线性数据集是“无力”的，所以最终训练的结果就瞎猜一样

## 实战Kaggle比赛：房价预测

### 1.数据读取加预处理

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E5%8A%A0%E9%A2%84%E5%A4%84%E7%90%86.png?raw=true)

### 2.损失函数加训练函数

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_%E6%8D%9F%E5%A4%B1%E5%8A%A0%E8%AE%AD%E7%BB%83.png?raw=true)

### 3.k折交叉验证

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_k%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.png?raw=true)

### 4.k折交叉验证效果图

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_k%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E6%95%88%E6%9E%9C%E5%9B%BE.png?raw=true)

### 5.利用数据集训练加预测

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_%E5%88%A9%E7%94%A8%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%AD%E7%BB%83%E5%B9%B6%E9%A2%84%E6%B5%8B.png?raw=true)

### 6.kaggle提及结果

![cmd-markdown-logo](https://github.com/TragedyN/images/blob/master/hp_%E7%BB%93%E6%9E%9C.png?raw=true)

