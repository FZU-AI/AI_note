# 模型
## 模型选择、⽋拟合和过拟合 
### 1.1训练误差和泛化误差 
**训练误差（training error）**指模型在训练数据集上表现出的误差  
**泛化误差（generalization error）**指模型在任意一个测试数据样本上表现出的误差的期望  
机器学习模型应关注降低泛化误差。  
### 1.2模型选择 
多层感知机，可以选择隐藏层的个数，以及每个隐藏层中隐藏单元个数和激活函数。  
#### 验证数据集
严格讲，测试集只能在所有超参数和模型参数选定后使用一次  
可以预留一部分在训练数据集和测试数据集以外的数据来进行模型选择。这部分数据被称为**验证数据集**，简称验证集（validation set）。
#### K折交叉验证 
k折交叉验证中，把原始训练数据集分割成k个不重合的子数据集，然后做k次模型训练和验证。每一次，使用一个子数据集验证模型，并使用其他k−1个子数据集来训练模型。  
最后，对这k次训练误差和验证误差分别求平均。
### 1.3⽋拟合和过拟合 
模型无法得到较低的训练误差，我们将这一现象称**作欠拟合（underfitting）**
模型的训练误差远小于它在测试数据集上的误差，我们称该现象为**过拟合（overfitting）**
#### 模型复杂度
多项式函数拟合的目标是找一个K阶多项式函数  
如果模型的复杂度过低，很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。  
//图片
#### 训练数据集⼤⼩
如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生
### 1.4多项式函数拟合实验 
以多项式函数拟合为例  
首先导入实验需要的包或模块。  
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
```
#### ⽣成数据集 
我们使用如下的三阶多项式函数来生成该样本的标签：
y=1.2x−3.4x²+5.6x³+5+ϵ,
其中噪声项ϵ服从均值为0、标准差为0.1的正态分布。训练数据集和测试数据集的样本数都设为100。  
```python
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2),
                          nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)
```
#### 定义、训练和测试模型
定义作图函数semilogy，其中y轴使用了对数尺度。
```python
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
```
尝试使用不同复杂度的模型来拟合生成的数据集，所以把模型定义部分放在fit_and_plot函数中
```python
num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())
```
#### 三阶多项式函数拟合（正常） 
先使用与数据生成函数同阶的三阶多项式函数拟合
```python
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])
```
#### 线性函数拟合（⽋拟合）
再试试线性函数拟合
```python
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])
```
#### 训练样本不⾜（过拟合）
只使用两个样本来训练模型
```python
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])
```
典型的过拟合
## 权重衰减
虽然增大训练数据集可能会减轻过拟合，但是获取额外的训练数据往往代价高昂。故使用权重衰减  
### 2.1方法
权重衰减等价于**L2范数正则化（regularization）**  
L2范数正则化在模型原损失函数基础上添加L2范数惩罚项，从而得到训练所需要最小化的函数。  
L2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积。  
//图片
L2范数正则化令权重w1和w2先⾃乘小于1的数，再减去不含惩罚项的梯度  
### 2.2⾼维线性回归实验
们以⾼维线性回归为例来引⼊⼀个过拟合问题，并使⽤权重衰减来应对过拟合。  
设维度为p  
们使⽤ 如下的线性函数来⽣成该样本的标签：  
y = 0.05 +p ∑ i=10.01xi + ϵ,
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, num_inputs = 20, 100, 200//维度200，训练集样本数20
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
```
### 2.3从零开始实现
#### 初始化模型参数 
定义随机初始化模型参数的函数  
该函数为每个参数附上梯度  
```python
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```
#### 定义L2范数惩罚项 
只惩罚模型的权重参数。
```python
def l2_penalty(w):
    return (w**2).sum() / 2
```
#### 定义训练和测试 
定义如何在训练数据集和测试数据集上分别训练和测试模型  
```python
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                //添加了L2范数惩罚项，广播机制使其变成长度为batch_size的向量
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())
```
#### 观察过拟合
训练并测试高维线性回归模型  
当lambd设为0时，没有使用权重衰减,出现典型的过拟合
```python
fit_and_plot(lambd=0)
```
#### 使⽤权重衰减
```python
fit_and_plot(lambd=3)
```
### 2.4简洁实现 
直接在构造Trainer实例时通过wd参数来指定权重衰减超参数  
```python
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # 对权重参数衰减。权重名称一般是以weight结尾
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate': lr, 'wd': wd})
    # 不对偏差参数衰减。偏差名称一般是以bias结尾
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # 对两个Trainer实例分别调用step函数，从而分别更新权重和偏差
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
```
## 丢弃法
丢弃法也可以用来应对过拟合问题  
### 3.1方法
当对隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。  
训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了得到更加确定性的结果，一般不使用丢弃法。  
### 3.2从零开始实现
dropout函数将以drop_prob的概率丢弃NDArray输入X中的元素。  
```python
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob
```
#### 定义模型参数 
使用Fashion-MNIST数据集
定义包括两个隐藏层的多层感知机，隐藏层的输出个数为256  
```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```
#### 定义模型 
定义模型将全连接层和激活函数ReLU串起来，并对每个激活函数的输出使用丢弃法  
分别设置各个层的丢弃概率。通常把靠近输入层的丢弃概率设得小一点  
第一层为0.2，第二层为0.5
```python
rop_prob1, drop_prob2 = 0.2, 0.5

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return nd.dot(H2, W3) + b3
```
#### 训练和测试模型
```python
num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```
### 3.3简洁实现
在Gluon中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。  
Dropout只会在训练时发挥作用  
```python
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob1),  # 在第一个全连接层后添加丢弃层
        nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob2),  # 在第二个全连接层后添加丢弃层
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```
训练并测试模型
```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```
## 正向传播、反向传播和计算图 
使⽤数学和计算图（computational graph）两个⽅式来描述正向传播和反向传播。
具体来说，将以带L2范数正则化的含单隐藏层的多层感知机为样例模型解释正向传播和反向传播。 
### 4.1正向传播
**正向传播**是指对神经⽹络沿着从输⼊层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）
```python
```
### 4.2正向传播的计算图
通常绘制计算图来可视化运算符和变量在计算中的依赖关系。  
//插图
### 4.3反向传播 
**反向传播**指的是计算神经⽹络参数梯度的⽅法。
总的来说，反向传播依据微积分中的链式法则，沿着从输出层到输⼊层的顺序，依次计算并存储⽬标函数有关神经⽹络各层的中间变量以及参数的梯度
### 4.4训练深度学习模型 
在训练深度学习模型时，正向传播和反向传播之间相互依赖。  
⼀⽅⾯，正向传播的计算可能依赖于模型参数的当前值，而这些模型参数是在反向传播的梯度 计算后通过优化算法迭代的。  
另⼀⽅⾯，反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传 播计算得到的。  
因此，在模型参数初始化完成后，我们交替地进⾏正向传播和反向传播，并根据反向传播计算的梯度迭代模型参数  
## 数值稳定性和模型初始化 
深度模型有关数值稳定性的典型问题是衰减（vanishing）和爆炸（explosion）。   
### 5.1衰减和爆炸 
当神经⽹络的层数较多时，模型的数值稳定性容易变差。
### 5.2随机初始化模型参数 
在神经⽹络中，通常需要随机初始化模型参数。  
```python
```
#### MXNet的默认随机初始化
如果不指定初始化⽅法，如net.initialize()，MXNet将使⽤默认的随机初始化⽅ 法：权重参数每个元素随机采样于-0.07到0.07之间的均匀分布，偏差参数全部清零。  
#### Xavier随机初始化 
假设某全连接层的输⼊个数为a， 输出个数为b，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布   
//插入公式
