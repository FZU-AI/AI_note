# Softmax回归
## Softmax回归
对于离散值预测问题，可以使用softmax分类模型
### 1.1分类问题
softmax的输出是离散的
### 1.2Softmax回归模型
softmax回归跟线性回归⼀样将输⼊特征与权重做线性叠加。与线性回归的⼀个主要不同在于， softmax回归的输出值个数等于标签⾥的类别数。  
有四个输入和三个输出，包含三个偏差
#### 1.2.1softmax运算
将输出当做预测类别的置信度，并将最大的值所对应的类作为预测输出。
softmax运算符（softmaxoperator）可以将输出值转变为正值且和为1的概率分布
//////////////////////////补充图片
### 1.3单样本分类的⽮量计算表达式
////////////////////////补充图片
### 1.4⼩批量样本分类的⽮量计算表达式
将小批量数据进行矢量计算
### 1.5交叉熵损失函数
交叉熵是常用的衡量两个概率分布差异的测量函数
/////////补充图片
### 1.6模型预测及评价
使用准确率来衡量模型的表现。它等于正确预测数量与总预测数量之⽐。  
##  图像分类数据集（Fashion-MNIST） 
### 2.1获取数据集
导入所需的包或模块
```python
%matplotlib inline
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time 
```
通过gluon中的data包来下载数据集
```python
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```
将不同类别的数值标签转化为文本标签
```python
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```   
### 2.2读取小批量
直接创建DataLoader实例。该实例每次读取一个样本数为batch_size的小批量数据。  
Gluon的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取（暂不支持Windows操作系统）。这里我们通过参数num_workers来设置4个进程读取数据。  
通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间。  
通过数据集的transform_first函数，我们将ToTensor的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上。  
```python
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```                            
## softmax回归的从零开始实现 
首先导入所需的包或模块
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, nd
```
### 3.1获取和读取数据 
使用Fashion-MNIST数据集，并设置批量大小为256。 
```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```
### 3.2初始化模型参数 
宽和高均为28像素的图像
单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。  
```python
num_inputs = 784
num_outputs = 10
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
```
梯度
```python
W.attach_grad()
b.attach_grad()
```
### 3.3实现softmax运算 
```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制
```
### 3.4定义模型 
```python
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
```
### 3.5定义损失函数 
使用pick函数。  
```python
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()
```
### 3.6计算分类准确率 
定义准确率accuracy函数。其中y_hat.argmax(axis=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y状相同。  
```python
#首先将变量y变换为浮点数，在进行判断
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
```
评价模型net在数据集data_iter上的准确率。  
```python
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n
```
### 3.7训练模型
用小批量随机梯度下降来优化模型的损失函数。在训练模型时，迭代周期数num_epochs和学习率lr都是可以调的超参数
```python
num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # “softmax回归的简洁实现”一节将用到
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)
```
### 3.8预测
```python
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```
## softmax回归的简洁实现
首先导入所需的包或模块。
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```
### 4.1获取和读取数据 
```python
atch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```
### 4.2定义和初始化模型 
使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数  
```python
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```
### 4.3softmax和交叉熵损失函数 
使用gluon提供的一个包括softmax运算和交叉熵损失计算的函数
```python
loss = gloss.SoftmaxCrossEntropyLoss()
```
### 4.4定义优化算法 
学习率为0.1的小批量随机梯度下降作为优化算法。
```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```
### 4.5训练模型 
使用上一节定义的训练函数来训练模型
```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
```
```python
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```
