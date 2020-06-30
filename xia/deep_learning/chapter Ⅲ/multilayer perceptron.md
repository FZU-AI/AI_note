# 多层感知机

## 多层感知机
多层感知机（multilayerperceptron，MLP）
### 1.1隐藏层
多层感知机在单层神经⽹络的基础上引⼊了⼀到多个隐藏层  
////图片
虽然神经⽹络引⼊了隐藏层，却依然等价于⼀个单层神经⽹络  
### 1.2激活函数
全连接层只是对数据做仿射变换（affinetransformation），而多个仿射变换的叠加仍然是⼀个仿射变换。  
可以引入非线性变换。  
定义xypolt函数，来观察非线性变换  
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
```
#### ReLU函数
ReLU（rectifiedlinearunit）函数提供了⼀个很简单的⾮线性变换。给定元素x，该函数定义为 ReLU(x) = max(x,0). 可以看出，ReLU函数只保留正数元素，并将负数元素清零。  
```python
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')
```
#### sigmoid函数
sigmoid函数可以将元素的值变换到0和1之间：
//公式
```python
with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')
```
依据链式法则，sigmoid函数的导数 sigmoid′(x) = sigmoid(x)(1−sigmoid(x)). 
sigmod函数导数图像
```python
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')
```
#### tanh函数
tanh,双曲正切函数，可以将元素的值变换为-1到1.
```python
with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')
```
依据链式法则，tanh函数的导数  
tanh′(x) = 1−tanh2(x).  
图像为：  
```python
 y.backward() xyplot(x, x.grad, 'grad of tanh')
```
### 1.3多层感知机
多层感知机就是含有⾄少⼀个隐藏层的由全连接层组成的神经⽹络，且每个隐藏层的输出通过激活函数进⾏变换
## 多层感知机的从零开始实现 
⾸ 先导⼊实现所需的包或模块。
```python
%matplotlib inline
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
```
### 2.1获取和读取数据 
使用Fashion-MNIST数据集
```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```
### 2.2定义模型参数 
使用长度为28×28=784的向量表示每一张图像  
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```
### 2.3定义激活函数 
使用基础的maximum函数来实现ReLU
```python
def relu(X):
    return nd.maximum(X, 0)
```
### 2.4定义模型
通过reshape函数将每张原始图像改成长度为num_inputs的向量
```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```
### 2.5定义损失函数 
使用Gluon提供的包括softmax运算和交叉熵损失计算的函数  
```python
loss = gloss.SoftmaxCrossEntropyLoss()
```
### 2.6训练模型 
直接调用d2lzh包中的train_ch3函数 
```python
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```
## 多层感知机的简洁实现
使用gluon实现
首先导入所需的包或模块。
```python
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```
### 3.1定义模型 
加了一个隐藏单元为256个的全连接层作为隐藏层  
使用ReLU函数作为激活函数。  
```python
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```
### 3.2读取数据并训练模型
```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```
                                                                            
