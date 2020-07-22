# 第三章

## 3.1 线性回归

### 3.1.1回归与分类

回归问题的输出是连续值，分类问题的输出是离散值

线性回归属于回归，softmax属于分类

### 3.1.2线性回归

模型：相当于建立一个函数

损失函数：衡量价格预测值与真实值之间的误差

优化算法：尽可能降低损失函数的值（比如小批量随机梯度下降调参）

神经网络图：

![image-3.1](https://github.com/FZU-AI/AI_note/blob/master/woo/deep_learning/photo/3.1.png)

此神经网络图的层数为1，输入层的输入个数有两个，输入个数也叫特指数或者特征向量维度
### 3.1.3实现线性回归

生成数据集：

```
from mxnet import autograd, nd
num_inputs = 2			#特征数
num_examples = 1000		#样本数量
true_w =[2,-3.4]		#真实的权重
true_b =4.2
#以均值为1的正态分布随机生成1000*2数据
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))	
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
labels +=  nd.random.normal(scale=0.01,shape=labels.shape)
features,labels
```

读取数据：

Gluon提供了data包来读取数据，每一次迭代随机读取包含10个数据样本的小批量

```
from mxnet.gluon import data as gdata
#数据样本大小
batch_size =10	
#把数据存入dataset
dataset = gdata.ArrayDataset(features,labels)		
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)
for X in data_iter:
    print(X)
    break
```

定义模型：

nn是neural networks神经网络的缩写,该模块定义了大量神经网络的层

```
from mxnet.gluon import nn
#sequential是一个存放神经网络层的容器
net = nn.Sequential()	
#这里不用定义这个层的输入节点是多少，只需定义输出节点
net.add(nn.Dense(1))

from mxnet import init
#初始化模型参数，权重参数（w）随机采样于均值为0标准差为0.01的正态分布，偏差参数（b）默认为0
net.initialize(init.Normal(sigma=0.01))	
```

定义损失函数：

```
from mxnet.gluon import loss as gloss
#平方损失函数
loss = gloss.L2Loss		
```

定义优化算法：

```
from mxnet import gluon
#指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})
```

训练模型：

```
num_epochs = 5
for epoch in range(1,num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l=loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l=loss(net(features),labels)
    print('epoch%d:,loss:%f'%(epoch,l.mean().asnumpy()))
```

### 3.2 softmax回归

和线性回归不同，softmax回归的输出是离散值，并且可以是多个，属于分类模型，也是单层神经网络。在本节有四种特征和三种输出类别：

![image-3.2](https://github.com/FZU-AI/AI_note/blob/master/woo/deep_learning/photo/3.2.png)

​		
$$
o1 = x1w11 + x2w21 + x3w31 + x4w41 + b1,\\
o2 = x1w12 + x2w22 + x3w32 + x4w42 + b2,\\
o3 = x1w13 + x2w23 + x3w33 + x4w43 + b3.
$$


#### 3.2.1  softmax运算

分类问题的输出是一个离散值，由于输出值的范围不确定，难以直观上判断这些值的意义，也难以衡量误差，因此引入了softmax运算符：
$$
y1, ˆ y2, ˆ y3 = softmax(o1,o2,o3),
$$
其中
$$
y1 =\frac{exp(o1)}{\sum_{i=1}^{3}{exp(oi)}},\\ 
y2 =\frac{exp(o2)}{\sum_{i=1}^{3}{exp(oi)}}, \\
y3 =\frac{exp(o3)}{\sum_{i=1}^{3}{exp(oi)}}.
$$
容易看出 
$$
y1 +  y2 +  y3 = 1 \quad且\quad 0 ≤  y1,  y2,  y3 ≤ 1，
$$
因此 y1,  y2,  y3是⼀个合法的概率分布，从而softmax运算将输出变换成一个合法的类别预测分布。

#### 3.2.2  图像分类数据集

获取数据集：

```
%matplotlib inline
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time
#通过data包下载这个数据集
mnist_train=gdata.vision.FashionMNIST(train=True)
mnist_test=gdata.vision.FashionMNIST(train=False)
#查看样本数
len(mnist_train),len(mnist_test)
#显示训练集前20个样本
X, y =mnist_train[0:20]
d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))
```

读取小批量：

```
batch_size=256
num-workers=0
transformer = gdata.vision.transforms.ToTensor()
train_iter=gdata.DataLoader(mnist_train.transform_first(transform),batch_size,shuffle=True,num-workers=num-workers)
test_iter=gdata.DataLoader(mnist_test.transform_first(transform),batch_size,shuffle=True,num-workers=num-workers)
```

#### 3.2.3  实现softmax回归

所需要的包：

```
%matplotlib inline
import d2lzh as d2l
from mxnet import gluon,init
from mxnet.gluon import loss as gloss,nn
```

读取数据：

```
#随机获取256个样本数据
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
```

初始化模型（函数）：

```
#存放神经网络层的容器
net=nn.Sequential()
#添加一个输出数为10的全连接层
net.add(nn.Dense(10))
#设置权重参数：均值=0，标准差=0.01
net.initialize(init.Normal(sigma=0.01))
```

损失函数：

```
loss=gloss.SoftmaxCrossEntropyLoss()
```

优化算法：

```
#使用学习率为0.1的小批量随机梯度下降算法
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
```

训练模型：

```
num_epochs=5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)
```

### 3.3  多层感知机

多层感知机在单层神经网络的基础上引入一到多个隐藏层，隐藏层位于输入层和输出层之间

![image-3.3](https://github.com/FZU-AI/AI_note/blob/master/woo/deep_learning/photo/3.3.png)

其输出O的计算为
$$
H = XWh + bh,\\
O = HWo + bo, 
$$
将上式联立可得
$$
O = (XWh + bh)Wo+bo = XWhWo + bhWo + bo.
$$
从联立后的式子可看出，虽然引入了隐藏层，却依然等价于一个单层神经网络。输出层权重为WhWo，偏差参数为bhWo+bo。更多隐藏层的神经网络也以此类推。
