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

