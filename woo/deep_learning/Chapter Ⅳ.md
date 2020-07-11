## 第四章

### 4.1  模型构造

Block是nn模块里提供的一个模型结构类，我们继承它并扩展我们想要的模型

```
from mxnet import nd
from mxnet.gluon import nn

#定义一个MLP类并继承于nn.Block
class MLP(nn.Block):

#自定义初始化函数
    def __init__(self,**kwargs):
    	#调用父类的初始化函数
        super(MLP,self).__init__(**kwargs)
        #定义成员变量
        self.hidden = nn.Dense(256,activation='relu')
        self.output = nn.Dense(10)
        
    def forward(self,x):
        return self.output(self.hidden(x))
```

### 4.2  模型参数的访问、初始化和共享

初始化一个模型并做前向计算，无需定义层的输入个数，容器会根据不同的X变化，这也会导致延后初始化

```
from mxnet import init,nd
from mxnet.gluon import nn

#new一个神经网络容器
net = nn.Sequential()
#加入第一层，256个输出作为下一层的输入
net.add(nn.Dense(256,activation='relu'))
#加入第二层，10个输出
net.add(nn.Dense(10))
net.initialize()


X=nd.random.uniform(shape=(2,20))

Y=net(X)

#查看各个层的形状
net.collect_params()

#以不同方式初始化模型参数
net.initialize(init=init.Normal(sigma=0.01),force_reinit=True)
```

自定义初始化方法：

```
class MyInit(init.Initializer): 
	def _init_weight(self, name, data): 
		print('Init', name, data.shape) 
		data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape) 
		data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True) 
```

### 4.3  使用GPU计算

#### 4.3.1  指定计算设备

```
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

#a默认为CPU
a=nd.array([1,2,3])
#b指定为GPU，可指定第i块GPU，默认为第0块
b=nd.array([1,2,3],ctx=mx.gpu(0))
a,b
```

Out:

```
(
 [1. 2. 3.]
 <NDArray 3 @cpu(0)>, 
 [1. 2. 3.]
 <NDArray 3 @gpu(0)>)
```

需要注意MXNT希望计算的输入数据都在内存或者同一块显卡的显存上

```
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
#若输入为a则会报错，因为a在内存中
net(b)
```

























