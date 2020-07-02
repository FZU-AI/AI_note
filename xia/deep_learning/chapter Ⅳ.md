# 深度学习计算
## 模型构造
### 1.1继承Block类来构造模型
Block类是nn模块⾥提供的⼀个模型构造类，可以继承它来定义想要的模型。  
这⾥定义的MLP类重载了Block类的__init__函 数和forward函数  
```python
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.output = nn.Dense(10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))
```
### 1.2Sequential类继承⾃Block类 
Block类是一个通用的部件。事实上，Sequential类继承自Block类 
Sequential类的目的：它提供add函数来逐一添加串联的Block子类实例，而模型的正向计算就是将这些实例按添加的顺序逐一计算。  
下面实现一个与Sequential类有相同功能的MySequential类
```python
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block是一个Block子类实例，假设它有一个独一无二的名字。我们将它保存在Block类的
        # 成员变量_children里，其类型是OrderedDict。当MySequential实例调用
        # initialize函数时，系统会自动对_children里所有成员初始化
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict保证会按照成员添加时的顺序遍历成员
        for block in self._children.values():
            x = block(x)
        return x
```
### 1.3构造复杂的模型
虽然Sequential类可以使模型构造更加简单，且不需要定义forward函数，但直接继承Block类可以极大地拓展模型构造的灵活性。  
下面构造一个网络FancyMLP。通过get_constant函数创建训练中的常数参数。在正向计算中，除了使用创建的常数参数外，还使用NDArray的函数和Python的控制流，并多次调用相同的层。
```python
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```
因为FancyMLP和Sequential类都是Block类的子类，所以可以嵌套调用。
```python
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())

net.initialize()
net(X)
```
## 模型参数的访问、初始化和共享
先定义一个与上一节中相同的含单隐藏层的多层感知机  
```python
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算
```
### 2.1访问模型参数
对于使用Sequential类构造的神经网络，可以通过方括号[]来访问网络的任一层。  
对于Sequential实例中含模型参数的层，可以通过Block类的params属性来访问该层包含的所有参数。  
```python
net[0].params, type(net[0].params)
```
Gluon里参数类型为Parameter类，它包含参数和梯度的数值，可以分别通过data函数和grad函数来访问。  
```python
net[0].weight.data()
```
可以使用collect_params函数来获取net变量所有嵌套（例如通过add函数嵌套）的层所包含的所有参数。  
```python
net.collect_params()
```
### 2.2 初始化模型参数 
MXNet的init模块  
将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。  
```python
# 非首次对模型初始化需要指定force_reinit为真
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```
下面使用常数来初始化权重参数。  
```python
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```
### 2.3⾃定义初始化⽅法 
令权重有一半概率初始化为0，有另一半概率初始化为[−10,−5]和[5,10]两个区间里均匀分布的随机数。  
```python
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```
还可以通过Parameter类的set_data函数来直接改写模型参数。  
例如，在下例中将隐藏层参数在现有的基础上加1。  
```python
net[0].weight.set_data(net[0].weight.data() + 1)
net[0].weight.data()[0]
```
### 2.4共享模型参数 
在构造层的时候指定使用特定的参数。如果不同层使用同一份参数，那么它们在前向计算和反向传播时都会共享相同的参数。  
在下面的例子里，让模型的第二隐藏层（shared变量）和第三隐藏层共享模型参数。  
```python
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
net(X)

net[1].weight.data()[0] == net[2].weight.data()[0]
```
## 模型参数的延后初始化
### 3.1延后初始化 
创建多层感知机，并使用MyInit实例来初始化模型参数。
```python
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())
```
### 3.2避免延后初始化 
如果系统在调用initialize函数时能够知道所有参数的形状，那么延后初始化就不会发生。  
第一种是对已初始化的模型重新初始化时。因为参数形状不会发生变化，所以系统能够立即进行重新初始化。  
```python
net.initialize(init=MyInit(), force_reinit=True)
```
第二种是在创建层的时候指定了它的输入个数，使系统不需要额外的信息来推测参数形状。  
通过in_units来指定每个全连接层的输入个数，使初始化能够在initialize函数被调用时立即发生。  
```python
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```
## 自定义层
本节将介绍如何使用NDArray来自定义一个Gluon的层，从而可以被重复调用。  
### 4.1不含模型参数的⾃定义层 
下面的CenteredLayer类通过继承Block类自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了forward函数里。这个层里不含模型参数。  
```python
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```
### 4.2含模型参数的⾃定义层 
在自定义含模型参数的层时，可以利用Block类自带的ParameterDict类型的成员变量params。  
它是一个由字符串类型的参数名字映射到Parameter类型的模型参数的字典。   
可以通过get函数从ParameterDict创建Parameter实例。  
```python
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```
实现一个含权重参数和偏差参数的全连接层。  
它使用ReLU函数作为激活函数。其中in_units和units分别代表输入个数和输出个数。  
```python
class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```
## 读取和存储
### 5.1读写NDArray 
可以直接使用save函数和load函数分别存储和读取NDArray
```python
from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)
```
将数据从存储的文件读回内存。  
```python
x2 = nd.load('x')
x2
```
还可以存储一列NDArray并读回内存。  
```python
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
(x2, y2)
```
可以存储并读取一个从字符串映射到NDArray的字典。
```python
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
mydict2
```
### 5.2读写Gluon模型的参数 
Gluon的Block类提供了save_parameters函数和load_parameters函数来读写模型参数。   
为了方便演示，首先创建一个多层感知机，并初始化  
```python
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
```
把该模型的参数存成文件，文件名为mlp.params
```python
filename = 'mlp.params'
net.save_parameters(filename)
```
再实例化一次定义好的多层感知机  
```python
net2 = MLP()
net2.load_parameters(filename)
```
## GPU计算
对复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效   
使用单块NVIDIA GPU来计算   
### 6.1计算设备 
MXNet可以指定用来存储和计算的设备   
```python
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

mx.cpu(), mx.gpu(), mx.gpu(1)
```
### 6.2NDArray的GPU计算
在默认情况下，NDArray存在内存上  
```python
x = nd.array([1, 2, 3])
x
```
可以通过NDArray的context属性来查看该NDArray所在的设备
```python
x.context
```
#### GPU上的存储 
有多种方法将NDArray存储在显存上  
可以在创建NDArray的时候通过ctx参数指定存储设备。   
```python
a = nd.array([1, 2, 3], ctx=mx.gpu())
a
```
也可以通过copyto函数和as_in_context函数在设备之间传输数据。  
```python
y = x.copyto(mx.gpu())
y
```
```python
z = x.as_in_context(mx.gpu())
z
```
如果源变量和目标变量的context一致，as_in_context函数使目标变量和源变量共享源变量的内存或显存。   
而copyto函数总是为目标变量开新的内存或显存。   
#### GPU上的计算
MXNet的计算会在数据的context属性所指定的设备上执行   
只需要事先将数据存储在显存上  
```python
(z + 2).exp() * y
```
### 6.3 Gluon的GPU计算 
同NDArray类似，Gluon的模型可以在初始化时通过ctx参数指定设备。
```python
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
```
