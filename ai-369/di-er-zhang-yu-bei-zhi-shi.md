# 第二章：预备知识

## 1 数据操作

在MXNet中，NDArray是⼀个类，也是存储和变换数据的主要⼯具。

### 1.1 创建NDArray

首先需要导入

`from mxnet import nd`

其中nd是ndarray的简写，如果使用

`from mxnet import ndarray`

相应后续调用时也要用ndarray，不能用简写。

`x=nd.arange(n)`创建一个大小为n的行向量，里面元素为0到n-1顺序排列

`x.shape`返回NDArray实例的行列大小，即几行几列

`x.size`返回元素的个数

`X = x.reshape((a, b))`将x改变形状，即变成a行b列，可省略一个括号，其中a和b其中一个可以用-1代替

当a和b都不省略时reshape函数改变后的元素数可以小于原来的元素数，但是不能大于原来的元素数

_-1_ 表示根据输入与输出中元素数量守恒的原则，根据已知的维度值，推导值为-1的位置的维度值，假设a用-1代替，而总元素数无法整除b时，就会取使a\*b小于原元素值的最大a值

`nd.zeros((a,b,c))`创建各元素为0，形状为\(a, b, c\)的张量

`nd.ones(a)`创建各元素为1的一维行向量

zeros和ones函数第一个参数都指的是形状，所以除非形状为一维，否则形状参数需要使用括号

`nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])`Python的列表（list）指定创建的NDArray中每个元素的值

nd.random.normal\(0, 1, shape=\(3, 4\)\)创建⼀个形状为\(3,4\)的NDArray，其中每个元素都随机采样于均值为0、标准差为1的正态分布，其中shape=可以省略

### 1.2 运算

直接用‘+’，‘-’，‘\*’，‘/’时，都是按元素进行，需要形状相同

exp函数为指数运算

dot函数为矩阵乘法，只能用nd调用，不能用nd对象调用

x.T得到的是x的转置

`nd.concat(X, Y, dim=0)`是将X和Y两个矩阵在维度0上合并

`nd.concat(X, Y, dim=1)`是将X和Y两个矩阵在维度1上合并

除合并维度的其他维度需要相同

`X==Y, X<Y, X>Y`条件判断式也是按元素进行的

`X.sum()`是对NDArray对象中的所有元素求和

asscalar函数将只有一个元素的NDArray对象变换为Python中的标量

norm函数计算NDArray对象的L2范数，即其中元素平方和开根号

函数exp，sum，norm可以直接用NDArray对象调用，不用参数，或者使用nd调用，参数为NDArray对象

### 1.3 索引

NDArray的索引（index）从0开始

可以使用X\[a: b\]提取X中第a行到b-1行的所有元素

X\[a:b, :\]中逗号后的‘:'代表该维度中的所有元素，

注意`nd.arange(n)`生成的NDArray对象是一维的

## 2 自动求梯度

MXNet提供的autograd模块来⾃动求梯度（gradient）

### 2.1 简单例子

如求函数

$$
y = 2x^⊤x
$$

求关于列向量x 的梯度

导入

`from mxnet import autograd,nd`

创建变量

`x = nd.arange(4).reshape((4, 1))`

申请存储梯度所需要的内存

`x.attach_grad()`

调⽤record函数来要求MXNet记录与求梯度有关的计算

`with autograd.record():`

​ `y = 2 * nd.dot(x.T, x)`

调⽤backward函数⾃动求梯度

`y.backward()`

`x.grad`为所求梯度

with

assert（断言）用于判断一个表达式，在表达式条件为 true 正常执行，条件为 false 的时候触发异常

### 2.2 训练模式和预测模式

`autograd.is_training()`可以查看模型是否在训练模式

### 2.3 查阅文档

dir函数获取模块中的所有成员

help函数查阅某个函数或者类的具体用法

在Jupyter记事本⾥，我们可以使⽤?来将⽂档显⽰在另外⼀个窗口中。例如，使⽤nd.random.uniform?将得到与help\(nd.random.uniform\)⼏乎⼀样的内容，但会显⽰在额外窗口⾥。此外，如果使⽤nd.random.uniform??，那么会额外显⽰该函数实现的代码。

也可访问MXNet⽹站[http://mxnet.apache.org/](http://mxnet.apache.org/) 查阅相关文档

