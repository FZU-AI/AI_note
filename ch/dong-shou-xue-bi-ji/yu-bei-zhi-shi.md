# 预备知识

\[TOC\]

## 预备知识

### 配置环境

#### —window10

1. 下载anaconda和d2l-zh-1.0.zip
2. 打开anaconda powershell prompt
3. 配置清华_PyPI_镜像：pip config set global.index -url [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
4. 创建运行环境：conda env create -f environment.yml
5. 运行环境：conda activate gluon

#### —Linux

### NDArray数据操作

mxnet的基础是对数据进行操作，**NDArray**类似NumPy,是MXNet中用于数据储存和变换的主要工具

```python
#创建一个ndarray数组，⽤arange函数创建⼀个⾏向量
from mxnet import nd
x = nd.arange(12)
```

#### NDArray的创建方法包括：

* 通过python的list或者tuple直接指定值
* 通过arange创建一维行向量之后reshape修改形状
* 通过ones，zeros创建指定形状的张量
* 通过nd.random生成随机值的张量，例如nd.random.normal生成的是正态分布的随机数

#### NDArray的运算方法：

* ndarray的运算包括基本的加减乘初，不过要求参与运算的**两个对象大小相同**，
* ndarray提供dot函数计算张量的矩阵乘法。
* ndarray提供concat函数支持多个张量连结
* 条件运算==判断对应位置的值是否相等
* ndarray提供sum函数计算特定维度的和，没有指定维度则计算所有元素的和
* ndarray提供asscalar函数将结果变成标量

#### NDArray的广播机制：

广播机制针对两个形状不同的NDArray进行按元素进行运算。

**广播机制原则**

* 输出的shape是输入中最大的值，输入的shape中不足的部分都通过在前面加1补齐
* 输入数组要求shape上的每个位置要么是**1**要么和**输出一致**

#### NDArray索引:

通俗说：索引就是用冒号分隔开，对张量内元素进行定位查找或者修改。

```python
x[:2]
#输出
[0. 1. 2.]
<NDArray 3 @cpu(0)>
```

#### NDArray与NumPy转化:

通过**array函数**和**asnumpy函数**令数据在NDArray和NumPy格式之间相互变换。

### NDArray自动求梯度

步骤：

1. 调⽤attach\_grad函数来申请存储梯度所需要的内存
2. 调⽤record函数来要求MXNet记录与求梯度有关的计算
3. 通过调⽤backward函数⾃动求梯度

```python
from mxnet import autograd, nd
x = nd.arange(4).reshape((4, 1))
x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)
y.backward()
x.grad
#输出
[[ 0.]
 [ 4.]
 [ 8.]
 [12.]]
<NDArray 4x1 @cpu(0)>
```

