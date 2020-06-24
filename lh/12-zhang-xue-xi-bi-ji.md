# 1-2章学习笔记

## 第一章

> * 环境配置
> * 获取代码

### 环境

Ubuntu18.04 + Anaconda4.4.10 Windows10 + Anaconda4.4.10

### 运行工具

win10: jupyter notebook ubuntu: xshell

### 代码地址

[https://zh.d2l.ai/d2l-zh-1.0.zip](https://zh.d2l.ai/d2l-zh-1.0.zip)

## 第二章

学习如何使用**NDArray**对数据进行处理

> * 创建NDArray
> * NDArray运算
> * 索引
> * NDArray和NumPy相互变换

### 创建NDArray

```python
# 从MXNet导入ndarray模块
from mxnet import nd 
# 创建一个全为0的3x2矩阵x
x = nd.zeros((3,2))
# 创建一个全为1的3x2矩阵x
x = nd.ones((3,2))
# 获取矩阵元素的总数
num = x.size 
#行向量x的形状改为(1, 6)，也就是一个1行6列的矩阵，并记作y。除了 形状改变之外，X中的元素保持不变
y = x.reshape(1,6)
#也可写成x.reshape((-1, 6))或x.reshape((6, -1))。由于x的元素个数是已知的，这里的-1是能够通过元素个数和 其他维度的大小推断出来的
#随机生成NDArray中每个元素的值。下面我们创建一个形状为(3,4)的NDArray。它的每个元素都随机采样于均值为0、标准差为1的正态分布
nd.random.normal(0, 1, shape=(3, 4)) 
# Python的列表（list）指定需要创建的NDArray中每个元素的值
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

### NDArray运算

```python
# 使用dot函数做矩阵乘法。下面将X与Y的转置做矩阵乘法。由 于X是3行4列的矩阵，Y转置为4行3列的矩阵，因此两个矩阵相乘得到3行3列的矩阵
z = nd.dot(X, Y.T) 
#将多个NDArray连结（concatenate） 。下面分别在行上（维度0，即形状中的最左边元素）和列上（维度1， 即形状中左起第二个元素） 连结两个矩阵。可以看到，输出的第一个NDArray在维度0的⻓度（6）为两个输入矩阵在维度0的⻓度之和（3 + 3） ，而输出的第二个NDArray在维 度1的⻓度（8）为两个输入矩阵在维度1的⻓度之和（4 + 4） 
nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1) 
# 矩阵内所有元素求和
X.sum()
```

### 索引

NDArray的索引从0开始逐一递增， 依据左闭右开指定范围 X\[1:3\]取第1和第2行

```python
#为行索引为1的每一列元素重新赋值为1
X[1:2, :] = 1
#为第1行第2列元素赋值为9
X[1, 2] = 9
```

### NDArray和NumPy相互变换

```text
# np->nd
import numpy as np 
P = np.ones((2, 3)) 
D = nd.array(P) 
# nd->np
D.asnumpy()
```

