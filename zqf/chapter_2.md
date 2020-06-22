
## 数据操作
**使用MXNet中的NDArray类**

```python
from mxnet import nd //引入
```
**一些基本创建NDArray操作**
```python
x=nd.arange(12) //创建一个行向量，从0开始12个整数
x.shape  //通过shape属性 得到元素总数
x.size  //使⽤reshape函数把⾏向量x的形状改为(3, 4) 3⾏4列的矩阵
x.reshape((3, 4)) //改变x形状
nd.zeros((2, 3, 4)) //生成两个3x4的矩阵里面元素都是0
nd.ones((3, 4)) //生成一个3x4的矩阵里面元素都是1

Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) 
//通过Python的列表（list）指定需要创建的NDArray中每个元素的值

nd.random.normal(0, 1, shape=(3, 4)) 
//创建⼀个形状为(3,4)的NDArray。它的每个元素都随机采样于均值为0、标准差为1的正态分布
```
**运算**

```python
x+y //两个NDArray中的相应元素相加，注意x与y形状要一致。
//如果不一致，两个矩阵会被复制成一样的然后进行加法。
x*y //乘，同上
x/y //除，同上
X == Y //判断相应位置的数是否相等，相等为1，不同为0
y.exp() //按元素做指数运算

In [18]: X.norm().asscalar()
Out[18]: 22.494442  //不是矩阵了
//通过asscalar函数将结果变换为Python中的标量

X.sum() 
//对NDArray中的所有元素求和得到只有⼀个元素的NDArray

nd.dot(X, Y.T)  //Y.T是Y矩阵的转置矩阵
//⽤dot函数做矩阵乘法。将X与Y的转置做矩阵乘法

nd.concat(X, Y, dim=0) 
//把X Y矩阵拼在一起，dim=0竖着拼，dim=1横着拼。默认dim=1
```
**索引**

```python
X[1:3]  //指定了NDArray的⾏索引截取范围[1:3]，左闭右开
X[1, 2] = 9 //该第1行第2列 元素重新赋值。下标从0开始
X[1:2, :] = 12 //为⾏索引为1的每⼀列元素重新赋值
```
**NDArray和NumPy相互变换**
首先，引入包

```python
import numpy as np
```
NumPy实例变换成NDArray实例

```python
In : 
P = np.ones((2, 3))  //创建nunmpy实例
D = nd.array(P)		//将numpy实例转换为NDArray
D
Out:
[[1. 1. 1.]
[1. 1. 1.]]
<NDArray 2x3 @cpu(0)>
```
将NDArray实例变换成NumPy实例

```python
In : D.asnumpy() //将NDArray实例转换为numpy
Out: array([[1., 1., 1.],
[1., 1., 1.]], dtype=float32)
```
